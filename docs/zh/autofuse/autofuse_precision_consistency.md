# 关于自动融合模块精度一致性的说明

## 结论

**自动融合模块无法保证与 Eager 模式下手写算子的输出在二进制上完全一致**；但通过保守的精度策略，可保证**精度不差于 Eager 模式**。本文说明其原理与适用边界，并给出希望继续使用自动融合时的处理路径。

---

## 一、精度差异的根本来源

### 1.1 代码生成路径不同

- **Eager 模式**：每个算子由**人工预先编写**、编译为固定的二进制 kernel。
- **自动融合**：通过 codegen 技术在运行时（JIT）生成算子代码。即使**不做任何融合**，同一语义的算子，自动融合生成的代码与手写 kernel 也不是同一份源码。

**源码不同，是一切数值差异的起点**。

### 1.2 源码差异如何演变为数值差异

同一算子的两份不同源码，至少会在以下四个层面引入可观察的末位差异：

1. **升精度策略不同**
   为保持 fp16 下的数值稳定性，算子内部常把中间计算提升到 fp32。是否升、在哪一步升、何时降回，手写实现与 codegen 实现的选择不一定一致。
2. **指令选择不同**
   同一语义可以用不同指令实现。例如 `Tensor * Scalar`：
   - 先 broadcast scalar，再走标准乘法指令；
   - 直接使用 `muls` 类的标量乘指令。

   两者数学上等价，但在硬件上走的是不同的乘法单元，舍入行为不完全一致。
3. **切块策略与 Reduction 累加顺序不同**
   浮点加法**不满足结合律**：`(a+b)+c` 与 `a+(b+c)` 的结果在最低位通常不同。`reduce`、`matmul`、`layernorm` 等计算的归约顺序由切块策略和并行度决定，而切块策略本身在自动融合与手写算子之间**从硬件资源约束上不可能一致**。

   展开来说，在相同大小切块下，一个融合 kernel 内部要承担多个算子的中间计算，单步 UB 占用会高于单算子 kernel。而手写单算子为了降低搬运开销，往往会把切块尽量做大（用满 ub）——这种切块尺寸在融合场景下可能放不下。
4. **算法实现不同**
   同一数学函数可以用不同算法逼近。例如 `rsqrt`、`div` 常采用多轮迭代逼近实现，迭代轮次、初值选取不同，末位误差就不同。

### 1.3 小结

源码不同 → 在升精度、指令、累加序、算法等任一维度产生差异 → 最终输出在二进制位上不一致。这是**原理层面**的限制，不是可通过工程对齐消除的 bug。

---

## 二、可以保证二进制一致的场景

精度差异来源于**计算**。对**纯搬运类算子**（不做任何数学运算，只重排数据）：

- `broadcast`
- `concat` / `split`
- `transpose` / `reshape` / `slice`

这类算子理论上可以做到二进制一致。

但**这类场景的融合收益通常很小**。自动融合的收益主要来自：

- 多个计算算子融合（消除中间结果的读写）；
- 多个计算 + 单个搬运的融合（计算与访存 overlap）。

纯搬运融合在真实业务中占比低，因此"能保证一致的场景"与"值得融合的场景"交集很窄。

---

## 三、自动融合提供的保证

虽然与手写算子之间不保证二进制一致，自动融合在精度与可复现性两个方向上提供以下保证：

### 3.1 融合区域内部强制 fp32 累积

自动融合在生成的融合 kernel 内部，默认将所有中间计算统一提升到 fp32，只在融合块的输入/输出边界保留原 dtype。以损失部分计算性能的代价，换取更高的中间精度。

### 3.2 精度不差于 Eager 模式

Eager 模式下，每个算子独立执行，算子与算子的中间结果必须按 tensor 原 dtype（如 fp16/bf16）落盘，再由下一个算子读入。这意味着：

- **Eager 模式**：在**每个算子边界**都发生一次 fp16/bf16 截断；
- **自动融合**：把一串算子合并为一个 kernel，**只在融合块边界**截断，块内全程 fp32。

因此，自动融合的累积舍入误差上界 ≤ Eager 模式下对应算子链的累积误差。即，精度不低于 Eager 模式。

### 3.3 自身的确定性（可复现性）

> 本节讨论的是**自动融合自身**的输出确定性，**不涉及与 Eager 模式的对比**。与 Eager 的关系参见 §1、§2、§3.2。

自动融合的确定性可以分两层来看：

- **编译确定性**：相同的编译期输入 → 相同的 kernel 二进制。编译期输入包括模型结构、shape、dtype、自动融合的版本与编译选项。
- **执行确定性**：相同的 kernel 二进制 + 相同的运行期输入 → 相同的输出。运行期输入包括张量数值、芯片型号等。

两层叠加，即可得到端到端的保证：**编译期输入与运行期输入都不变时，多次运行的输出在二进制上完全一致**。

也就是说，§1.2 列举的差异来源（升精度策略、指令选择、切块/累加顺序、算法实现），在自动融合**自身的两次编译+运行之间是完全一致的**——它们只在"自动融合 vs 手写算子"之间产生分歧。自动融合本身不引入额外的不确定性。

---

## 四、PyTorch Inductor 的相关说明

PyTorch Inductor 社区不保证精度的二进制一致，其逻辑与自动融合一致。摘录原文如下：

### 4.1 PyTorch 文档《Numerical accuracy》

涉及浮点结合律与 bit-exact，原文：

> "PyTorch is not guaranteed to produce bitwise identical results for floating point computations that are mathematically identical."
>
> "floating point addition and multiplication are not associative, so the order of the operations affects the results."

来源：[https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)

### 4.2 Edward Yang《Ways to use torch.compile》（2024）

作者为 PyTorch 核心开发者，文章在 "Improve training efficiency on a small-medium scale" 一节中说明 `torch.compile` 与 eager 的数值关系：

> "Unfortunately, the compiler does not guarantee exact bitwise equivalence with eager code; we reserve the right to do things like select different matrix multiply algorithms with different numerics or eliminate unnecessary downcast/upcasts when fusing half precision compute together."

来源：[https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/](https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/)

### 4.3 PyTorch 文档《torch.compile Troubleshooting》

"Accuracy Debugging" 一节中关于下游编译器数值表现的说明：

> "the reason we need this is downstream compilers will codegen code whether it's Triton code or the C++ backend, the numerics from those downstream compilers can be different in subtle ways yet have dramatic impact on your training stability."

来源：[https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting_old.html](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting_old.html)

---

## 五、对照表

| 维度 | Eager 手写算子 | 自动融合 |
|------|---------------|----------|
| 源代码 | 人工固定 | JIT 生成 |
| 二进制一致性 | 基准 | **不保证**（计算类）/ 可一致（纯搬运类） |
| 中间计算精度 | 受限于算子间 dtype（常为 fp16） | 融合块内部 fp32 |
| 累积误差上界 | 基准 | **≤ Eager 基准** |
| 性能 | 基准 | 显著优于基准 |

---

## 六、使用建议：希望继续使用自动融合时的处理路径

若用户关注的是**数值正确性**（模型精度、收敛性、下游业务指标），自动融合在精度上无劣势、在性能上有显著收益，可直接启用，无需额外处理。

若用户有**与 Eager 严格对齐**的诉求，又希望继续使用自动融合的性能收益，是十分有挑战的，参考以下策略组合使用。

### 6.1 PyTorch-Inductor 场景：状态和使用建议

与 Eager 模式严格对齐涉及 Inductor 融合前的图处理与融合阶段两部分。Inductor 对多数相关流程未提供原生关闭开关，需通过 patch 或修改源码实现。

#### 融合前处理

Inductor 在自动融合前会执行以下可能引入数值差异的流程：

- **Inplace 算子函数化**：例如 `relu_` → `relu`，函数化后的 Kernel 实现与 Eager 可能存在差异。
- **Decompose**：将复杂算子分解为基础算子以扩大融合范围，与单算子实现相比算法有差异。
- **FX 图 Pass**：主要为 pattern 替换类融合 Pass，涉及 Kernel 替换。
- **前反向切图**：基于自动重计算的切图策略，重计算的融合 Kernel 与 Eager 模式可能存在差异。

建议措施：

1. **Inplace 算子函数化**为融合前提、非功能项，无法关闭。实践中建议仅实现 Inplace 版本，通过自动函数化减少 Kernel 差异带来的数值差异。
2. **Decompose**：Inductor 原生不提供关闭选项，需通过 patch 或修改源码实现。
3. **FX 图 Pass**：Inductor 不提供全部关闭的选项，且每个版本生效的 Pass 可能不同，需根据执行时实际情况依次关闭。
4. **前反向切图**：Inductor 使用最大流最小割算法切图，会引入重计算，可通过配置 `custom_partitioner_fn` 替换为无重计算的切图策略。

#### 自动融合

Inductor 融合阶段因重新生成执行 Kernel 引入数值差异，涉及的融合类型如下：

| 融合类型 | 影响二进制精度 | 数值差异来源 | 用户可配置 |
|----------|----------------|--------------|------------|
| Reduction 类 | 是 | 累加顺序不同 | 需 patch 或修改源码 |
| Pointwise 类 | 是 | 类型提升规则、指令映射、迭代算法差异 | 需 patch 或修改源码 |
| MM/FA 模板类 | 是 | 算法实现差异 | 需 patch 或修改源码 |

使用须知：

- 如追求与 Eager 严格对齐，需关闭上述所有融合类型。Inductor 未提供相应配置项，需通过 patch 或修改源码实现。
- 建议显式开启 `TORCHINDUCTOR_EMULATE_PRECISION_CASTS`，避免 Lowering 过程中的 Cast 消除优化引入数值误差。

### 6.2 TensorFlow 场景：状态和使用建议

#### 通用优化

TensorFlow 场景默认使用 GE 作为图编译器后端。GE 自带的融合 pass 与硬件无关优化 pass（例如常量折叠、等价公式变换）都会导致 kernel 源代码变化，使二进制精度一致无法保障。

建议措施：

1. 将图编译优化等级设为 O0（仅保留功能类优化）。
2. O0 等级下仍会执行功能相关与静态 shape 相关优化。可打开 DUMP 图开关观察 GE 中实际生效的 pass，对照 1.2 节判断对精度的影响；若有影响，可通过手工调整脚本规避相应优化（难度较高）。

#### 自动融合

自动融合目前支持以下类别算子的融合，默认策略与可控性如下：

| 融合类型 | 默认状态 | 影响二进制精度 | 用户可配置 |
|----------|----------|----------------|------------|
| elemwise（含 broadcast） | 开启 | 是 | 否 |
| reduce    | 关闭 | 是 | 是 |
| concat    | 关闭 | 否 | 是 |
| slice     | 关闭 | 否 | 是 |
| gather    | 关闭 | 否 | 是 |
| transpose | 关闭 | 否 | 是 |

默认关闭的融合可通过环境变量显式开启，以 concat 为例：

```shell
export AUTOFUSE_FLAGS="--enable_autofuse=true;--autofuse_enable_pass=concat"
```

使用须知：

- **elemwise 融合暂不支持关闭**。该类融合既是二进制精度差异的主要来源，也是融合收益的主要来源，关闭后整体收益将显著降低。如确有关闭需求，可提交 issue 反馈。
- **默认关闭的融合属于实验特性**，开启后可能出现功能异常或性能劣化，建议在目标网络上实测后再决定是否保留。

