# SK 算子代码生成

`sk-operator-codegen` 负责算子源码形态识别、SK bind 适配、聚合目录生成，以及 standalone
compare 代码生成。端到端场景优先使用 `sk-operator-pipeline run-sk-pipeline`，本工具适合单独定位生成阶段问题。

## 支持的源码形态

| 输入形态 | 典型特征 | 处理结果 |
| --- | --- | --- |
| 当前 SK bind | 已包含当前框架需要的 bind 结构 | 复用并生成 manifest |
| 历史 `__spk__` | 旧版 SuperKernel 入口或绑定方式 | 迁移到当前 SK bind |
| 普通 `__global__` kernel | 只有 device kernel 或 host wrapper 不完整 | 生成 SK bind 封装 |
| 无法识别 | 缺少入口、规格或语义信息 | 生成诊断结果，交由人工处理 |

## 常用命令

识别单个算子形态：

```bash
python3 <skills_root>/sk-operator-codegen/scripts/operator_codegen.py detect-sk-form \
  $OPERATOR_ASSET \
  --output-dir build/examples/sk-codegen/detect
```

把普通 `__global__` 或历史形态适配为 SK bind：

```bash
python3 <skills_root>/sk-operator-codegen/scripts/operator_codegen.py adapt-sk-from-global \
  $OPERATOR_ASSET \
  --output-dir build/examples/sk-codegen/adapted/my_op \
  --io-contract operator-io-contract.json
```

`--io-contract` 是算子 IO 语义契约。固化脚本不会根据变量名猜测输入输出；当 kernel 有多个
tensor-like 参数时，必须由用户、adapter skill 或上游资产契约明确说明每个 tensor-like 参数属于
`inputs`、`outputs` 还是 `workspaces`，以及 pybind 单返回值应该返回哪个 tensor。最小格式：

```json
{
  "schema_version": 1,
  "entries": {
    "add_custom": {
      "inputs": ["x", "y"],
      "outputs": ["z"],
      "pybind_return_tensor": "z"
    }
  }
}
```

如果一个 entry 有多个 tensor-like 参数但没有匹配的 `--io-contract`，`adapt-sk-from-global`
会返回 `needs-human`，并在 `operator-sk-adapted.json` 中给出
`codegen.pybind-return-tensor-unresolved`。如果契约匹配但遗漏了某个 tensor-like 参数，会给出
`codegen.io-contract-tensor-incomplete`。struct-valued 运行时参数需要在
`parameters` 中声明，例如 `"tiling": {"kind": "host_struct"}`；`GM_ADDR workspace` 或
`GM_ADDR tiling` 这类地址参数仍应放入 `workspaces`，不要声明成 host struct。

### 图捕获前准备状态

有些算子需要在图捕获前准备运行状态，例如 TensorList descriptor、workspace tail
元数据或持久缓存。生成规则是：

- 这些状态必须来自显式 contract，不从源码变量名或参数顺序推断。
- runtime wrapper 只能消费已经准备好的状态。
- 不允许在 forward/capture 路径中生成隐藏 helper kernel 来临时准备状态。
- descriptor 顺序必须和 contract 中声明的 TensorList 参数顺序一致；不一致时返回
  `needs-human` 或生成错误。

这条规则适用于所有需要 prepared runtime state 的算子，不是某个样例的特殊逻辑。

聚合多个已适配算子：

```bash
python3 <skills_root>/sk-operator-codegen/scripts/operator_codegen.py aggregate-sk-adapted \
  --adapted-output-dir build/examples/sk-codegen/adapted/op_a \
  --adapted-output-dir build/examples/sk-codegen/adapted/op_b \
  --output-dir build/examples/sk-codegen/aggregate \
  --aggregate-wheel-name op_extension \
  --package-version 0.1.0
```

生成 standalone compare 工程：

```bash
python3 <skills_root>/sk-operator-codegen/scripts/operator_codegen.py generate-standalone-compare \
  build/examples/sk-codegen/aggregate \
  --output-dir build/examples/sk-codegen/standalone \
  --target-chip ascend-910b \
  --npu-arch dav-2201
```

standalone compare 必须有明确的 NPU arch。优先显式传 `--npu-arch`；未传时只会在
`--target-chip` 能通过官方来源映射到唯一 arch 时继续生成可编译 CMake，否则输出
`needs-target-arch`，不会静默回退到某个默认 arch。

生成的 ACLGraph wheel 包支持多芯片版本原生产物：

- 构建时优先读取 `SK_NPU_ARCHS`，可用逗号或分号传多个值，例如
  `SK_NPU_ARCHS=dav-2201,dav-3510`。
- wheel 内 native module 按 `op x arch` 拆分，例如
  `op_extension.add_custom_dav_2201.so` 和 `op_extension.mul_custom_dav_3510.so`；
  可用 `SK_BISHENG_JOBS` 或流水线 `--jobs` 控制并行 bisheng 编译数。
- 两者都未设置时，只使用有官方源码依据的当前环境检测。目前自动映射只覆盖
  `Ascend950*` / torch_npu SoC enum `260` 到 `dav-3510`；其他芯片不会静默
  fallback，需显式设置 `SK_NPU_ARCHS`。
- 运行时优先读取 `SK_ACLGRAPH_NPU_ARCH`；未设置时才尝试有来源依据的 SoC 自动映射。即使 wheel 里只有一个
  `.so`，也不会在无法确认目标架构时静默选择。

基于检查结果应用自动修复：

```bash
python3 <skills_root>/sk-operator-codegen/scripts/operator_codegen.py apply-remediation \
  build/examples/sk-codegen/adapted/my_op \
  build/examples/sk-codegen/spec/operator-validation-findings.json
```

查看模板能力：

```bash
python3 <skills_root>/sk-operator-codegen/scripts/operator_codegen.py list-templates
python3 <skills_root>/sk-operator-codegen/scripts/operator_codegen.py generate-from-template \
  TEMPLATE_ID \
  --param name=value \
  --output-dir build/examples/sk-codegen/template-out
```

## 输出约定

生成阶段通常产出：

- 适配后的源码目录。
- 描述算子、输入、输出、构建配置的 manifest。
- 聚合目录 `_aggregate`，供 pybind、wheel 和 standalone 阶段继续使用。
- 诊断 JSON，用于说明为什么某个算子不能自动生成。

在总流水线中，这些文件会落到：

```text
01-detect-form/<op>/{inputs,outputs}
02-adapt-sk-from-global/<op>/{inputs,outputs}
02-adapt-sk-from-global/_aggregate/{inputs,outputs}
```

## 何时直接运行本工具

- 只想确认一个算子是否能被识别。
- 生成阶段失败，需要单独重跑 `adapt-sk-from-global`。
- 想检查聚合后的目录是否满足后续 pybind/wheel 构建要求。
- 需要用 `apply-remediation` 对静态检查结果做自动修复尝试。

## 验证

```bash
python3 <skills_root>/sk-operator-codegen/scripts/operator_codegen.py --help
```
