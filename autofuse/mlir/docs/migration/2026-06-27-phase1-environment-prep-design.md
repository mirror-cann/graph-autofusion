# Phase 1 环境准备工作方案

> 状态：Closed（2026-07-04，Phase 1 已完成）
> 日期：2026-06-27
> 范围：Phase 1「正式方案准备」中的环境准备工作：工程集成 MLIR、测试工具适配、回归基线冻结
> 关联：主设计 `2026-06-25-codegen-mlir-migration-design.md` 第 6.2 节（Stage 1 基础设施前准备）。主设计的 Stage 0-6 是 tracker Phase 2（Codegen 交付）的内部实施阶段，和本文 Phase 1 不是同一层概念。
> 参考：mlir-af（`/Users/neo/Code/Ascend-MLIR-gsr`，分支 `dev-gaoshuer`）作为可行性样本，只参考集成形态，不直接照搬代码。

---

## 0. 结论先行

Phase 1 的目标是把 MLIR/PyAsc 基础设施安全接入 Autofuse，并建立后续 Codegen/ATT/Schedule 迁移可复用的验证网。本阶段不替换现有业务路径，不改变 `--no-autofuse` 语义。

2026-07-04 关闭结论：Phase 1 按上述边界完成。下一阶段进入 Phase 2 Codegen 模块交付；真实 NPU 链路未打通和 legacy CANN cpudebug 兼容问题已在验证矩阵中分层记录，不作为 Phase 1 关闭阻塞项。

Phase 1 必选交付：

- **104 工程集成 MLIR**：默认关闭，显式打开后可编译 MLIR/PyAsc，提供最小 `af-opt` 冒烟和 op 覆盖盘点。
- **105 测试工具适配**：继续以 `tests/st/codegen/ascir_tool` 作为现有 E2E 基线，补齐 MLIR 路径 dump/diff/NPU 实测/精度校验。
- **107 回归基线冻结**：冻结现有 legacy codegen 行为基线，定义等价判据和一键复现方式。

任务关系：

```text
104 工程集成 MLIR
  ├─ 105 测试工具适配：复用 ascir_tool，补 MLIR dump/diff/NPU 实测
  └─ 107 回归基线冻结：为 legacy/MLIR 双路判等价提供基准
```

核心原则：

- 默认主线稳定优先：`BUILD_AUTOFUSE=ON, ENABLE_AUTOFUSE_MLIR=OFF, AF_MLIR_CODEGEN=off` 必须不发现、不依赖 LLVM/MLIR/PyAsc。
- `--no-autofuse` 已有语义不变：仍表示跳过整个 Autofuse 后端构建，不复用为 MLIR 开关。
- 构建能力和运行路由分离：`ENABLE_AUTOFUSE_MLIR` 控制是否编译 MLIR 能力；`AF_MLIR_CODEGEN=off|on|compare` 控制运行时是否走 MLIR codegen。
- `ascir_tool` 是当前真实 NPU E2E 基线。
- 默认 run 包不得安装 MLIR 工具或 LLVM/MLIR/PyAsc 运行时依赖。
- 验证入口统一见 `autofuse/mlir/docs/verification_matrix.md`；构建命令使用 `bash build.sh ... -j 8`，Docker 编译环境显式挂载 CANN volume。

---

## 1. 现状基线

### 1.1 Autofuse 现状

- 构建入口：上层 `build.sh` 支持 `--pkg`、`--no-autofuse`、`--module=<name>`、`-j`、`--build-type`、`--cann_3rd_lib_path`、`--output_path`。仓库要求构建限制并行度，优先 `-j 8`。
- 顶层 CMake：当前 `autofuse/CMakeLists.txt` 只有 Python3 和现有子目录集成，无 `find_package(MLIR)`、无 `externals/`、无 MLIR 相关目标。
- 单 kernel E2E 工具：`tests/st/codegen/ascir_tool` 已覆盖三种模式：
  - mode 0：`input_ascir.py` 构图 -> codegen -> 编译 -> 按 `ascir.json` launch -> `gen_input.py` 生成输入和 golden -> 输出比对。
  - mode 1：已有 host/device 代码编译、执行、比对。
  - mode 2：已有 device 代码仅编译。
  - 支持 `--prof` profiling。
- runtime stub：`tests/depends/runtime` 提供 `autofuse_runtime_stub`，用于测试隔离，不是生产 runtime。
- 现有 codegen 测试：`tests/{ut,st}/codegen/`、`tests/v35/{ut,st}/`、`tests/st/codegen/e2e/` 等。

### 1.2 mlir-af 对照样本

- LLVM/MLIR：mlir-af 脚本 pin 到 LLVM 21.1.8 commit `2078da43...`，CMake flags 包括 `LLVM_ENABLE_PROJECTS=mlir`、`LLVM_TARGETS_TO_BUILD=host`、`LLVM_ENABLE_RTTI=ON`、`MLIR_ENABLE_BINDINGS_PYTHON=ON`。源码构建产物约 5-10GB，编译耗时约 20-40min。
- CMake 集成：通过 `-DLLVM_BUILD_DIR` 推导 `MLIR_DIR`，校验 `MLIRConfig.cmake`，再 `find_package(MLIR REQUIRED CONFIG)`，并加载 `TableGen/AddLLVM/AddMLIR/HandleLLVMOptions`。
- PyAsc：`externals/pyasc` 复用主工程 `MLIR_DIR`，通过 `lib/TableGen`、`include`、`lib`、`bin` 四个 `add_subdirectory(... EXCLUDE_FROM_ALL)` 接入。实际目标包括 `MLIRAsc`、`MLIREmitAsc`、`MLIRTargetAsc`、`MLIRAscTransforms` 等。
- stablehlo：mlir-af 主路径未实际依赖。Autofuse Phase 1 不引入 stablehlo。
- 工具入口：`tools/afir-opt` 和 `tools/afir-translate` 可作为 `af-opt` / translate 工具形态参考。

---

## 2. Phase 1 边界

### 2.1 必须做到

- 默认构建不依赖 LLVM/MLIR/PyAsc。
- MLIR 构建通过显式开关进入，并能复用预编译依赖，避免每个开发者重复编 LLVM。
- `ascir_tool` 现有 mode 0/1/2 能力不被破坏。
- MLIR 路径的 dump/diff/NPU 实测结果可归档、可复现。
- legacy codegen golden 和等价判据可支撑后续 MLIR codegen 阶段性里程碑的整路径双路 A/B 验证。

### 2.2 明确不做

- 不改变 `--no-autofuse` 含义。
- 不默认启用 MLIR codegen。
- 不把大体积生成物、临时 `kernel_meta`、完整 generated code、coverage 输出直接入库。

### 2.3 依赖分发原则

LLVM/MLIR/PyAsc 依赖只在 `ENABLE_AUTOFUSE_MLIR=ON` 时解析。解析顺序建议如下：

```text
-DLLVM_BUILD_DIR 显式指定
-> AF_LLVM_ROOT 环境变量
-> 团队/公司预编译依赖目录
-> scripts/prepare_mlir_deps.sh 显式准备命令
-> 清晰 fatal error + 安装指引
```

预编译制品建议同时覆盖 x86_64 和 aarch64。每个制品必须带 manifest，至少记录：

- 架构、OS、工具链版本、编译器版本。
- `_GLIBCXX_USE_CXX11_ABI`，需与 Autofuse 当前 `_GLIBCXX_USE_CXX11_ABI=0` 保持一致。
- LLVM commit、CMake flags、PyAsc commit。
- checksum、是否启用 Python bindings/assertions。

禁止默认联网 clone 或静默编译 LLVM；需要下载或编译时必须由开发者显式触发。

---

## 3. 104 工程集成 MLIR（必选）

### 3.1 目标

让 MLIR/PyAsc 在 Autofuse 构建体系中可选编译通过，建立最小 MLIR pipeline 触发入口，并确认 PyAsc ascendc 方言覆盖度。不改现有 codegen 行为。

### 3.2 子任务

**T0 引入构建开关**

- 新增 CMake 开关 `ENABLE_AUTOFUSE_MLIR`，默认 `OFF`。
- `ENABLE_AUTOFUSE_MLIR` 与 `BUILD_AUTOFUSE` 解耦。
- 三条路径必须同时成立：

```text
BUILD_AUTOFUSE=ON,  ENABLE_AUTOFUSE_MLIR=OFF  -> 默认 legacy Autofuse 构建，不依赖 LLVM
BUILD_AUTOFUSE=ON,  ENABLE_AUTOFUSE_MLIR=ON   -> MLIR 开发构建，发现并链接 LLVM/MLIR/PyAsc
BUILD_AUTOFUSE=OFF                            -> 既有 --no-autofuse 行为
```

**T1 接入 LLVM/MLIR 发现逻辑**

- 仅在 `ENABLE_AUTOFUSE_MLIR=ON` 分支内执行 `find_package(MLIR REQUIRED CONFIG)`。
- 复用 mlir-af 的 `LLVM_BUILD_DIR -> MLIR_DIR` 发现思路，但嵌入 Autofuse 既有 `build.sh` / CMake 体系。
- `OFF` 分支不得出现 MLIR include、link、TableGen、AddMLIR 等逻辑。

**T2 接入 PyAsc**

- 新建 `autofuse/externals/`，引入 `externals/pyasc`。
- PyAsc 复用主工程 `MLIR_DIR`，不重复发现或编译 LLVM。
- 只在 `ENABLE_AUTOFUSE_MLIR=ON` 分支内添加 PyAsc 子目录。
- 不引入 stablehlo。

**T3 新增最小工具入口**

- 新增 `tools/af-opt/`，参考 `afir-opt` 的 `MlirOptMain` 模板。
- Phase 1 只要求 parse/print 和 dialect 注册，不接业务 pass。
- `af-opt` 不安装进默认 run 包。

**T4 链接 codegen 相关 MLIR 库**

- 在 MLIR 开关打开时，按需链接 `MLIRAsc`、`MLIREmitAsc`、`MLIRTargetAsc`、`MLIRSCF`、`MLIRArith`、`MLIRMath`、`MLIRMemRef` 等目标。
- 所有新增链接必须隔离在 `ENABLE_AUTOFUSE_MLIR=ON` 分支。

**T5 最小冒烟和 lit suite**

- 新增 `autofuse/test/` lit suite。
- 最小用例：`af-opt` 能 parse/print 一个含 `ascendc` op 或最小 `func.func` 的 MLIR 文件。
- 新增 `check-autofuse-mlir` 或同等 target，仅在 `ENABLE_AUTOFUSE_MLIR=ON` 时可用。

**T6 PyAsc ascendc 覆盖度盘点**

- 盘点 Autofuse 算子族和 PyAsc ascendc op 的覆盖关系：elewise、reduce、brc、concat、gather、transpose、datacopy、cube 等。
- 产出覆盖矩阵：已覆盖、部分覆盖、缺失。
- 产出缺口补法决策：
  - 向 PyAsc 补 op。
  - Autofuse 用 `codegen.*` 自定义属性或已有 op 组合补。
  - 暂时保留 legacy 双路。
- 同步确认 emitasc 对 TPipe/TQue/TBuf、tiling data、infer shape 等状态的承载边界。

### 3.3 验收标准

- 默认路径 `ENABLE_AUTOFUSE_MLIR=OFF` 构建通过，且无 LLVM/MLIR/PyAsc 环境也能编译。
- `ENABLE_AUTOFUSE_MLIR=ON` 路径 `bash build.sh --pkg --enable-mlir -j 8` 可编译通过。
- `--no-autofuse` 行为不变。
- `af-opt` 最小 parse/print 冒烟通过。
- 现有 UT/ST 通过，现有 codegen 行为不变。
- 预编译依赖的来源、版本、路径、ABI manifest 写入文档。
- 覆盖矩阵、缺口清单、首条纵向切片选择结论完成评审。

### 3.4 风险与缓解

- 默认构建被 MLIR 污染：所有 MLIR 逻辑必须在 `ENABLE_AUTOFUSE_MLIR=ON` 分支内；每步后验证默认构建和 `--no-autofuse`。
- 预编译依赖 ABI 不匹配：manifest 强制记录 `_GLIBCXX_USE_CXX11_ABI`、编译器和 LLVM/PyAsc commit。
- LLVM 编译耗时/OOM：统一使用 `-j 8`，并优先复用预编译 x86/aarch64 制品。
- PyAsc 版本不匹配：先单独打通 PyAsc 探针，必要时锁定 commit。

---

## 4. 105 测试工具适配（必选）

### 4.1 目标

在 104 构建产物基础上，复用 `ascir_tool` 作为真实 NPU E2E 基线，补齐 MLIR 路径的结构化 dump、diff、NPU 实测和精度校验能力。

### 4.2 子任务

**T1 MLIR 路径 artifact dump**

- 复用 `ascir_tool` 的 dump 目录规范。
- 新增 MLIR 路径 dump：MLIR region、tiling schema、CANN C++ 串、manifest。
- dump 产物进入忽略目录或 CI artifact，不直接污染源码树。

**T2 结构化 diff**

- `diff-kernel-abi`：对比 kernel signature、输入输出、workspace、tiling data ABI。
- `diff-tiling-schema`：对比 tiling fields、field order、shape source、block dim。
- `diff-fused-result`：对比 schedule group、impl graph，为后续 ATT/Schedule 预留。
- diff 判据只断言关键语义，不要求生成文本逐字符一致。

**T3 结构化输出边界**

- 105 只保留结构化输入边界：`artifact + manifest + npy/golden + tiling`。

**T4 NPU 实测和精度校验**

- 继续复用 `ascir_tool` mode 0 的 launch、`gen_input.py`、`verify_result.py`。
- 适配 MLIR 路径产出的 kernel 或后续 CodegenResult dump。
- profiling、coverage、kernel_meta 等输出进入忽略目录或 CI artifact。

**T5 一键化入口**

- 提供与 `test_ascir.sh` 风格一致的一键入口。
- 典型 case 一键产出 dump/diff/NPU 实测/精度报告。
- `case-minimizer` 作为增强项：失败时缩小到单 graph/task/schedule group。

### 4.3 验收标准

- 典型 case（elewise+brc）一键生成 dump/diff/NPU 实测/精度校验报告。
- diff 工具能判断 legacy vs MLIR 的 ABI/tiling 关键语义是否一致。
- `ascir_tool` 现有 mode 0/1/2 能力不被破坏。

### 4.4 风险与缓解

- MLIR kernel 尚未生成：Phase 1 可先用 legacy kernel 跑通 dump/diff/归档链路，MLIR kernel 待 Phase 2 接入。
- diff 判据过严：先固定关键语义字段，数值和性能等价交给 E2E 与 107 判据。
- 结果目录污染源码树：所有大体积生成物放入忽略目录或 CI artifact。

---

## 5. 107 回归基线与等价判据（必选）

### 5.1 目标

冻结现有 legacy codegen 行为基线，定义后续 MLIR/legacy 双路 A/B 的等价判据和 CI 复现方式。107 是迁移安全网。

### 5.2 子任务

**T1 用例盘点**

- 盘点现状必跑集：`tests/st/codegen`、`tests/v35/st`、`tests/ut/codegen`、`tests/st/codegen/e2e`、`ascir_tool` testcase。
- 标记每类用例覆盖的算子族、tiling 形态、shape 特征、NPU 运行要求。

**T2 golden 冻结**

- 冻结结构化语义 manifest、小型必要样例和等价判据输入。
- 不直接提交大体积 generated code、临时 `kernel_meta`、完整 coverage、profiling 输出。
- 大体积产物通过忽略目录、CI artifact 或外部归档保存。

**T3 等价判据定义**

- 定义数值容差、性能不劣化阈值、必跑集分层。
- 明确哪些差异允许存在，例如变量名、临时 buffer 名、非语义代码顺序。
- 明确哪些差异必须阻断，例如 ABI、tiling field 顺序、workspace、shape 推导、输出 dtype/shape。

**T4 一键复现和 CI 接入**

- 提供一键复现脚本。
- CI 至少覆盖默认 legacy 路径；MLIR 开关路径可作为显式开发/预合入门禁。
- 比对脚本可被 105 复用。

### 5.3 验收标准

- 必跑用例清单和覆盖矩阵完成。
- golden 存储策略符合红线：结构化、小型、可复现；大体积产物不入库。
- 等价判据文档化。
- 一键复现脚本可运行，CI 接入策略明确。

### 5.4 风险与缓解

- 判据过严导致迁移卡死：按场景分级容差，边界场景单独登记。
- golden 漂移：Phase 1 冻结后，Phase 2 在 MLIR codegen 可运行里程碑上对代表性 case 集做整路径 A/B，比对发现漂移即停。
- golden 入库过重：只入结构化 manifest 和小型样例，大文件走 CI artifact。

---

## 6. Phase 1 出口门禁

| 类别 | 必选/可选 | 交付物 | 出口门禁 |
|---|---|---|---|
| 104 工程集成 MLIR | 必选 | `ENABLE_AUTOFUSE_MLIR`、MLIR/PyAsc 集成、`af-opt`、lit 冒烟、覆盖矩阵、预编译依赖 manifest | 默认构建不依赖 LLVM；MLIR 开关打开可 `bash build.sh ... -j 8` 编过；`--no-autofuse` 不变；现有 UT/ST 通过 |
| 105 测试工具适配 | 必选 | `ascir_tool` 兼容的 MLIR dump/diff/NPU 实测/精度报告链路 | 典型 case 一键报告；diff 可判 ABI/tiling；既有 `ascir_tool` 回归通过 |
| 107 回归基线 | 必选 | 必跑集、结构化 golden、等价判据、一键复现/CI 策略 | golden 可复现；判据文档化；大体积生成物不入库 |

Phase 1 整体出口：Autofuse 具备默认关闭的 MLIR 构建能力，具备基于 `ascir_tool` 的端到端验证网，具备 legacy 行为基线和等价判据，可支撑 Phase 2 Codegen 迁移在可运行里程碑上的整路径双路 A/B 验证。

2026-07-04 关闭状态：上述出口按 Phase 1 范围关闭。`verification_matrix.md` 记录的 `isinf_maskedfill_test_e2e` CANN cpudebug `vabs(int*)` 兼容问题和真实 NPU 链路不通，移入 Phase 2/后续真实链路验证继续处理。

---

## 7. 与主设计和 tracker 的同步

- 主设计 `2026-06-25-codegen-mlir-migration-design.md` 第 6.2 节应引用本文作为 Phase 1 环境准备方案。
- 入库权威状态以本文和 `2026-06-29-phase1-environment-prep-development-plan.md` 为准：Phase 1 已完成，Phase 2 Codegen 模块交付作为下一阶段。若本地周报 tracker 同步 Phase 1 状态，只保留环境底座、LLVM/MLIR 预编译依赖准备和 task 104/105/107：
  - 开发/编译验证环境准备：Docker 编译验证环境、CANN volume、环境 manifest。
  - LLVM/MLIR 预编译依赖准备：显式依赖输入、manifest、禁止默认路径静默下载/编译。
  - 104：工程集成 MLIR，默认关闭，预编译依赖和覆盖矩阵是出口条件。
  - 105：测试工具适配，复用 `ascir_tool`。
  - 107：回归基线和等价判据。
