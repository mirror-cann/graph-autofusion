# AFIR Codegen Demo — 三维 LoadAddStore 穿刺

autofuse 的 codegen 正在从字符串拼接迁移到 MLIR 基础设施。本 demo 用一条最小纵向切片
打通完整链路，并以真实用例 **LoadAddStore**（`out = x0 + x1`，float16，三维 shape）为主线，
在 **CPU 功能仿真**与**真机 910C** 上执行、数值全对。

```
语义图 ─①schedule─▶ FusedScheduledResult ─②桥接─▶ AFIR 方言 ─③lowering─▶ ascendc 方言 ─④翻译─▶ Ascend C 源码 ─⑤编译执行─▶ NPU
```

| 环节 | 吃进 | 吐出 | 代码 | 自研 |
|---|---|---|---|---|
| ① schedule | 语义图 HintGraph | `FusedScheduledResult`（调度轴 / position / tensor_id） | autofuse 既有 | 否 |
| ② 桥接 | `FusedScheduledResult` | AFIR module（`.mlir`） | `ScheduleToAfir/` + `AfirBuilder/` | **是** |
| ③ lowering | AFIR module | ascendc 队列 IR（tiling + 多核 + 尾块） | `AFIRToAscendCQueue/` | **是** |
| ④ 翻译 | ascendc IR | Ascend C 源码（C++） | PyAsc `translateToAscendC` | 否 |
| ⑤ 执行 | Ascend C 源码 | NPU 运行结果 | CANN 工具链 | 否 |

**术语**：MLIR 层没有名为 "ascir" 的方言，真实方言是 `ascendc` / `emitasc`（均来自 PyAsc）；
"ascir" 只是 PyAsc 的命名空间与工具名。本 demo 自研的高层方言称为 **AFIR**。

---

## 一句话现状

- **用例**：LoadAddStore，`out = in0 + in1`，f16，具体 shape `<100x16x8xf16>`（12800 elems）。
- **kernel 完全 shape-agnostic**：`tile_len` / `total` 是运行期 `TilingData` 字段，核数由 launch 的
  `blockDim` 决定。**同一个 `kernel.bin`** 换运行期参数即覆盖：对齐/非对齐尾块（DataCopyPad）、
  单核/多核（GetBlockIdx cyclic 分配）。
- **两级执行均已实测数值全对**：
  - CPU 功能仿真：`total` = 12800 / 12808 / 12840（50/51/51 tiles）全量 PASS。
  - 真机 910C（device 5）：`total`(12800/12808/12840) × `blockDim`(1/2/8) 共 **9 组合全 PASS**。
  - camodel 周期精确仿真：对齐用例（12800，1/2/8 核）PASS（camodel 极慢，尾块组合未逐一跑完）。

一键复现见下方“复现”。

---

## 主线用例 LoadAddStore

**代码路径**：`tests/ut/python/test_python_ascir.py::TestAutofuseLoadAddStore`。
计算 `y = x0 + x1`，f16，三维符号 shape `[100 + s0, s1, s2]`（`s0/s1/s2` 为 `create_size` 符号尺寸）：

```
Data(arg0) ─┐
            ├─▶ Load(load0) ─┐
Data(arg1) ─┤                ├─▶ Add(add) ─▶ Store(store) ─▶ Output(buf1)
            └─▶ Load(load1) ─┘
```

本 demo 的可执行切片用**具体化** shape `<100x16x8xf16>`（模拟符号 shape 在运行期定形），
lowering 时把三维**展平为一维** tiling（kernel 只吃运行期 flat `total`）。即“三维输入 + 一维
tiling”，非多维 tiling —— 这是当前边界，见文末“边界与后续”。

---

## pass 业务代码怎么设计的

业务代码（方言/桥接/lowering pass）已从 demo 目录挪到 `mlir/` 下的正式目录，demo 只留复现材料：

```
mlir/
├── dialect/AFIR/           AFIR 方言定义（.td + .cpp）
├── bridge/                 ② 桥接：FusedScheduledResult → AFIR
│   ├── ScheduleToAfir/       进程内适配器：真实 FusedScheduledResult → 中性结构体
│   └── AfirBuilder/          后端中性：中性结构体 → AFIR module（纯 MLIR，可独立编译测试）
├── conversion/             ③ lowering：AFIR → ascendc
│   ├── AFIRToAscendC/        最小基础版（仅 binary elewise → L3，讲概念）
│   └── AFIRToAscendCQueue/   队列忠实版（tiling + 多核 + 尾块，可上真机）★主
├── tools/
│   ├── af-afir-gen/          手工驱动：内置 KernelInfo 例子，无需真实后端也能出 AFIR
│   └── af-opt/               方言 opt 驱动（注册 pass）
└── demo/                   ★ 本用例复现材料
    ├── README.md              本文（介绍 + pass 设计 + 复现）
    ├── reproduce.sh           一键复现脚本（AFIR 由真实调度现场生成，见 ②）
    └── loadaddstore_3d/       版本化的 kernel wrapper / harness / launcher
```

### 设计主线：桥接两层解耦 + lowering 是“纯发射器”

**② 桥接 = `ScheduleToAfir` + `AfirBuilder` 两层**，故意拆开：

- `AfirBuilder`（`AfirBuilder.h/.cpp`）是**后端中性**的一半：只吃朴素结构体
  （`KernelInfo{inputs, outputs, graphs}`，其中 `NodeInfo{type, input_names, output, sched_axis}`、
  `TensorInfo{shape, dtype, tensor_id, position, axis_ids}`、`AxisInfo{id, name, type, size}`），
  用 OpBuilder 造 AFIR module。它只依赖 MLIR + AFIR 方言，**不碰任何 autofuse 后端类型**，
  所以能进 mlir/ 独立 build、能被 lit 单测。核心 `buildNodeOp` 按 `node.type`（"Load"/"Add"/
  "Store"/…）建对应 AFIR op，并把 `position/tensor_id` 挂进 `outputs` 属性；节点按拓扑序、
  用名字表把 `input_names` 解析成前序 op 的结果值，建立数据流。
- `ScheduleToAfir`（`ScheduleToAfir.h/.cpp`）是**进程内适配器**：遍历真实
  `ascir::FusedScheduledResult` 的图结构，填出上面那些中性结构体，再调 `BuildAfirModule`。**零序列化**。它挂在 `codegen.cpp` 的 codegen
  钩子上，`ENABLE_AUTOFUSE_MLIR` 时编入 codegen 库；`MaybeDumpAfirFromSchedule` 在设了
  `AF_MLIR_AFIR_DUMP_DIR` 时把 AFIR dump 到磁盘，默认关（不改 legacy 行为）。

这样切的意义：**“怎么把调度决策变成 MLIR”这段知识（AfirBuilder）与“autofuse 的后端数据结构”
（ScheduleToAfir）解耦**——codegen 迁移工程师要学的是前者，且前者可脱离全量 autofuse 编译/测试。

**③ lowering = `AFIRToAscendCQueue`（队列忠实版，主）**，核心设计是**读调度、不重造**：

- pass（`ConvertAFIRToAscendCQueuePass::lowerFunc`）把 AFIR func 重建成队列驱动的 ascendc IR：
  函数签名 tensor 参数 → GM `memref`（内存空间标记 22），追加 workspace + `TilingData` py_struct，
  附 `ascendc.aicore` / `cann.num_inputs` 属性。
- 每个 op 的 buffer 落哪个队列（VECIN/VECOUT/VECCALC）**读自 op 的 `outputs` 属性里 schedule 已定
  的 position**（`scheduledPosition`），lowering 不自己重新决定 placement —— 这印证设计文档的核心
  论点“**codegen 是上游决策的纯发射器**”。
- 带 `afir.sched_axis` 的真实调度 AFIR 走 **tiling-aware 路径**：发射一个 tile 循环，trip =
  `ceildiv(total, tile_len)`、每次 `actual = min(tile_len, total-offset)`（尾块自动收缩）、GM
  offset 全是从运行期 `TilingData` 读出的 SSA 值，**无静态 shape 常量**。多核用 cyclic 分配
  （`lb = GetBlockIdx()`, `step = GetBlockNum()`），单核 launch 自动退化为 `0..trip step 1`。
  非对齐尾块用 `DataCopyPad` + `DataCopyExtParams`（`types` 属性让 emitter 生成 `static_cast`）。
- **关键约束（曾是 bug，现已修）**：队列的 `InitBuffer`（UB 池线性分配）必须提到 tile 循环**外**
  一次性分配，循环内只 `AllocTensor` / `FreeTensor` 回收 depth-1 buffer。否则每 tile 泄漏一块 buffer，
  跑满 UB 池就在固定 tile 数后失败（CPU 功能仿真能抓到；真机因 UB 够大在这些 shape 下侥幸不炸）。

`AFIRToAscendC`（基础版）只把 binary elewise 降为 ascendc L3 op，用来讲最小转换概念，不上真机。

### 扩展一个新算子落在哪

| 目标 | 改动位置 |
|---|---|
| 新增 elementwise 算子（方言） | `dialect/AFIR/Ops/Basic/math.td` 加一行 `def`；`math.cpp` 补 verify |
| 桥接新算子类型 | `bridge/AfirBuilder/AfirBuilder.cpp:buildNodeOp` 加 `node.type` 分支（真实/手工两侧共用） |
| 队列版增 unary/binary | `conversion/AFIRToAscendCQueue/AFIRToAscendCQueue.cpp` 的 `l2Mnemonic`/`l2UnaryMnemonic` 加映射 |
| 队列版增 reduce 等复杂算子 | 参照 `lowerBroadcast`：`lowerFunc` 加分派；PyAsc 通用发射器表达不了就手写 `Basic/Vec*.cpp` |

方言与桥接是填充式扩展；主要工程量集中在 ③ lowering —— 每个算子族如何展开为 ascendc 队列序列。

---

## 复现

### 前置

- docker + 镜像 `autofuse-mlir-dev:arm64`（预编译 LLVM/MLIR 为 Linux aarch64，宿主 macOS 只编辑代码）。
- CANN 在 docker named volume `af-cann-910`（完整 9.1.0）。
- 真机额外需 real-npu 环境文件（SSH 连接信息，勿入库），**只碰 `/data/nyh`、只用 device 5**。

### 一键脚本 `reproduce.sh`

```bash
cd mlir/demo
./reproduce.sh build      # 构建 af-opt + ascir-translate（首次，数分钟）
./reproduce.sh sim        # 桥接产物→lowering→翻译→CPU 功能仿真，全量数值校验
./reproduce.sh camodel    # 同一 kernel.bin 跑 camodel 周期精确仿真
./reproduce.sh realnpu    # 真机 910C：total×blockDim 9 组合
./reproduce.sh all        # = build + pipeline + sim（默认）
```

脚本先把 `loadaddstore_3d/` 里版本化的源 stage 到 build 工作目录，再在容器内跑各级。

路径不写死：`SRC`（仓库根）从脚本自身位置推导，`BUILD` 默认 `$HOME/afmlir-build`。
都可用环境变量覆盖：

```bash
BUILD=/data/me/afbuild ./reproduce.sh sim          # 换持久 build 目录
IMAGE=my-mlir-dev:arm64 ./reproduce.sh build       # 换 docker 镜像
CANN_VOL=my-cann ./reproduce.sh sim                # 换 CANN 卷
REALNPU_ENV=/path/to/real-npu.local.env ./reproduce.sh realnpu   # 真机必需
```

`realnpu` 需 `REALNPU_ENV` 指向一个导出 SSH 连接信息的文件（`ASCEND_MLIR_CI_REMOTE` /
`_REMOTE_PORT` / `_SSH_USERNAME` / `_SSH_PASSWORD`）；该文件含密码，勿入库。

### 各级预期输出

**③④ pipeline**：`lowered.mlir` 含 `scf.for` / `ceildivsi` / `data_copy_pad`；`kbody.inc` 里
`InitBuffer` × 3 全在 `for` **之前**，循环内有 `FreeTensor` × 3（这就是 InitBuffer-in-loop 修复的证据）。

**CPU 功能仿真**：
```
total=12800 (tiles=50): [RESULT] PASS all 12800 elements (out = in0 + in1)
total=12808 (tiles=51): [RESULT] PASS all 12808 elements (out = in0 + in1)
total=12840 (tiles=51): [RESULT] PASS all 12840 elements (out = in0 + in1)
```

**真机 910C**（device 5，同一 kernel.bin，仅运行期参数不同）：
```
aclInit rc=0 / aclrtSetDevice(5) rc=0 / rtDevBinaryRegister rc=0 / rtFunctionRegister rc=0
  total=12800 tail=0   blockDim=1  errors=0/12800 -> PASS
  ... (9 组合) ...
  total=12840 tail=40  blockDim=8  errors=0/12840 -> PASS
==================== ALL PASS (one kernel: 3 totals x 3 blockDims, tail + multi-core) ====================
```

| 维度 | 取值 | 覆盖 |
|---|---|---|
| `total`（尾块 = `total%256`） | 12800(0) / 12808(8) / 12840(40) | 对齐 + 两种非 32B 对齐尾块（DataCopyPad） |
| `blockDim`（核数） | 1 / 2 / 8 | 单核 + 多核 cyclic 分配（GetBlockIdx/GetBlockNum） |

### 三级执行环境（对照）

| 级别 | 编译 | 执行 | 用途 |
|---|---|---|---|
| CPU 功能仿真 | `ccec --run-mode=cpu`（`-D_GLIBCXX_USE_CXX11_ABI=0` 对齐 cpudebug 旧 ABI） | `libpem_davinci` | 快速验证数值逻辑 |
| camodel 周期精确 | `bisheng -x cce --cce-aicore-arch=dav-c220-vec` | rt* 直调 `libruntime_camodel` | 指令级仿真 |
| 真机 910C | 同上（device binary 通用） | ACL `aclInit` + rt* 直调真机 libruntime | 真实硬件 |

要点：device 二进制由 `bisheng` 编 + `ld.lld -m aicorelinux` 链为 `kernel.bin`（aicore ELF），
camodel 与真机**共用同一二进制**，仅运行时库与是否经 ACL 建会话不同（`launcher.cpp` 的
`#ifdef AF_REAL_NPU`；runtime 路径由 argv 传入，不写死某个 CANN 版本）。

### 复现材料清单（`loadaddstore_3d/`）

| 文件 | 作用 |
|---|---|
| `kernel_wrapper.cpp` | device 编译入口：定义 `TilingData{tile_len,total}` 后 `#include "kbody.inc"` |
| `harness_cpusim.cpp` | CPU 功能仿真 main：填 f16 输入、`ICPU_RUN_KF` 调 kernel、全量容差校验 |
| `launcher.cpp` | camodel/真机通用 launcher：runtime 路径走 argv，`AF_REAL_NPU` 决定是否经 ACL |

③ 的输入（AFIR）不再版本化在本目录，改由 `reproduce.sh` 的 stage 阶段跑真实调度现场 dump
（动态 shape `<?x?x?xf16>`，见 ② 桥接）。

---

## 边界与后续

- **一维 tiling**：当前把三维 shape 展平成一维 tiling（kernel 吃 flat `total`）。真正多维 tiling
  需 `AFIRToAscendCQueue` 支持多维 stride/offset，或在 AFIR 层先 reshape 为一维再 lower。
- **算子范围**：队列版覆盖 load/store + unary/binary elementwise（+ broadcast 手写发射器）；
  reduce 队列下降尚未实现（AFIR 层已能表达）。
- **符号 shape 上机**：符号 `[100+s0,s1,s2]` 不直接上机，需具体化为运行期尺寸——这是设计选择，
  非技术限制（kernel 本就 shape-agnostic，只是 launcher 要给定 `total`）。
- **tiling 写死**：demo launcher 手工给 `TilingData`，未接 ATT host tiling。

## 附：单命令速查（脚本背后的原始命令）

```bash
# 别名（容器内）
AFOPT=/build/b/mlir/tools/af-opt/af-opt ; TR=/build/b/bin/ascir-translate
# ③ lowering + ④ 翻译（afir_LoadAddStore_3D.mlir 由 ② 现场 dump，见下）
$AFOPT --convert-afir-to-ascendc-queue afir_LoadAddStore_3D.mlir > lowered.mlir
$TR -mlir-to-ascendc lowered.mlir > kbody.inc
# ② 桥接（真实 schedule，进程内直连；需完整构建容器 + pyautofuse.so + CANN）
AF_MLIR_AFIR_DUMP_DIR=/tmp/afir_dump \
  pytest tests/ut/python/test_python_ascir.py::TestAutofuseLoadAddStore::test_codegen
# ② 桥接（手工 case，无需真实后端）
/build/b/mlir/tools/af-afir-gen/af-afir-gen
# 方言 roundtrip
$AFOPT mlir/test/demo/afir_ops_roundtrip.mlir
```
