# PyAsc AscendC 算子覆盖矩阵

## 范围

本矩阵记录 PyAsc `ascendc` / `emitasc` 是否能表达 Phase 2 需要迁移的 Autofuse codegen 算子族。
当前结论基于 PyAsc TableGen `.td` 源文件提取出的 op inventory；具体语义、tiling、shape、
buffer 生命周期和代码生成稳定性仍需在后续纵向切片中验证。

## 扫描原理

`scan_pyasc_coverage.py` 递归扫描 PyAsc 的 `include/ascir/Dialect/Asc/IR/**/*.td` 和
`include/ascir/Dialect/EmitAsc/IR/**/*.td`：

- 对显式 `def ... : SomeOp<"..."> { let arguments/results = ... }`，直接提取 op 名、参数、结果和 TableGen 类型约束。
- 对 `def ... : SomeBaseOp<"...">;`，从对应 `class SomeBaseOp` 继承 `arguments/results`。
- 对 PyAsc 常见的 `defm` / `multiclass`，做轻量展开，覆盖 `baseMnemonic # "_l0"` 这类字符串拼接，生成 `ascendc.add_l0` 等 op。
- 输出中的 `constraint` / `type_hints` 是 MLIR TableGen 类型约束，不等同于运行时 dtype 白名单。

脚本不做完整 TableGen 语义解释；如果后续遇到复杂条件、复杂 DAG 或 generated `.inc` 才能确定的信息，需要再接入
`mlir-tblgen` 生成物做交叉校验。

## 初始矩阵

| Autofuse 算子族 | PyAsc 支持情况 | 决策 | 说明 |
|---|---|---|---|
| elewise | 当前扫描提取 132 个相关 op | 作为首条候选切片 | 覆盖首条纵向切片，仍需验证 dtype、broadcast 和 scalar 变体 |
| brc | 当前扫描提取 2 个相关 op | 作为首条候选切片 | 适合和 elewise 一起验证基础广播表达 |
| reduce | 当前扫描提取 31 个相关 op | tiling/state 支持不完整时后置 | 需要验证 reduce axis、workspace、tiling 和同步 |
| concat | 当前扫描提取 18 个 data move 相关 op，未检测到直接 concat op | 后置 | 需要确认 offset、多输入和连续拷贝表达方式 |
| gather | 当前扫描提取 6 个相关 op | index/shape 语义确认后迁移 | 对 index 边界、mask 和动态 shape 敏感 |
| transpose | 当前扫描提取 5 个相关 op | layout 表达确认后迁移 | 对 layout、stride、连续性和 workspace 敏感 |
| datacopy | 当前扫描提取 17 个相关 op | 可作为内存搬运验证切片 | 对 GM/UB 地址空间、对齐和 pipeline 生命周期敏感 |
| cube | 当前扫描提取 57 个相关 op | 不放入首条切片 | 编译、tiling、workspace 和执行风险更高 |

当前脚本总计提取 535 个 PyAsc `ascendc` / `emitasc` op。其中 268 个映射到上表 8 个 Autofuse
候选 family，另外 267 个归为 `other`，表示暂未映射到 Phase 2 首批候选 family 的 PyAsc op。

## 扫描命令

```bash
python3 autofuse/mlir/tools/af-mlir-tools/scan_pyasc_coverage.py
python3 autofuse/mlir/tools/af-mlir-tools/scan_pyasc_coverage.py --format json > /tmp/pyasc_ops.json
python3 autofuse/mlir/tools/af-mlir-tools/scan_pyasc_coverage.py --format csv --output autofuse/mlir/docs/pyasc_ascendc_op_inventory.csv
```

Markdown 输出包含 family summary 和完整 op inventory。CSV 快照保存在
`autofuse/mlir/docs/pyasc_ascendc_op_inventory.csv`，便于表格工具筛选。JSON 输出面向后续自动化消费。
主要字段包括：

- `op_name`：如 `ascendc.data_copy_l0`、`ascendc.add_l0`、`emitasc.call_opaque`。
- `td_symbol`：TableGen def 或 defm 展开后的 symbol。
- `arguments` / `results`：每项包含 `name`、`constraint`、`type_hints`、`raw`。
- `param_type_lists`：PyAsc emitter 使用的参数模板映射信息。
- `status`：`explicit-def`、`inherited-class`、`expanded-defm` 或 `no-signature`。

该脚本输出的是候选证据和类型约束，不替代人工语义评审。
