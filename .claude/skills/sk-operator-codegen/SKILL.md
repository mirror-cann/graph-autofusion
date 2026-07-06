---
name: sk-operator-codegen
description: SK 内置代码生成 skill，负责识别普通 global 算子和当前 SK bind 源码，生成当前 SK binding（Args struct + __sk__ template + SK_BIND），应用可机器修复项，并生成最小非 SK 算子模板。
---

# SK 算子代码生成

这个 skill 用于把普通 AscendC `__global__` kernel 适配为 SuperKernel 入口。已经是当前 SK bind 的源码会按字节复用；混合形态或信息不足的输入会输出明确的人工处理结论。

自动生成逻辑以本 skill 的 `scripts/`、`templates/` 和 `references/` 为稳定契约。

入口：

```
python3 <skills_root>/sk-operator-codegen/scripts/operator_codegen.py <subcommand> ...
```

## 命令

### 自动生成

- `adapt-sk-from-global <asset> --output-dir DIR`
  统一处理已分类的输入形态。普通或可修复的非 SK 源码会先走 codegen 拥有的预适配自动修复，再生成当前 SK bind。模板化 global 会保留模板参数，并追加 `splitidx`；只有字段类型依赖模板参数时，Args struct 才模板化。

  生成包遵循 aclgraph-canonical 布局：`csrc/<op>.asc`、`csrc/pybind11.asc`、`op_extension/`、`setup.py`。ACLGraph wheel 通过 `SK_NPU_ARCHS` 支持多 arch native extension；native module 按 `entry x arch` 拆分，便于 `SK_BISHENG_JOBS` 并行编译。

  多 tensor-like 参数必须通过 `--io-contract FILE` 声明 `inputs`、`outputs`、`workspaces` 和 `pybind_return_tensor`。脚本不得根据变量名或参数顺序推断输出语义。struct-valued runtime 参数需要声明 `parameters.<name>.kind = "host_struct"`。契约缺失或不完整时，输出 `needs-human`。

- `aggregate-sk-adapted --adapted-output-dir DIR ... --output-dir DIR [--aggregate-wheel-name NAME] [--package-version VERSION]`
  把多个 aclgraph-canonical 适配输出合并成一个聚合包树。输出仍保持 `operator-sk-adapted/csrc/*.asc`、`csrc/pybind11.asc`、`op_extension/`、`setup.py` 布局，但 `pybind11.asc` 和 `_torch_library.py` 会注册所有算子 entry。wheel native module 仍按 entry 和 NPU arch 拆分。

- `generate-standalone-compare <aggregate_output_dir> --output-dir DIR [--runtime-fixture-dir DIR] [--target-chip CHIP] [--npu-arch ARCH]`
  生成 `operator-sk-standalone-verify/`，包含 `runtime_compare.asc`、`CMakeLists.txt`、复制后的 adapted csrc 和 `operator-sk-standalone-verify.json`。生成源码会输出 `launch_<op>_baseline` 和 `launch_<op>_sk` wrapper；两者调用同一个 `bind_target`，由 runtime context 区分 baseline 与 SK。

  如果 fixture 声明 device buffers/scalars，会分配独立 baseline/SK buffer、回拷可比输出，并输出 byte/hash 对比结果。没有显式 device plan 时，真实设备运行返回 `skipped-insufficient-runtime-spec`，不能伪造通过。standalone 的 `--npu-arch` 必须显式，或只能由 `--target-chip` 解析到唯一 source-backed arch；否则输出 `needs-target-arch`。

- `detect-sk-form <asset> --output-dir DIR`
  将源码分类为 `none`、`current-sk-bind`、`partial` 或 `unknown`，输出 `operator-sk-form-analysis.json`。

- `apply-remediation <asset_dir> <findings.json>`
  应用自动可修复项，支持 `rename-symbol`、`remove-line-containing`、`add-include`、`replace-pattern`。不可自动修复项会作为人工处理项输出。

### 模板生成

- `generate-from-template <id> --param k=v --output-dir DIR`
  渲染 `templates/<id>.yaml`，用于闭环流水线的干净输入起点。
- `list-templates`
  列出可用模板。

### 历史 scaffold 入口

`intake`、`plan`、`analyze-sk-conversion`、`adapt-sk-binding-scaffold`、`generate-sk-source-scaffold` 仍保留，详见 `references/workflow.md`。

## 扩展点

- 新基础算子模板：新增 `templates/<id>.yaml`。
- 新自动修复规则：扩展 `scripts/sk_codegen_lib.py` 中的 `AUTO_REMEDIATION_KINDS`，并在 `apply_remediation` 中增加处理分支。

## 交付边界

本 skill 的输出交给：

- `sk-operator-validate`：执行 contract/spec/compat 规则包并输出统一 findings。
- `sk-operator-build-package`：消费 `operator-sk-adapted.json` 和 `operator-sk-adapted/`，生成 pybind binding、wheel，并构建 standalone compare 工程。
- `sk-operator-pipeline run-sk-pipeline`：编排完整闭环。

## 运行时状态规则

codegen 必须保持 graph capture 语义：

- tensor-like 参数角色不明确时，必须要求 IO contract。
- capture 前必须准备的 runtime state，例如 TensorList descriptor 或持久 workspace 元数据，必须由显式契约表达，并由 runtime wrapper 消费。
- 生成 wrapper 不得在 captured forward 路径里插入隐藏 helper kernel 来临时制造 descriptor 或状态。
- descriptor 顺序必须由 contract 驱动，并和声明的 TensorList 参数校验一致；不一致时是生成错误，不做 best-effort fallback。

## 参考

- `references/sk-adaptation-cookbook.md`：SK 适配规则和代码生成形态说明。
