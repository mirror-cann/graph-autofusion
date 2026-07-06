---
name: sk-operator-sample-gen
description: SK 内置样例生成与验证 contract skill，用于构造运行输入、oracle、runner、单算子 differential contract 和整网 wheel 验证 contract。
---

# SK 算子样例生成

这个 skill 把已构建的 SK 算子转换为可运行的单算子或整网验证样例，并产出正确性/融合验证所需的 JSON contract。

能力分三层：

1. 自动构造：生成 input values、oracle、runner command 等闭合 JSON spec。
2. 验证链路：校验闭合 spec，运行目标 runtime，提取输出，对比 expected/actual，并输出 verdict。
3. 验证 contract：记录单算子 standalone differential 验证与整网 wheel 验证的边界。

入口：

```bash
python3 <skills_root>/sk-operator-sample-gen/scripts/operator_sample_gen.py <subcommand> ...
```

## 验证层级

单算子验证不使用 wheel。它在同一个显式 runtime fixture 下比较非 SK standalone entry 和 SK standalone entry。必须有参数角色和可比较输出声明；缺少可比较输出时返回 `needs-user-confirmation`，不能根据名字或参数顺序猜测。

整网验证使用已打包 wheel。它需要显式 network sample contract，包括 package/wheel、runner adapter、nodes、edges、inputs、comparable outputs、prepare steps 和 expected SK fusion ops。真实网络构造委托给 contract 中声明的 adapter；拓扑、tensor 构造、状态准备和领域断言属于用户/adapter 层，不属于核心 skill。

## 通用规则

- 验证是分层的：单算子 differential 验证证明某个 entry 的 SK 语义；整网验证证明 wheel 可以参与用户级 graph。两者不能互相替代。
- 参数角色必须显式：`input`、`output`、`workspace`、`tiling`、`scalar` 或 `descriptor`。
- 比较目标必须显式声明 comparator：`exact`、`allclose` 或 `bytewise`。
- graph capture 前需要存在的状态必须在 `prepare` 或等价 runtime contract 字段中声明。
- runner 不得在 captured graph 内发明 helper kernel 或隐式 runtime work。
- 网络语义属于 adapter。核心 runner 只负责加载 wheel、调用 adapter、校验结构化结果。

## 自动构造命令

- `auto-construct-runtime-input-values <output_dir> [--shape SHAPE] [--dtype DTYPE] [--fill zero]`
  读取 `operator-sk-runtime-input-spec.json`，输出 `operator-sk-auto-input-values-spec.json`。
- `auto-build-correctness-oracle <output_dir> [--oracle-source bind-target-on-wheel|reference-impls-numpy]`
  默认 `bind-target-on-wheel`；`reference-impls-numpy` 保留给简单数学算子的 numpy 参考实现。
- `generate-runner-script <output_dir>`
  输出 `operator-sample-runner.py` 和 `operator-sk-target-runtime-command-spec.json`。
- `build-single-op-verification-contract <output_dir> <operator-runtime-contract.json>`
  输出 `operator-single-op-verification-contract.json`。
- `collect-network-sample-contract <output_dir> <network-contract.json>`
  规范化用户/adapter 提供的整网 contract，输出 `operator-network-sample-contract.json` 和 `operator-network-fusion-expectation.json`。
- `generate-network-runner-script <output_dir>`
  根据 canonical network contract 输出 `operator-network-sample-runner.py` 和 `operator-network-target-runtime-command-spec.json`。

network runner adapter 的推荐函数签名：

```python
def run_network(contract: dict, context: dict) -> dict:
    ...
```

`context` 包含 `contract_path`、`contract_dir` 和 `cwd`。旧的一参 adapter 仍可接受，但新 adapter 应使用两参形式。

## 验证链

典型链路：

```text
collect-runtime-input-spec
-> provide-sk-runtime-input-values
-> collect-correctness-oracle-spec
-> run-sk-target-runtime-validation
-> extract-sk-runtime-outputs
-> compare-sk-runtime-outputs
-> validate-sk-operator-correctness
```

## 预期结果来源

- `bind-target-on-wheel`：通用路径。runner 调用 package 的 `run_<op>` bind target，baseline 作为 expected，SK 路径作为 actual。
- `reference_impls/`：适合简单数学算子。每个 `reference_impls/<entry_name>.py` 暴露 `REFERENCE_IMPL` 和 `compute(inputs)`。

## stage 06 状态语义

- 缺少 runtime fixture：记录 `skipped-insufficient-runtime-spec`。
- NPU runtime 不可用：记录 `skipped-no-npu`。
- mock runner 只用于 routing/schema 检查。即使 JSON 状态是 `mock-passed`，进程也必须返回非 0，不能被当作发布正确性。

## 扩展点

- 新输入构造策略：扩展 `scripts/sk_sample_gen_lib.py` 中的 `build_input_values_spec`。
- 新参考实现：新增 `reference_impls/<entry>.py`。
- 自定义 runner 模板：替换 `sk_sample_gen_lib.py` 中的 runner template。
- standalone fixture metadata：使用 `build_standalone_runtime_fixture_spec` 和 `build_standalone_insufficient_fixture_verdict`。
