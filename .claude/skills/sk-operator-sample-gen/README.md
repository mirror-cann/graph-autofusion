# SK 算子样例生成

`sk-operator-sample-gen` 生成运行样例、正确性 oracle、runner 和验证 contract。
它服务于 stage 06 的 standalone 或 wheel 正确性验证，也可以单独用于构造最小复现。

## 两层验证模型

单算子验证不经过 whl 包。它基于 `operator-runtime-contract.json`，要求用户或
adapter 明确声明参数角色和可比较输出，然后对同一组输入分别运行非 SK 入口和
SK 入口，比较输出是否一致。缺少输出声明时工具会返回
`needs-user-confirmation`，不会根据变量名或参数顺序猜测。

整网验证经过 whl 包。它基于用户或 adapter 提供的 network contract，声明包名、
wheel、网络节点、边、输入、输出、prepare 步骤和预期 SK 融合范围。工具负责把
contract 规范化，生成 runner 和 fusion expectation；具体网络构造逻辑由 contract
里的 runner adapter 承担。

真实 runner adapter 推荐签名：

```python
def run_network(contract: dict, context: dict) -> dict:
    ...
```

`context` 包含 `contract_path`、`contract_dir` 和 `cwd`。adapter 应返回结构化
JSON dict，至少包含 `status` 和两种运行模式的结果。发布包不绑定固定样例路径；
如维护侧提供参考网络 contract，可作为 adapter 写法参考。

## 泛化规则

本 skill 固化的是验证规则，不固化某个样例的网络结构。

- 验证分两层：单算子差分验证用于定位单个 SK 入口是否保持语义；整网验证用于确认
  whl 交付物能被用户网络加载并在 `aclgraph` / `aclgraph+sk` 下保持一致。
- 参数语义必须显式声明：`input`、`output`、`workspace`、`tiling`、`scalar`、
  `descriptor`。脚本不能从变量名、参数顺序、shape 或 compare 标记反推语义。
- 输出比较必须显式声明：哪些输出参与比较、比较方式和容差都来自 contract。缺失时
  返回 `needs-user-confirmation`。
- capture 前准备状态必须显式声明：例如 descriptor、workspace tail、静态缓存等。
  runner 只能调用声明好的 prepare hook，不能在图内临时插入 helper kernel。
- 用户网络语义属于 adapter：核心 runner 只调用 `run_network(contract, context)`；
  网络拓扑、输入构造、中间状态准备和业务断言由 adapter 或用户代码负责。

## 常用命令

自动构造 runtime 输入建议：

```bash
python3 <skills_root>/sk-operator-sample-gen/scripts/operator_sample_gen.py auto-construct-runtime-input-values \
  build/examples/sk-codegen/standalone \
  --shape 1,16,64 \
  --dtype float16 \
  --fill zero
```

该命令读取 `operator-sk-runtime-input-spec.json`，写出
`operator-sk-auto-input-values-spec.json`。它是可审核的输入建议，不会直接替代
canonical `operator-sk-runtime-input-values.json`。

把审核后的输入值登记为 canonical runtime values：

```bash
python3 <skills_root>/sk-operator-sample-gen/scripts/operator_sample_gen.py provide-sk-runtime-input-values \
  build/examples/sk-codegen/standalone \
  build/examples/sk-codegen/standalone/operator-sk-auto-input-values-spec.json
```

生成正确性 oracle 建议：

```bash
python3 <skills_root>/sk-operator-sample-gen/scripts/operator_sample_gen.py auto-build-correctness-oracle \
  build/examples/sk-codegen/standalone \
  --oracle-source reference-impls-numpy
```

该命令要求目录内已经存在 canonical `operator-sk-runtime-input-values.json`，
并写出 `operator-sk-auto-oracle-spec.json`。

Mock runner 只用于开发期 route/schema 检查。它会在 JSON 中记录
`mock-passed`，但进程返回码为 1，不能作为发布或 CI 正确性通过信号。

把审核后的 oracle 登记为 canonical oracle spec：

```bash
python3 <skills_root>/sk-operator-sample-gen/scripts/operator_sample_gen.py collect-correctness-oracle-spec \
  build/examples/sk-codegen/standalone \
  build/examples/sk-codegen/standalone/operator-sk-auto-oracle-spec.json
```

生成 runner：

```bash
python3 <skills_root>/sk-operator-sample-gen/scripts/operator_sample_gen.py generate-runner-script \
  build/examples/sk-codegen/standalone
```

生成单算子差分验证 contract：

```bash
python3 <skills_root>/sk-operator-sample-gen/scripts/operator_sample_gen.py build-single-op-verification-contract \
  build/examples/sk-codegen/runtime-contracts/add_custom \
  build/examples/sk-codegen/runtime-contracts/add_custom/operator-runtime-contract.json
```

登记整网验证 contract：

```bash
python3 <skills_root>/sk-operator-sample-gen/scripts/operator_sample_gen.py collect-network-sample-contract \
  build/examples/sk-codegen/network \
  network-contract.json
```

生成整网 runner：

```bash
python3 <skills_root>/sk-operator-sample-gen/scripts/operator_sample_gen.py generate-network-runner-script \
  build/examples/sk-codegen/network
```

查看完整帮助：

```bash
python3 <skills_root>/sk-operator-sample-gen/scripts/operator_sample_gen.py --help
```

## 预期结果来源

常见来源：

- `reference-impls-numpy`：使用 numpy 参考实现构造期望输出。
- `bind-target-on-wheel`：通过已构建 wheel 运行目标绑定生成对照结果。

如果缺少 shape、dtype 或语义信息，本工具只能生成诊断结果，不能可靠构造运行样例。

## 输出和使用方式

生成结果通常包括：

- `operator-sk-auto-input-values-spec.json`：自动生成的输入值建议。
- `operator-sk-runtime-input-values.json`：审核后登记的 canonical runtime values。
- `operator-sk-auto-oracle-spec.json`：自动生成的 oracle 建议。
- `operator-sk-correctness-oracle-spec.json`：审核后登记的 canonical oracle spec。
- `operator-sample-runner.py`：运行脚本。
- `operator-sk-target-runtime-command-spec.json`：目标运行命令规格。
- 差分验证所需的 stdout schema 或记录文件。
- `operator-single-op-verification-contract.json`：单算子非 SK / SK 入口差分验证 contract。
- `operator-network-sample-contract.json`：整网 whl 验证 contract。
- `operator-network-fusion-expectation.json`：整网 SK 融合期望，供网络分析阶段消费。
- `operator-network-sample-runner.py`：整网 runner。
- `operator-network-target-runtime-command-spec.json`：整网 runner 命令规格。

总流水线中这些文件会被 stage 06 消费：

```text
06-build-and-verify/{inputs,standalone,wheel,verify}
```

## 验证

```bash
python3 <skills_root>/sk-operator-sample-gen/scripts/operator_sample_gen.py --help
```

维护侧 pytest 回归由发布准备流程维护，不作为发布包内命令。
