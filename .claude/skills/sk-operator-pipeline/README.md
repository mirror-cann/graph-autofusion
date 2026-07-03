# SK 算子流水线

`sk-operator-pipeline` 是算子生成链路的总入口。它负责发现算子输入、调度 `sk-operator-*`
子能力、落盘每个阶段的输入输出，并生成清晰的交付物目录和资产地图。

## 什么时候使用

优先使用本工具完成端到端闭环：

- 批量把普通 `__global__` 或历史 `__spk__` 算子适配为当前 SK bind 形态。
- 对一批算子做规格检查，并按需要显式运行 source-backed 兼容性风险检查。
- 生成 standalone 或 wheel 构建产物，并在有运行条件时做正确性校验。
- 在自动修复无法收敛时，保留每轮诊断结果供人工处理。

## 快速运行

批量处理目录：

```bash
python3 <skills_root>/sk-operator-pipeline/scripts/operator_pipeline.py run-sk-pipeline \
  --asset-root $OPERATOR_ASSET_ROOT \
  --output-dir build/examples/sk-operator-pipeline \
  --target-chip ascend-910b \
  --profile fast \
  --io-contract operator-io-contract.json \
  --jobs 4
```

处理单个算子：

```bash
python3 <skills_root>/sk-operator-pipeline/scripts/operator_pipeline.py run-sk-pipeline \
  --asset $OPERATOR_ASSET_ROOT/my_op \
  --output-dir build/examples/sk-operator-pipeline/my_op \
  --target-chip ascend-910b \
  --io-contract operator-io-contract.json
```

发布前完整构建：

```bash
python3 <skills_root>/sk-operator-pipeline/scripts/operator_pipeline.py run-sk-pipeline \
  --asset-root $OPERATOR_ASSET_ROOT \
  --output-dir build/examples/sk-operator-release \
  --target-chip ascend-910b \
  --profile release \
  --aggregate-wheel-name op_extension \
  --package-version 0.1.0 \
  --wheel-mode cache \
  --build-cache-dir build/sk-operator-build-cache \
  --jobs 8
```

只跑前半段生成和检查：

```bash
python3 <skills_root>/sk-operator-pipeline/scripts/operator_pipeline.py run-sk-pipeline \
  --asset-root $OPERATOR_ASSET_ROOT \
  --output-dir build/examples/sk-operator-analysis \
  --stages 01,02,03 \
  --no-package \
  --no-verify
```

## 输入方式

`run-sk-pipeline` 支持以下输入方式，可以组合使用：

```bash
# 指定一个或多个 asset
run-sk-pipeline --asset $OP_A --asset $OP_B --output-dir build/examples/out

# 扫描目录
run-sk-pipeline --asset-root $OPERATOR_ASSET_ROOT --output-dir build/examples/out
```

支持的算子源码形态：

| 形态 | 处理方式 |
| --- | --- |
| 当前 SK bind | 复用现有结构，继续规格、兼容性、构建验证 |
| 历史 `__spk__` | 迁移到当前 SK bind 结构 |
| 普通 `__global__` kernel | 生成 SK bind 封装和聚合产物 |
| 缺少规格或无法识别 | 标记为 `needs-human` 并保留诊断输出 |

## 常用参数

CLI 默认 profile 是 `release`。如果只是开发验证，建议显式传 `--profile fast`。

| 参数 | 作用 |
| --- | --- |
| `--output-dir DIR` | 必填，保存流水线所有产物；默认要求为空目录 |
| `--clean-output` | 输出目录已存在时先清理再运行 |
| `--resume-from artifact-map.json` | 显式复用同一输出目录中的已有资产地图 |
| `--asset-root-mode auto|aggregate|separate` | 扫描目录时自动聚合或按资产分别运行 |
| `--duplicate-entry-policy reject|namespace` | 聚合模式遇到同名入口时默认拒绝；显式选择 `namespace` 后按资产命名空间改名并保留原始 launch target |
| `--io-contract FILE` | 明确算子 tensor IO 语义；多 tensor 参数的输入、输出、workspace 和 pybind 返回值不能由脚本猜测，需通过该契约提供 |
| `--operator-build-config FILE` | 声明用户自定义 include、support source、链接库、编译参数、环境变量和需要打包的资源 |
| `--operator-build-config-set FIELD=JSON_VALUE` | 临时覆盖 build config 字段；可重复传入，适合调试 |
| `--stages 01,02,03` | 只运行指定阶段 |
| `--profile fast|release` | 选择快速或发布 profile |
| `--target-chip ascend-910b` | 目标芯片；仅对有来源背书的芯片映射生成 `--npu-arch` |
| `--verify-backend standalone|wheel|both|none` | 选择正确性验证后端 |
| `--wheel-mode never|cache|always` | 控制 wheel 构建或复用策略 |
| `--reuse-wheel WHL` | 直接复用已有 wheel |
| `--build-cache-dir DIR` | 保存可复用构建缓存 |
| `--jobs N` | 并行处理算子数量，并传递给 wheel 阶段作为 bisheng 并行编译上限 |
| `--max-iterations N` | 自动修复最大迭代次数 |
| `--no-package` | 默认停在检查阶段；只有显式 `--stages` 选择 06 时才进入构建验证 |
| `--no-verify` | 跳过运行正确性验证 |

## 阶段和产物

输出目录采用 Artifact Layout v2。用户入口是 `artifact-map.md` 和
`artifact-map.json`；真正需要交付或检查的文件都有唯一规范位置。

```text
artifact-map.md                  # 人读入口：本次运行有哪些资产、终态、交付物
artifact-map.json                # 机器读入口
artifact-layout-lint.json        # 产物布局自检
run-manifest.json                # 运行摘要
deliverables/                    # 面向交付的产物
  wheels/<asset>/                # wheel 包，按资产分区，文件名避免冲突
  sk-source/<asset>/             # 可交付 SK 源码
  pybind-projects/<asset>/       # 可交付 pybind 工程
artifacts/                       # 规范化中间资产，供追溯和后续阶段复用
  sources/<asset>/               # 输入源码快照
  sk-source/<asset>/             # 生成后的 SK 源码快照
  pybind-projects/<asset>/       # 生成后的 pybind 工程快照
  baseline-so/<asset>/<op>/      # 原始分支构建出的 baseline so
  sk-extensions/<asset>/         # SK 分支构建出的 native extension so
  operator-units/<asset>/        # 按 op 拆出的结构化描述
assets/<asset>/                  # 每个资产的报告和索引
  asset-manifest.json
  ops/<op>.json
  stages/<stage>/stage-manifest.json
work/stage-work/<asset>/         # 内部调试工作区；阶段 inputs/outputs 在这里
```

当多个资产里的 kernel 入口同名时，默认行为是拒绝聚合打包，避免生成一个用户无法区分的 wheel。
如果确认这些算子需要进入同一个 wheel，可以显式传
`--duplicate-entry-policy namespace`。此时只有发生冲突的 entry 会改名为
`<asset_namespace>__<source_entry_name>`；非冲突 entry 保持原名。真实 SK launch
仍绑定到原始入口名，改名只影响 Python 包中的公开 wrapper 名称。根目录会生成
`name-resolution-report.md/json`，并且 wheel 内包含 `_name_resolution.json`，用于说明每个公开名对应的原始入口。

IO 契约示例：

```json
{
  "schema_version": 1,
  "entries": {
    "add_custom": {
      "inputs": ["x", "y"],
      "outputs": ["z"],
      "workspaces": ["workspace", "tiling"],
      "pybind_return_tensor": "z"
    },
    "rms_norm": {
      "inputs": ["x", "gamma"],
      "outputs": ["y", "rstd"],
      "pybind_return_tensor": "y",
      "parameters": {
        "tiling": {"kind": "host_struct"}
      }
    }
  }
}
```

`run-sk-pipeline` 会把同一个 `--io-contract` 传给每个 Stage 02 子任务。asset-root
场景可以使用一个包含多个 entry 的全集契约；每个子任务只消费与自身 entry 匹配的部分。同名 entry
被 `--duplicate-entry-policy namespace` 改名后，会优先匹配公开名，再回退到原始 entry 名。没有契约时，
多 tensor-like 参数的算子会进入 `needs-human`，避免把变量名当成输出语义。契约匹配但遗漏 tensor-like
参数时同样会进入 `needs-human`，需要把遗漏项归类到 `inputs`、`outputs` 或 `workspaces`。

构建配置示例：

```json
{
  "schema_version": "sk.operator.build_config.v1",
  "include_dirs": ["include"],
  "support_dirs": ["common"],
  "force_includes": ["include/force_config.h"],
  "compile_options": ["-DEXTRA_OPTION=1"],
  "compile_definitions": ["EXTRA_DEFINE=1"],
  "link_dirs": ["lib"],
  "link_libraries": ["custom_runtime"],
  "link_options": ["-Wl,--as-needed"],
  "build_env": {"CUSTOM_BUILD_FLAG": "1"},
  "runtime_env": {"CUSTOM_RUNTIME_FLAG": "1"},
  "package_files": ["resources/table.json"]
}
```

`operator-build-config.json` 只表达用户私有依赖，是流水线输入契约，不是 wheel 的运行交付物。CANN 标准依赖由 pipeline 根据
`--target-cann`、`ASCEND_HOME_PATH` 或 `ASCEND_TOOLKIT_HOME` 推导，并在
`work/stage-work/<asset>/operator-build-config.resolved.json` 记录候选路径和存在性。
使用配置文件时，相对路径按 JSON 文件所在目录解析；只使用 CLI 覆盖时，相对路径按当前执行目录解析。
核心脚本不会全仓搜索 include/lib。显式声明的绝对路径会按用户输入直接使用；
路径不存在时报错，路径位于 repo/CANN 之外时只在 resolved 记录中标记为 `external-explicit`，不阻断构建。
只有 `package_files` 中声明的运行期资源会被复制进 wheel 的 `_resources/`。

调试时可以用 `--operator-build-config-set FIELD=JSON_VALUE` 临时覆盖已支持的 build config 字段：

```bash
--operator-build-config-set include_dirs='["include"]'
--operator-build-config-set build_env.DEBUG='"1"'
--operator-build-config-set compile_options='["-DEXTRA=1"]'
```

覆盖值必须是合法 JSON。使用配置文件时，相对路径仍按配置文件所在目录解析；只使用
`--operator-build-config-set` 时，相对路径按当前执行目录解析。

阶段说明：

| 阶段 | 说明 | 主要关注文件 |
| --- | --- | --- |
| 01-detect-form | 判断输入是当前 SK bind、历史 `__spk__` 还是普通 `__global__` | form report |
| 02-adapt-sk-from-global | 生成 per-op SK bind，并生成 `_aggregate` 聚合目录 | adapted source、manifest |
| 03-validate-spec | 扫描静态规则，给出问题和修复建议 | findings JSON |
| 04-validate-compat | 可显式开启的 source-backed 兼容性风险检查 | coverage metadata、findings JSON |
| 05-generate-pybind-binding | 生成 pybind 绑定 | binding manifest、C++ source |
| 06-build-and-verify | 构建 standalone/wheel 并做运行正确性验证 | build records、verify records |

## 读状态和排障

先看 `artifact-map.md`：

- `deliverables.wheels`：最终 wheel 包。
- `deliverables.sk_source`：生成后的 SK 源码交付目录。
- `assets/<asset>/asset-manifest.json`：单个资产的阶段报告、op 清单、交付物指针。
- `assets/<asset>/ops/<op>.json`：单个 op 的 baseline so、SK extension、wheel 指针。
- `work/stage-work/<asset>/pipeline-state.json`：内部调试状态；只有排查阶段细节时再看。

常见终态：

- `verified`：构建和正确性验证通过。
- `packaged`：构建通过，未运行验证或验证被跳过。
- `skipped-no-npu`：当前环境缺少 NPU，构建产物已保留。
- `skipped-insufficient-runtime-spec`：缺少可安全运行的输入规格或 fixture，不能做数值验证。
- `skipped-by-user`：用户显式跳过验证。
- `skipped-target-arch`：缺少显式 arch，且 `--target-chip` 不能解析到唯一 source-backed arch。
- `clean`：检查阶段通过，但未进入构建验证。
- `adapted` / `pybind-generated` / `analyzed`：只选择了部分阶段，流水线停在对应中间状态。
- `structural-only` / `mock-only`：显式开发开关下完成结构化或 Mock 路径检查，不代表发布正确性。
- `needs-human`：自动适配或检查发现问题，需要人工处理。

CLI 返回码按发布/CI 语义收窄：只有 `verified` 和 `packaged` 返回 0；其余状态即使命令正常结束，也返回 1，避免被消费方误当成真实发布验证通过。

排障顺序：

1. 查看根目录 `artifact-map.md`，确认本次运行终态和失败资产。
2. 查看 `assets/<asset>/asset-manifest.json`，定位失败阶段。
3. 进入 `work/stage-work/<asset>/<stage>/<op>/outputs` 查看该阶段原始输出。
3. 根据 findings 或 compatibility report 修改源算子。
4. 用新的 `--output-dir` 重跑；如果要复用同一目录，显式传 `--clean-output` 或 `--resume-from`。

## 子命令

查看入口和能力索引：

```bash
python3 <skills_root>/sk-operator-pipeline/scripts/operator_pipeline.py --help
python3 <skills_root>/sk-operator-pipeline/scripts/operator_pipeline.py route --help
python3 <skills_root>/sk-operator-pipeline/scripts/operator_pipeline.py index --help
python3 <skills_root>/sk-operator-pipeline/scripts/operator_pipeline.py run-sk-pipeline --help
```

当只想定位某一阶段时，直接使用对应子工具：

- 形态识别和适配：[../sk-operator-codegen/README.md](../sk-operator-codegen/README.md)
- 统一校验：[../sk-operator-validate/README.md](../sk-operator-validate/README.md)
- 构建打包：[../sk-operator-build-package/README.md](../sk-operator-build-package/README.md)
- 样例和正确性验证生成：[../sk-operator-sample-gen/README.md](../sk-operator-sample-gen/README.md)

## 验证

```bash
python3 <skills_root>/sk-operator-pipeline/scripts/operator_pipeline.py run-sk-pipeline --help
```
