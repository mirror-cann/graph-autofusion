---
name: sk-operator-pipeline
description: SK 算子交付流水线总入口，负责运行可定制资产适配器、校验稳定契约、调度核心 SK 算子 skill（codegen / validate / sample-gen / build-package），并提供路由和索引辅助命令。
---

# SK 算子流水线

这是 SK 算子交付工作的主入口，承担三类职责：

1. 闭环编排：把资产适配器和核心 SK 算子 skill 组织成一个可迭代流水线，通过自动修复收敛，不能收敛时明确升级给人工。
2. 阶段产物治理：为每个阶段落盘稳定的 inputs、outputs、deliverables 和 artifact map，方便检查和交付。
3. 路由与索引：当用户不需要完整流水线时，把自由文本问题路由到合适的能力 skill。

入口：

```
python3 <skills_root>/sk-operator-pipeline/scripts/operator_pipeline.py <subcommand> ...
```

## 闭环流水线

```
run-sk-pipeline [--asset OP ...] [--asset-root OPS_DIR]
                --output-dir DIR
                [--stages 01,02,03,05,06]
                [--aggregate-wheel-name op_extension] [--package-version 0.1.0]
                [--target-chip ascend-910b]
                [--profile fast|release]
                [--io-contract operator-io-contract.json]
                [--operator-build-config operator-build-config.json]
                [--operator-build-config-set FIELD=JSON_VALUE]
                [--verify-backend standalone|wheel|both|none]
                [--wheel-mode never|cache|always] [--reuse-wheel WHL]
                [--duplicate-entry-policy reject|namespace]
                [--build-cache-dir DIR] [--jobs N]
                [--max-iterations 5] [--no-package] [--no-verify]
```

`--asset` 可重复，并且可以和 `--asset-root` 混用。`asset-root` 只扫描直接子目录，并筛选像算子源码资产的目录。

`--io-contract` 为 codegen 声明 tensor IO 语义，包括 `inputs`、`outputs`、`workspaces` 和 `pybind_return_tensor`。核心脚本不根据参数名推断输出 buffer；多 tensor 参数语义不明确或契约不完整时，进入 `needs-human`，除非用户或 asset adapter 提供明确契约。asset-root 场景下，契约可以包含所有 entry 的全集；每个 Stage 02 子任务只消费匹配自己的 entry，优先匹配公开 namespace 名，再回退到源码 entry 名。

`--operator-build-config` 为 pipeline 声明用户私有构建依赖，包括 include 目录、support source、强制 include、编译参数、链接参数、构建环境、运行环境和需要进入 wheel 的资源文件。调试时可用 `--operator-build-config-set FIELD=JSON_VALUE` 临时覆盖配置字段，例如 `build_env.DEBUG='"1"'`。CANN 标准依赖由 pipeline 根据 `--target-cann`、`ASCEND_HOME_PATH` 或 `ASCEND_TOOLKIT_HOME` 推导；核心脚本不全仓搜索 include/lib。显式声明的绝对路径会按用户输入直接使用，路径不存在时报错，repo/CANN 外部路径只在 resolved 记录中标记为 `external-explicit`。

输出布局以阶段为主：

```
artifact-map.md / artifact-map.json
deliverables/{wheels,sk-source,pybind-projects}/<asset>/
artifacts/{sources,sk-source,pybind-projects,baseline-so,sk-extensions,operator-units}/...
assets/<asset>/{asset-manifest.json,ops,stages}
work/stage-work/<asset>/
  00-pipeline-config.json
  pipeline-state.json
  01-detect-form/<op>/{inputs,outputs}
  02-adapt-sk-from-global/<op>/{inputs,outputs}
  02-adapt-sk-from-global/_aggregate/{inputs,outputs}
  03-validate-spec/<op>/{inputs,outputs}
  04-validate-compat/<op>/{inputs,outputs}
  05-generate-pybind-binding/{inputs,outputs}
  06-build-and-verify/{inputs,standalone,wheel,verify}
```

先看 `artifact-map.md` 或 `artifact-map.json`。`deliverables/` 是交付面，`work/` 是内部调试工作区。

阶段选择会检查依赖，例如 `--stages 01,05` 会被拒绝，因为 05 依赖 02。Stage 02 同时保留 per-op adapted tree 和一个聚合 aclgraph-canonical tree；Stage 05/06 消费聚合 tree，生成包含所有算子 entry 的单个 wheel。Stage 06 默认做 differential verification；没有 NPU 时显式记录 `skipped-no-npu`。`--no-verify` 只跳过 differential step，不跳过 wheel build。

asset-root 聚合默认拒绝重复 kernel entry 名，避免生成无法区分的 Python wrapper。如果确认需要进入同一个 wheel，显式传 `--duplicate-entry-policy namespace`；只有冲突的公开 wrapper 名会改成 `<asset_namespace>__<source_entry_name>`，生成的 SK launch 仍绑定原始 source entry。根目录的 `name-resolution-report.md/json` 和 wheel 内的 `_name_resolution.json` 用于审计映射。

`--profile fast` 是开发验证路径，默认跑 `01,02,03,06`，使用 standalone differential verification，不构建 wheel。`--profile release` 是交付默认路径，会保留 Stage 05，并在 `deliverables/wheels/<asset>/` 构建或复用一个聚合 wheel。

Stage 04 (`04-validate-compat`) 是显式高级阶段，不属于默认 fast/release。它只报告内置官方来源声明能支撑的兼容性事实和覆盖率元数据，不是 CANN 版本支持白名单。

Stage 06 会区分验证产物和交付产物。standalone 后端比较原始 global chevron 输出和 SK launch 路径；无法执行时记录 `skipped-no-npu` 或 `skipped-insufficient-runtime-spec`。wheel 后端是可选交付打包，受 `--wheel-mode never|cache|always`、`--reuse-wheel`、`--build-cache-dir` 控制。`--jobs` 并行 per-op 阶段，同时保持状态按算子名稳定排序。

状态记录在 `work/stage-work/<asset>/pipeline-state.json`，用于检查每阶段状态、每 op 状态、聚合 entry、cache key、wheel 路径和验证结论。

## 状态和返回码

- `verified`：构建和正确性验证通过，返回 0。
- `packaged`：wheel 已产出且没有失败验证结论，返回 0。
- `skipped-no-npu` / `skipped-insufficient-runtime-spec` / `skipped-by-user` / `skipped-target-arch`：命令完成，但未证明发布正确性，返回 1。
- `clean` / `adapted` / `pybind-generated` / `analyzed`：只完成部分阶段，返回 1，避免 CI 把中间进展当成发布验证成功。
- `structural-only` / `mock-only`：只完成开发用 structural/mock 检查，返回 1。
- `needs-human`：存在人工处理 blocker，返回 1。

## 路由和索引

- `route <query>`：根据关键词把问题路由到合适的 SK skill。
- `index`：构建本地能力索引；默认只扫描 `skills_root`，只有显式 `--index-root` 才扫描用户工作区路径。

路由表：

| Topic | Skill |
|---|---|
| asset / layout / adapter / 用户仓接入 | `sk-operator-asset-adapter` |
| codegen / SK 源码生成 / scaffold / intake | `sk-operator-codegen` |
| 校验 / validate / 规范 / spec / compat / CANN | `sk-operator-validate` |
| 样例 / sample / runtime / correctness / oracle | `sk-operator-sample-gen` |
| 编译 / build / 打包 / wheel / pybind | `sk-operator-build-package` |
| 融合分析 / 性能 | `sk-network-analysis` |

## 参考

- `references/routing.md`：完整路由规则。
- `references/dependencies.md`：skill 间依赖边界。
- `references/workflow.md`：operator pipeline 工作流说明。
