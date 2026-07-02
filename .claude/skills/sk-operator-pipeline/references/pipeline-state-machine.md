# SK 闭环流水线状态机

`run-sk-pipeline` 会在 `--output-dir` 写入面向用户的产物布局 v2（Artifact Layout v2）。阶段优先状态仍会记录，但位于内部调试工作区：`work/stage-work/<asset>/pipeline-state.json`。

## 阶段图

```
assets
  |
  v
01-detect-form/<op>
  |
  v
02-adapt-sk-from-global/<op>
  |
  +--> 03-validate-spec/<op>
  |
  +--> 04-validate-compat/<op>
  |
  v
02-adapt-sk-from-global/_aggregate
  |
  +--> 06-build-and-verify/standalone
  |
  +--> 05-generate-pybind-binding
         |
         v
      06-build-and-verify/wheel
```

阶段依赖是强约束：

| 阶段 | 前置依赖 |
|---|---|
| `01` | 输入 asset |
| `02` | `01` |
| `03` | `02` |
| `04` | `02`；如果存在 target context，可生成 source-backed compat coverage metadata |
| `05` | `02/_aggregate` |
| `06` standalone | `02/_aggregate` |
| `06` wheel | `05`，除非 `--reuse-wheel` 提供交付 wheel |

如果选择的阶段缺少前置依赖，orchestrator 会用 usage error 退出，例如 `05 needs 02`。

`--profile fast --stages 01,02,06` 是合法的，因为 standalone validation 依赖聚合 adapted tree，不依赖 pybind generation。release wheel build 仍然需要 stage 05。

## 布局

面向用户的布局：

```text
artifact-map.md
artifact-map.json
artifact-layout-lint.json
run-manifest.json
deliverables/
  wheels/<asset>/
  sk-source/<asset>/
  pybind-projects/<asset>/
artifacts/
  sources/<asset>/
  sk-source/<asset>/
  pybind-projects/<asset>/
  baseline-so/<asset>/<op>/
  sk-extensions/<asset>/
  operator-units/<asset>/<op>.json
assets/<asset>/
  asset-manifest.json
  ops/<op>.json
  stages/<stage>/stage-manifest.json
work/stage-work/<asset>/...
```

内部阶段工作区：

```
00-pipeline-config.json
pipeline-state.json
01-detect-form/<op>/{inputs,outputs}
02-adapt-sk-from-global/<op>/{inputs,outputs}
02-adapt-sk-from-global/_aggregate/{inputs,outputs}
03-validate-spec/<op>/{inputs,outputs}
04-validate-compat/<op>/{inputs,outputs}
05-generate-pybind-binding/{inputs,outputs}
06-build-and-verify/
  inputs/aggregate-output
  standalone/{inputs,outputs}
  wheel/{inputs,outputs}
  verify/<op>
```

`inputs/` 需要自包含。orchestrator 会把上游资产和阶段输出 materialize 到每个阶段的 `inputs/` 目录中，通常使用浅拷贝，让被复制出来的阶段目录不依赖 sibling stages。公开 artifact map 只指向相对路径；外部源码输入会复制到 `artifacts/sources/<asset>/`。

## 状态结构

```json
{
  "status": "verified | packaged | clean | adapted | pybind-generated | analyzed | skipped-no-npu | skipped-insufficient-runtime-spec | skipped-by-user | skipped-target-arch | structural-only | mock-only | needs-human",
  "assets": [{"name": "clear_ops", "path": "..."}],
  "selected_stages": ["01", "02", "03", "05", "06"],
  "ops": {
    "clear_ops": {
      "form": "legacy-spk",
      "stage02": {"outputs": "...", "pybind_layout": "aclgraph-canonical"}
    }
  },
  "aggregate": {"entries": ["clear_ops", "..."]},
  "binding": {"package_name": "op_extension", "kernel_entries": []},
  "standalone": {
    "status": "passed | reused | failed | skipped-no-npu | skipped-insufficient-runtime-spec | skipped-by-user | skipped-target-arch | mock-passed",
    "verify_status": "passed | skipped-no-npu | skipped-insufficient-runtime-spec | skipped-by-user | skipped-target-arch | mock-passed",
    "cache_key": "...",
    "cache_hit": false,
    "reused_from": null
  },
  "wheel": {
    "mode": "never | cache | always",
    "status": "skipped | built | reused | structural-built | structural-reused | failed",
    "cache_key": "...",
    "cache_hit": false,
    "wheel_paths": []
  },
  "verification": {
    "backend": "standalone | wheel | both | none",
    "standalone": {"status": "skipped-no-npu"},
    "wheel": null
  },
  "stages": []
}
```

`06-build-and-verify/standalone/outputs/` 包含 standalone executable project 和 per-op verdict。启用 wheel mode 时，`06-build-and-verify/wheel/outputs/wheels/` 包含最终交付 wheel。多算子 release 运行仍产出一个 wheel；native extension 按 entry/arch 拆分。
