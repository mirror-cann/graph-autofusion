# Closed-loop SK pipeline state machine

`run-sk-pipeline` writes a user-facing Artifact Layout v2 at `--output-dir`.
Stage-first state is still recorded, but it lives in the internal debug
workspace: `work/stage-work/<asset>/pipeline-state.json`.

## Stage Graph

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

Stage dependencies are strict:

| Stage | Prerequisite |
|---|---|
| `01` | input asset |
| `02` | `01` |
| `03` | `02` |
| `04` | `02`; optional source-backed compat coverage metadata uses target context when present |
| `05` | `02/_aggregate` |
| `06` standalone | `02/_aggregate` |
| `06` wheel | `05` unless `--reuse-wheel` supplies the delivery wheel |

If a selected stage misses a prerequisite, the orchestrator exits with a usage
error such as `05 needs 02`.

`--profile fast --stages 01,02,06` is valid because standalone validation
depends on the aggregate adapted tree, not on pybind generation. Release wheel
builds still require stage 05.

## Layout

User-facing layout:

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

Internal stage workspace:

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

Inputs are self-contained. The orchestrator materializes upstream assets and
stage outputs into each `inputs/` directory with shallow copies so a copied
stage directory has no dependency on sibling stages. The public artifact map
points to relative paths only; external source inputs are copied under
`artifacts/sources/<asset>/`.

## State Shape

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

`06-build-and-verify/standalone/outputs/` contains the standalone executable
project and per-op verdicts. `06-build-and-verify/wheel/outputs/wheels/`
contains the final delivery wheel when wheel mode is enabled. Multi-operator
release runs still produce one wheel with one native extension module.
