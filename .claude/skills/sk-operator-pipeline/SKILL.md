---
name: sk-operator-pipeline
description: Built-in SK operator delivery pipeline entrypoint -- runs the customizable asset adapter, validates stable contracts, orchestrates the core SK operator skills (codegen / validate / sample-gen / build-package), and exposes auxiliary routing/index commands.
---

# SK Operator Pipeline

The front door for SK operator delivery work. Three responsibilities:

1. **Closed-loop orchestration**: drive asset adapter plus the core SK operator skills as one
   iterative pipeline that converges via auto-remediation or escalates to a
   human.
2. **Stage artifact governance**: write stable per-stage inputs, outputs,
   deliverables, and artifact maps for inspection and handoff.
3. **Auxiliary routing/indexing**: classify a free-form SK question and point
   at the right capability skill when a user does not need the whole pipeline.

Top-level entry:

```
python3 <skills_root>/sk-operator-pipeline/scripts/operator_pipeline.py <subcommand> ...
```

## Closed-loop pipeline (the main user-facing command)

```
run-sk-pipeline [--asset OP ...] [--asset-root OPS_DIR]
                --output-dir DIR
                [--stages 01,02,03,05,06]
                [--aggregate-wheel-name op_extension] [--package-version 0.1.0]
                [--target-chip ascend-910b]
                [--profile fast|release]
                [--io-contract operator-io-contract.json]
                [--verify-backend standalone|wheel|both|none]
                [--wheel-mode never|cache|always] [--reuse-wheel WHL]
                [--duplicate-entry-policy reject|namespace]
                [--build-cache-dir DIR] [--jobs N]
                [--max-iterations 5] [--no-package] [--no-verify]
```

`--asset` is repeatable and may be mixed with `--asset-root`; asset-root scans
direct child directories only and filters to operator-like source assets.
`--io-contract` declares tensor IO semantics (`inputs`, `outputs`,
`workspaces`, and `pybind_return_tensor`) for codegen. The core scripts do not
infer output buffers from parameter names; ambiguous or incompletely classified
multi-tensor entries escalate to `needs-human` unless the user or asset adapter
supplies the contract. In asset-root runs the contract may contain a superset of
entries; each Stage 02 subtask consumes only the matching entry, preferring the
public namespace name before falling back to the source entry name.

The output layout is stage-first:

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

Start from `artifact-map.md` or `artifact-map.json`. `deliverables/` is the
handoff surface; `work/` is the internal debug workspace.

Stage selection is dependency-checked: for example `--stages 01,05` is rejected
because 05 needs 02. Stage 02 keeps both per-op adapted trees and one aggregate
aclgraph-canonical tree; stage 05/06 consume the aggregate tree to produce one
wheel containing all operator entries. Stage 06 runs differential verification
by default and records `skipped-no-npu` verdicts when NPU execution is not
available; `--no-verify` skips only that differential step, not the wheel build.

Aggregate asset-root runs reject duplicate kernel entry names by default. When a
single wheel is required, pass `--duplicate-entry-policy namespace`; only
colliding public wrapper names are changed to
`<asset_namespace>__<source_entry_name>`, while the generated SK launch still
binds to the original source entry. Inspect `name-resolution-report.md/json` at
the output root or `_name_resolution.json` inside the wheel package to audit the
mapping.

`--profile fast` runs the development validation path: stages `01,02,03,06`
by default, standalone differential verification under
`work/stage-work/<asset>/06-build-and-verify/standalone/`, and no wheel build. `--profile release` is
the delivery default: it keeps stage 05 and builds or reuses one aggregate
wheel under `deliverables/wheels/<asset>/`.

Stage 04 (`04-validate-compat`) is an explicit advanced stage, not part of the
default fast or release path. It reports only compatibility facts backed by the
bundled official-source declarations and writes coverage metadata; it is not a
CANN version support whitelist.

Stage 06 has separate validation and delivery artifacts. The standalone backend
compares original global chevron output with the SK launch path and records
`skipped-no-npu` or `skipped-insufficient-runtime-spec` explicitly when it
cannot execute. The wheel backend remains optional delivery packaging:
`--wheel-mode never|cache|always`, `--reuse-wheel`, and `--build-cache-dir`
control build reuse. `--jobs` parallelizes per-op stages while keeping state
ordered by operator name.

State is accumulated in `work/stage-work/<asset>/pipeline-state.json`, so a
human can inspect per-stage status, per-op status, aggregate entries, cache
keys, wheel paths, and verification verdicts.

Status and CLI success policy:

- `verified` -- build and correctness verification passed; `run-sk-pipeline` returns 0.
- `packaged` -- wheel produced without a failing verification verdict; `run-sk-pipeline` returns 0.
- `skipped-no-npu` / `skipped-insufficient-runtime-spec` / `skipped-by-user` /
  `skipped-target-arch` -- the command completed but did not prove release
  correctness; `run-sk-pipeline` returns 1.
- `clean` / `adapted` / `pybind-generated` / `analyzed` -- selected partial
  stages completed; `run-sk-pipeline` returns 1 so CI cannot confuse partial
  progress with release validation.
- `structural-only` / `mock-only` -- development-only structural or mock checks
  completed; `run-sk-pipeline` returns 1.
- `needs-human` -- a human-only blocker was found; `run-sk-pipeline` returns 1.

## Auxiliary routing & index

- `route <query>` — heuristic routing to one of the SK skills based on
  keywords (`codegen` / `spec` / `compat` / `runtime` / `pybind` / etc.).
- `index` — build a local capability index; by default it scans `skills_root`, and `--index-root` opt-in scans user-selected workspace paths.

Route table:

| Topic | Skill |
|---|---|
| asset / layout / adapter / 用户仓接入 | `sk-operator-asset-adapter` |
| codegen / SK 源码生成 / scaffold / intake | `sk-operator-codegen` |
| 校验 / validate / 规范 / spec / compat / CANN | `sk-operator-validate` |
| 样例 / sample / runtime / correctness / oracle | `sk-operator-sample-gen` |
| 编译 / build / 打包 / wheel / pybind | `sk-operator-build-package` |
| 融合分析 / 性能 | `sk-network-analysis` |

## References

- `references/routing.md` — full routing rules.
- `references/dependencies.md` — inter-skill dependency map.
- `references/workflow.md` — operator-pipeline workflow details.
