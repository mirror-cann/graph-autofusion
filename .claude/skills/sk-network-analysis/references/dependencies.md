# dependencies

## Portable Skill Inputs

`sk-network-analysis` consumes user-provided SK result directories. It does not
require the target workspace to carry any external source documents or samples.

Preferred user-provided inputs:

- result root containing `sk_meta/`, logs, `kernel_meta/`, and profiling/event outputs
- model asset root such as `sk_meta/<pid>/`
- single `model_*` directory

Primary evidence files:

- `super_kernel.log`
- `sk_scope_split.log`
- `sk_node_detail.log`
- `sk_fused_nodes.log`
- `sk_device_args.log`
- `sk_task_queue.json` as shadow validation during the compatibility period
- `sk_event_dev_device_*.json` / `sk_prof_device_*.json`

## Portable Knowledge Sources

- `references/workflow.md`
- `references/artifact-contract.md`
- `references/script-index.md`
- `references/diagnosis-matrix.md`
- `references/update-view-registry-guide.md`
- `references/html-design-workflow.md`

## Current Gaps

- Profiling auto-root-cause attribution is still incomplete.
- `performance-report.html` still needs deeper event-to-structure explanation.
- Launch-info payload and full graph writeback details depend on upstream logs.
- Missing `sk_meta` / event / profiling assets force summary-only or
  insufficient diagnostics.
