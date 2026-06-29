---
name: sk-network-analysis
description: Built-in SK domain skill for SK network-level diagnostics, including result-path asset discovery, hang and coredump triage, performance diagnosis, scope/task graph visualization, and node tracing for SuperKernel scenarios.
---

# SK Network Analysis

Use this skill when the task is about whole-network SK diagnostics, especially:

- analysis of user-provided SK result directories
- hang / coredump / exception localization
- performance diagnosis
- scope/task graph visualization

Top-level entry must use the Python CLI:

- `python3 <skills_root>/sk-network-analysis/scripts/network_analysis.py <subcommand> ...`

Execution modes:

- `base mode`
  - default mode
  - does not require network or a large model
  - generates machine-consumable reports and indexes
- `ai mode`
  - optional, enabled by `--with-ai`
  - must build on top of base-mode artifacts
  - if AI is not configured, the base artifacts still remain valid and usable

Output contract:

- base artifacts
  - `run-portal.html`
  - HTML / Markdown / JSON reports
  - per-report generation status and diagnostic completeness
  - asset guidance, collection hints, and next-information-needed guidance
- AI artifacts
  - optional hints or routing suggestions
  - must not replace base artifacts

Current direct assets are under `scripts/`.

Read these references first:

- `references/workflow.md`
- `references/dependencies.md`
- `references/artifact-contract.md`
- `references/script-index.md`
- `references/diagnosis-matrix.md`
- `references/update-view-registry-guide.md`
- `references/html-design-workflow.md`

Primary workflows:

- `analyze`
  - analyze one user-provided result path
  - detect recognized assets, explain why missing evidence matters, and suggest what to collect next
  - classify each report as generated vs diagnostically complete/limited/insufficient
- `diagnose-hang-crash`
  - correlate `sk_meta`, plog, and device logs for hang / coredump / exception triage
- `diagnose-performance`
  - correlate `sk_event_dev_device_*.json`、task structure、fused nodes and time events for performance diagnosis
  - event/prof JSON stats are lazy-loaded, stream parsed with `ijson` when available, and batch-parallelized by process when possible
  - large event/prof JSON files are first-class performance inputs; task / scope remain secondary drill-down context
- `trace-nodes`
  - export tracing-compatible node trace artifacts for `edge://tracing` and `chrome://tracing`
  - emit cross-report node/task/scope linkage metadata

Current result guidance:

- if an asset exists
  - the skill should explain what diagnostic value it provides
- if an asset is missing
  - the skill should explain why it matters, what reports it blocks or downgrades, and what to preserve next time
- if current evidence is still insufficient
  - the skill should explicitly list the next information needed instead of only returning a generic failure

Current source-of-truth guidance:

- use this skill's bundled `references/` and generated JSON artifacts as the portable source of truth
- when a target workspace provides extra design notes, treat them as optional context rather than required inputs
- keep `SKILL.md` as the concise routing/operation surface; put detailed contracts in `references/`

Current navigation expectations:

- `run-portal.html`
  - is the default workspace entry
  - should show current evidence, diagnostic completeness, recommended next step, and report/object entry points
  - should include dedicated `Interactive Views`, `Structured Views`, and `Summary Views` sections
  - should include a dedicated `What To Read Next` section with direct object-level links
  - should expose `Current Capability` and `Still Missing` so users can see diagnostic ability, not just file presence
  - should distinguish `graph-capable` runs from `summary-only` runs when interactive graph evidence is unavailable
- `scope-graph.html`
  - must remain a first-class interactive graph view
  - should preserve the old high-quality graph exploration experience instead of being replaced by summary tables
  - should accept lightweight object context such as `scopeId` / `nodeId` and surface that context in-page
  - should support lightweight in-page object focus instead of only showing passive context text
- `task-queue-graph.html`
  - must remain a first-class interactive queue view
  - should preserve the old high-quality queue exploration experience instead of being replaced by summary tables
  - should accept lightweight object context such as `taskIndex` / `nodeId` and surface that context in-page
  - should support lightweight in-page object focus instead of only showing passive context text
- `scope-graph.html`
  - should prefer a canonical graph-centric scope view and demote repeated rounds to reference-only context
- `task-queue-graph.html`
  - should expose canonical task anchors for cross-report linking
- `hang-crash-report.html` / `performance-report.html`
  - should exist as browser-friendly companions to the Markdown summaries
  - should expose stable section anchors for portal deep links
  - `performance-report.html` should keep event/prof data first:
    - `sk_event_dev_device_*.json` / `sk_prof_device_*.json` are primary performance assets
    - task / scope remain secondary drill-down links

Current AOT/task queue compatibility expectations:

- log parsing remains the canonical compatibility path during the transition
- task queue JSON is used as a shadow-validation source when it can be aligned safely
- support `scopes[].taskQueues` aligned by `scopeId` / `skId`
- support root-level `taskQueues` only when they can be matched to a unique candidate section
- do not report mismatches for JSON-only fields when the log side has no comparable value
- exclude sync/event/custom tasks from graph-bound duplicate identity checks when their graph identity is not valid

Current performance / progress expectations:

- expose `Stage 0/4` through `Stage 4/4` for full/performance analysis
- make event/prof parsing visible as `Stage 2/4`
- use `tqdm` progress bars on TTY and compact start/done lines in non-TTY logs
- suppress noisy worker parser/render output in `network_analysis.py` runs
- write timing details to `reports/data/diagnose-profile.json` when `--profile` is enabled

Atomic implementation expectations:

- high-quality graph scripts should be wrapped into stable source/model/renderer boundaries
- graph views, structured data, and summary views are all required outputs
- `diagnose_run.py` should orchestrate these atomic capabilities instead of inlining graph logic

External HTML guidance expectations:

- external design guidance is optional and advisory only
- optional design review may reshape page hierarchy, reading order, interaction grouping, and visual style
- optional design review must not redefine diagnostic semantics, evidence-tier logic, capability-mode logic, or JSON report contracts
- first implement changes inside `sk-network-analysis`; then apply optional page-design review if available in the target environment
- first-wave HTML redesign targets are:
  - `run-portal.html`
  - `scope-graph.html`
  - `task-queue-graph.html`
- `hang-crash-report.html` and `performance-report.html` should follow the same design system, but they are secondary to the first-wave pages

Internal regression helper:

- `python3 <skills_root>/sk-network-analysis/scripts/regression_runner.py <sample-root>`
  - runs the current skill against an existing sample inside a temporary workspace
  - auto-cleans generated files after each case

Verification expectations when editing this skill:

- for docs-only changes, at least validate command help and stale-reference search
- for parser/report changes, run targeted `sk_network_analysis` unit tests
Prefer the bundled analysis scripts when the task matches:

- SK log extraction
- scope split visualization
- task queue visualization
- node tracing
- scope/task graph visualization
