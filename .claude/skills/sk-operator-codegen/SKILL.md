---
name: sk-operator-codegen
description: Built-in SK domain skill that converts supported operator source forms to current SK binding (Args struct + __sk__ template + SK_BIND), identifies SK adaptation form (none / legacy-spk / current-sk-bind / partial / unknown), applies machine-remediable findings to source, and renders minimal non-SK operator templates.
---

# SK Operator Codegen

Use this skill to **derive a SuperKernel adaptation from an ordinary AscendC
`__global__` kernel, migrate legacy `__spk__` assets to current `SK_BIND`,
pass through already-current SK source byte-for-byte, or classify unsupported
mixed forms for human escalation**. The auto-generation logic follows
the stable contracts implemented in this skill's `scripts/`, `templates/`,
and `references/`.

Top-level entry:

```
python3 <skills_root>/sk-operator-codegen/scripts/operator_codegen.py <subcommand> ...
```

## Commands

Auto-generation:

- `adapt-sk-from-global <asset> --output-dir DIR` — handle all classified
  input forms in one pass. Pure or remediable non-SK source is cleaned with
  codegen-owned pre-adapt auto-remediation before adaptation; legacy `__spk__`
  variants are migrated to current `__sk__` + `SK_BIND` by copying the matched
  `__global__` body into the SK body. Legacy helper-forward `param` bodies are
  not used as the generated SK implementation unless a future explicit user
  override says so; helper bodies may only provide specialization evidence for
  templated multi-specialization groups. Templated globals keep their template
  parameters on the SK function, with `splitidx` appended and concrete
  instantiations emitted in `SK_BIND`; the Args struct is templated only when
  a field type depends on a template parameter.
  generated package files follow the aclgraph-canonical layout
  (`csrc/<op>.asc`, `csrc/pybind11.asc`, `op_extension/`, `setup.py`). The
  generated ACLGraph wheel package supports multi-arch native extensions via
  `SK_NPU_ARCHS`.
  Native modules are split per `entry x arch` so multi-op and multi-arch wheel
  builds can run bisheng in parallel through `SK_BISHENG_JOBS`.
  If neither variable is set, build-time auto-detection only uses source-backed
  SoC mappings and otherwise fails fast.
  When an entry has multiple tensor-like parameters, pass
  `--io-contract FILE` to declare `inputs`, `outputs`, `workspaces`, and
  `pybind_return_tensor`; this skill must not infer output semantics from
  parameter names. A matching IO contract must classify every tensor-like
  parameter. Struct-valued runtime parameters require `parameters.<name>.kind =
  "host_struct"`. Without a matching or complete IO contract, ambiguous
  multi-tensor entries produce `needs-human` findings and require human or
  adapter input.
  Already-current SK source is copied byte-for-byte; `partial` / `unknown`
  forms produce actionable human escalations. Writes `operator-sk-adapted/`
	  + `operator-sk-adapted.json`.
- `aggregate-sk-adapted --adapted-output-dir DIR ... --output-dir DIR
  [--aggregate-wheel-name NAME] [--package-version VERSION]` — combine multiple
  aclgraph-canonical adapted outputs into one aggregate package tree. The output
  keeps the canonical layout (`operator-sk-adapted/csrc/*.asc`,
  `csrc/pybind11.asc`, `op_extension/`, `setup.py`) but `pybind11.asc` and
  `_torch_library.py` register every operator entry. Wheel native modules are
  emitted per entry and per selected NPU arch.
- `generate-standalone-compare <aggregate_output_dir> --output-dir DIR
  [--runtime-fixture-dir DIR] [--target-chip CHIP] [--npu-arch ARCH]` — generate
  `operator-sk-standalone-verify/` with `runtime_compare.asc`, `CMakeLists.txt`,
  copied adapted csrc, and `operator-sk-standalone-verify.json`. The generated
  source emits `launch_<op>_baseline` and `launch_<op>_sk` wrappers that both
  call the same `bind_target`; the runtime context supplies the baseline vs SK
  distinction. If `runtime-fixture-dir/<op>/operator-sk-runtime-fixture.json`
  declares device buffers/scalars, the source allocates independent baseline/SK
  buffers, copies compare buffers back to host, and reports byte/hash comparison
  results. Without that explicit device plan, real-device runs report
  `skipped-insufficient-runtime-spec` instead of a fake pass. This is the
  standalone differential backend used by the fast pipeline. Standalone
  `--npu-arch` is explicit; when omitted, `--target-chip` may be used only if it
  resolves to exactly one source-backed arch. Otherwise generation reports
  `needs-target-arch` and never silently emits a default arch.
  ACLGraph wheel runtime selection uses `SK_ACLGRAPH_NPU_ARCH` first, then only
  source-backed SoC mappings, and never silently picks the sole packaged `.so`.
- `detect-sk-form <asset> --output-dir DIR` — classify each source file as
  `none` / `legacy-spk` / `current-sk-bind` / `partial` / `unknown`. Writes
  `operator-sk-form-analysis.json`.
- `apply-remediation <asset_dir> <findings.json>` — apply auto-remediable
  findings (kinds: `rename-symbol`, `remove-line-containing`, `add-include`,
  `replace-pattern`) to source files. Non-auto-remediable findings are
  reported as escalations. Writes `operator-sk-remediation.json`.

Template-driven generation (convenience):

- `generate-from-template <id> --param k=v --output-dir DIR` — render a
  `templates/<id>.yaml` into a fresh source tree (the "clean input" entry
  point of the closed-loop pipeline).
- `list-templates` — discover available templates.

Existing scaffold path (kept from the original):

- `intake` / `plan` / `analyze-sk-conversion` / `adapt-sk-binding-scaffold` /
  `generate-sk-source-scaffold` — see `references/workflow.md`.

## Extension points

- New base operator: drop `templates/<id>.yaml` (kernel-launch-adapt §4-style
  body); loader auto-discovers.
- New auto-remediation rule kind: extend `AUTO_REMEDIATION_KINDS` in
  `scripts/sk_codegen_lib.py` and add the handler branch in `apply_remediation`.

## Hand-off boundary

Outputs of this skill feed:

- `sk-operator-validate` (runs contract/spec/compat validation rule packs and
  produces unified findings; legacy `spec-check` and `compat-check` remain
  compatibility wrappers).
- `sk-operator-build-package` (consumes `operator-sk-adapted.json` +
  `operator-sk-adapted/` for pybind binding + wheel build; in multi-op runs it
  consumes the aggregate tree; it also builds the standalone compare project
  emitted by `generate-standalone-compare`).
- `sk-operator-pipeline.run-sk-pipeline` orchestrates the whole loop.

## Runtime State Rules

Codegen must preserve graph-capture semantics:

- Tensor-like parameters with ambiguous role require an IO contract. Do not
  infer inputs, outputs, workspaces, or return tensors from names or order.
- Runtime state that must be prepared before capture, such as TensorList
  descriptors or persistent workspace metadata, must be represented by an
  explicit contract field and consumed by a runtime wrapper.
- Generated wrappers must not launch hidden helper kernels in the captured
  forward path to manufacture descriptor or prepare state. If preparation is
  required, expose it as prepared runtime state for sample-gen or the user
  adapter to initialise before capture.
- Descriptor ordering must be contract-driven and validated against the declared
  TensorList parameters. A mismatch is a generation error, not a best-effort
  fallback.

## References

- `references/sk-adaptation-cookbook.md` — source-backed adaptation notes and
  code-generation patterns.
