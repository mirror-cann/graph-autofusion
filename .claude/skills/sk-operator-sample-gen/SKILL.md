---
name: sk-operator-sample-gen
description: Built-in SK domain skill that auto-constructs input values, declares bind-target-on-wheel or reference-impl correctness oracles, and renders a dual-path sample runner -- producing the closed JSON spec files the existing runtime / oracle / verdict chain validates and consumes.
---

# SK Operator Sample Generation

Turns built SK operators into runnable single-op and network-level samples with
scoped correctness/fusion verdict inputs. The skill has three halves:

1. **Auto-construction** (new): synthesise the closed JSON spec files (input
   values, oracle, runner command) a user would otherwise hand-write.
2. **Validation chain** (existing): the manual commands canonicalise the
   closed specs, run the target runtime, extract outputs, compare against
   expected, and emit a scoped correctness verdict.
3. **Verification contracts**: records the boundary between single-op
   standalone differential verification and network-level wheel verification.

The orchestrator wires construction → validation; schemas stay
guaranteed-consistent.

## Verification Levels

Single-op verification does **not** use a wheel. It compares a non-SK standalone
entry and an SK standalone entry under the same explicit runtime fixture. It
requires declared parameter roles and comparable outputs. If comparable outputs
are not declared, return `needs-user-confirmation`; do not infer outputs from
names or order.

Network-level verification **does** use the packaged wheel. It requires an
explicit network sample contract: package name/wheel, runner adapter, nodes,
edges, inputs, comparable outputs, prepare steps, and expected SK fusion ops.
The generated network runner delegates real network construction to the declared
adapter, so custom network semantics live in user/adapter code instead of this
core skill.

## Generalised Knowledge Rules

Treat sample generation as a contract problem, not a pattern-matching problem.
This skill may construct files only after the relevant semantics are explicit in
the contract.

- **Verification is layered**: single-op differential verification proves the
  SK entry preserves one operator's semantics; network verification proves the
  packaged wheel can participate in a user-level graph. Do not use one as a
  substitute for the other.
- **Roles are explicit**: parameters must be classified as `input`, `output`,
  `workspace`, `tiling`, `scalar`, or `descriptor` by an upstream contract.
  This skill must not infer roles from variable names, pointer order, shape, or
  whether a value is later compared.
- **Comparison targets are explicit**: outputs to compare and their comparator
  (`exact`, `allclose`, or `bytewise`) must be declared. Missing comparable
  outputs produce `needs-user-confirmation`.
- **Prepare state is explicit**: any state that must exist before graph capture
  must be declared in `prepare` or an equivalent runtime contract field. The
  runner may call a declared prepare hook; it must not invent helper kernels or
  hidden runtime work inside the captured graph.
- **Network semantics belong to adapters**: core runner code loads the wheel,
  calls the declared adapter, and validates structured results. Topology,
  tensor construction, state preparation, and domain-specific assertions belong
  to the user/adapter layer.

Stage 06 can now feed this skill from two runtime backends. The wheel backend
uses the generated Python runner and `bind-target-on-wheel` oracle. The
standalone backend supplies runner stdout directly from the ASC executable
schema:

```json
{"backend": "standalone", "outputs": {"op": {"baseline": [], "sk": []}}}
```

`compare-sk-runtime-outputs` treats standalone `baseline` as expected and `sk`
as actual. If runtime fixtures are missing, callers record
`skipped-insufficient-runtime-spec`; if NPU runtime is unavailable, callers
record `skipped-no-npu`. Mock runs may validate routing only. Real device
standalone runs are expected to provide baseline/SK hashes or captured outputs
from an explicit runtime fixture plan before they can be reported as `passed`.
Generated mock runners deliberately return non-zero even when their JSON status
is `mock-passed`; direct callers must not treat mock execution as release
correctness.

Top-level entry:

```
python3 <skills_root>/sk-operator-sample-gen/scripts/operator_sample_gen.py <subcommand> ...
```

## Auto-construction commands (new)

- `auto-construct-runtime-input-values <output_dir> [--shape SHAPE] [--dtype DTYPE] [--fill zero]`
  — read `operator-sk-runtime-input-spec.json`, emit
  `operator-sk-auto-input-values-spec.json` (zero-filled tensor for GM_ADDR
  params; scalar params get the element count with a dtype derived from the
  C type). Comma-separated `--shape` for multi-dim (e.g. `--shape 2,3`).
- `auto-build-correctness-oracle <output_dir>
  [--oracle-source bind-target-on-wheel|reference-impls-numpy]` — default
  mode is `bind-target-on-wheel`, which declares that the runner's bind-target
  baseline path supplies expected values. `reference-impls-numpy` is preserved as
  a compatibility path and loads `reference_impls/<entry_name>.py`.
- `generate-runner-script <output_dir>` — emit `operator-sample-runner.py`
  (loads the values manifest passed via argv[2] and prints structured stdout
  JSON keyed by operator name, with `baseline` and `sk` values) plus
  `operator-sk-target-runtime-command-spec.json` (binds the values manifest as
  `values_manifest_argv` argv_index=2).
- `build-single-op-verification-contract <output_dir> <operator-runtime-contract.json>`
  — emit `operator-single-op-verification-contract.json`. This is the canonical
  single-op differential contract: `baseline=non_sk_entry`,
  `actual=sk_entry`, `requires_wheel=false`.
- `collect-network-sample-contract <output_dir> <network-contract.json>` —
  canonicalise a user/adapter-provided network contract and emit
  `operator-network-sample-contract.json` plus
  `operator-network-fusion-expectation.json`.
- `generate-network-runner-script <output_dir>` — emit
  `operator-network-sample-runner.py` and
  `operator-network-target-runtime-command-spec.json` from the canonical network
  contract.

Network runner adapters use this callable shape:

```python
def run_network(contract: dict, context: dict) -> dict:
    ...
```

`context` contains `contract_path`, `contract_dir`, and `cwd`. Older one-arg
adapters are still accepted, but new adapters should use the two-arg shape so
they can place outputs next to the canonical contract or a user-provided output
root.

## Validation chain (existing)

- `collect-runtime-input-spec` -> `provide-sk-runtime-input-values` (validates
  the auto-constructed closed spec) -> `collect-correctness-oracle-spec`
  (validates the auto-built oracle spec) -> `run-sk-target-runtime-validation`
  (runs the auto-generated runner with `shell=False`, controlled env,
  bounded stdout tails) -> `extract-sk-runtime-outputs` ->
  `compare-sk-runtime-outputs` -> `validate-sk-operator-correctness` (terminal
  scoped verdict).

## Oracle Sources

`bind-target-on-wheel` is the general path for generated SK deliveries: the
runner calls the package's `run_<op>` bind target for the baseline and SK
execution contexts and comparison treats `baseline` as expected and `sk` as actual. The
`reference_impls/` registry remains useful for simple mathematical operators
that are naturally expressible in numpy.

## Contract Rules

- Do not guess input/output roles. Use explicit `role` and `compare` fields.
- Single-op verification must block if no comparable output is declared.
- TensorList/descriptor preparation must be represented as explicit runtime or
  network `prepare` information; do not insert graph-captured helper kernels in
  the runner path.
- Network verification must compare `aclgraph` and `aclgraph+sk` outputs and
  carry an expected fusion contract that downstream SK analysis can validate.

## Reference-impl registry

Each `reference_impls/<entry_name>.py` exposes:

```python
REFERENCE_IMPL = {
    "entry_name": "<kernel_entry>",   # used as the registry key
    "description": "<short summary>",
    "comparator": "exact" | "allclose",
    "tolerance": {"rtol": <num>, "atol": <num>},
}

def compute(inputs: dict) -> object:
    """inputs maps each kernel param name to {"shape", "dtype", "value"}.
       Returns the expected primary output value (the JSON-list shape the
       runner emits)."""
```

Bundled: `reference_impls/add_custom.py` (numpy: z = x + y).

## Extension points

- New auto-input strategy (e.g. `--fill random`): extend
  `build_input_values_spec` in `scripts/sk_sample_gen_lib.py`.
- New operator reference: drop a `reference_impls/<entry>.py`; registry
  auto-discovers.
- Custom runner template: replace `_RUNNER_TEMPLATE` in `sk_sample_gen_lib.py`
  (real-deployment shim that imports the pybind module instead of echoing the
  first tensor parameter).
- Standalone fixture metadata: use
  `build_standalone_runtime_fixture_spec` and
  `build_standalone_insufficient_fixture_verdict` in
  `scripts/sk_sample_gen_lib.py`.
