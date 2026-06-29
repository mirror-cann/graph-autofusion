# SK adaptation cookbook

Quick reference for the rules `adapt-sk-from-global` encodes. This document is
the skill-local behavior contract for generated SK binding shape.

## What gets generated

For every supported source form the adapter either emits current SK binding or
returns an explicit human escalation. For clean non-SK `__global__` kernels it
emits, in order, after the original function:

1. An **Args struct** named `<NameCamel>Args` with one field per kernel
   parameter, preserving order. Fields whose C type is < 4 bytes
   (`int8_t`/`uint8_t`/`int16_t`/`uint16_t`/`bool`) get `alignas(4)`.
   If the original kernel is templated, the Args struct is templated only over
   template parameters referenced by field types; body-only or kernel-type-only
   template parameters stay on the SK function but do not affect the runtime
   parameter package layout.
2. A **templated `__sk__` function**:
   ```cpp
   template<uint32_t splitidx>
   __sk__ <kernel_type> void <name>_sk(const <NameCamel>Args *args
                                       [, sk::SkSystemArgs *sysArgs]) {
       <c_type> <param> = args-><param>;   // one line per parameter
       // ... original body verbatim ...
   }
   ```
   The original body is reused unchanged except that
   `AscendC::GetBlockNum()` is rewritten to `sysArgs->skNumBlocks` (and the
   `sysArgs` parameter is injected) when the original body references it.
3. An **`SK_BIND(<orig>, <mask>, <name>_sk<0>, <name>_sk<1>, <name>_sk<2>, <name>_sk<3>)`**
   line. `mask` defaults to 4 (DCCI); allowed values are 0..7, where 0 means
   no capability bits and 1/2/4 are bit flags. `--num-splits` controls how many
   `<name>_sk<N>` symbols are bound (1..4).

The original `__global__` function is **preserved unchanged**.

## Multi-operator aggregate rendering

Single-op adaptation still writes one aclgraph-canonical tree per asset:

```text
operator-sk-adapted/
  csrc/<op>.asc
  csrc/pybind11.asc
  op_extension/__init__.py
  op_extension/_torch_library.py
  setup.py
```

`aggregate-sk-adapted` consumes multiple such outputs and renders one aggregate
tree with all `csrc/<op>.asc` files, one `pybind11.asc`, one
`_torch_library.py`, and one `setup.py`. Entry names must be unique across the
aggregate. The generated pybind layer exposes the customer-facing bind target
entry per operator:

| Function | Purpose |
|---|---|
| `run_<op>` | SK-facing bind target registered through `torch.library`; differential validation reuses this same entry for baseline and SK contexts |

The aggregate `setup.py` keeps the Python import package as `op_extension` while
using the requested distribution name and version for wheel naming.

## Five input forms

| Input form | Adapter behavior |
|---|---|
| `none` | Generate Args struct, templated `__sk__`, and `SK_BIND`. |
| remediable `none` | Run codegen-owned pre-adapt auto-remediation on a temp copy, then generate current SK binding. |
| `legacy-spk` | Remove legacy `__spk__` variants and `.ascend.meta` / `FunLevel*` metadata, then generate current SK binding from the common legacy body. |
| `current-sk-bind` | Copy the source byte-for-byte and mark it `already_current`. |
| `partial` / `unknown` | Do not guess; emit `codegen.unknown-sk-form` for human action. |

## Kernel-type mapping

| Original qualifier | SK qualifier |
|---|---|
| `__vector__` | `__vector__` |
| `__cube__` | `__cube__` |
| `__mix__(c, v)` general | `__mix__(c, v)` |
| `__mix__(1, 0)` | `__cube__` (special case) |
| `__mix__(0, 1)` | `__vector__` (special case) |
| bare `__aicore__` | `__aicore__` |

## When to inject `sysArgs`

`--with-sys-args=auto` (default): inject iff the original body contains
`AscendC::GetBlockNum()`. `--with-sys-args=always` / `=never` force the
choice.

When injected, the API names are the **current** ones:
`sysArgs->skNumBlocks` / `sysArgs->SkGetNumBlocks()`. Legacy
`skBlockNum` / `SkGetBlockNum` fail to compile under current CANN headers --
`sk-operator-validate --rule-pack spec` flags them as `sk.sys-args-api-current` (auto-
remediable rename).

## Legacy migration boundaries

- `legacy-spk` migration requires 1..4 variants per legacy stem. The generated
  SK body is derived from the matched `__global__` body, not from legacy
  `__spk__` wrapper/helper bodies.
- `FunLevelMixCoreType`, `FunLevelKType`, `.ascend.meta.*`, and legacy-only
  `__DAV_CUBE__` / `__DAV_VEC__` shells are removed from migrated output.
- Complex KernelLaunch wrappers are preserved as-is and recorded as warnings.
- Helper-forward legacy bodies of the form `<helper>(param);` are not used as
  implementation by default, because the adapter cannot prove that helper still
  matches the customer-facing global entry. They may only be used as evidence
  to select a concrete template specialization when multiple launch targets
  exist. When such a legacy-only helper is identified, it is removed from the
  generated output together with the legacy `__spk__` wrappers.
- Templated `__global__` entries with multiple launch specializations are bound
  per legacy stem. The migration deduces the concrete `SK_BIND(<op><T>, ...)`
  target from launch sites and, when needed, tiling types found in helper bodies.
  The generated SK function preserves the original global template parameters
  and appends `uint32_t splitidx`; `SK_BIND` explicitly instantiates the full
  template argument list for each split. The Args struct is templated only when
  a kernel parameter type depends on those template parameters.
- Zero variants, too many variants, or undeducible template specialization
  produce human escalations instead of partial output.

See the published guide for the full specification and corner cases.
