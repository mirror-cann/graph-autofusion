---
name: sk-operator-asset-adapter
description: Customizable SK operator asset intake skill that turns user-owned operator repositories or source trees into stable core contracts consumed by sk-operator-codegen, sk-operator-validate, sk-operator-build-package, and sk-operator-sample-gen.
---

# SK Operator Asset Adapter

Use this skill when a user-provided operator asset must be translated into the
stable contracts required by the SK core skills. This layer is intentionally
customizable: users may copy or modify adapter recipes for their own repository
layout, naming conventions, build commands, and verification setup.

Top-level entry:

```
python3 <skills_root>/sk-operator-asset-adapter/scripts/operator_asset_adapter.py <subcommand> ...
```

## Commands

- `adapt-asset <asset> --output-dir DIR [--target-chip CHIP] [--target-arch ARCH]`
  scans the asset, applies the default recipes, and writes:
  - `operator-asset-inventory.json`
  - `operator-asset-layout.json`
  - `operator-build-context.json`
  - `operator-verify-context.json`
  - `adapter-report.json`
- `validate-contracts --layout LAYOUT --build-context BUILD --verify-context VERIFY`
  validates the contracts without running the SK core pipeline.

## Boundary

This skill owns user-asset interpretation: directory traversal, host/kernel
matching, default include discovery, and build/verify context scaffolding.
It does not own SK_BIND generation, package build internals, or correctness
verdict semantics.

Core skills must consume the emitted contracts and fail closed when the adapter
cannot produce enough information.

## Contract Extraction Rules

The adapter may use repository conventions, user prompts, or project-specific
recipes to produce contracts, but it must keep uncertainty visible:

- Emit explicit parameter roles for runtime verification: `input`, `output`,
  `workspace`, `tiling`, `scalar`, or `descriptor`.
- Emit explicit comparable outputs and comparator policy. If the asset does not
  declare them, mark the contract as requiring user confirmation instead of
  guessing from names like `out`, `y`, or `workspace`.
- Emit explicit prepare requirements for graph capture, such as descriptor
  materialisation, persistent workspace state, or static cache initialisation.
- For network-level verification, emit a network sample contract with package,
  runner adapter, nodes, edges, inputs, outputs, prepare steps, and expected
  fusion scope when these are known. If topology or output semantics are
  unknown, stop at a partial contract and ask for user/adapter input.
- Keep user-specific repository rules in this adapter layer. Core skills should
  only consume stable JSON contracts and should not learn per-repository layout
  or naming conventions.
