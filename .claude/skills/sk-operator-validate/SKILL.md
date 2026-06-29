---
name: sk-operator-validate
description: Core SK operator validation skill that unifies contract validation, source/spec rule packs, and optional source-backed compatibility checks into one findings envelope.
---

# SK Operator Validate

Use this core skill after asset adapter output and before build/package. It
validates stable contracts and runs validation rule packs. This skill is core:
users should configure inputs and rule data, not rewrite the execution engine.

Top-level entry:

```
python3 <skills_root>/sk-operator-validate/scripts/operator_validate.py <subcommand> ...
```

## Commands

- `validate-operator --asset ASSET --output-dir DIR [--layout LAYOUT]
  [--build-context BUILD] [--verify-context VERIFY] [--target-chip CHIP]
  [--rule-pack all|contract|spec|compat]`
- `list-rule-pack --rule-pack spec|compat`

Writes:

- `operator-validation-findings.json`
- `operator-validation-report.md`

## Rule Packs

- `contract`: validates layout/build/verify contracts.
- `spec`: runs the bundled SK spec rules.
- `compat`: optional advanced static checks. It reports only facts covered by
  bundled official-source-backed declarations and writes coverage metadata; it
  is not the default CANN support certification path.
