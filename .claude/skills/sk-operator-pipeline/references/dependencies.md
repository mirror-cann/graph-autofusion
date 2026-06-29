# dependencies

## Skill Package Boundary

- `sk-operator-pipeline` locates sibling skills through `skills_root`.
- User assets, outputs, and build cache are interpreted through explicit CLI
  arguments plus `repo_root`.
- `index` scans `skills_root` by default; user workspace indexing requires
  explicit `--index-root`.

## Skill Links

- `sk-operator-asset-adapter`
- `sk-operator-codegen`
- `sk-operator-validate`
- `sk-operator-sample-gen`
- `sk-operator-build-package`
- `sk-network-analysis`
