---
name: sk-operator-build-package
description: Built-in SK domain skill that compiles SK source via bisheng, validates aclgraph-canonical pybind11 package trees from codegen, and packages the result as a distributable Python wheel containing the bisheng-built extension module.
---

# SK Operator Build + Package

Two-layer build/package surface:

1. **SK source build chain** (the original layer): controlled CMake builds for
   a generated SK source scaffold and for an exported SK source version tree.
2. **pybind binding + native wheel**: consumes codegen's aclgraph-canonical
   `operator-sk-adapted/` package tree directly and produces a Python wheel with
   `op_extension.<entry>_<arch>`. In multi-operator runs this input is the aggregate
   tree from `02-adapt-sk-from-global/_aggregate/outputs/`, so one build creates
   one wheel containing all entries; native extensions are split per entry and
   per selected NPU architecture for parallel bisheng compilation.
3. **standalone differential executable**: consumes codegen's
   `operator-sk-standalone-verify/` project and runs CMake directly, without
   pybind, torch, torch_npu, or wheel packaging.

Top-level entry:

```
python3 <skills_root>/sk-operator-build-package/scripts/operator_build_package.py <subcommand> ...
```

## SK source build chain

- `run-sk-build-validation <scaffold_dir>` — controlled CMake configure/build
  for `operator-sk-source-scaffold/`.
- `prepare-sk-source-version <scaffold_dir>` — export build-validated SK
  source version tree.
- `validate-sk-source-version <scaffold_dir>` — re-run CMake on the exported
  source-version tree itself.
- `prepare-validated-sk-source-version <asset>` — one-shot orchestrator from
  conversion analysis through validated source version.

## pybind + native wheel

- `generate-pybind-binding <adapted_output_dir>` — reads either a single-op or
  aggregate
	  `operator-sk-adapted.json` and validates the codegen-owned aclgraph layout:
  `operator-sk-adapted/csrc/<op>.asc`, `csrc/pybind11.asc`,
  per-entry `csrc/pybind11_<op>.asc`,
  `op_extension/__init__.py`, `op_extension/_arch_selector.py`,
  `op_extension/_torch_library.py`, and `setup.py`. No per-entry `_pybind.cpp`
  or aggregate `_module.cpp` files are generated here.
	  Manifest: `operator-sk-pybind-binding.json`.
- `build-native-wheel <adapted_output_dir> [--jobs N]` — runs `pip wheel --isolated
  --no-build-isolation --no-cache-dir` against `operator-sk-adapted/`.
  `setup.py` uses `setuptools.Extension(language="asc")` and
  `AscendBuildExtension.build_extension()` to invoke
  `bisheng -x asc --npu-arch=<arch> -shared -fPIC -std=c++17
  -D_GLIBCXX_USE_CXX11_ABI=<torch ABI>
  -DTORCH_EXTENSION_NAME=<entry>_<arch_suffix> -ltorch_npu -ltorch -lc10 ...`.
  Build arch selection uses `SK_NPU_ARCHS` first, then source-backed
  current-environment detection; otherwise it fails fast. Runtime selection
  uses `SK_ACLGRAPH_NPU_ARCH` first, then the same source-backed SoC mapping,
  and does not silently choose a single packaged `.so`.
  `--jobs` sets `SK_BISHENG_JOBS` for setuptools `build_ext` parallelism.
	  The resulting `.whl` ships `op_extension/<entry>_<arch_suffix>*.so` plus
	  `_arch_selector.py` and `_torch_library.py`. Manifest:
	  `operator-sk-native-wheel.json` (status, wheels, return_code, log_path).
  With an explicitly enabled structural fake toolchain the manifest status is
  `structural-passed`, but the CLI exits non-zero so direct callers cannot
  confuse structural output with a release build.

`sk-operator-pipeline.run-sk-pipeline` copies these build artifacts into
`06-build-and-verify/wheel/outputs/` and `deliverables/wheels/`, then runs or
records wheel-backed differential verification when that backend is selected.

## Standalone executable

- `build-standalone-executable <standalone_output_dir> [--target-chip CHIP]`
  — runs `cmake -S operator-sk-standalone-verify -B
  operator-sk-standalone-verify/build` and `cmake --build ...`. It writes
  `operator-sk-standalone-build.json`, `build-log.txt`, and
  `executable-path.txt`.
  Structural executable placeholders follow the same rule: JSON may say
  `structural-passed`, but the process exits non-zero.

`sk_build_cache_lib.py` provides the shared content-hash cache helpers used by
the pipeline for standalone executables and wheels. Stage 06 records
`cache_key`, `cache_hit`, and `reused_from` in `pipeline-state.json`.

## Extension points

- Package tree validation: edit `scripts/sk_pybind_lib.py` when the
  aclgraph-canonical contract changes.
- setup.py compile command: update the renderer in
  `sk-operator-codegen/scripts/sk_codegen_lib.py` to plumb extra bisheng
  flags or deeper compile-time code generation.

## Hand-off

The wheel goes downstream to whatever installs / publishes it. Runtime
correctness verification is owned by `sk-operator-sample-gen` (the runner
script there can `import` the built package once deployed on NPU hardware).
