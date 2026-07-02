---
name: sk-operator-build-package
description: SK 内置编译打包 skill，负责校验 aclgraph-canonical pybind11 包结构、调用 bisheng 编译 SK/ACLGraph 扩展，并打包为可分发 Python wheel。
---

# SK 算子编译与打包

这个 skill 提供三类构建能力：

1. SK 源码构建链：对生成的 SK source scaffold 或导出的 SK source version tree 做受控 CMake 构建。
2. pybind 绑定与 native wheel：消费 codegen 生成的 aclgraph-canonical `operator-sk-adapted/` 包树，生成包含 native extension 的 Python wheel。
3. standalone 差分可执行程序：消费 `operator-sk-standalone-verify/` 工程，直接用 CMake 构建 standalone 对比可执行文件，不依赖 pybind、torch、torch_npu 或 wheel。

入口：

```bash
python3 <skills_root>/sk-operator-build-package/scripts/operator_build_package.py <subcommand> ...
```

## SK 源码构建链

- `run-sk-build-validation <scaffold_dir>`：对 `operator-sk-source-scaffold/` 执行 CMake configure/build。
- `prepare-sk-source-version <scaffold_dir>`：导出已通过构建验证的 SK source version tree。
- `validate-sk-source-version <scaffold_dir>`：在导出的 source-version tree 上重新执行 CMake。
- `prepare-validated-sk-source-version <asset>`：从转换分析到 validated source version 的一键编排。

## pybind 与 native wheel

- `generate-pybind-binding <adapted_output_dir>`：读取单算子或聚合的 `operator-sk-adapted.json`，校验 codegen 负责的 aclgraph layout：
  - `operator-sk-adapted/csrc/<op>.asc`
  - `csrc/pybind11.asc`
  - `csrc/pybind11_<op>.asc`
  - `op_extension/__init__.py`
  - `op_extension/_arch_selector.py`
  - `op_extension/_torch_library.py`
  - `setup.py`

  输出：`operator-sk-pybind-binding.json`。

- `build-native-wheel <adapted_output_dir> [--jobs N]`：对 `operator-sk-adapted/` 执行 `pip wheel --isolated --no-build-isolation --no-cache-dir`。

  `setup.py` 使用 `setuptools.Extension(language="asc")` 和 `AscendBuildExtension.build_extension()` 调用 `bisheng`。arch 选择规则：

  - 构建期优先使用 `SK_NPU_ARCHS`。
  - 运行时优先使用 `SK_ACLGRAPH_NPU_ARCH`。
  - 未显式指定时，只允许使用有来源支撑的当前环境/SoC 映射。
  - 不能静默选择唯一 `.so` 或默认 `dav-2201`。

  `--jobs` 会设置 `SK_BISHENG_JOBS`，用于 `build_ext` 并行。

  输出 wheel 包含：

  - `op_extension/<entry>_<arch_suffix>*.so`
  - `_arch_selector.py`
  - `_torch_library.py`

  输出 manifest：`operator-sk-native-wheel.json`。

显式启用 structural fake toolchain 时，JSON 可以记录 `structural-passed`，但 CLI 必须返回非 0，避免被当作真实发布构建成功。

## Standalone 可执行程序

- `build-standalone-executable <standalone_output_dir> [--target-chip CHIP]`
  执行：
  - `cmake -S operator-sk-standalone-verify -B operator-sk-standalone-verify/build`
  - `cmake --build ...`

  输出：
  - `operator-sk-standalone-build.json`
  - `build-log.txt`
  - `executable-path.txt`

structural executable placeholder 也遵循同样规则：JSON 可记录 structural 状态，但进程返回非 0。

## 缓存

`sk_build_cache_lib.py` 提供内容 hash 缓存能力。pipeline stage 06 会在 `pipeline-state.json` 里记录：

- `cache_key`
- `cache_hit`
- `reused_from`

## 扩展点

- 包树 contract 变化：修改 `scripts/sk_pybind_lib.py`。
- bisheng 编译命令或额外 flag：修改 `sk-operator-codegen/scripts/sk_codegen_lib.py` 中的 renderer。

## 下游交付

wheel 交给下游安装或发布。运行时正确性验证由 `sk-operator-sample-gen` 负责；在具备 NPU 的环境中，runner 可以 import 构建好的 package 并执行验证。
