# SK 算子编译与打包

`sk-operator-build-package` 负责把已适配或已聚合的 SK bind 产物继续转换为可构建、可发布、可验证的工程产物。端到端场景优先使用 `sk-operator-pipeline run-sk-pipeline`，本工具适合定位 pybind、wheel 或 standalone 构建问题。

## 输入要求

输入一般来自 `sk-operator-codegen`：

- 单个算子的 adapted output。
- 多个算子的 `_aggregate` output。
- 已有 standalone compare output。

如果目录缺少 manifest、源码或构建配置，本工具会在传入目录内保留失败记录，方便回到生成阶段修复。

显式结构化工具链只用于开发期结构检查。相关 manifest 会记录
`structural-passed`，但子命令返回码为 1，不能作为发布或 CI 构建通过信号。

## 常用命令

生成 pybind 绑定：

```bash
python3 <skills_root>/sk-operator-build-package/scripts/operator_build_package.py generate-pybind-binding \
  build/examples/sk-codegen/aggregate
```

构建 native wheel：

```bash
python3 <skills_root>/sk-operator-build-package/scripts/operator_build_package.py build-native-wheel \
  build/examples/sk-codegen/aggregate
```

构建 standalone executable：

```bash
python3 <skills_root>/sk-operator-build-package/scripts/operator_build_package.py build-standalone-executable \
  build/examples/sk-codegen/standalone \
  --target-chip ascend-910b
```

查看完整命令：

```bash
python3 <skills_root>/sk-operator-build-package/scripts/operator_build_package.py --help
```

## 在总流水线中的位置

`run-sk-pipeline` 会把相关产物落到：

```text
deliverables/wheels/<asset>/
deliverables/pybind-projects/<asset>/
artifacts/sk-extensions/<asset>/
work/stage-work/<asset>/05-generate-pybind-binding/{inputs,outputs}
work/stage-work/<asset>/06-build-and-verify/{inputs,standalone,wheel,verify}
```

`fast` profile 通常跳过 wheel，以 standalone 差分验证为主。`release` profile 会进入 pybind 和 wheel 构建路径，并可通过 `--wheel-mode cache` 和 `--build-cache-dir` 复用构建结果。

## 常见问题

- 没有 CANN 或编译器环境：优先用 `--no-package` 跑前置阶段，或者只保留生成结果。
- 没有 NPU：构建可继续，运行校验可能进入 `skipped-no-npu`。
- wheel 已经存在：在总流水线中使用 `--reuse-wheel WHL` 或 `--wheel-mode cache`。
- standalone 缺少运行输入：先用 `sk-operator-sample-gen` 生成 runtime fixture。

## 验证

```bash
python3 <skills_root>/sk-operator-build-package/scripts/operator_build_package.py --help
```
