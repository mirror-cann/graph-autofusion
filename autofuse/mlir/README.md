# Autofuse MLIR

本目录提供 Autofuse MLIR 开发所需的环境脚本、LLVM/MLIR 依赖接入、PyAsc submodule
管理和轻量验证工具。

## 目录说明

```text
cmake/              LLVM/MLIR 依赖解析
docker/             dev+LLVM 镜像、CANN volume、CLion Docker Toolchain 指导
docs/               迁移设计、开发计划、依赖策略
externals/pyasc     PyAsc 官方仓 submodule，只在 ENABLE_AUTOFUSE_MLIR=ON 时使用
patches/pyasc       Autofuse 维护的 PyAsc 最小 patchset
scripts/            环境进入、依赖准备、PyAsc 同步和轻量检查脚本
```

## 工具索引

| 工具 | 使用者 | 当前阶段 | 作用 | 产物 |
|---|---|---|---|---|
| `tools/af-opt/af-opt` | MLIR 开发者、lit 测试 | 可用 | 最小 MLIR opt 入口，用于 parse/print、`--version` 和后续 pass 冒烟 | 构建目录中的 `af-opt`，不进默认 run 包 |
| `tools/af-mlir-tools/scan_pyasc_coverage.py` | MLIR 迁移开发者、reviewer | 可用 | 从 PyAsc `.td` 提取 `ascendc` / `emitasc` op inventory，辅助判断 PyAsc 能表达哪些算子族 | Markdown/JSON/CSV，CSV 快照在 `docs/pyasc_ascendc_op_inventory.csv` |
| `tools/af-mlir-tools/af_mlir_case.py` | MLIR 迁移开发者、NPU 验证者 | 可用入口 | 包装既有 `tests/st/codegen/ascir_tool/test_ascir.sh`，设置 `MAKEFLAGS=-jN`，记录 case 返回码 | `mlir/.artifacts/<case>.status` |

## 使用入口

命令默认从 Autofuse 组件目录执行：

```bash
cd /Volumes/GM9/code/autofuse/autofuse
```

开发环境、镜像、CANN volume、CLion 配置和完整编译流程见
[docker/README.md](docker/README.md)。

## PyAsc

PyAsc 来源固定为官方仓：

```text
https://gitcode.com/cann/pyasc.git
```

父仓 gitlink 固定 PyAsc upstream commit，默认不会自动拉取官方仓最新 HEAD。初始化固定基线并
应用本仓 patchset：

```bash
bash mlir/scripts/sync_pyasc_upstream.sh
```

查看当前固定 commit：

```bash
git ls-files --stage mlir/externals/pyasc
```

只恢复固定 upstream 基线、不应用 patchset：

```bash
bash mlir/scripts/sync_pyasc_upstream.sh --no-apply-patches
```

本地应用 patchset 后，父仓会看到 `mlir/externals/pyasc` gitlink 变化。该变化通常不应提交；gitlink
仍应作为官方 upstream 固定基线，Autofuse 改动通过 `mlir/patches/pyasc/*.patch` 维护。
`sync_pyasc_upstream.sh` 默认会在成功应用 patchset 后保护本地 gitlink，避免 `git status` 噪音和
`git add -A` 误提交。该保护等价于：

```bash
git config submodule.autofuse/mlir/externals/pyasc.ignore all
git update-index --skip-worktree mlir/externals/pyasc
```

如需临时跳过该保护：

```bash
bash mlir/scripts/sync_pyasc_upstream.sh --no-protect-gitlink
```

后续新增 PyAsc 修改时，推荐在独立 PyAsc checkout 中开发并形成 commit，再同步成 Autofuse patchset；
不要依赖本仓受保护 submodule 的 dirty worktree：

```bash
bash mlir/scripts/export_pyasc_patch.sh --pyasc-dir /path/to/pyasc --commit <sha>
bash mlir/scripts/export_pyasc_patch.sh --pyasc-dir /path/to/pyasc --range <start>..<end>
```

确认屏蔽生效：

```bash
git status --short
git add -n -A
git ls-files -v mlir/externals/pyasc
```

`git ls-files -v` 输出前缀为 `S` 表示 `skip-worktree` 生效。以上设置只影响本地工作区，不会进入
commit。若确实需要评审并提交 PyAsc upstream 基线更新，先解除屏蔽：

```bash
bash mlir/scripts/sync_pyasc_upstream.sh --unprotect-gitlink
```

导出 Autofuse 需要维护的最小 patchset 时，必须显式选择 commit 或 range：

```bash
bash mlir/scripts/export_pyasc_patch.sh --commit <sha>
bash mlir/scripts/export_pyasc_patch.sh --range <start>..<end>
bash mlir/scripts/export_pyasc_patch.sh --pyasc-dir /path/to/pyasc --commit <sha>
```

## 验证

完整 Phase 1 验证矩阵见 [docs/verification_matrix.md](docs/verification_matrix.md)。

轻量检查：

```bash
bash mlir/scripts/check_mlir_build_switch.sh
bash mlir/scripts/check_llvm_dependency_resolution.sh
bash mlir/scripts/check_pyasc_integration.sh
```

Docker 编译验证：

```bash
# 验证 --no-autofuse 仍只构建 SuperKernel/package 路径，不进入 Autofuse/MLIR
bash mlir/scripts/enter_dev_container.sh -- bash build.sh --pkg --no-autofuse -j 8

# 验证默认 Autofuse 构建路径，MLIR 保持关闭
bash mlir/scripts/enter_dev_container.sh -- bash build.sh --pkg -j 8

# 验证显式打开 MLIR 工程接入后，LLVM/MLIR/PyAsc 路径可配置并可编译
bash mlir/scripts/enter_dev_container.sh -- bash build.sh --pkg --enable-mlir -j 8

# 验证 MLIR 构建路径下的 Autofuse UT coverage
bash mlir/scripts/enter_dev_container.sh -- \
  bash build.sh -u --module=autofuse_framework --enable-mlir -c -j 8
```

所有 `build.sh` 构建命令必须带 `-j 8`。
