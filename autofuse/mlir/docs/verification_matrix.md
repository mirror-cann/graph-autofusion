# Phase 1 验证矩阵

## 执行约定

- 默认从仓库根目录执行：`/Volumes/GM9/code/autofuse`；容器内对应 `/workspace`。
- 除本地静态检查、镜像构建和 NPU 真机命令外，构建和测试命令优先通过
  `bash autofuse/mlir/scripts/enter_dev_container.sh -- <command>` 执行。
- 所有 `build.sh` 和 `cmake --build` 命令必须带 `-j 8`。
- `.artifacts/`、coverage、`kernel_meta` 和 run 包输出不得入库。

## 矩阵

| 路径 | 环境 | 命令 | 预期结果 | 产物/说明 |
|---|---|---|---|---|
| Docker 开发镜像 | 宿主机 + Docker | `bash autofuse/mlir/docker/build_llvm_image.sh --base-only` | 可一键构建基础编译镜像；基础镜像不内置 CANN | Docker image |
| 用户自定义镜像 | 宿主机 + Docker | `AF_MLIR_DEV_IMAGE=<image> bash autofuse/mlir/scripts/enter_dev_container.sh -- bash autofuse/mlir/scripts/collect_env_manifest.sh` | 能进入用户镜像并输出环境 manifest | `autofuse/mlir/.artifacts/env/dev_env_manifest.json` |
| CANN/tikicpulib 环境 | Docker 编译环境 | `AF_CANN_VOLUME=<cann-volume> bash autofuse/mlir/scripts/enter_dev_container.sh -- bash -lc 'cmake -S <probe> -B <probe-build> -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH"'` | 容器入口保留 CANN 安装前缀，`find_package(tikicpulib)` 可找到官方 config | `ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0` |
| Docker 默认包 | Docker 编译环境 | `AF_CANN_VOLUME=<cann-volume> bash autofuse/mlir/scripts/enter_dev_container.sh -- bash build.sh --pkg -j 8` | 默认路径不依赖 LLVM/MLIR/PyAsc；当前执行状态见“当前本地状态” | 目标产物 `build_out/` |
| Docker 跳过 Autofuse | Docker 编译环境 | `AF_CANN_VOLUME=<cann-volume> bash autofuse/mlir/scripts/enter_dev_container.sh -- bash build.sh --pkg --no-autofuse -j 8` | 保持既有 `--no-autofuse` 行为，不进入 Autofuse/MLIR 路径 | `build_out/` |
| 默认包边界 | 宿主机或 Docker 编译环境 | `find build_out -maxdepth 2 -type f \| sort \| rg 'af-opt\|MLIR\|LLVM\|pyasc' \|\| true` | 默认 run 包中没有 `af-opt`、LLVM、MLIR 或 PyAsc 运行时制品 | 任务 13 的包边界证据 |
| LLVM 本地依赖 | 宿主机或 Docker 编译环境 | `AF_LLVM_ROOT=/path/to/prebuilt/llvm bash autofuse/mlir/scripts/prepare_mlir_deps.sh` | 找到 `llvm-config` 和 `MLIRConfig.cmake` 并输出 manifest | `autofuse/mlir/.artifacts/env/llvm_manifest.json` |
| PyAsc 官方仓同步 | 宿主机 | `bash autofuse/mlir/scripts/sync_pyasc_upstream.sh` | 基于官方仓固定 commit 应用本仓 patchset，不依赖个人 fork；默认保护 gitlink | `autofuse/mlir/externals/pyasc` 本地 worktree |
| Autofuse UT | Docker 编译环境 | `AF_CANN_VOLUME=<cann-volume> bash autofuse/mlir/scripts/enter_dev_container.sh -- bash build.sh -u --module=autofuse_framework -j 8` | 既有 Autofuse framework UT 通过 | UT 日志 |
| Autofuse E2E ST | Docker 编译环境 | `AF_CANN_VOLUME=<cann-volume> bash autofuse/mlir/scripts/enter_dev_container.sh -- bash build.sh -s --module=autofuse_e2e -j 8` | 既有 Autofuse E2E ST 通过 | ST 日志 |
| MLIR 包 | Docker 编译环境 + LLVM/PyAsc | `AF_CANN_VOLUME=<cann-volume> bash autofuse/mlir/scripts/enter_dev_container.sh -- bash build.sh --pkg --enable-mlir -j 8` | LLVM/PyAsc 配置正确时构建通过；默认关闭路径不受影响 | `build/`、`build_out/` |
| `af-opt` 构建 | Docker 编译环境 + LLVM/PyAsc | `bash autofuse/mlir/scripts/enter_dev_container.sh -- cmake --build build --target af-opt -j 8` | 生成开发工具 `af-opt`；不安装进默认 run 包 | `build/autofuse/mlir/tools/af-opt/af-opt` |
| MLIR 冒烟 | Docker 编译环境 + LLVM/PyAsc | `bash autofuse/mlir/scripts/enter_dev_container.sh -- cmake --build build --target check-autofuse-mlir -j 8` | lit 冒烟通过 | lit 日志 |
| PyAsc op inventory | 宿主机或 Docker 编译环境 | `python3 autofuse/mlir/tools/af-mlir-tools/scan_pyasc_coverage.py --format csv --output autofuse/mlir/docs/pyasc_ascendc_op_inventory.csv` | 从 PyAsc TableGen 源提取 `ascendc` / `emitasc` op 清单 | CSV 快照 |
| ascir tool 兼容入口 | NPU 环境 | `MAKEFLAGS=-j8 bash autofuse/tests/st/codegen/ascir_tool/test_ascir.sh --mode=0 --case=isinf_maskedfill_fusion` | 既有入口参数保持不变；真实链路结果按日志记录 | 当前真实 NPU 链路已知未打通 |
| ascir tool wrapper | NPU 环境 | `python3 autofuse/mlir/tools/af-mlir-tools/af_mlir_case.py --case isinf_maskedfill_fusion --mode 0 -j 8` | wrapper 设置 `MAKEFLAGS=-j8` 并记录 case 返回码 | `autofuse/mlir/.artifacts/isinf_maskedfill_fusion.status` |

## 当前本地状态

- 2026-07-04：`AF_CANN_VOLUME=af-cann-910 bash autofuse/mlir/scripts/enter_dev_container.sh -- bash build.sh --pkg -j 8`
  已验证 `tikicpulib` 路径问题修复，原先被跳过的 AscendC/backend E2E 目标会进入编译。
  当前阻塞在 `isinf_maskedfill_test_e2e`，CANN cpudebug 头报 `vabs(int*)` 重载不匹配。

## Reviewer 快速检查

```bash
rg -n -- 'build_llvm_image|collect_env_manifest|prepare_mlir_deps|sync_pyasc_upstream|-j 8|--no-autofuse|--enable-mlir|check-autofuse-mlir|af_mlir_case' \
  autofuse/mlir/docs/verification_matrix.md
```
