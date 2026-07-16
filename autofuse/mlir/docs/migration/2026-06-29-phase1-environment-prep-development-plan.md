# Phase 1 环境准备详细开发方案

> **给执行 agent 的要求：** 实施本计划时必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，按任务逐项推进。任务步骤使用 `- [ ]` 复选框，便于跟踪执行状态。

**状态：** Closed（2026-07-04，Phase 1 已完成）。下一阶段进入 Phase 2 Codegen 迁移。

**目标：** 在 `af-mlir` 分支上完成 Phase 1 环境准备：可复用的开发/编译验证 Docker 环境、可显式配置的 LLVM/MLIR 预编译依赖、默认关闭的 MLIR/PyAsc 工程集成、基于 `ascir_tool` 的测试工具适配、legacy 回归基线冻结。

**总体架构：** 从 `main` 新建 `af-mlir` 分支。先建立 Docker 编译验证环境和 LLVM/MLIR 依赖输入，再开发默认关闭的 MLIR 工程接线。MLIR 相关新增代码尽量集中在 `autofuse/mlir/`，只在 `build.sh`、`autofuse/CMakeLists.txt`、`.gitmodules`、`.gitignore` 等少量入口做薄接线。默认主线仍走 legacy codegen；`ENABLE_AUTOFUSE_MLIR=ON` 只表示编译 MLIR 能力，运行路由由后续 `AF_MLIR_CODEGEN=off|on|compare` 控制。

**技术栈：** Bash、Docker、CMake 3.16+、C++17、LLVM/MLIR、PyAsc、lit/FileCheck、Python3、gtest、现有 `tests/st/codegen/ascir_tool`。

## 全局约束

- 基于 `main` 新建并使用 `af-mlir` 分支开发。
- 开发模式遵循 `autofuse/mlir/docs/agent_development_guide.md`：本地编辑，Docker/Linux 环境编译验证，NPU 环境只做真机实测。
- 必须先完成 Docker 编译验证环境和 LLVM/MLIR 依赖输入，再进入 MLIR 代码开发任务。
- 除明确标注为宿主机或 NPU 环境的步骤外，后续构建和测试命令默认在任务 1 的 Docker 编译验证环境中执行；文档中保留原始命令时，执行 agent 应使用 `autofuse/mlir/scripts/enter_dev_container.sh -- <command>` 包裹。
- Docker 编译验证环境同时支持用户自定义镜像和项目一键构建镜像。
- CANN Toolkit 不内置到 LLVM 镜像；通过宿主机挂载、CI 注入或远端 NPU 环境预装方式提供，并记录版本。
- LLVM/MLIR 依赖同时支持团队提供的预编译镜像/制品，以及用户通过 `AF_LLVM_ROOT` 或 `LLVM_BUILD_DIR` 指定的本地已编译目录。
- 构建脚本不得静默下载或编译 LLVM、PyAsc、CANN Toolkit；所有重型依赖准备必须是显式命令。
- 所有开发、编译验证、NPU 实测和 CI 环境必须输出环境 manifest，至少记录架构、OS、编译器、CMake、Python、CANN、LLVM/MLIR 路径和镜像 digest。
- 默认路径必须是 `BUILD_AUTOFUSE=ON, ENABLE_AUTOFUSE_MLIR=OFF, AF_MLIR_CODEGEN=off`，且不发现、不依赖 LLVM/MLIR/PyAsc。
- `--no-autofuse` 语义不变，仍表示 `BUILD_AUTOFUSE=OFF`，不复用为 MLIR 开关。
- 新增 MLIR 代码尽量集中在 `autofuse/mlir/`；现有模块只做必要入口接线。
- PyAsc 来源必须是官方仓 `https://gitcode.com/cann/pyasc.git` 加本仓维护的 patch，不得依赖个人 fork。
- 所有构建命令限制并行度，优先 `-j 8`。
- 默认 run 包不得安装 MLIR 工具或 LLVM/MLIR/PyAsc 运行时依赖。
- 复用现有 `tests/st/codegen/ascir_tool` 做 E2E/NPU 基线。
- Phase 1 只覆盖工程集成、测试工具适配和回归基线冻结。
- 测试临时产物、generated code、coverage、profiling、`kernel_meta` 不得入库。

---

## 目录规划

新增目录尽量集中在：

```text
autofuse/mlir/
  CMakeLists.txt
  cmake/
    AutofuseMlirDeps.cmake
    AutofuseMlirTargets.cmake
  docker/
    README.md
    Dockerfile.llvm
    build_llvm_image.sh
  externals/
    pyasc/                       # git submodule, only used when ENABLE_AUTOFUSE_MLIR=ON
  tools/
    af-opt/
      CMakeLists.txt
      main.cpp
    af-mlir-tools/
      af_mlir_diff.py
      af_mlir_case.py
      freeze_codegen_baseline.py
  scripts/
    collect_env_manifest.sh
    enter_dev_container.sh
    resolve_llvm_env.sh
    prepare_mlir_deps.sh
    sync_pyasc_upstream.sh
    export_pyasc_patch.sh
  patches/
    pyasc/
      README.md
  test/
    CMakeLists.txt
    lit.cfg.py
    lit.site.cfg.py.in
    smoke/
      parse_print.mlir
  docs/
    pyasc_ascendc_op_coverage.md
    baseline_policy.md
```

必要入口接线文件：

```text
build.sh
autofuse/CMakeLists.txt
.gitmodules
.gitignore
```

原则：除入口接线外，不直接改 `autofuse/codegen/`、`autofuse/optimize/`、`autofuse/att/` 的业务逻辑。后续真正实现 MLIR codegen 路径时再进入这些目录。

---

## 任务 0：分支和工作区准备

**涉及文件：**

- 不修改源码。

**产出接口：**

- 基于最新 `main` 的干净开发分支 `af-mlir`。

- [ ] **步骤 1：检查当前工作区**

在仓库根目录 `/Volumes/GM9/code/autofuse` 执行：

```bash
git status --short
git branch --show-current
```

预期：切分支前确认当前脏文件。不得丢弃与本任务无关的用户改动。

- [ ] **步骤 2：从 main 新建分支**

```bash
git fetch origin
git switch main
git pull --ff-only
git switch -c af-mlir
```

预期：当前分支切换到 `af-mlir`。

- [ ] **步骤 3：确认分支**

```bash
git branch --show-current
```

预期：

```text
af-mlir
```

---

## 任务 1：准备开发和 Docker 编译验证环境

**涉及文件：**

- 新增：`autofuse/mlir/docker/README.md`
- 新增：`autofuse/mlir/docker/Dockerfile.llvm`
- 新增：`autofuse/mlir/docker/build_llvm_image.sh`
- 新增：`autofuse/mlir/scripts/enter_dev_container.sh`
- 新增：`autofuse/mlir/scripts/collect_env_manifest.sh`
- 修改：`.gitignore`

**接口关系：**

- 输入：用户自定义 Docker 镜像，或项目一键构建的 `AF_MLIR_DEV_IMAGE`。
- 输出：
  - 可重复进入的 Linux 编译验证容器。
  - 环境 manifest `autofuse/mlir/.artifacts/env/dev_env_manifest.json`。
  - 默认构建 baseline 结果。

- [ ] **步骤 1：定义 Docker 环境契约**

新增 `autofuse/mlir/docker/README.md`：

```markdown
# Autofuse MLIR Docker 环境

## 支持模式

1. 用户自定义镜像：设置 `AF_MLIR_DEV_IMAGE=<image>` 后直接进入容器验证。
2. 项目一键构建镜像：执行 `bash autofuse/mlir/docker/build_llvm_image.sh` 构建 dev+LLVM 镜像；只构建内部工具底座时使用 `--base-only`。

## CANN Toolkit 约束

CANN Toolkit 不内置到 LLVM 镜像。开发者通过 `AF_CANN_ROOT` 指向宿主机或 CI/NPU 环境中的 Toolkit 安装目录，容器启动时挂载到 `/opt/cann`。

## 必须记录的环境信息

- Docker image tag 和 digest
- CPU 架构
- OS 版本
- gcc/g++ 版本
- CMake/Ninja 版本
- Python 版本
- CANN Toolkit 路径和版本
- LLVM/MLIR 路径和版本
```

- [ ] **步骤 2：新增基础开发镜像 Dockerfile**

新增 `autofuse/mlir/docker/Dockerfile.llvm`：

```dockerfile
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    cmake \
    g++ \
    gcc \
    git \
    make \
    ninja-build \
    python3 \
    python3-pip \
    python3-venv \
    rsync \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
```

- [ ] **步骤 3：新增统一镜像构建脚本**

新增 `autofuse/mlir/docker/build_llvm_image.sh`：

```bash
#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
image="${AF_MLIR_DEV_BASE_IMAGE:-autofuse-mlir-dev-base:$(uname -m)}"
base="${AF_MLIR_DEV_BASE_OS_IMAGE:-ubuntu:22.04}"

docker build \
  --target dev-base \
  --build-arg BASE_IMAGE="${base}" \
  -t "${image}" \
  -f "${repo_root}/autofuse/mlir/docker/Dockerfile.llvm" \
  "${repo_root}"

echo "${image}"
```

- [ ] **步骤 4：新增进入开发容器脚本**

新增 `autofuse/mlir/scripts/enter_dev_container.sh`：

```bash
#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
image="${AF_MLIR_DEV_IMAGE:-autofuse-mlir-dev:$(uname -m)}"
cann_root="${AF_CANN_ROOT:-}"

mount_args=(-v "${repo_root}:/workspace" -w /workspace)
env_args=(-e CANN_ROOT=/opt/cann)
if [ -n "${cann_root}" ]; then
  mount_args+=(-v "${cann_root}:/opt/cann:ro")
fi

if [ "$#" -gt 0 ] && [ "$1" = "--" ]; then
  shift
fi

tty_args=()
if [ -t 0 ] && [ -t 1 ]; then
  tty_args=(-it)
fi

docker run --rm "${tty_args[@]}" "${mount_args[@]}" "${env_args[@]}" "${image}" "$@"
```

- [ ] **步骤 5：新增环境 manifest 脚本**

新增 `autofuse/mlir/scripts/collect_env_manifest.sh`：

```bash
#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
out_dir="${repo_root}/autofuse/mlir/.artifacts/env"
mkdir -p "${out_dir}"
manifest="${out_dir}/dev_env_manifest.json"

cat > "${manifest}" <<EOF
{
  "arch": "$(uname -m)",
  "os": "$(uname -s)",
  "gcc": "$(gcc --version | head -n 1 || true)",
  "cmake": "$(cmake --version | head -n 1 || true)",
  "python": "$(python3 --version 2>&1 || true)",
  "cann_root": "${CANN_ROOT:-}",
  "llvm_root": "${AF_LLVM_ROOT:-${LLVM_BUILD_DIR:-}}",
  "docker_image": "${AF_MLIR_DEV_IMAGE:-}",
  "docker_digest": "${AF_MLIR_DEV_IMAGE_DIGEST:-}"
}
EOF

echo "${manifest}"
```

- [ ] **步骤 6：忽略环境产物**

`.gitignore` 追加：

```gitignore

# Autofuse MLIR environment artifacts
autofuse/mlir/.artifacts/
```

- [ ] **步骤 7：在 Docker 中冻结默认构建 baseline**

如果使用项目镜像：

```bash
bash autofuse/mlir/docker/build_llvm_image.sh --base-only
```

如果使用自定义镜像：

```bash
export AF_MLIR_DEV_IMAGE=<your-image>
```

在同一个 Docker 环境中执行：

```bash
bash autofuse/mlir/scripts/enter_dev_container.sh -- bash autofuse/mlir/scripts/collect_env_manifest.sh
bash autofuse/mlir/scripts/enter_dev_container.sh -- sh build.sh --pkg -j 8
bash autofuse/mlir/scripts/enter_dev_container.sh -- sh build.sh --pkg --no-autofuse -j 8
```

预期：

- 默认构建和 `--no-autofuse` baseline 在统一 Docker 环境中完成记录。
- 失败时先记录环境和失败原因，不进入 MLIR 代码改造。

---

## 任务 2：准备 LLVM/MLIR 镜像和本地依赖输入

**涉及文件：**

- 修改：`autofuse/mlir/docker/README.md`
- 修改：`autofuse/mlir/docker/Dockerfile.llvm`
- 新增：`autofuse/mlir/docker/build_llvm_image.sh`
- 新增：`autofuse/mlir/docs/llvm_dependency_policy.md`

**接口关系：**

- 输入：
  - 团队预编译 LLVM/MLIR 镜像或制品。
  - 用户本地已编译目录 `AF_LLVM_ROOT` 或 `LLVM_BUILD_DIR`。
- 输出：
  - `MLIRConfig.cmake` 和 `bin/llvm-config` 可被后续任务发现。
  - 统一 dev+LLVM 镜像，默认镜像名为 `autofuse-mlir-dev:<arch>`。
  - LLVM/MLIR manifest，记录版本、架构、ABI、镜像名、可用时的镜像 digest 和构建参数。

- [x] **步骤 1：新增 LLVM 依赖策略文档**

新增 `autofuse/mlir/docs/llvm_dependency_policy.md`：

```markdown
# LLVM/MLIR 依赖策略

## 目标场景

1. 本地开发：开发者可以直接使用团队 LLVM 镜像，也可以通过 `AF_LLVM_ROOT` 或 `LLVM_BUILD_DIR` 指向本地已编译目录。
2. Docker 编译验证：容器内通过统一 dev+LLVM 镜像内置路径提供 LLVM/MLIR。
3. 远端 NPU 实测：推送代码和已确认的 LLVM/MLIR 依赖信息到 NPU 服务器，不要求服务器重新编译 LLVM。
4. CI 流水线：流水线拉取团队镜像或预编译制品，禁止在普通验证任务中临时全量编译 LLVM。

## 非目标

- 不把 CANN Toolkit 打进 LLVM 镜像。
- 不在 `build.sh` 中静默下载或编译 LLVM。
- 不依赖个人仓库作为默认依赖来源。

## 必须校验

- `bin/llvm-config` 可执行。
- `lib/cmake/mlir/MLIRConfig.cmake` 存在。
- 架构与编译环境匹配。
- `_GLIBCXX_USE_CXX11_ABI` 与 Autofuse 构建约束一致。
```

- [x] **步骤 2：扩展统一 Dockerfile 的 LLVM 镜像 target**

在 `autofuse/mlir/docker/Dockerfile.llvm` 中新增 `dev-llvm` target：

```dockerfile
ARG BASE_IMAGE=autofuse-mlir-dev-base:local
FROM ${BASE_IMAGE}

ARG LLVM_INSTALL_PREFIX=/opt/llvm
COPY llvm/ ${LLVM_INSTALL_PREFIX}/
ENV AF_LLVM_ROOT=${LLVM_INSTALL_PREFIX}
ENV LLVM_BUILD_DIR=${LLVM_INSTALL_PREFIX}
ENV LLVM_DIR=${LLVM_INSTALL_PREFIX}/lib/cmake/llvm
ENV MLIR_DIR=${LLVM_INSTALL_PREFIX}/lib/cmake/mlir
ENV PATH=${LLVM_INSTALL_PREFIX}/bin:${PATH}
```

该 Dockerfile 只定义运行环境边界。LLVM 编译或制品拷贝由显式脚本完成，不在 `build.sh` 中隐式发生。

- [x] **步骤 3：新增 LLVM 镜像构建脚本**

新增 `autofuse/mlir/docker/build_llvm_image.sh`：

```bash
#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
image="${AF_MLIR_LLVM_IMAGE:-${AF_MLIR_DEV_IMAGE:-autofuse-mlir-dev:$(uname -m)}}"
base="${AF_MLIR_LLVM_BASE_IMAGE:-${AF_MLIR_DEV_BASE_IMAGE:-autofuse-mlir-dev-base:$(uname -m)}}"
llvm_root="${AF_LLVM_ROOT:-}"

if [ -z "${llvm_root}" ]; then
  echo "ERROR: set AF_LLVM_ROOT to an existing prebuilt LLVM/MLIR directory." >&2
  exit 1
fi
if [ ! -x "${llvm_root}/bin/llvm-config" ]; then
  echo "ERROR: ${llvm_root}/bin/llvm-config is missing or not executable." >&2
  exit 1
fi
if [ ! -f "${llvm_root}/lib/cmake/mlir/MLIRConfig.cmake" ]; then
  echo "ERROR: ${llvm_root}/lib/cmake/mlir/MLIRConfig.cmake is missing." >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
mkdir -p "${tmp_dir}/llvm"
rsync -a "${llvm_root}/" "${tmp_dir}/llvm/"

cat > "${tmp_dir}/Dockerfile" <<EOF
ARG BASE_IMAGE=${base}
FROM \${BASE_IMAGE}
COPY llvm/ /opt/llvm/
ENV AF_LLVM_ROOT=/opt/llvm
ENV LLVM_BUILD_DIR=/opt/llvm
ENV PATH=/opt/llvm/bin:\${PATH}
EOF

docker build -t "${image}" "${tmp_dir}"
docker run --rm "${image}" llvm-config --version
docker run --rm "${image}" test -f /opt/llvm/lib/cmake/mlir/MLIRConfig.cmake
echo "${image}"
```

`AF_MLIR_LLVM_IMAGE` 和 `AF_MLIR_LLVM_BASE_IMAGE` 只作为兼容别名保留；新文档和 README 均使用
`AF_MLIR_DEV_IMAGE` 与 `AF_MLIR_DEV_BASE_IMAGE`。

- [x] **步骤 4：记录 LLVM manifest**

`build_llvm_image.sh` 会写入 `/opt/llvm/share/autofuse-mlir/llvm-image-manifest.json`：

```json
{
  "llvm_root": "/opt/llvm",
  "llvm_version": "from llvm-config --version",
  "llvm_commit": "from llvm-project checkout or release tag",
  "arch": "x86_64 or aarch64",
  "abi": "_GLIBCXX_USE_CXX11_ABI value",
  "image": "autofuse-mlir-dev:<arch>",
  "image_digest": "registry digest when available"
}
```

- [ ] **步骤 5：验证团队镜像和本地目录两种输入**

团队镜像路径：

```bash
export AF_MLIR_DEV_IMAGE=<team-llvm-image>
bash autofuse/mlir/scripts/enter_dev_container.sh -- llvm-config --version
bash autofuse/mlir/scripts/enter_dev_container.sh -- test -f "${AF_LLVM_ROOT:-/opt/llvm}/lib/cmake/mlir/MLIRConfig.cmake"
```

本地目录路径：

```bash
export AF_LLVM_ROOT=/path/to/prebuilt/llvm
bash autofuse/mlir/docker/build_llvm_image.sh
```

预期：两种输入都能提供 `llvm-config` 和 `MLIRConfig.cmake`；CANN Toolkit 不出现在 LLVM 镜像层。

当前进度：

- 已验证本地候选 dev+LLVM 镜像 `autofuse-mlir-dev:arm64`，`llvm-config --version` 为 `21.1.8`，
  `/opt/llvm/lib/cmake/mlir/MLIRConfig.cmake` 和
  `/opt/llvm/share/autofuse-mlir/llvm-image-manifest.json` 均存在。
- 本地目录输入仍待可用的宿主机 `AF_LLVM_ROOT` 后验证。

---

## 任务 3：新增默认关闭的 MLIR 构建开关

**涉及文件：**

- 修改：`build.sh`
- 修改：`autofuse/CMakeLists.txt`
- 新增：`autofuse/mlir/CMakeLists.txt`
- 新增：`autofuse/mlir/scripts/check_mlir_build_switch.sh`

**接口关系：**

- 输入：既有 `BUILD_AUTOFUSE` 行为。
- 输出：
  - CMake option `ENABLE_AUTOFUSE_MLIR`，默认 `OFF`。
  - `build.sh` 参数 `--enable-mlir`，默认关闭。
  - 空的 `autofuse/mlir` 子目录，后续任务再补 target。

- [x] **步骤 1：先验证默认路径现状**

```bash
sh build.sh --pkg -j 8
sh build.sh --pkg --no-autofuse -j 8
```

预期：当前行为只允许因为已有环境问题失败；记录失败原因，不在本任务中扩大修改范围。

- [x] **步骤 2：给 `build.sh` 增加参数解析**

在 `build.sh` 中：

- usage 增加 `--enable-mlir`。
- `checkopts` 默认值增加 `ENABLE_AUTOFUSE_MLIR="off"`。
- `getopt` 长参数列表增加 `enable-mlir`。
- 参数分支增加：

```bash
--enable-mlir)
  ENABLE_AUTOFUSE_MLIR="on"
  shift
  ;;
```

- `cmake_config` 中追加：

```bash
if [ "X$ENABLE_AUTOFUSE_MLIR" == "Xon" ]; then
  extra_option="${extra_option} -DENABLE_AUTOFUSE_MLIR=ON"
else
  extra_option="${extra_option} -DENABLE_AUTOFUSE_MLIR=OFF"
fi
```

- [x] **步骤 3：给 CMake 增加开关和隔离目录**

在 `autofuse/CMakeLists.txt` 的全局设置区域附近增加：

```cmake
option(ENABLE_AUTOFUSE_MLIR "Enable Autofuse MLIR migration infrastructure" OFF)

if(ENABLE_AUTOFUSE_MLIR)
    add_subdirectory(mlir)
endif()
```

新增 `autofuse/mlir/CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.16)

message(STATUS "Enable Autofuse MLIR migration infrastructure")
```

- [x] **步骤 4：验证默认路径不依赖 MLIR**

```bash
sh build.sh --pkg -j 8
sh build.sh --pkg --no-autofuse -j 8
```

预期：

- 默认构建不需要 `LLVM_BUILD_DIR`、`AF_LLVM_ROOT`、`MLIR_DIR` 或 PyAsc。
- `--no-autofuse` 仍映射为 `BUILD_AUTOFUSE=OFF`。

- [x] **步骤 5：验证打开开关后能进入占位目录**

```bash
sh build.sh --pkg --enable-mlir -j 8
```

预期：CMake 输出 `Enable Autofuse MLIR migration infrastructure`。如果后续依赖逻辑尚未实现而失败，失败不能影响默认构建。

当前进度：

- 已新增 `autofuse/mlir/scripts/check_mlir_build_switch.sh`，用 stub `cmake` 验证默认 `OFF`、
  显式 `ON` 和 `--no-autofuse` 参数映射。
- 已在 Docker dev+LLVM 镜像 + CANN volume 中验证：
  - `bash build.sh --pkg --no-autofuse -j 8` 成功，CMake 参数包含
    `-DBUILD_AUTOFUSE=OFF -DENABLE_AUTOFUSE_MLIR=OFF`。
  - `bash build.sh --pkg --enable-mlir -j 8` 成功，CMake 输出
    `Enable Autofuse MLIR migration infrastructure`。
  - `bash build.sh --pkg -j 8` 成功，CMake 参数包含
    `-DBUILD_AUTOFUSE=ON -DENABLE_AUTOFUSE_MLIR=OFF`。
  - 默认包 staging 与 `build_out` 未出现 `af-opt`、`MLIR`、`LLVM` 或 `pyasc` 文件名。

---

## 任务 4：新增 LLVM/MLIR 依赖解析

**涉及文件：**

- 新增：`autofuse/mlir/cmake/AutofuseMlirDeps.cmake`
- 新增：`autofuse/mlir/scripts/resolve_llvm_env.sh`
- 新增：`autofuse/mlir/scripts/prepare_mlir_deps.sh`
- 修改：`autofuse/mlir/CMakeLists.txt`
- 修改：`.gitignore`

**接口关系：**

- 输入：`ENABLE_AUTOFUSE_MLIR=ON`、任务 2 准备的团队 LLVM 镜像/制品、可选 `LLVM_BUILD_DIR`、可选 `AF_LLVM_ROOT`。
- 输出：只在 MLIR 开关打开时执行 `find_package(MLIR REQUIRED CONFIG)`。

- [x] **步骤 1：忽略本地依赖和生成物目录**

`.gitignore` 追加：

```gitignore

# Autofuse MLIR 本地依赖和生成产物
autofuse/mlir/.deps/
autofuse/mlir/.baseline_artifacts/
```

- [x] **步骤 2：新增依赖路径解析脚本**

新增 `autofuse/mlir/scripts/resolve_llvm_env.sh`：

```bash
#!/bin/bash
set -euo pipefail

candidate="${LLVM_BUILD_DIR:-}"
if [ -z "${candidate}" ] && [ -n "${AF_LLVM_ROOT:-}" ]; then
  candidate="${AF_LLVM_ROOT}"
fi
if [ -z "${candidate}" ]; then
  candidate="$(cd "$(dirname "$0")/.." && pwd)/.deps/llvm"
fi

if [ ! -x "${candidate}/bin/llvm-config" ]; then
  echo "ERROR: LLVM/MLIR build not found." >&2
  echo "Set LLVM_BUILD_DIR or AF_LLVM_ROOT to a directory containing bin/llvm-config." >&2
  exit 1
fi

echo "${candidate}"
```

- [x] **步骤 3：新增显式依赖准备脚本**

新增 `autofuse/mlir/scripts/prepare_mlir_deps.sh`：

```bash
#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
llvm_root="$(bash "${repo_root}/autofuse/mlir/scripts/resolve_llvm_env.sh")"
out_dir="${repo_root}/autofuse/mlir/.artifacts/env"
mkdir -p "${out_dir}"
manifest="${out_dir}/llvm_manifest.json"

cat > "${manifest}" <<EOF
{
  "llvm_root": "${llvm_root}",
  "llvm_version": "$("${llvm_root}/bin/llvm-config" --version)",
  "llvm_bindir": "$("${llvm_root}/bin/llvm-config" --bindir)",
  "llvm_libdir": "$("${llvm_root}/bin/llvm-config" --libdir)",
  "mlir_config": "${llvm_root}/lib/cmake/mlir/MLIRConfig.cmake",
  "arch": "$(uname -m)",
  "docker_image": "${AF_MLIR_DEV_IMAGE:-}",
  "docker_digest": "${AF_MLIR_DEV_IMAGE_DIGEST:-}"
}
EOF

echo "${manifest}"
```

- [x] **步骤 4：新增 CMake 依赖发现逻辑**

新增 `autofuse/mlir/cmake/AutofuseMlirDeps.cmake`：

```cmake
if(NOT ENABLE_AUTOFUSE_MLIR)
    return()
endif()

set(AF_LLVM_ROOT "$ENV{AF_LLVM_ROOT}" CACHE PATH "Prebuilt LLVM/MLIR root for Autofuse MLIR")
set(LLVM_BUILD_DIR "" CACHE PATH "LLVM/MLIR build directory for Autofuse MLIR")

if(LLVM_BUILD_DIR)
    set(_af_llvm_root "${LLVM_BUILD_DIR}")
elseif(AF_LLVM_ROOT)
    set(_af_llvm_root "${AF_LLVM_ROOT}")
else()
    set(_af_llvm_root "${CMAKE_CURRENT_LIST_DIR}/../.deps/llvm")
endif()

set(_af_mlir_cmake_dir "${_af_llvm_root}/lib/cmake/mlir")
if(NOT EXISTS "${_af_mlir_cmake_dir}/MLIRConfig.cmake")
    message(FATAL_ERROR
        "ENABLE_AUTOFUSE_MLIR=ON requires MLIRConfig.cmake. "
        "Set -DLLVM_BUILD_DIR=/path/to/llvm-build or AF_LLVM_ROOT=/path/to/prebuilt/llvm. "
        "Checked: ${_af_mlir_cmake_dir}")
endif()

list(APPEND CMAKE_PREFIX_PATH "${_af_llvm_root}")
set(MLIR_DIR "${_af_mlir_cmake_dir}" CACHE PATH "MLIR CMake config directory" FORCE)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
```

修改 `autofuse/mlir/CMakeLists.txt`：

```cmake
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AutofuseMlirDeps.cmake)
```

- [x] **步骤 5：验证缺失依赖只影响打开开关的路径**

```bash
sh build.sh --pkg -j 8
sh build.sh --pkg --enable-mlir -j 8
```

预期：

- 默认构建仍不依赖 LLVM。
- 开启 MLIR 但未配置依赖时，报错包含 `ENABLE_AUTOFUSE_MLIR=ON requires MLIRConfig.cmake`。

当前进度：

- 已新增 `autofuse/mlir/scripts/check_llvm_dependency_resolution.sh`，用 fake LLVM/MLIR 目录验证：
  - `LLVM_BUILD_DIR`、`AF_LLVM_ROOT`、`MLIR_DIR` 和默认 `.deps/llvm` 解析优先级。
  - `resolve_llvm_env.sh` 在缺失 `bin/llvm-config` 时给出明确错误。
  - `prepare_mlir_deps.sh` 生成 `autofuse/mlir/.artifacts/env/llvm_manifest.json`。
  - `AutofuseMlirDeps.cmake` 在 `ENABLE_AUTOFUSE_MLIR=OFF` 时直接返回。
  - `ENABLE_AUTOFUSE_MLIR=ON` 且缺失 `MLIRConfig.cmake` 时失败信息包含
    `ENABLE_AUTOFUSE_MLIR=ON requires MLIRConfig.cmake`。
  - `ENABLE_AUTOFUSE_MLIR=ON` 且提供有效 LLVM/MLIR CMake config 时配置通过。
- 已在 Docker dev+LLVM 镜像 + CANN volume 中验证：
  - `bash autofuse/mlir/scripts/check_mlir_build_switch.sh` 成功。
  - `bash autofuse/mlir/scripts/check_llvm_dependency_resolution.sh` 成功。
  - `bash autofuse/mlir/scripts/prepare_mlir_deps.sh` 成功生成
    `/workspace/autofuse/mlir/.artifacts/env/llvm_manifest.json`，内容包含
    `/opt/llvm/lib/cmake/mlir/MLIRConfig.cmake`。
  - `bash build.sh --pkg --enable-mlir -j 8` 成功，CMake 输出
    `Enable Autofuse MLIR migration infrastructure`，最终生成
    `/workspace/build_out/cann-graph-autofusion_9.1.0_linux-aarch64.run`。
  - `bash build.sh --pkg --no-autofuse -j 8` 成功，CMake 参数包含
    `-DBUILD_AUTOFUSE=OFF -DENABLE_AUTOFUSE_MLIR=OFF`。

---

## 任务 5：在隔离目录接入 PyAsc

**涉及文件：**

- 修改：`.gitmodules`
- 新增：`autofuse/mlir/externals/pyasc` submodule
- 新增：`autofuse/mlir/patches/pyasc/README.md`
- 新增：`autofuse/mlir/scripts/sync_pyasc_upstream.sh`
- 新增：`autofuse/mlir/scripts/export_pyasc_patch.sh`
- 新增：`autofuse/mlir/scripts/check_pyasc_integration.sh`
- 修改：`autofuse/mlir/CMakeLists.txt`

**接口关系：**

- 输入：任务 4 已发现 MLIR。
- 输出：只在 `ENABLE_AUTOFUSE_MLIR=ON` 时可见的 PyAsc 目标；PyAsc 修改以官方仓加本仓 patch 的方式集成。

- [x] **步骤 1：在隔离目录新增 submodule**

在仓库根目录执行：

```bash
git submodule add https://gitcode.com/cann/pyasc.git autofuse/mlir/externals/pyasc
```

预期：`.gitmodules` 中新增 `autofuse/mlir/externals/pyasc`，URL 指向官方仓，不指向个人 fork。

- [x] **步骤 2：新增 PyAsc patch 目录说明**

新增 `autofuse/mlir/patches/pyasc/README.md`：

```markdown
# PyAsc patches

PyAsc submodule follows upstream `https://gitcode.com/cann/pyasc.git`.
Autofuse-specific changes are maintained as ordered patches in this directory.

Rules:

- Do not point `.gitmodules` to a personal fork.
- Treat the PyAsc submodule gitlink as the pinned upstream baseline.
- Keep each patch reviewable and related to one PyAsc change.
- Export only selected Autofuse-required commits or ranges; do not export the full upstream history.
- Refresh patches after syncing upstream.
- Apply patches before building `ENABLE_AUTOFUSE_MLIR=ON`.
```

- [x] **步骤 3：新增同步官方仓脚本**

新增 `autofuse/mlir/scripts/sync_pyasc_upstream.sh`：

```bash
#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
pyasc_dir="${repo_root}/autofuse/mlir/externals/pyasc"
base_ref="${AF_PYASC_BASE_REF:-}"
patch_dir="${repo_root}/autofuse/mlir/patches/pyasc"
expected_url="https://gitcode.com/cann/pyasc.git"
apply_patches=1

while [ "$#" -gt 0 ]; do
  case "$1" in
    --ref)
      base_ref="$2"
      shift 2
      ;;
    --no-apply-patches)
      apply_patches=0
      shift
      ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

actual_url="$(git -C "${repo_root}" config --file .gitmodules --get submodule.autofuse/mlir/externals/pyasc.url)"
if [ "${actual_url}" != "${expected_url}" ]; then
  echo "ERROR: PyAsc submodule must use official upstream: ${expected_url}" >&2
  echo "Actual: ${actual_url}" >&2
  exit 2
fi
git -C "${repo_root}" submodule update --init --recursive autofuse/mlir/externals/pyasc
if [ -n "${base_ref}" ]; then
  git -C "${pyasc_dir}" fetch origin
  git -C "${pyasc_dir}" checkout "${base_ref}"
fi
if [ "${apply_patches}" -eq 1 ] && compgen -G "${patch_dir}/*.patch" > /dev/null; then
  git -C "${pyasc_dir}" am "${patch_dir}"/*.patch
fi
```

说明：默认使用 Autofuse 仓库 gitlink 固定的 PyAsc commit，不自动更新到官方仓最新 HEAD。
只有设置 `AF_PYASC_BASE_REF=<ref>` 或执行脚本时传 `--ref <ref>`，才会显式 fetch 并切换到目标
ref。更新 PyAsc 基线时使用 `--no-apply-patches` 停在官方 upstream commit，再单独评审
submodule gitlink 变更和 patchset 变更；构建准备时再应用 patchset。

- [x] **步骤 4：新增导出 patch 脚本**

新增 `autofuse/mlir/scripts/export_pyasc_patch.sh`：

```bash
#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
pyasc_dir="${repo_root}/autofuse/mlir/externals/pyasc"
patch_dir="${repo_root}/autofuse/mlir/patches/pyasc"

commits=""
ranges=""
commit_count=0
range_count=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --commit)
      commits="${commits}${2}"$'\n'
      commit_count=$((commit_count + 1))
      shift 2
      ;;
    --range)
      ranges="${ranges}${2}"$'\n'
      range_count=$((range_count + 1))
      shift 2
      ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ "${commit_count}" -eq 0 ] && [ "${range_count}" -eq 0 ]; then
  echo "ERROR: choose PyAsc commits explicitly with --commit or --range." >&2
  echo "Refusing to export the full upstream-to-HEAD range implicitly." >&2
  exit 2
fi

mkdir -p "${patch_dir}"
while IFS= read -r commit; do
  [ -n "${commit}" ] || continue
  git -C "${pyasc_dir}" format-patch -1 --output-directory "${patch_dir}" "${commit}"
done <<EOF
${commits}
EOF
while IFS= read -r range; do
  [ -n "${range}" ] || continue
  git -C "${pyasc_dir}" format-patch --output-directory "${patch_dir}" "${range}"
done <<EOF
${ranges}
EOF
```

说明：patchset 只保存 Autofuse 必须维护的最小 PyAsc 修改。导出时必须显式选择 commit 或
range，不允许默认把 `origin/master..HEAD` 的所有 PyAsc upstream commit 搬入本仓。

- [x] **步骤 5：接入 PyAsc CMake**

在 `autofuse/mlir/CMakeLists.txt` 的依赖 include 之后增加：

```cmake
if(NOT ENABLE_AUTOFUSE_MLIR)
    return()
endif()

set(AF_PYASC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/pyasc" CACHE PATH "PyAsc upstream submodule path")
if(NOT EXISTS "${AF_PYASC_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "PyAsc submodule is missing. Run: bash autofuse/mlir/scripts/sync_pyasc_upstream.sh")
endif()

set(ASCIR_LLT_TEST OFF CACHE BOOL "Build PyAsc lit tests" FORCE)
if(NOT pybind11_DIR)
    if(NOT Python3_EXECUTABLE)
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
    endif()
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _af_pybind11_dir
        RESULT_VARIABLE _af_pybind11_result
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_af_pybind11_result EQUAL 0 AND EXISTS "${_af_pybind11_dir}/pybind11Config.cmake")
        set(pybind11_DIR "${_af_pybind11_dir}" CACHE PATH "pybind11 CMake config directory" FORCE)
    endif()
endif()
add_subdirectory(${AF_PYASC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/pyasc EXCLUDE_FROM_ALL)
```

说明：实际接入官方 PyAsc 顶层 `CMakeLists.txt`，由 PyAsc 自身维护 `include`、`lib`、`bin` 和
`python` 子目录关系；`ASCIR_LLT_TEST=OFF` 避免 Phase 1 默认拉起 PyAsc lit 测试。
`pybind11_DIR` 按 PyAsc `setup.py` 的做法从 Python 包自动解析，匹配 dev+LLVM 镜像内的
`pybind11==2.13.1`。

- [x] **步骤 6：验证默认路径不需要 PyAsc**

```bash
sh build.sh --pkg -j 8
```

预期：默认构建不要求 checkout PyAsc。

- [x] **步骤 7：验证打开开关后才要求 PyAsc**

在有效 LLVM 路径存在时执行：

```bash
sh build.sh --pkg --enable-mlir -j 8
```

预期：若 PyAsc 缺失，报错明确提示初始化 `autofuse/mlir/externals/pyasc`；默认路径不受影响。

当前进度：

- 已新增 `autofuse/mlir/scripts/check_pyasc_integration.sh`，用 fake LLVM 和 fake PyAsc 验证：
  - `.gitmodules` path 和 URL 指向官方仓 `https://gitcode.com/cann/pyasc.git`。
  - patch README 声明官方 upstream 和本仓 patchset 规则。
  - `sync_pyasc_upstream.sh`、`export_pyasc_patch.sh` 可执行且 bash 语法正确。
  - `export_pyasc_patch.sh` 无参数时拒绝隐式导出全量 range，显式 `--commit HEAD` 时可导出到临时目录。
  - 仓内存在 `patches/pyasc/*.patch` 时，验证这些 patch 能从 PyAsc gitlink 固定 commit apply。
  - `ENABLE_AUTOFUSE_MLIR=OFF` 时 CMake 不要求 LLVM/PyAsc。
  - `ENABLE_AUTOFUSE_MLIR=ON` 且缺失 PyAsc 时失败信息包含 `PyAsc submodule is missing`。
  - `ENABLE_AUTOFUSE_MLIR=ON` 且提供有效 LLVM/PyAsc 时 CMake 配置通过。
- 已执行 `bash autofuse/mlir/scripts/sync_pyasc_upstream.sh`，默认不拉取最新 upstream，PyAsc
  submodule 当前锁定 `72712f6e17758bea5e1e01c6e38a078c2d0ff24b`。
- 当前无本仓 PyAsc patch 需要导出；后续导出必须使用
  `bash autofuse/mlir/scripts/export_pyasc_patch.sh --commit <sha>` 或
  `bash autofuse/mlir/scripts/export_pyasc_patch.sh --range <start>..<end>`。
- 已在宿主机和 Docker dev+LLVM 镜像 + CANN volume 中验证
  `bash autofuse/mlir/scripts/check_pyasc_integration.sh` 成功。
- 已在 Docker dev+LLVM 镜像 + CANN volume 中验证：
  - `bash build.sh --pkg --enable-mlir -j 8` 成功，CMake 输出
    `Enable Autofuse MLIR migration infrastructure` 和
    `Found pybind11: ... (found version "2.13.1")`，最终生成
    `/workspace/build_out/cann-graph-autofusion_9.1.0_linux-aarch64.run`。
  - `bash build.sh --pkg --no-autofuse -j 8` 成功，CMake 参数包含
    `-DBUILD_AUTOFUSE=OFF -DENABLE_AUTOFUSE_MLIR=OFF`。

---

## 任务 6：新增最小 `af-opt`

**涉及文件：**

- 新增：`autofuse/mlir/tools/af-opt/CMakeLists.txt`
- 新增：`autofuse/mlir/tools/af-opt/main.cpp`
- 修改：`autofuse/mlir/CMakeLists.txt`

**接口关系：**

- 输出：只在 `ENABLE_AUTOFUSE_MLIR=ON` 时可构建的 `af-opt` target。

- [x] **步骤 1：新增 `af-opt` 入口**

新增 `autofuse/mlir/tools/af-opt/main.cpp`：

```cpp
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/Utils/Registration.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

namespace {
constexpr const char *kAutofuseMlirVersion = "af-mlir-phase1";

void PrintAutofuseMlirVersion(llvm::raw_ostream &os)
{
    os << "  Autofuse MLIR version " << kAutofuseMlirVersion << "\n";
}
} // namespace

int main(int argc, char **argv)
{
    llvm::cl::AddExtraVersionPrinter(PrintAutofuseMlirVersion);

    mlir::DialectRegistry registry;
    mlir::ascir::registerDialects(registry);
    mlir::ascendc::registerInlinerInterfaces(registry);
    mlir::ascir::registerExtensions(registry);
    mlir::ascir::registerPasses();
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Autofuse MLIR migration optimizer\n", registry));
}
```

- [x] **步骤 2：新增 CMake target**

新增 `autofuse/mlir/tools/af-opt/CMakeLists.txt`：

```cmake
add_executable(af-opt main.cpp)
target_link_libraries(af-opt PRIVATE
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRPass
    MLIRTransforms
)
```

修改 `autofuse/mlir/CMakeLists.txt`：

```cmake
add_subdirectory(tools/af-opt)
```

- [x] **步骤 3：验证工具构建**

```bash
sh build.sh --pkg --enable-mlir -j 8
cmake --build build --target af-opt -j 8
```

预期：LLVM/MLIR/PyAsc 配置正确时，`af-opt` 构建通过。

- [x] **步骤 4：验证默认包不安装工具**

```bash
sh build.sh --pkg -j 8
find build_out -name 'af-opt' -print
```

预期：默认 run 包中没有 `af-opt`。

---

## 任务 7：新增 MLIR lit 冒烟测试

**涉及文件：**

- 新增：`autofuse/mlir/test/CMakeLists.txt`
- 新增：`autofuse/mlir/test/lit.cfg.py`
- 新增：`autofuse/mlir/test/lit.site.cfg.py.in`
- 新增：`autofuse/mlir/test/smoke/parse_print.mlir`
- 修改：`autofuse/mlir/CMakeLists.txt`

**接口关系：**

- 输出：`check-autofuse-mlir` target。

- [x] **步骤 1：新增 smoke MLIR 文件**

新增 `autofuse/mlir/test/smoke/parse_print.mlir`：

```mlir
// RUN: af-opt %s | FileCheck %s

func.func @main() {
  return
}

// CHECK: func.func @main()
```

- [x] **步骤 2：新增 lit 配置**

新增 `autofuse/mlir/test/lit.cfg.py`：

```python
import lit.formats

config.name = "AutofuseMLIR"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.autofuse_mlir_test_exec_root

llvm_config.use_default_substitutions()
llvm_config.with_environment("PATH", config.autofuse_mlir_tools_dir, append_path=True)
```

新增 `autofuse/mlir/test/lit.site.cfg.py.in`：

```python
import os

config.llvm_tools_dir = "@LLVM_TOOLS_DIR@"
config.autofuse_mlir_tools_dir = "@CMAKE_BINARY_DIR@/autofuse/mlir/tools/af-opt"
config.autofuse_mlir_test_exec_root = "@CMAKE_CURRENT_BINARY_DIR@"

import lit.llvm
from lit.llvm import llvm_config

lit_config.load_config(config, "@CMAKE_CURRENT_SOURCE_DIR@/lit.cfg.py")
```

- [x] **步骤 3：新增测试 target**

新增 `autofuse/mlir/test/CMakeLists.txt`：

```cmake
configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-autofuse-mlir
    "Running Autofuse MLIR smoke tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS af-opt
)
```

修改 `autofuse/mlir/CMakeLists.txt`：

```cmake
add_subdirectory(test)
```

- [x] **步骤 4：验证冒烟测试**

```bash
cmake --build build --target check-autofuse-mlir -j 8
```

预期：`parse_print.mlir` 通过。

---

## 任务 8：新增 PyAsc AscendC op inventory 和覆盖矩阵

**涉及文件：**

- 新增：`autofuse/mlir/docs/pyasc_ascendc_op_coverage.md`
- 新增：`autofuse/mlir/docs/pyasc_ascendc_op_inventory.csv`
- 新增：`autofuse/mlir/tools/af-mlir-tools/scan_pyasc_coverage.py`

**接口关系：**

- 输出：
  - 从 PyAsc `.td` 提取的 `ascendc` / `emitasc` op inventory。
  - 可用表格工具筛选的 CSV inventory 快照。
  - Phase 2 MLIR codegen 可运行里程碑所需的算子覆盖矩阵文档。

- [x] **步骤 1：新增 PyAsc op inventory 扫描脚本**

新增 `autofuse/mlir/tools/af-mlir-tools/scan_pyasc_coverage.py`：

```python
#!/usr/bin/env python3
# 扫描 PyAsc include/ascir/Dialect/{Asc,EmitAsc}/IR/**/*.td
# 输出 family summary 和 op inventory，JSON 中包含 arguments/results 类型约束。
```

- [x] **步骤 2：新增覆盖矩阵文档**

新增 `autofuse/mlir/docs/pyasc_ascendc_op_coverage.md`：

```markdown
# PyAsc AscendC 算子覆盖矩阵

## 范围

本矩阵记录 PyAsc ascendc/emitasc 是否能表达 Phase 2 需要迁移的 Autofuse codegen 算子族。
完整 op inventory 由 scan_pyasc_coverage.py 从 PyAsc .td 中生成。

| Autofuse 算子族 | PyAsc 支持情况 | 决策 | 说明 |
|---|---|---|---|
| elewise | 当前扫描提取相关 op | 作为首条候选切片 | 覆盖首条纵向切片 |
| brc | 当前扫描提取相关 op | 作为首条候选切片 | 覆盖首条纵向切片 |
| reduce | 当前扫描提取相关 op | tiling/state 支持不完整时后置 | 需要验证 tiling |
| concat | 当前扫描提取 data move 相关 op，未检测到直接 concat op | 后置 | 对多输出和 offset 敏感 |
| gather | 当前扫描提取相关 op | index/shape 语义确认后再迁移 | 对 index 边界敏感 |
| transpose | 当前扫描提取相关 op | layout 表达确认后再迁移 | 对 layout/stride 敏感 |
| datacopy | 当前扫描提取相关 op | 内存搬运语义确认后再迁移 | 对 AscendC API 敏感 |
| cube | 当前扫描提取相关 op | 不放入首条切片 | 编译和执行风险更高 |
```

- [x] **步骤 3：验证扫描脚本**

```bash
python3 autofuse/mlir/tools/af-mlir-tools/scan_pyasc_coverage.py
```

预期：

- 如果 submodule 已初始化，脚本打印 family summary 和完整 op inventory。
- `--format json` 输出结构化 op 清单，包含 `op_name`、`arguments`、`results` 和 `type_hints`。
- `--format csv --output autofuse/mlir/docs/pyasc_ascendc_op_inventory.csv` 保存表格快照。
- summary 中 8 个 Autofuse 候选 family 之外的 PyAsc op 归为 `other`，保证 summary 合计等于 op inventory 总数。
- 如果没有初始化，脚本输出清晰的 PyAsc 缺失错误。

---

## 任务 9：新增结构化 diff 工具骨架

**涉及文件：**

- 新增：`autofuse/mlir/tools/af-mlir-tools/af_mlir_diff.py`
- 新增：`autofuse/mlir/testdata/diff/legacy_manifest.json`
- 新增：`autofuse/mlir/testdata/diff/mlir_manifest.json`
- 新增：`autofuse/mlir/manifest/ManifestSnapshot.{h,cpp}`
- 修改：`autofuse/codegen/CMakeLists.txt`
- 修改：`autofuse/codegen/codegen.cpp`
- 修改：`autofuse/compiler/py_module/CMakeLists.txt`
- 修改：`autofuse/compiler/py_module/pyascir_types.{h,cpp}`
- 修改：`autofuse/compiler/py_module/pyautofuse.cpp`
- 修改：`autofuse/tests/ut/python/test_python_ascir.py`

**接口关系：**

- 输入：`FusedScheduledResult.to_manifest_json()` 或后续 MLIR 路径生成的 JSON manifest。
- 输出：确定性的 diff 退出码；`0` 表示等价，`1` 表示不匹配，`2` 表示输入非法。

- [x] **步骤 1：新增 legacy manifest dump 入口**

在 `FusedScheduledResult` Python binding 上新增 `to_manifest_json()`，直接从现有 `FusedScheduledResult`、`ScheduleGroup`、`ImplGraph`、`AscNode`、tensor attr、IR attr 和 `ext_attrs` 中已知的 `CodegenApiParam` 生成结构化 JSON。dump helper 仅在 `ENABLE_AUTOFUSE_MLIR=ON` 时编译；默认 legacy 构建下误调用该方法会返回 Python `RuntimeError`，不自动 dump、不引入 dumper 编译风险。

当前 manifest 分两层记录：

- 静态结构：kernel name、inputs、outputs、workspaces、schedule groups、impl graphs、axes、nodes、tensor attr、ir attrs。
- codegen/runtime 摘要：tiling stage、origin vars，以及 codegen 后已经挂到 node `ext_attrs` 上的 API 参数；最终 workspace size/block dim 仍由 host tiling 运行时产生，不从裸 graph 推断。

- [x] **步骤 2：新增手动 C++ manifest 快照埋点**

新增 `autofuse/mlir/manifest/ManifestSnapshot.{h,cpp}`，作为 MLIR 迁移期公共 manifest dump helper。该 helper 不依赖 LLVM/MLIR/PyAsc，只依赖 legacy `FusedScheduledResult`，因此可以被 `autofuse/codegen/` 手动调用。

开发人员可在需要观察的 C++ 阶段显式加一行：

```cpp
AF_MLIR_DUMP_MANIFEST("my_custom_stage", fused_schedule_result);
```

运行时仅在设置 `AF_MLIR_MANIFEST_DUMP_DIR=<dir>` 时输出 JSON；不设置环境变量时不创建目录、不序列化、不写文件。可选 `AF_MLIR_MANIFEST_DUMP_FILTER=stage_a,stage_b` 只输出指定 stage。当前已埋两个示例点：

- `after_schedule`：`Autofuser::Schedule` 成功后。
- `after_codegen`：`Codegen::GenerateForInductor` 成功后。

- [x] **步骤 3：新增样例 manifest**

新增 `autofuse/mlir/testdata/diff/legacy_manifest.json` 和 `autofuse/mlir/testdata/diff/mlir_manifest.json`，采用同一 schema。`producer` 字段可不同，不参与等价比较。

```json
{
  "schema_version": 1,
  "producer": "legacy-autofuse",
  "kernel": {"name": "elewise_brc", "inputs": [], "outputs": [], "workspaces": []},
  "tiling": {"stage": "codegen_runtime", "origin_vars": ["M", "N"]},
  "schedule": [{"graph_index": 0, "results": [{"schedule_groups": [{"impl_graphs": []}]}]}]
}
```

- [x] **步骤 4：新增 diff 脚本**

新增 `autofuse/mlir/tools/af-mlir-tools/af_mlir_diff.py`：

```python
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise ValueError(f"cannot read JSON manifest {path}: {exc}") from exc

def compare_manifest(legacy, mlir):
    diffs = []
    for key in [
        ("schema_version",),
        ("kernel", "name"),
        ("kernel", "inputs"),
        ("kernel", "outputs"),
        ("kernel", "workspaces"),
        ("schedule",),
        ("tiling",),
    ]:
        left = legacy
        right = mlir
        for part in key:
            left = left.get(part)
            right = right.get(part)
        if left != right:
            diffs.append(".".join(key))
    return diffs

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--mlir", required=True)
    args = parser.parse_args()
    try:
        legacy = load_json(Path(args.legacy))
        mlir = load_json(Path(args.mlir))
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    diffs = compare_manifest(legacy, mlir)
    if diffs:
        print("DIFF:", ",".join(diffs))
        return 1
    print("PASS: manifests are equivalent")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **步骤 5：验证等价 manifest**

```bash
python3 autofuse/mlir/tools/af-mlir-tools/af_mlir_diff.py \
  --legacy autofuse/mlir/testdata/diff/legacy_manifest.json \
  --mlir autofuse/mlir/testdata/diff/mlir_manifest.json
```

预期：

```text
PASS: manifests are equivalent
```

补充验证：

- manifest 字段不一致时返回 `1`，并输出 `DIFF: <field>`。
- JSON 文件缺失或非法时返回 `2`，并输出 `ERROR: ...`。

---

## 任务 10：新增 `ascir_tool` 包装入口

**涉及文件：**

- 新增：`autofuse/mlir/tools/af-mlir-tools/af_mlir_case.py`

**接口关系：**

- 输入：现有 `autofuse/tests/st/codegen/ascir_tool/test_ascir.sh`。
- 输出：`autofuse/mlir/.artifacts/` 下的运行状态报告。

- [x] **步骤 1：新增包装脚本**

新增 `autofuse/mlir/tools/af-mlir-tools/af_mlir_case.py`：

```python
#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
ASCIR_TOOL = REPO_ROOT / "autofuse" / "tests" / "st" / "codegen" / "ascir_tool" / "test_ascir.sh"
ARTIFACT_ROOT = REPO_ROOT / "autofuse" / "mlir" / ".artifacts"

def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("--jobs must be a positive integer.")
    return number

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--mode", default="0")
    parser.add_argument("-j", "--jobs", type=positive_int, default=8)
    args = parser.parse_args()

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    cmd = ["bash", str(ASCIR_TOOL), f"--mode={args.mode}", f"--case={args.case}"]
    env = os.environ.copy()
    env["MAKEFLAGS"] = f"{env.get('MAKEFLAGS', '')} -j{args.jobs}".strip()
    result = subprocess.run(cmd, cwd=str(ASCIR_TOOL.parent), env=env, text=True)
    report = ARTIFACT_ROOT / f"{args.case}.status"
    report.write_text(
        f"case={args.case}\nmode={args.mode}\njobs={args.jobs}\nreturncode={result.returncode}\n",
        encoding="utf-8",
    )
    return result.returncode

if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **步骤 2：验证 help**

```bash
python3 autofuse/mlir/tools/af-mlir-tools/af_mlir_case.py --help
```

预期：argparse help 中包含 `--case` 和 `--mode`。

- [x] **步骤 3：在 NPU 环境验证已知用例（已执行：真实链路不通）**

```bash
python3 autofuse/mlir/tools/af-mlir-tools/af_mlir_case.py --case isinf_maskedfill_fusion --mode 0 -j 8
```

预期：现有 `ascir_tool` 行为保持不变；包装脚本写出 `autofuse/mlir/.artifacts/isinf_maskedfill_fusion.status`。

2026-07-04 真机实测记录：

- 状态：真实 NPU 链路未打通；任务 10 先按“入口、编译和失败分层记录完成”继续向后推进，真机 pass 后续单独修复。
- 当前实现不修改既有 `ascir_tool`；wrapper 通过 `MAKEFLAGS=-j8` 约束内部裸 `make` 并保留既有命令接口。
- 远端 CANN 需显式选择 `/data/.../Ascend/cann-9.1.0/set_env.sh`，并设置 `ASCEND_CUSTOM_PATH=${ASCEND_HOME_PATH}`；否则会退回 `/usr/local/Ascend/latest` 并缺少 `acl/acl_prof.h`、`libascendcl.so`。
- `isinf_maskedfill_fusion` 在 codegen 后的 device compile 阶段失败，`bisheng` 报 `Cannot select: intrinsic %llvm.hivm.VABS.s16`，未进入 launch。
- `reduce_sum_arar_4dim_not_align` 在临时诊断改动下完成 codegen、runner 编译、输入生成并打开真实 device 7；临时 `AutofuseLaunch` fallback 可进入 launch，但 kernel 在 `aclrtSynchronizeStream` 处超过 3 分钟未完成。该诊断改动不保留，NPU 实测仍记为未通过。

---

## 任务 11：新增 baseline 冻结策略和工具骨架

**涉及文件：**

- 新增：`autofuse/mlir/docs/baseline_policy.md`
- 新增：`autofuse/mlir/tools/af-mlir-tools/freeze_codegen_baseline.py`

**接口关系：**

- 输出：`autofuse/mlir/.baseline_artifacts/` 下的 baseline manifest。
- 当前边界：`freeze_codegen_baseline.py` 冻结已有 legacy manifest dump；manifest 生成仍复用任务 9
  `to_manifest_json()` / `ManifestSnapshot`，本工具不负责自己跑 legacy codegen。

- [x] **步骤 1：新增策略文档**

新增 `autofuse/mlir/docs/baseline_policy.md`：

```markdown
# Codegen 基线策略

## 入库内容

- 结构化 manifest JSON 文件。
- 测试必需的小型手写样例。
- 等价判据和必跑 case 清单。

## 不入库内容

- 完整 generated C++ code。
- `kernel_meta` 临时目录。
- coverage 输出。
- profiling 输出。
- run 包输出。

## 必须判等的字段

- kernel 输入和输出。
- 输出 dtype 和 shape。
- workspace size。
- tiling 字段名、顺序和类型。
- 可获取时检查 block dim。
```

- [x] **步骤 2：新增 baseline 工具骨架**

新增 `autofuse/mlir/tools/af-mlir-tools/freeze_codegen_baseline.py`，读取已有 legacy manifest dump 并写入
`.baseline_artifacts`：

```python
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = REPO_ROOT / "autofuse" / "mlir" / ".baseline_artifacts"

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.case}.baseline.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **步骤 3：验证 baseline 产物被忽略**

```bash
python3 autofuse/mlir/tools/af-mlir-tools/freeze_codegen_baseline.py \
  --case elewise_brc \
  --manifest autofuse/mlir/testdata/diff/legacy_manifest.json
git status --short autofuse/mlir/.baseline_artifacts
```

预期：`.baseline_artifacts` 被忽略，不出现在 git status 中。

2026-07-04 验证：

- `python3 -m py_compile autofuse/mlir/tools/af-mlir-tools/freeze_codegen_baseline.py` 通过。
- `python3 autofuse/mlir/tools/af-mlir-tools/freeze_codegen_baseline.py --help` 输出 `--case` 和 `--manifest`。
- `freeze_codegen_baseline.py --case elewise_brc --manifest autofuse/mlir/testdata/diff/legacy_manifest.json`
  生成 `autofuse/mlir/.baseline_artifacts/elewise_brc.baseline.json`。
- `af_mlir_diff.py --legacy autofuse/mlir/.baseline_artifacts/elewise_brc.baseline.json --mlir
  autofuse/mlir/testdata/diff/mlir_manifest.json` 输出 `PASS: manifests are equivalent`。
- `git status --short autofuse/mlir/.baseline_artifacts` 无输出。

- [x] **步骤 4：接入已有 legacy manifest dump**

`freeze_codegen_baseline.py` 已从占位 manifest 扩展为读取已有 legacy manifest dump 的工具。输入 manifest
至少应包含 `af_mlir_diff.py` 可比较的字段：

- `schema_version`
- `kernel.name`
- `kernel.inputs`
- `kernel.outputs`
- `kernel.workspaces`
- `schedule`
- `tiling`

输入来源应优先复用任务 9 已接入的 `to_manifest_json()` / `ManifestSnapshot`。本工具不直接运行 legacy
codegen，不保存完整 generated C++、`kernel_meta`、coverage、profiling 或 run 包输出。

---

## 任务 12：新增验证矩阵

**涉及文件：**

- 新增：`autofuse/mlir/docs/verification_matrix.md`

**接口关系：**

- 输出：给 reviewer 和 CI owner 使用的明确验证命令。

- [x] **步骤 1：新增验证矩阵**

新增 `autofuse/mlir/docs/verification_matrix.md`：

```markdown
# Phase 1 验证矩阵

| 路径 | 命令 | 预期结果 |
|---|---|---|
| Docker 开发镜像 | `bash autofuse/mlir/docker/build_llvm_image.sh --base-only` | 可一键构建基础编译镜像 |
| 用户自定义镜像 | `AF_MLIR_DEV_IMAGE=<image> bash autofuse/mlir/scripts/enter_dev_container.sh -- bash autofuse/mlir/scripts/collect_env_manifest.sh` | 能进入用户镜像并输出环境 manifest |
| Docker 默认包 | `AF_CANN_VOLUME=<cann-volume> bash autofuse/mlir/scripts/enter_dev_container.sh -- bash build.sh --pkg -j 8` | 不依赖 LLVM/MLIR/PyAsc 且构建通过 |
| Docker 跳过 Autofuse | `AF_CANN_VOLUME=<cann-volume> bash autofuse/mlir/scripts/enter_dev_container.sh -- bash build.sh --pkg --no-autofuse -j 8` | 保持既有跳过行为 |
| LLVM 本地依赖 | `AF_LLVM_ROOT=/path/to/prebuilt/llvm bash autofuse/mlir/scripts/prepare_mlir_deps.sh` | 找到 `llvm-config` 和 `MLIRConfig.cmake` 并输出 manifest |
| PyAsc 官方仓同步 | `bash autofuse/mlir/scripts/sync_pyasc_upstream.sh` | 基于官方仓应用本仓 patch，不依赖个人 fork |
| Autofuse UT | `AF_CANN_VOLUME=<cann-volume> bash autofuse/mlir/scripts/enter_dev_container.sh -- bash build.sh -u --module=autofuse_framework -j 8` | 既有 UT 通过 |
| Autofuse E2E ST | `AF_CANN_VOLUME=<cann-volume> bash autofuse/mlir/scripts/enter_dev_container.sh -- bash build.sh -s --module=autofuse_e2e -j 8` | 既有 ST 通过 |
| MLIR 包 | `AF_CANN_VOLUME=<cann-volume> bash autofuse/mlir/scripts/enter_dev_container.sh -- bash build.sh --pkg --enable-mlir -j 8` | LLVM/PyAsc 配置正确时构建通过 |
| MLIR 冒烟 | `cmake --build build --target check-autofuse-mlir -j 8` | lit 冒烟通过 |
| ascir tool case | `bash autofuse/tests/st/codegen/ascir_tool/test_ascir.sh --mode=0 --case=isinf_maskedfill_fusion` | 既有 E2E 行为保持不变 |
```

- [x] **步骤 2：验证关键命令存在**

```bash
rg -n -- 'build_llvm_image|collect_env_manifest|prepare_mlir_deps|sync_pyasc_upstream|-j 8|--no-autofuse|--enable-mlir|check-autofuse-mlir|af_mlir_diff|freeze_codegen_baseline|af_mlir_case' \
  autofuse/mlir/docs/verification_matrix.md
```

预期：能匹配所有关键命令。

2026-07-04 验证：

- 新增 `autofuse/mlir/docs/verification_matrix.md`，按本地、Docker 编译环境、LLVM/PyAsc 专项依赖、
  manifest/baseline 工具和 NPU 环境分层列出命令。
- 所有 `build.sh` 和 `cmake --build` 命令均显式包含 `-j 8`。
- 将原计划中不存在的 `build_dev_image` 检查项修正为真实脚本 `build_llvm_image`。
- `rg` 快速检查已覆盖 `build_llvm_image`、`collect_env_manifest`、`prepare_mlir_deps`、
  `sync_pyasc_upstream`、`-j 8`、`--no-autofuse`、`--enable-mlir`、`check-autofuse-mlir`、
  `af_mlir_diff`、`freeze_codegen_baseline` 和 `af_mlir_case`。

---

## 任务 13：检查默认包边界

**涉及文件：**

- 必要时修改：`autofuse/CMakeLists.txt`
- 必要时修改：`build.sh`
- 修改：`autofuse/mlir/scripts/enter_dev_container.sh`
- 修改：`autofuse/mlir/docker/README.md`

**接口关系：**

- 确认默认包不安装 MLIR 工具。

- [x] **步骤 1：构建默认包**

```bash
AF_CANN_VOLUME=af-cann-910 bash autofuse/mlir/scripts/enter_dev_container.sh -- bash build.sh --pkg -j 8
```

预期：使用既有 package 逻辑构建成功。

- [x] **步骤 2：检查包输出**

```bash
find build_out -maxdepth 2 -type f | sort | rg 'af-opt|MLIR|LLVM|pyasc' || true
```

预期：默认包中没有 `af-opt`、LLVM、MLIR 或 PyAsc 运行时制品。

- [x] **步骤 3：如确需安装开发工具，必须加显式安装开关**

如果 reviewer 要求开发包安装 MLIR 工具，在 `autofuse/mlir/CMakeLists.txt` 中增加独立开关：

```cmake
option(ENABLE_AUTOFUSE_MLIR_TOOLS_INSTALL "Install Autofuse MLIR developer tools" OFF)
if(ENABLE_AUTOFUSE_MLIR_TOOLS_INSTALL)
    install(TARGETS af-opt RUNTIME DESTINATION tools OPTIONAL)
endif()
```

该开关不得默认开启。

2026-07-04 实测结果：

- `sh build.sh --pkg -j 8` 会因 `build.sh` 使用 Bash array 语法失败，后续验证命令统一改为 `bash build.sh ... -j 8`。
- 2026-07-04 复查确认不应修改 legacy 测试 CMake 来跳过 `tikicpulib`。真实根因是容器入口只挂载 `/opt/cann`，而 CANN `set_env.sh` 和官方 `tikicpulib-config.cmake` 依赖安装前缀 `/usr/local/Ascend/cann-9.1.0`。`enter_dev_container.sh` 已改为同时挂载该前缀并在执行命令前 source CANN 环境；临时 CMake probe 验证 `tikicpulib_FOUND=1`。
- 路径修正后重新执行 `AF_CANN_VOLUME=af-cann-910 bash autofuse/mlir/scripts/enter_dev_container.sh -- bash build.sh --pkg -j 8`，CMake 配置为 `-DBUILD_AUTOFUSE=ON -DENABLE_AUTOFUSE_MLIR=OFF`，并成功编译原先被跳过的 `test_ascendc_api`、`test_ascendc_api_v35`、`backend_e2e` 和 `codegen/e2e` 多数目标。
- 当前 full package 构建仍未完成：在 77% 左右失败于 `autofuse/tests/st/backend_e2e/isinf_maskedfill_test/isinf_maskedfill_test_kernel.cpp`，CANN cpudebug 头报 `no matching function for call to 'vabs(int*&, int*&, int, uint16_t, uint16_t, uint8_t, uint8_t)'`。这是路径修正后暴露出的 legacy kernel 与当前 CANN AscendC/cpudebug API 兼容问题，不再是 `tikicpulib_DIR=NOTFOUND`。
- 包边界检查命令保持为 `find build_out -maxdepth 2 -type f | sort | rg 'af-opt|MLIR|LLVM|pyasc' || true`；当前 full package 未完成时，不更新默认包产物结论。

---

## 任务 14：同步主设计和 tracker

**涉及文件：**

- 修改：`autofuse/mlir/docs/migration/2026-06-25-codegen-mlir-migration-design.md`
- 修改：`autofuse/mlir/docs/migration/2026-06-27-phase1-environment-prep-design.md`
- 如本次纳入 tracker，则修改：`build/migration_data.mjs`

**接口关系：**

- 输出：开发环境、LLVM/MLIR 依赖输入、104/105/107 内容一致。

- [x] **步骤 1：同步主设计开关名**

在 `2026-06-25-codegen-mlir-migration-design.md` 中统一使用：

```text
ENABLE_AUTOFUSE_MLIR=OFF|ON
AF_MLIR_CODEGEN=off|on|compare
```

除非后续单独设计更通用的 pipeline map，否则不要使用 `AF_CODEGEN_BACKEND`。

- [x] **步骤 2：保持 Phase 1 范围为环境底座加 104/105/107**

```bash
rg -n '开发环境|Docker|LLVM/MLIR|104 工程集成 MLIR|105 测试工具适配|107 回归基线' \
  autofuse/mlir/docs/migration/2026-06-27-phase1-environment-prep-design.md
```

预期：环境准备方案描述 Docker/LLVM 环境底座和 104/105/107，不引入 Runtime 预研或迁移任务。

- [x] **步骤 3：如纳入 tracker，同步环境底座和三项迁移任务**

如果更新 `build/migration_data.mjs`，只保留：

```text
开发/编译验证环境准备
LLVM/MLIR 预编译依赖准备
104 工程集成 MLIR
105 测试工具适配
107 回归基线冻结
```

不要新增 Runtime 预研或 runtime 迁移项。

2026-07-04 同步结果：

- 主设计中继续统一使用 `ENABLE_AUTOFUSE_MLIR=OFF|ON` 和 `AF_MLIR_CODEGEN=off|on|compare`，未发现 `AF_CODEGEN_BACKEND` 残留。
- 主设计 Stage 1 基础设施条目已同步为当前真实工件：`prepare_mlir_deps.sh`、`build_llvm_image.sh`、官方 PyAsc submodule、`patches/pyasc`、`af-opt`、`check-autofuse-mlir`。
- 主设计和 Phase 1 工作方案均明确 Phase 1 不引入 Runtime 预研、runtime 迁移或 artifact runner，只复用现有 `ascir_tool`/测试 stub 作为验证入口。
- 入库权威状态已在主设计、Phase 1 工作方案和本文中同步：Phase 1 100% 完成，Phase 2 Codegen 模块交付为下一阶段。本地周报 tracker 目录当前受 `.gitignore` 的 `tracker` 规则忽略，不作为本次入库交付物。

## Phase 1 阶段关闭结论

- 状态：已完成（2026-07-04）。
- 范围：关闭于环境底座、LLVM/MLIR/PyAsc 工程接入、`af-opt`/lit smoke、PyAsc op inventory、manifest dump/diff/baseline 工具、`ascir_tool` wrapper 和验证矩阵状态同步。
- 遗留项：真实 NPU 链路未打通、`isinf_maskedfill_test_e2e` 暴露 CANN cpudebug `vabs(int*)` 兼容问题，均已在验证矩阵分层记录；不作为 Phase 1 关闭阻塞项。
- 下一阶段：Phase 2 Codegen 迁移，围绕 legacy codegen 分阶段边界、MLIR lowering、代表性 case A/B 等价和真实链路验证推进。

---

## 特性交叉影响检查

| 场景 | 适用性 | 结论 |
|---|---|---|
| SuperKernel Python 接口 | 不适用 | 不修改 SuperKernel Python 包、pytest 入口或 wheel 内容。 |
| SuperKernel C++/AOT 接口 | 不适用 | 不修改 `super_kernel/kernel`、AOT ABI 或 `libascendsk.so`。 |
| Autofuse 图优化 | 部分适用 | Phase 1 不改 optimize 逻辑；后续 diff/fused-result 会读取 schedule/graph 语义。 |
| Autofuse Codegen/Backend | 适用 | 环境底座和 104/105/107 均服务 codegen 迁移；必须覆盖 generated code、tiling、kernel ABI、NPU E2E。 |
| AscendC API 交互 | 部分适用 | 不新增运行时调用；但 MLIR codegen 产物和 `ascir_tool` E2E 必须验证 AscendC/tiling 行为。 |
| Python/C++ 混合绑定 | 部分适用 | 默认不改 `pyautofuse`；Python 工具新增在 `autofuse/mlir/tools`，异常必须清晰返回。 |
| 构建与打包 | 适用 | 修改 Docker、`build.sh`、CMake、第三方依赖和 package 边界；必须验证默认包不含 MLIR 工具。 |
| 测试与覆盖率 | 适用 | 新增 lit smoke、diff tool fixtures、`ascir_tool` wrapper 和 baseline policy。 |
| 性能与日志 | 适用 | MLIR enabled build 增加编译时长；默认 path 不应增加编译/运行耗时；工具不新增默认高频日志。 |
| 兼容性 | 适用 | `--no-autofuse`、默认 build、run 包布局、现有 `ascir_tool` mode 0/1/2 均必须保持兼容。 |

## 编码红线检查

- G10 构建脚本：默认路径不得联网、不得静默 clone/build LLVM；MLIR 依赖必须显式配置；CANN Toolkit 不内置到 LLVM 镜像。
- G11 测试产物：`.artifacts`、`.baseline_artifacts`、coverage、kernel_meta 不入库。
- ABI/API：新增 CMake option 不改变既有 ABI/API；`_GLIBCXX_USE_CXX11_ABI=0` 必须体现在依赖 manifest。
- 资源生命周期：Phase 1 不新增 runtime/aclrt 调用，不引入新的 stream/event 生命周期。
- 图确定性：diff 和 baseline manifest 输出必须排序稳定，JSON 使用 `sort_keys=True`。
- 硬编码：不得硬编码公网地址、密钥、芯片型号作为逻辑分支；PyAsc submodule URL 必须指向官方仓，不得使用个人 fork 作为依赖来源。

## 最终验证命令

在仓库根目录优先准备并进入 Docker 编译验证环境：

```bash
bash autofuse/mlir/docker/build_llvm_image.sh --base-only
bash autofuse/mlir/scripts/enter_dev_container.sh -- bash autofuse/mlir/scripts/collect_env_manifest.sh
AF_CANN_VOLUME=<cann-volume> bash autofuse/mlir/scripts/enter_dev_container.sh -- bash build.sh --pkg -j 8
AF_CANN_VOLUME=<cann-volume> bash autofuse/mlir/scripts/enter_dev_container.sh -- bash build.sh --pkg --no-autofuse -j 8
AF_CANN_VOLUME=<cann-volume> bash autofuse/mlir/scripts/enter_dev_container.sh -- bash build.sh -u --module=autofuse_framework -j 8
AF_CANN_VOLUME=<cann-volume> bash autofuse/mlir/scripts/enter_dev_container.sh -- bash build.sh -s --module=autofuse_e2e -j 8
```

配置好 MLIR 依赖后执行：

```bash
AF_LLVM_ROOT=/path/to/prebuilt/llvm bash autofuse/mlir/scripts/prepare_mlir_deps.sh
bash autofuse/mlir/scripts/sync_pyasc_upstream.sh
AF_CANN_VOLUME=<cann-volume> bash autofuse/mlir/scripts/enter_dev_container.sh -- bash build.sh --pkg --enable-mlir -j 8
cmake --build build --target af-opt -j 8
cmake --build build --target check-autofuse-mlir -j 8
```

在具备 NPU 的环境执行：

```bash
bash autofuse/tests/st/codegen/ascir_tool/test_ascir.sh --mode=0 --case=isinf_maskedfill_fusion
python3 autofuse/mlir/tools/af-mlir-tools/af_mlir_case.py --case isinf_maskedfill_fusion --mode 0
```
