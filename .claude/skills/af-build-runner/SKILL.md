---
name: af-build-runner
description: |
  graph-autofusion 项目编译构建辅助。
  **必须触发的场景**：用户提到编译、构建、build、cmake、编译报错、make、编译选项、build.sh、打包、pkg、编译类型(Debug/Release)、第三方依赖下载等。
---

# graph-autofusion 编译构建辅助

辅助用户完成项目的编译、构建、打包和依赖管理。

## 构建入口

项目使用 `build.sh` 作为统一构建入口，位于项目根目录。

```bash
sh build.sh [-h|--help] [--pkg] [-u|--ut] [-s|--st] [--impl=<py|cpp|all>] [--module=<name>]
             [-c|--coverage] [-j <N>] [--build-type=<TYPE>] [--no-autofuse]
             [--output_path=<PATH>] [--cann_3rd_lib_path=<PATH>] [--run_example]
             [--test_case=<FILTER>]
```

## 重要：编译必须加 -j 限制

**所有编译命令必须指定 `-j 8`（或合理线程数），否则默认无限制会导致 OOM（Killed）。**

## 重要：编译超时处理

全量编译 autofuse 组件耗时较长（通常 15-30 分钟），可能超过工具的执行超时限制。

**解决方案**：

1. **优先使用增量编译**（推荐）：
   ```bash
   # 已有 build/ 目录时，只重编变更的源文件
   cmake --build build --target <target> -j 8
   ```

2. **后台运行全量编译**：
   ```bash
   # 使用 nohup 后台运行，避免超时中断
   nohup sh build.sh --pkg -j 8 > build.log 2>&1 &
   
   # 查看编译进度
   tail -f build.log
   
   # 检查编译是否完成
   ps aux | grep "build.sh" | grep -v grep
   ```

3. **分阶段编译**：
   ```bash
   # 先编译核心库
   cmake --build build --target aihac_codegen -j 8
   
   # 再编译测试目标
   cmake --build build --target test_common -j 8
   ```

4. **检查编译产物**：
   ```bash
   # 编译完成后检查产物是否存在
   ls -la build/cann-graph-autofusion_*.run
   ls -la build/autofuse/tests/ut/common/test_common
   ```

## 增量编译（日常开发首选）

CMake 天然支持增量编译：`build/` 目录已存在时，只重编译变更的源文件。**日常开发应优先使用增量编译，避免全量重建。**

### 判断是否需要全量重建

| 场景 | 是否需要全量 | 说明 |
|------|-------------|------|
| 修改了 `.cpp` / `.h` / `.py` 源码 | **否** | 增量编译即可 |
| 修改了 `CMakeLists.txt` | **否** | CMake 自动检测并重新 configure |
| 切换 Debug/Release | **是** | 编译选项变化，需清理 `build/` |
| 首次编译 | **是** | 无 `build/` 目录 |
| 切换了 CANN Toolkit 版本 | **是** | 头文件和库路径变化 |
| 添加了新的源文件 | **否** | CMake 的 `CONFIGURE_DEPENDS` / `GLOB_RECURSE` 会自动发现 |

### 增量编译命令

```bash
# 方式一：通过 build.sh（自动 configure + build，增量编译未变更的文件）
sh build.sh --pkg -j 8

# 方式二：直接 cmake --build（跳过 configure，最快，适用于源码修改）
cmake --build build --target ascendsk -j 8        # 只重编 SuperKernel
cmake --build build --target aihac_codegen -j 8    # 只重编 autofuse 核心库
cmake --build build --target pyautofuse -j 8       # 只重编 Python 绑定
cmake --build build -j 8                           # 增量编译所有目标

# 方式三：只编译指定目标后直接打包
cmake --build build --target ascendsk -j 8 && cmake --build build --target package -j 8
```

### 单目标增量编译参考

根据修改的源码区域，选择对应的编译目标：

| 修改的源码目录 | 编译目标 | .so 产物 |
|---------------|---------|---------|
| `super_kernel/src/` | `ascendsk` | `build/super_kernel/libascendsk.so` |
| `super_kernel/kernel/` | `ascendsk`（含 sk_scope 子目标） | `build/super_kernel/libascendsk.so` |
| `super_kernel/*.py` | `superkernel_whl` | `build/super_kernel/superkernel-*.whl` |
| `autofuse/codegen/` | `aihac_codegen` | `build/autofuse/libaihac_codegen.so` |
| `autofuse/optimize/` | `aihac_codegen` | `build/autofuse/libaihac_codegen.so` |
| `autofuse/ascir/` | `aihac_codegen`（含 `ascir`、`ascir_builtin_ops`） | `build/autofuse/libaihac_codegen.so` |
| `autofuse/graph_metadef/graph/` | `graph_af`、`graph_base_af`、`aihac_ir` 等 | 各自 `.so` |
| `autofuse/compiler/py_module/` | `pyautofuse` | `build/autofuse/compiler/py_module/pyautofuse.so` |
| `autofuse/att/` | `aihac_codegen` | `build/autofuse/libaihac_codegen.so` |
| `autofuse/common/` | `aihac_codegen` | `build/autofuse/libaihac_codegen.so` |

### 快速迭代工作流

修改代码后快速验证，不打包：

```bash
# 1. 增量编译目标库
cmake --build build --target ascendsk -j 8

# 2. 直接运行测试验证
sh build.sh -u --module=superkernel --impl=py
# 或
sh build.sh -u --module=superkernel --impl=cpp --test_case="*SkEntry*"
```

### 换 .so（替换动态库快速验证）

当需要把编译产物部署到 CANN 安装目录或其他环境时：

```bash
# 增量编译目标
cmake --build build --target ascendsk -j 8
cmake --build build --target aihac_codegen -j 8

# 替换到部署目录
cp build/super_kernel/libascendsk.so <部署路径>/lib64/
cp build/autofuse/libaihac_codegen.so <部署路径>/lib64/

# 或通过 LD_LIBRARY_PATH 优先加载
export LD_LIBRARY_PATH=$(pwd)/build/super_kernel:$(pwd)/build/autofuse:$LD_LIBRARY_PATH
```

**注意**：换 `.so` 需确保 ABI 兼容（相同编译选项、相同编译器版本、相同 Build Type）。

## 全量编译

仅在必要时使用（首次编译、切换 Build Type、切换 Toolkit 版本）：

```bash
# 清理后全量重建
rm -rf build && sh build.sh --pkg -j 8

# 切换 Debug 模式（需全量）
rm -rf build && sh build.sh --pkg --build-type=Debug -j 8
```

## 常用构建命令

### 编译

注意：`build.sh` 要求至少指定一个参数，不能无参数运行。

```bash
# 编译并打包（Release 模式，最常用；已有 build/ 时自动增量）
sh build.sh --pkg -j 8

# Debug 模式编译（切换 Build Type 需全量）
rm -rf build && sh build.sh --pkg --build-type=Debug -j 8

# 跳过 autofuse 后端编译（只编 SuperKernel）
sh build.sh --pkg --no-autofuse -j 8
```

**注意**：`build.sh --module=<name>` 只影响测试运行范围，**不影响编译范围**（编译始终全量）。要限制编译范围，使用 `cmake --build build --target <target>` 代替。

### 打包

```bash
# 构建运行包（已有 build/ 时增量编译后打包）
sh build.sh --pkg -j 8

# 只重新打包（不重编，适用于已编译完成只需生成 .run 包）
cmake --build build --target package -j 8

# 指定输出路径
sh build.sh --pkg --output_path=/path/to/output -j 8
```

### 运行测试

```bash
# SuperKernel Python UT
sh build.sh -u --module=superkernel --impl=py

# SuperKernel C++ UT
sh build.sh -u --module=superkernel --impl=cpp

# SuperKernel Python ST
sh build.sh -s --module=superkernel --impl=py

# Autofuse Framework UT
sh build.sh -u --module=autofuse_framework

# Autofuse AscendC API UT
sh build.sh -u --module=autofuse_ascendc_api

# Autofuse E2E ST
sh build.sh -s --module=autofuse_e2e

# 带覆盖率报告
sh build.sh -u --module=superkernel -c

# 运行指定的 C++ UT 用例（gtest filter）
sh build.sh -u --module=superkernel --impl=cpp --test_case="*SkEntry*"
```

注意：`autofuse_e2e` 模块只支持 ST（`-s`），不支持 UT（`-u`）。

### 运行示例

```bash
sh build.sh --run_example --module=superkernel
```

## 支持的模块

| 模块名 | 说明 | 支持的测试类型 |
|--------|------|---------------|
| `superkernel` | SuperKernel 组件 | UT（py/cpp）、ST（py） |
| `autofuse_framework` | Autofuse 框架 | UT、ST |
| `autofuse_ascendc_api` | Autofuse AscendC API | UT、ST |
| `autofuse_e2e` | Autofuse 端到端测试 | 仅 ST |

## 环境依赖

- CANN Toolkit >= 9.1.0（使用 `/cann-toolkit-installer` 安装）
- Python3 >= 3.8.0（建议 3.9+，使用虚拟环境）
- CMake >= 3.16.0
- bash >= 5.0

## 环境变量

编译前必须 source CANN 环境变量。`set_env.sh` 位于 `ASCEND_HOME_PATH` 目录下：

```bash
# 如果 ASCEND_HOME_PATH 已设置
source ${ASCEND_HOME_PATH}/set_env.sh

# 常见安装路径（按优先级尝试）
source /home/developer/Ascend/master/cann-9.1.0/set_env.sh   # master 构建版本
source /home/developer/Ascend/cann-9.0.0/set_env.sh           # 发布版本
source /usr/local/Ascend/cann/set_env.sh                       # 系统级安装
```

**注意**：项目源码依赖 CANN 9.1.0+ 的 API（如 `sk::SkSystemArgs`），使用低版本 Toolkit 会编译报错 `no type named 'SkSystemArgs'`。优先使用 master 路径下的 Toolkit。

### 验证环境

```bash
# 检查 ASCEND_HOME_PATH 是否设置
echo $ASCEND_HOME_PATH

# 检查 atc 工具是否存在
ls ${ASCEND_HOME_PATH}/bin/atc

# 检查 set_env.sh 是否存在
ls ${ASCEND_HOME_PATH}/set_env.sh
```

## 第三方依赖

首次编译会自动下载第三方依赖到 `output/third_party/`。后续编译跳过此步骤。

离线环境需手动下载并放到 `open_source/` 目录，详见 `docs/zh/build.md`。

指定第三方库路径：
```bash
sh build.sh --cann_3rd_lib_path=/path/to/third_party -j 8
```

## 编译产物

### 目录结构

```
build/                              # 编译中间文件（CMAKE_INSTALL_PREFIX）
├── super_kernel/
│   ├── libascendsk.so              # SuperKernel 动态库
│   └── superkernel-*.whl           # Python wheel 包
├── autofuse/
│   ├── libaihac_codegen.so         # autofuse 编译器核心库
│   ├── compiler/py_module/pyautofuse.so  # Python 绑定
│   ├── ascir/meta/libascir.so
│   ├── ascir/generator/libascir_builtin_ops.so
│   ├── graph_metadef/graph/
│   │   ├── libgraph_af.so
│   │   ├── libgraph_base_af.so
│   │   ├── expression/libaihac_symbolizer_af.so
│   │   └── ascendc_ir/
│   │       ├── libaihac_ir.so
│   │       └── generator/
│   │           ├── libascir_generate.so
│   │           └── libaihac_ir_register.so
│   └── tests/                      # 测试专用 .so（仅测试时生成）
└── _CPack_Packages/                # 打包中间文件

build_out/                          # 最终输出
└── cann-graph-autofusion_*.run     # 运行包（可部署）
```

### 正式产物 .so 清单（11 个）

| .so | 编译目标 | 说明 |
|-----|---------|------|
| `libascendsk.so` | `ascendsk` | SuperKernel 动态库 |
| `libaihac_codegen.so` | `aihac_codegen` | Autofuse 编译器核心 |
| `libgraph_af.so` | `graph_af` | 图定义库 |
| `libgraph_base_af.so` | `graph_base_af` | 图基础设施 |
| `libaihac_ir.so` | `aihac_ir` | AIHAC IR |
| `libaihac_ir_register.so` | `aihac_ir_register` | AIHAC IR 注册 |
| `libaihac_symbolizer_af.so` | `aihac_symbolizer_af` | 符号化库 |
| `libascir.so` | `ascir` | ASCIR 中间表示 |
| `libascir_builtin_ops.so` | `ascir_builtin_ops` | ASCIR 内置算子 |
| `libascir_generate.so` | `ascir_generate` | ASCIR 代码生成 |
| `pyautofuse.so` | `pyautofuse` | Python 绑定模块 |

## 常见编译错误排查

1. **`no type named 'SkSystemArgs'`**：CANN Toolkit 版本过低，需使用 >= 9.1.0。执行 `source /home/developer/Ascend/master/cann-9.1.0/set_env.sh` 切换到 master 版本
2. **CANN Toolkit 未找到**：检查 `ASCEND_HOME_PATH` 环境变量是否设置，执行 `source ${ASCEND_HOME_PATH}/set_env.sh`。如未安装 Toolkit，使用 `/cann-toolkit-installer` 安装
3. **编译 OOM（Killed）**：编译默认并行度无限制，在多核机器上导致内存不足。**必须使用 `-j 8` 限制并行线程数**
4. **第三方依赖下载失败**：检查网络或使用离线模式（手动下载到 `open_source/` 目录）
5. **CMake 版本过低**：升级到 >= 3.16.0
6. **Python 版本不兼容**：使用 Python 3.9+
7. **增量编译后链接错误**：切换了 Build Type 或 Toolkit 版本但未清理 `build/`，执行 `rm -rf build` 后全量重建
