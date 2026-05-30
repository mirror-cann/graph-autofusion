---
name: af-test-developer
description: |
  graph-autofusion 项目 UT/ST 测试开发辅助。
  **必须触发的场景**：用户提到写测试、UT、ST、单元测试、系统测试、测试用例、gtest、mockcpp、pytest、测试覆盖率、coverage、测试编译、运行测试、测试规范等。
---

# graph-autofusion UT/ST 测试开发

辅助用户编写和运行 super_kernel 和 autofuse 组件的单元测试（UT）和系统测试（ST）。

---

## 测试开发完整工作流

**开发完测试用例后，必须执行验证，不可跳过。**

### Step 1: 编写测试用例

按本 skill 的规范编写测试用例（见后续章节）。

### Step 2: 编译验证

触发 af-build-runner skill（使用 `Skill` 工具加载 `af-build-runner`），按其指导执行编译。

**C++ 测试**：
```bash
# 增量编译指定 target（推荐，最快）
cmake --build build --target <test_target> -j8

# 或通过 run_autofuse_test.sh 编译运行
bash scripts/test/run_autofuse_test.sh -u -m <module> -j 8
```

**Python 测试**：
```bash
# 确保 pyautofuse 已编译
cmake --build build --target pyautofuse -j8

# 运行 pytest
PYTHONPATH=build/autofuse/compiler/py_module:$PYTHONPATH \
  python3 -m pytest autofuse/tests/ut/python/<test_file>.py -v
```

### Step 3: 运行测试并确认结果

```bash
# C++ 运行指定用例
./build/<path_to_binary> --gtest_filter="<TestClass>.<TestName>"

# C++ 运行整个 target
./build/<path_to_binary>

# Python 运行指定类/方法
python3 -m pytest <test_file>::<TestClass>::<test_method> -v
```

**必须确认以下全部通过**：
- [ ] 编译无错误
- [ ] 测试全部 PASS（无 FAIL、无 ERROR）
- [ ] 无新增 warning（关注 `-Werror` 相关）
- [ ] 无超时或死循环

---

## 测试运行命令

### SuperKernel

```bash
# Python UT
sh build.sh -u --module=superkernel --impl=py

# Python ST
sh build.sh -s --module=superkernel --impl=py

# C++ UT (AOT)
sh build.sh -u --module=superkernel --impl=cpp

# 带覆盖率
sh build.sh -u --module=superkernel -c
```

### Autofuse

注意：Autofuse 编译默认并行度无限制，在多核机器上可能导致 OOM，建议加 `-j 8`。

```bash
# Framework UT（att + optimize + common + codegen + py_module）
sh build.sh -u --module=autofuse_framework -j 8

# AscendC API UT
sh build.sh -u --module=autofuse_ascendc_api -j 8

# E2E ST
sh build.sh -s --module=autofuse_e2e -j 8
```

### run_autofuse_test.sh 模块名完整列表

UT 模块（`-u -m <module>`）：

| 模块名 | 说明 |
|--------|------|
| `att` | ATT UT |
| `optimize` | 优化 pass UT |
| `common` | 通用工具 UT |
| `codegen` | 代码生成 UT + py_module UT |
| `ascendc_api` | AscendC API UT |
| `autofuse_utils` | Autofuse 配置 UT |
| `framework` | 全部 framework UT（att + optimize + common + codegen + py_module） |
| `all` | 所有 UT |

ST 模块（`-s -m <module>`）：

| 模块名 | 说明 |
|--------|------|
| `att` | ATT ST |
| `ascir` | ASCIR ST |
| `optimize` | 优化 ST |
| `common` | 通用 ST |
| `codegen` | 代码生成 ST + E2E ST + py_module ST |
| `tools` | kernel tool ST |
| `backend` | Backend ST |
| `e2e` | Codegen E2E ST |
| `ascendc_api` | ASCIR + codegen + backend + kernel tool ST |
| `framework` | att + common + optimize + py_module ST |
| `all` | 所有 ST |

示例：

```bash
bash scripts/test/run_autofuse_test.sh -u -m common -j 8
bash scripts/test/run_autofuse_test.sh -u -m optimize -j 8
bash scripts/test/run_autofuse_test.sh -s -m backend -j 8
```

---

## 测试目录结构

### SuperKernel

```
super_kernel/tests/
├── conftest.py              # pytest 配置，自动标记 ut/st（仅 super_kernel 有）
├── fixtures/                # 共享 pytest fixtures
├── utils/                   # 测试工具函数
├── ut/                      # Python 单元测试
├── st/                      # Python 系统测试
│   └── scenarios/
└── aot/                     # C++ AOT 测试
    ├── ut/                  # C++ 单元测试 (gtest + mockcpp)
    │   ├── main.cpp         # gtest main() 入口
    │   ├── test_sk_*.cpp    # 测试文件
    │   └── stub/            # Mock/Stub 头文件
    └── rdv/                 # 设备端验证测试（见"RDV 设备端测试"章节）
```

### Autofuse

```
autofuse/tests/
├── common/stub/             # 共享 stubs（att 模块使用）
├── depends/                 # 测试依赖 stubs（共享库形式）
│   ├── slog/                # 日志 stub（asc_slog_stub）
│   ├── runtime/             # Runtime stub（autofuse_runtime_stub）
│   ├── trace/               # Atrace stub（atrace_stub）
│   ├── common/              # 通用测试工具（common_stub）
│   └── securec/             # Secure C 库 stub
├── framework/               # 测试框架辅助（AscGraphBuilder 等）
├── graph_metadef/           # Graph metadef 测试工具
├── ut/                      # 单元测试
│   ├── att/                 # ATT UT（独立二进制 att_ut）
│   ├── ascendc/             # AscendC UT
│   │   ├── api/             # AscendC API UT
│   │   └── compile/         # AscendC 编译 UT
│   ├── ascir/               # ASCIR IR UT（OBJECT 库，链接到 test_main）
│   ├── autofuse/            # Autofuse 配置 UT（独立二进制 autofuse_utils_ut）
│   ├── codegen/             # 代码生成 UT（OBJECT 库，链接到 test_main）
│   ├── common/              # 通用工具 UT（独立二进制 test_common）
│   ├── e2e/                 # 端到端 UT（OBJECT 库，链接到 test_main）
│   ├── optimize/            # 优化 pass UT（独立二进制 optimize_ut）
│   ├── py_module/           # Python 模块 C++ binding UT（OBJECT 库）
│   └── python/              # Python UT (pytest)
├── st/                      # 系统测试
│   ├── ascir/               # ASCIR ST
│   ├── att/                 # ATT ST
│   ├── backend_e2e/         # Backend 端到端 ST
│   ├── codegen/             # Codegen ST
│   │   ├── e2e/             # Codegen E2E ST（67 场景，见"E2E ST 场景"章节）
│   │   ├── ascir_tool/      # ASCIR tool ST
│   │   └── kernel_tool/     # Kernel tool ST
│   ├── common/              # 通用 ST
│   ├── optimize/            # 优化 ST
│   └── python/              # Python ST
└── v35/                     # v3.5 架构特定测试
    ├── ut/
    └── st/
```

---

## 测试目标与运行方式

### 各模块测试二进制和 CTest 集成

| 模块 | 二进制名 | 运行方式 | CTest 集成 | CTest 标签 |
|------|----------|----------|------------|------------|
| att UT | `att_ut` | 直接运行 | 无 | — |
| optimize UT | `optimize_ut` | 直接运行 | 无 | — |
| common UT | `test_common` | 直接运行 | 无 | — |
| autofuse UT | `autofuse_utils_ut` | 直接运行 | `gtest_discover_tests` | — |
| ascendc API UT | `test_ascendc_api` | 直接运行 | `gtest_discover_tests` | — |
| ascendc API UT (v35) | `test_ascendc_api_v35` | 直接运行 | `gtest_discover_tests` | — |
| ascir/codegen/py_module/e2e UT | `test_main` | 直接运行 | `gtest_discover_tests` | — |
| e2e UT (codegen) | `test_load_broadcast_*_codegen` | 直接运行 | `gtest_discover_tests` | — |
| att ST | — | 直接运行 | `add_test` | `att_st` |
| codegen ST | — | 直接运行 | `add_test` | `codegen_st` |
| common ST | — | 直接运行 | `add_test` | `test_common_st` |
| optimize ST | — | 直接运行 | `add_test` | `optimize_st` |
| backend ST | — | 直接运行 | `add_test` | `build_backend_test1/2` |
| codegen E2E ST | — | 直接运行 | `add_test` | `codegen_e2e_st_test1/2` |

注意：`att_ut`、`optimize_ut`、`test_common` 是独立二进制，不通过 CTest 发现，需直接执行。

### CTest 命令示例

```bash
# ST 标签（有效标签）
ctest --test-dir build/tests/st --output-on-failure -j8 -L att_st
ctest --test-dir build/tests/st --output-on-failure -j8 -L codegen_st
ctest --test-dir build/tests/st --output-on-failure -j8 -L optimize_st
ctest --test-dir build/tests/st --output-on-failure -j8 -L test_common_st
```

---

## C++ 测试编写规范 (gtest)

### 文件命名

- `test_<module>.cpp`：如 `test_sk_common.cpp`
- `<feature>_unittest.cpp`：如 `codegen_infershape_unittest.cpp`

### 测试类命名

使用 PascalCase + `Test` 后缀：

```cpp
class SkCommonTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(SkCommonTest, TestFunctionName) {
    // Arrange
    // Act
    // Assert
    EXPECT_EQ(result, expected);
}
```

### CMake 集成

**场景 1：向已有 target 添加测试文件**

如果 CMakeLists.txt 使用 `GLOB_RECURSE`（如 `test_common`、`optimize_ut`），新文件会自动发现，无需修改 CMake。

如果 CMakeLists.txt 显式列出源文件，需添加：

```cmake
target_sources(<test_target> PRIVATE test_new_feature.cpp)
```

**场景 2：新增测试模块（完整流程）**

1. 创建 `autofuse/tests/ut/new_module/CMakeLists.txt`：
   ```cmake
   file(GLOB_RECURSE NEW_MODULE_TEST_SRCS CONFIGURE_DEPENDS "*.cpp")
   add_executable(new_module_ut test_main.cpp ${NEW_MODULE_TEST_SRCS})
   target_link_libraries(new_module_ut PRIVATE GTest::gtest GTest::gtest_main ...)
   ```

2. 修改 `autofuse/tests/ut/CMakeLists.txt`：
   ```cmake
   add_subdirectory(new_module)
   ```

3. 如果是 OBJECT 库（链接到 `test_main`）：
   ```cmake
   target_link_libraries(test_main PRIVATE new_module_tests)
   ```

4. 可选：在 `scripts/test/run_autofuse_test.sh` 中添加模块分发逻辑。

### Stub/Mock 模式

**各模块 stub 模式不同**：

| 模块 | Stub 位置 | 形式 | 说明 |
|------|-----------|------|------|
| att | `ut/att/testcase/stub/` | 本地头文件 | Mock tiling API、model info |
| optimize | `depends/runtime/`、`depends/slog/` | 共享库 | 链接 `autofuse_runtime_stub`、`asc_slog_stub` |
| common | `depends/` | 共享库 | 同上 |
| codegen | `depends/` | 共享库 | 同上 |
| super_kernel AOT | `aot/ut/stub/` | 本地头文件 | Mock ACL、runtime |

编写新测试时，优先复用已有 stub。optimize/common/codegen 模块不需要本地 stub 目录，通过链接共享 stub 库隔离依赖。

---

## 复杂图构造指导

### 问题

optimize 模块测试需要构造 `af::AscGraph` 对象，每个节点需手动设置 dtype、axis、repeats、strides、sched.axis、compute_type、ir_attr，非常冗长（30+ 行/节点）。

### 方案 1：使用 AscGraphBuilder（推荐）

`autofuse/tests/framework/easy_asc_graph/asc_graph_builder.h` 提供 fluent API：

```cpp
#include "framework/easy_asc_graph/asc_graph_builder.h"

auto graph = AscGraphBuilder("test_graph")
    .AddData("x", dtype::FLOAT16, {s0, s1})
    .AddLoad("load1")
    .AddOp<Abs>("abs1")
    .AddStore("store1")
    .Build();
```

### 方案 2：复用已有工厂函数

`test_optimizer.cpp` 中有多个图构造工厂函数可复用：
- `CreateAscBackendGraphTwoInTwoOut()`
- `CreateOneNodeAscGraph()`
- `CreateTailPackAscGraph()`

### 方案 3：提取共享 fixture

将常用图结构提取到 `tests/framework/` 下的共享 fixture 中。

---

## Python 测试编写规范 (pytest)

### 文件命名

- `test_<module>.py`：如 `test_super_kernel.py`
- `test_<module>_<feature>.py`：如 `test_super_kernel_option_parse.py`

### 目录约定

**仅 super_kernel 有 conftest.py 自动标记**：
- `super_kernel/tests/ut/` 下的文件自动标记为 `@pytest.mark.ut`
- `super_kernel/tests/st/` 下的文件自动标记为 `@pytest.mark.st`

**autofuse 没有 conftest.py**，Python 测试通过 CMake `add_test()` 注册，不依赖 pytest marker。

### 示例

```python
import pytest

class TestMyFeature:
    def test_basic_functionality(self):
        result = my_function(input_data)
        assert result == expected

    def test_edge_case(self):
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

### Fixtures

共享 fixtures 放在 `tests/fixtures/` 目录下，通过 `conftest.py` 自动发现（仅 super_kernel）。

---

## Python + C++ 混合测试

### pyautofuse 模块

`autofuse/compiler/py_module/` 提供 CPython 扩展模块 `pyautofuse`，暴露 `ascir`、`Autofuser`、`CodeGen` 等接口。

**Python 端测试**（`ut/python/`）：

```python
from autofuse.pyautofuse import ascir, Autofuser, AutofuserOptions, Schedule, CodeGen

def test_autofuse_pipeline():
    graph = construct_test_graph()
    options = AutofuserOptions()
    autofuser = Autofuser(options)
    result = autofuser.fuse(graph)
    # verify result
```

**C++ 端测试**（`ut/py_module/`）：

测试 CPython 扩展的内部实现，需调用 `Py_Initialize()` 和 `PyInit_pyautofuse()`。

运行命令：

```bash
bash scripts/test/run_autofuse_test.sh -u -m codegen -j 8  # 包含 py_module UT
```

---

## E2E ST 场景

### 添加新场景

`autofuse/tests/st/codegen/e2e/` 下有 67 个 E2E 场景，每个场景使用 CMake 宏 `add_codegen_e2e_st_test()`。

**步骤**：

1. 创建场景目录 `st/codegen/e2e/my_scenario_expect_code/`

2. 创建源文件：
   - `my_scenario_codegen.cpp` — 生成 kernel 代码
   - `my_scenario_codegen_tiling.cpp` — tiling 逻辑
   - `test_e2e_my_scenario_expect_kernel.cpp` — 验证 kernel 执行

3. 创建 `CMakeLists.txt`：
   ```cmake
   add_codegen_e2e_st_test(my_scenario_expect_code
       TILING my_scenario_codegen_tiling.cpp
       CODEGEN my_scenario_codegen.cpp
       KERNEL_SRC
           my_scenario_kernel.cpp
           my_scenario_tiling.cpp
           autofuse_tiling_data.h
       TEST_SRC test_e2e_my_scenario_expect_kernel.cpp)
   ```

4. 在父 `CMakeLists.txt` 中添加：
   ```cmake
   add_subdirectory(my_scenario_expect_code)
   ```

**两阶段执行**：
- Phase 1：编译 codegen 可执行文件，运行生成 kernel 源码
- Phase 2：编译生成的 kernel + 测试代码，在 Ascend 模拟器上执行验证

---

## RDV 设备端测试

### 概述

`super_kernel/tests/aot/rdv/` 包含设备端验证测试（Real Device Verification），用于在真实设备或模拟器上验证 super kernel 算子的正确性。

### 目录结构

```
aot/rdv/
├── tensor_list/             # 共享 tensor list 工具
├── ops/
│   ├── interface/           # 算子调度接口（OpsInterface）
│   ├── common/              # 共享 kernel 工具
│   ├── rms_norm/            # RMSNorm 算子
│   │   ├── tests/main.cpp   # 测试入口
│   │   └── tests/CMakeLists.txt
│   ├── weight_quant_batch_matmul_v2/
│   └── ...
```

### 编写新 RDV 测试

1. 在 `ops/` 下创建算子目录
2. 实现 kernel（`.h` + `.asc` 文件）
3. 在 `tests/main.cpp` 中编写测试逻辑（分配设备内存、拷贝数据、launch kernel、验证结果）
4. 在 `OpsInterface` 中注册算子到 `g_sk_fun_map`

### 运行

RDV 测试不集成到 `build.sh`，需手动编译运行：

```bash
cd super_kernel/tests/aot
mkdir build && cd build
cmake .. -DENABLE_CPP_UTEST=ON
make <op_test_target>
./<op_test_target>
```

---

## 覆盖率

### C++ 覆盖率

```bash
# 编译时启用 gcov
sh build.sh -u --module=superkernel --impl=cpp -c
# 使用 lcov 生成报告（由 generate_cpp_cov.sh 脚本处理）
# 报告位置：build/coverage/html/index.html
```

Autofuse C++ 覆盖率：

```bash
bash scripts/test/run_autofuse_test.sh -u -m common -c -j 8
# lcov 数据在 build/ 下 .gcda/.gcno 文件中
```

### Python 覆盖率

```bash
sh build.sh -u --module=superkernel --impl=py -c
# 使用 pytest-cov 生成报告，配置文件为 sk_ut_cfg.toml / sk_st_cfg.toml
# 报告位置：super_kernel/coverage/ut/html/ 或 super_kernel/coverage/st/html/
```

### 覆盖率分析

生成报告后，关注：
- 未覆盖的分支（`show_missing = true` 配置已启用）
- 低于 80% 的文件（`generate_cpp_cov.sh` 会列出）
- 排除项：`__repr__`、`__main__`、`NotImplementedError` 等已配置排除

---

## 常见编译错误排查

### FAQ

**Q1: `version.info does not exist`**

CANN toolkit 路径不匹配。检查 `/usr/local/Ascend/ascend-toolkit/latest/` 是否存在，或创建符号链接：

```bash
sudo ln -sf /home/developer/Ascend/cann-X.Y.Z /usr/local/Ascend/ascend-toolkit/latest
```

**Q2: `undefined symbol: _ZTVN2af11AscNodeAttrE`**

`LD_LIBRARY_PATH` 未包含所有必要的共享库路径。确保包含：

```bash
export LD_LIBRARY_PATH=\
  <build_dir>/autofuse/graph_metadef/graph/ascendc_ir/generator:\
  <build_dir>/autofuse/graph_metadef/graph:\
  <build_dir>/autofuse/graph_metadef/graph/expression:\
  <build_dir>/autofuse/graph_metadef/graph/ascendc_ir:\
  $ASCEND_HOME_PATH/lib64:\
  $LD_LIBRARY_PATH
```

**Q3: `opening dependency file ... No such file or directory`**

protobuf 编译时的依赖文件路径问题。清理 build 目录重新编译：

```bash
rm -rf build && bash scripts/test/run_autofuse_test.sh -u -m <module> -j 8
```

**Q4: 链接时找不到 `-lasc_slog_stub` 或 `-lautofuse_runtime_stub`**

确保 CMakeLists.txt 中正确链接了 stub 库：

```cmake
target_link_libraries(<target> PRIVATE
    asc_slog_stub
    autofuse_runtime_stub
    atrace
)
```

这些库在 `autofuse/tests/depends/` 下定义，由 `autofuse/tests/CMakeLists.txt` 构建。

**Q5: `gtest_discover_tests` 未生效**

只有部分 target 使用 `gtest_discover_tests`（`test_main`、`autofuse_utils_ut`、`test_ascendc_api`）。`att_ut`、`optimize_ut`、`test_common` 是独立二进制，直接运行即可。

### 通用排查流程

1. **检查环境变量**：`ASCEND_HOME_PATH`、`LD_LIBRARY_PATH`、`CMAKE_PREFIX_PATH`
2. **检查 CANN 安装**：`ls $ASCEND_HOME_PATH/lib64/libruntime.so`
3. **清理重建**：`rm -rf build && bash scripts/test/run_autofuse_test.sh ...`
4. **查看 CMake 日志**：`build/CMakeFiles/CMakeOutput.log` 和 `CMakeError.log`

---

## 测试失败排查

### 排查流程

1. **确认失败类型**：
   - 断言失败（`EXPECT_EQ` 等）→ 检查输入数据和预期值
   - 崩溃（segfault）→ 检查空指针、未初始化对象
   - 超时 → 检查无限循环（如 `SubStringReplace` 空 `from` 参数）

2. **检查 stub 是否正确**：
   - Runtime stub 返回的 SoC 版本是否匹配预期
   - Slog stub 是否正确初始化

3. **检查环境**：
   - `LD_LIBRARY_PATH` 是否包含所有必要的 `.so`
   - 是否有残留的 `.gcda` 文件影响覆盖率构建

4. **隔离测试**：
   ```bash
   ./test_binary --gtest_filter="TestClass.TestName"
   ```

---

## CMake 验证清单

新增测试后，执行以下验证：

1. **编译验证**：
   ```bash
   cmake --build . --target <test_target> -j8
   ```
   确保无编译错误。

2. **发现验证**（仅使用 `gtest_discover_tests` 的 target）：
   ```bash
   ctest --test-dir build/tests/ut -N | grep <test_name>
   ```
   确保测试被 CTest 发现。

3. **运行验证**：
   ```bash
   ./<test_binary> --gtest_filter="<TestClass>.<TestName>"
   ```
   确保测试通过。

4. **GLOB 验证**（使用 `GLOB_RECURSE` 的 target）：
   - 确认新文件在 glob 路径范围内
   - 如不在，需手动添加到 `target_sources`


