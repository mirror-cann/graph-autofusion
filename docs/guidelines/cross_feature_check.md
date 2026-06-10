# 特性交叉影响检查

> **目的**：确保 Graph-autofusion 新需求在 SuperKernel、Autofuse、构建交付和测试场景下均被覆盖，避免只验证单一路径导致功能缺失、性能退化或交付失败。

逐项分析本需求与以下场景的关系，填写“适用/不适用”。适用场景需在设计章节给出对应方案；不适用场景需说明理由。

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| SuperKernel Python 接口 | {适用/不适用} | {是否影响 Python 包、选项解析、pytest UT/ST、wheel 交付} |
| SuperKernel C++/AOT 接口 | {适用/不适用} | {是否影响 `libascendsk.so`、AOT 测试、设备端 RDV、ABI/API} |
| Autofuse 图优化 | {适用/不适用} | {是否影响 ASCIR 图结构、融合 pass、图改写等价性、pass 时序} |
| Autofuse Codegen/Backend | {适用/不适用} | {是否影响代码生成、tiling、kernel 编译、backend E2E} |
| AscendC API / Runtime 交互 | {适用/不适用} | {是否新增 runtime/aclrt/AscendC 调用，是否满足资源生命周期和同步约束} |
| Python/C++ 混合绑定 | {适用/不适用} | {是否影响 pyautofuse、CPython 扩展、Python 导入路径和异常传播} |
| 构建与打包 | {适用/不适用} | {是否影响 CMake、build.sh、第三方依赖、run 包内容、安装路径} |
| 测试与覆盖率 | {适用/不适用} | {是否需要新增 UT/ST/RDV/pytest/gtest/覆盖率验证} |
| 性能与日志 | {适用/不适用} | {是否影响编译时长、融合收益、执行耗时、日志量和内存占用} |
| 兼容性 | {适用/不适用} | {是否影响已有 API/ABI、配置、产物布局、历史模型或脚本调用方式} |

## 分析指引

- **SuperKernel Python 接口**：检查 `super_kernel/*.py`、`super_kernel/src/`、`super_kernel/tests/ut`、`super_kernel/tests/st`。若修改选项解析、异常行为或 wheel 内容，需补充 pytest 用例。
- **SuperKernel C++/AOT 接口**：检查 `super_kernel/kernel/`、`super_kernel/tests/aot/`。对外符号、结构体、枚举和动态库行为变化需评估 ABI/API 兼容。
- **Autofuse 图优化**：检查 `autofuse/optimize/`、`autofuse/ascir/`、`autofuse/graph_metadef/`。图改写必须保持数据边和控制边等价；新增 pass 需说明时序依赖。
- **Autofuse Codegen/Backend**：检查 `autofuse/codegen/`、`autofuse/backend/`、`autofuse/tests/st/codegen/e2e/`。新增 codegen 行为需覆盖生成代码、tiling 和 kernel 执行验证。
- **AscendC API / Runtime 交互**：新增 `rt*`、`aclrt*`、AscendC API 调用时，检查内存有效期、异步拷贝同步、size 为 0 的行为和错误码处理。
- **Python/C++ 混合绑定**：检查 `autofuse/compiler/py_module/`。新增接口需同时验证 C++ 绑定层和 Python 使用方式。
- **构建与打包**：修改 CMake 或 install/package 逻辑时，确认 `build.sh --pkg -j 8`、`--no-autofuse`、模块化测试路径是否仍可用。
- **测试与覆盖率**：新功能优先按 TDD 添加最小失败用例；Bug 修复必须有复现用例。C++ 使用 gtest/mockcpp，Python 使用 pytest。
- **性能与日志**：编译期循环、图遍历、字符串处理、日志和运行时调度路径需评估性能；执行高频路径禁止新增默认开启的海量日志。
- **兼容性**：对外接口、安装路径、脚本参数、配置项和产物命名变化需说明兼容策略；不确定时先提 Issue 讨论。
