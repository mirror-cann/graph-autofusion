# af-reg-ascir

graph-autofusion 项目 ASCIR 注册、新增/修改、dtype、tmp buffer、UT/ST 辅助 Skill。

## 功能

- 指导新增或修改 ASCIR 算子注册流程
- 辅助维护 V35/V2 ASCIR 注册、ATT、Codegen 和性能模型
- 指导 dtype 扩展、类型转换和 tmp buffer 注册
- 每个新增 ASCIR 默认考虑并补充 backend E2E；更新 ASCIR 时强制审计现有 E2E 是否需要同步修改
- 参考 BesselJ0 生成 share graph、backend generator、ICPU 运行与精度验证、CMake 和统一测试入口
- 分析 ASCIR API 名、OpType、Python API 和测试用例的影响范围
- 按完整矩阵检查 regbase 构建/注册、ATT、Codegen、Python ops/IR Attr、perf、UT、share graph、backend E2E、脚本与打包
- 识别 regbase 内部 include、官方 AscendC API 头缺失、Device 编译选项、旧包加载等上板故障
- 根据输入输出、模板属性、多输出和 dtype 变化给出最小修改方案

## 触发场景

用户提到以下关键词时自动触发：
- ASCIR 注册、ascir op、REG_ASC_IR
- 新增算子、修改算子、算子注册
- dtype 注册、类型转换、cast-map
- tmp buffer、CalcTmpBufSize、CalcXxxTmpSizeV2
- ASCIR UT、ASCIR ST、E2E 测试、端到端测试生成

## 验证方法

在 opencode 中输入以下指令验证 Skill 是否生效：
- "帮我新增一个 ASCIR 算子注册"
- "给 ASCIR 算子增加一个 dtype 支持"
- "这个 ASCIR 算子需要注册 tmp buffer 吗？"
- "修改 ASCIR API 名称需要同步改哪些地方？"
