# af-build-runner

graph-autofusion 项目编译构建辅助 Skill。

## 功能

- 指导使用 `build.sh` 的各种参数和编译模式
- 解析 CMake 编译错误并给出修复建议
- 管理 CANN Toolkit 依赖版本匹配
- 指导环境配置和第三方依赖管理

## 触发场景

用户提到以下关键词时自动触发：
- 编译、构建、build、cmake、make
- 编译报错、编译错误、build error
- 打包、pkg、package
- build.sh、编译选项、编译类型

## 验证方法

在 opencode 中输入以下指令验证 Skill 是否生效：
- "如何编译项目？"
- "build.sh 有哪些参数？"
- "编译报错找不到 CANN Toolkit 怎么办？"
