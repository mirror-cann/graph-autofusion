# af-code-reviewer

graph-autofusion 项目代码审查和贡献规范辅助 Skill。

## 功能

- PR 提交前自查清单生成
- 基于 `docs/guidelines/编码红线.md` 检查安全、资源、ABI/API、图改写和 runtime 生命周期红线
- 基于 `docs/guidelines/cross_feature_check.md` 检查 SuperKernel、Autofuse、构建交付和测试场景的交叉影响
- 基于 `.clang-format` 检查代码风格
- 基于 `CONTRIBUTING.md` 检查贡献规范合规性
- Commit message 规范检查
- Pre-commit 检查指导
- 开源贡献礼仪和 CLA/DCO 指导

## 触发场景

用户提到以下关键词时自动触发：
- 代码审查、review、代码规范
- 检视 PR、检查 PR、编码红线、设计规范
- 贡献规范、CONTRIBUTING
- 代码风格、clang-format
- commit message、PR 模板
- pre-commit、CLA、DCO

## 验证方法

在 opencode 中输入以下指令验证 Skill 是否生效：
- "帮我检查代码规范"
- "提交 PR 前需要做什么？"
- "commit message 格式是什么？"
