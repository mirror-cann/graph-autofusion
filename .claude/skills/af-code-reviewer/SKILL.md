---
name: af-code-reviewer
description: |
  graph-autofusion 项目代码审查和贡献规范辅助。
  **必须触发的场景**：用户提到代码审查、review、代码规范、贡献规范、CONTRIBUTING、代码风格、clang-format、DCO、CLA、commit message、PR 模板、pre-commit、代码检查等。
---

# graph-autofusion 代码审查与贡献规范

辅助用户进行代码审查、遵循贡献规范和提交合规的 PR。

## PR 提交前自查清单

在提交 PR 前，请确认以下事项：

### 1. 代码规范

- [ ] 代码符合 Google [开源代码规范](https://google.github.io/styleguide/)
- [ ] C++ 代码已通过 `clang-format` 格式化
- [ ] 变量、函数、类命名符合规范
- [ ] 注释完整且准确
- [ ] 无硬编码的密钥或敏感信息

### 2. 代码格式化

项目使用 `.clang-format` 配置，关键规则：

| 规则 | 值 |
|------|-----|
| 标准 | C++11 |
| 缩进宽度 | 4 空格 |
| 行宽限制 | 120 字符 |
| 大括号风格 | Custom（函数定义后换行） |
| 指针对齐 | Left |
| Tab | 不使用（UseTab: Never） |
| 命名空间缩进 | None |
| 排序 includes | 不自动排序 |

格式化命令：

```bash
# 格式化单个文件
clang-format -i path/to/file.cpp

# 使用 pre-commit 自动格式化
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### 3. Commit Message 规范

格式：`<类型>: <简短描述>`

| 类型 | 说明 | 示例 |
|------|------|------|
| feat | 新功能 | feat: 添加用户注册功能 |
| fix | 修复 bug | fix: 修复登录态过期问题 |
| docs | 文档更新 | docs: 更新 API 使用说明 |
| style | 代码格式调整 | style: 调整代码缩进 |
| refactor | 重构 | refactor: 优化服务类结构 |
| perf | 性能优化 | perf: 减少查询次数 |
| test | 测试相关 | test: 添加登录功能单元测试 |
| chore | 构建/工具链 | chore: 更新配置 |
| ci | CI 配置 | ci: 添加自动化测试流程 |

提交 PR 前建议 rebase 合并多个 commit：

```bash
git rebase -i HEAD~N  # N 为要合并的 commit 数
```

### 4. PR 模板

项目 PR 模板位于 `.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md`，提交 PR 时请按模板填写以下章节：
- **描述**：清晰准确地描述本次 PR 的意图和变更内容
- **变更类型**：选择 Bug 修复 / 新功能 / 代码风格更新 / 重构 / 构建变更 / 文档更新
- **关联的 Issue**：如有关联 Issue，在右侧"关联 Issue"部分添加链接
- **如何测试**：描述测试此变更的步骤和前提条件
- **核对清单**：确认代码风格、自测、文档更新、commit 规范等
- **其他信息**：补充说明

### 5. 协议签署

首次贡献前需完成：
- CLA 协议签署（通过 [cann-community](https://gitcode.com/cann/community)）
- 了解 GitCode 工作流

## 代码审查要点

### 审查者检查清单

1. **功能正确性**：代码逻辑是否正确实现了需求
2. **代码规范**：是否符合项目代码规范和 clang-format 配置
3. **测试覆盖**：新增/修改的代码是否有对应测试
4. **性能影响**：是否引入性能退化
5. **安全性**：是否有敏感信息泄露风险
6. **文档更新**：接口变更是否同步更新了文档
7. **Commit 规范**：commit message 是否清晰规范

### 常见审查问题

- 缺少错误处理
- 硬编码的魔数（magic number）
- 未释放的资源
- 不完整的注释
- 过长的函数（建议拆分）
- 重复代码（建议抽取公共函数）

## Pre-commit 检查

项目配置了 `.pre-commit-config.yaml`：

1. **clang-format** (v16.0.0)：自动格式化 C/C++ 代码
2. **OAT Compliance Check**：开源审计工具检查

```bash
# 安装 pre-commit
pip install pre-commit
pre-commit install

# 手动运行所有检查
pre-commit run --all-files

# 仅运行 OAT 检查
pre-commit run oat-check --all-files
```

## 贡献场景

| 场景 | 流程 |
|------|------|
| Bug 修复 | 新建 `Bug-Report` Issue → `/assign` → 修复 → 提交 PR |
| 新功能 | 新建 `Requirement` Issue → 方案讨论 → `/assign` → 实现 → 提交 PR |
| 文档纠错 | 新建 `Documentation` Issue → `/assign` → 修复 → 提交 PR |
| 帮助他人 | 在 Issue 中评论交流 → `/assign` → 协助解决 |
