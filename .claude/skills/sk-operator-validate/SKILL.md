---
name: sk-operator-validate
description: 校验 SK 算子适配产物的规范 contract、源码结构和可选兼容性事实，输出统一 findings，供 pipeline 决策是否继续、自动修复或要求人工确认。
---

# SK 算子校验

这个 skill 负责对适配后的算子资产做结构化校验。它只报告明确规则覆盖到的问题，不把版本或芯片支持范围伪装成完整官方白名单。

入口：

```bash
python3 <skills_root>/sk-operator-validate/scripts/operator_validate.py validate-operator \
  --asset <operator-sk-adapted-dir> \
  --output-dir <dir> \
  --rule-pack spec
```

## 规则包

- `spec`：默认规则集，检查 SK 适配结构、关键源码形态、contract 完整性和生成产物一致性。
- `compat`：显式高级规则集，只检查已经有来源覆盖的兼容性事实；它不是默认 CANN 版本支持白名单。

## 输出

主要输出文件：

- `operator-validation-findings.json`

每条 finding 都包含：

- `severity`
- `code`
- `message`
- `path`
- `stage`
- `iteration_index`
- `auto_remediable`

## 边界

这个 skill 只负责校验和报告，不负责：

- 修改源码
- 生成 SK binding
- 构建 wheel
- 运行正确性验证

自动修复由 `sk-operator-codegen apply-remediation` 或上层 pipeline 决定。硬件运行正确性由 `sk-operator-sample-gen` 和 `sk-operator-build-package` 参与闭环。
