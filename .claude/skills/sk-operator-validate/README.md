# SK 算子校验

`sk-operator-validate` 是 SK 算子校验的唯一公开入口，负责 contract 和
spec rule pack。compat rule pack 是可显式开启的高级静态风险检查，只报告已有
官方来源覆盖到的内容，不作为默认 CANN 版本支持白名单。端到端场景优先通过
`sk-operator-pipeline run-sk-pipeline` 调用；定位某个校验阶段时可直接运行本工具。

## 常用命令

```bash
python3 <skills_root>/sk-operator-validate/scripts/operator_validate.py validate-operator \
  --asset build/examples/sk-codegen/adapted/my_op/operator-sk-adapted \
  --output-dir build/examples/sk-validate/my_op \
  --rule-pack spec
```

```bash
python3 <skills_root>/sk-operator-validate/scripts/operator_validate.py validate-operator \
  --asset build/examples/sk-codegen/adapted/my_op/operator-sk-adapted \
  --output-dir build/examples/sk-validate/my_op \
  --rule-pack compat \
  --target-chip ascend-910b
```

列出内置 rule pack 元数据：

```bash
python3 <skills_root>/sk-operator-validate/scripts/operator_validate.py list-rule-pack --rule-pack spec
python3 <skills_root>/sk-operator-validate/scripts/operator_validate.py list-rule-pack --rule-pack compat
```

## 输出

- `operator-validation-findings.json`
- `operator-validation-report.md`

finding envelope 使用统一 schema：`skill_source` 固定为 `operator-validate`。
compat 输出会额外包含 `metadata.compat`，用于说明本次 target context 下实际检查了哪些
source-backed rules。
