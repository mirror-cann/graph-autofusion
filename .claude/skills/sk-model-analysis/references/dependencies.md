# 依赖边界

## 可迁移输入

`sk-model-analysis` 消费用户提供的 SK 运行结果目录。目标环境不需要携带维护侧的设计文档、测试样例或规划材料。

推荐用户输入：

- 包含 `sk_meta/`、日志、`kernel_meta/`、profiling/event 输出的结果根目录。
- `sk_meta/<pid>/` 这类 model asset 根目录。
- 单个 `model_*` 目录。

主要证据文件：

- `super_kernel.log`
- `sk_scope_split.log`
- `sk_node_detail.log`
- `sk_fused_nodes.log`
- `sk_device_args.log`
- `sk_task_queue.json` as shadow validation during the compatibility period
- `sk_event_dev_device_*.json` / `sk_prof_device_*.json`

## 可迁移知识源

- `references/workflow.md`
- `references/artifact-contract.md`
- `references/script-index.md`
- `references/diagnosis-matrix.md`
- `references/update-view-registry-guide.md`

## 已知限制

- profiling 自动根因归因仍不完整。
- `performance-report.html` 的 event 到结构解释仍需要继续增强。
- launch-info payload 和完整 graph writeback 细节依赖上游日志是否保留。
- 缺少 `sk_meta`、event 或 profiling 资产时，只能生成 summary-only 或 insufficient diagnostics。
