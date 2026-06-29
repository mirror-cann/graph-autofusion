# diagnosis matrix

| 问题类型 | 先看哪些资产 | 入口脚本 | 当前能回答什么 | 当前缺口 |
| --- | --- | --- | --- | --- |
| 优化是否进入 SK | `super_kernel.log` | `diagnose_run.py` | 是否进入 `aclskOptimize`、是否完成 build tasks | 若缺 `super_kernel.log`，需保留 model 目录下的 SK 主日志 |
| scope 切分异常 | `sk_scope_split.log`、`sk_node_detail.log` | `sk_scope_visualizer.py` | scope、stream、processed node；优先通过 `scope-graph.html` 做直观定位，并查看 `graph-library.json` / `scope-library.json` 的汇总 | 若缺这两类资产，先保留 scope split 和 node detail dump |
| 融合结果判断 | `sk_fused_nodes.log` | `sk_library_extractor.py`, `sk_scope_visualizer.py` | 哪些节点将融合 | 不能单独代表 update 结果；若缺失需保留 fused-node dump |
| update 方案理解 | `sk_scope_split.log`、`sk_fused_nodes.log`、`sk_device_args.log`、`super_kernel.log` | `sk_library_extractor.py`, `sk_scope_visualizer.py`, `sk_task_queue_visualizer.py` | scope 结构、update 结构、内存配对与图谱基础 | launch-info payload 和完整 graph writeback 细节仍有缺口；结果中会提示下一步还缺哪些证据 |
| task queue 异常 | `sk_device_args.log` | `sk_task_queue_visualizer.py` | AIC / AIV task 结构、task index、entry、args；优先通过 `task-queue-graph.html` 做直观定位 | 若缺失需保留 device args dump；没有它时只能先看 scope/fused 结构 |
| 卡死 / coredump | `super_kernel.log`、`sk_device_args.log`、plog、device log | `diagnose_run.py` | phase、signal confidence、hard failure/runtime warning/noise 分类、runtime 辅助线索 | 如果当前信号不足，结果中会提示继续保留 runtime/device/exception 侧证据 |
| 性能问题 | `sk_event_dev_device_*.json`、`sk_device_args.log`、`sk_fused_nodes.log` | `diagnose_run.py` / `sk_node_tracing.py` | 当前已接入 time event 资产，能做结构 + 队列 + 时间事件矩阵、诊断完整度，以及当前诊断焦点 | 当前主组织仍偏向 update/function；后续需改成“事件优先”，让 `sk_event_dev_device_*.json` 成为性能主资产，若时间覆盖不足则继续保留 event/profiling 相关资产 |

## HTML 设计改造约束

- 当任务是“优化页面层级、导航、风格、信息降噪”时：
  - 可以使用目标环境提供的外部设计评审能力作为可选建议
  - 先收口页面结构和阅读顺序，再收口视觉系统和组件组织
- 当任务是“修诊断逻辑、修证据提取、修 JSON 契约”时：
  - 不交给第三方设计 skill
  - 继续直接在 `sk-network-analysis` 内实现
- 当前允许第三方设计指导优先覆盖的页面：
  - `run-portal.html`
  - `scope-graph.html`
  - `task-queue-graph.html`
