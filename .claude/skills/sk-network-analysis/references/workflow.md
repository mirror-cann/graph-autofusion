# 工作流

## 统一运行合同

- 顶层入口固定为：
  - `python3 <skills_root>/sk-network-analysis/scripts/network_analysis.py <subcommand> ...`
- 默认先跑 `base mode`
  - 不依赖联网
  - 不依赖大模型
  - 负责生成稳定报告和中间数据
- 可选再跑 `ai mode`
  - 通过 `--with-ai` 触发
  - 只做进一步提示
  - 不覆盖基础结果

## 主入口

- `analyze`
  - 分析用户给定的结果路径
  - 先识别 `log / sk_meta / kernel_meta / model_*` 等资产
  - 缺资产时明确返回：
    - 为什么这类资产重要
    - 缺它会影响哪些诊断能力
    - 如果后续要继续定位，应该补什么信息
  - 对每类报告同时给出：
    - 是否成功生成
    - 当前诊断完整度是 `complete / limited / insufficient`
- `diagnose-hang-crash`
  - 聚合 `sk_meta + plog + device log`
  - 输出 hang / coredump / exception 报告
- `diagnose-performance`
  - 当前聚合 `sk_event_dev_device_*.json + task queue + fused nodes`
  - 已能读取 event 文件并做基础相关，但后续要把性能线改成“事件优先”，让 event 成为主资产、scope / task 退到辅助跳转层
  - 输出性能摘要
- `trace-nodes`
  - 输出 tracing 兼容的 `node-trace.json`

## 当前基础范围

- 支持对已有结果目录做多报告输出
- 支持资产识别、缺失资产说明、补采建议和部分成功状态
- 为后续统一 HTML 预埋稳定主键和报告索引
- 在信息不足时给出下一步还需要哪类证据
- `scope` / `task queue` 现已补结构化 JSON 产物，供 portal 和后续联动消费
- `update view` 的信息库建设遵循 `update-view-registry-guide.md`
  - 当前主骨架是 `scope_library.scopes[]`
  - scope 内 update 载荷放在 `scope_library.scopes[].update`
  - graph-backed update 标准行放在 `graph_library.node_update_registry.rows`
  - `task queue` 与 `device_task_library` 只作为下钻和 synthesized custom 绑定层，不反推主图
- `scope-graph.html` / `task-queue-graph.html` 作为正式 interactive views 保留，不再被简化摘要页替代
- `run-portal.html` 统一组织：
  - `Interactive Views`
  - `Structured Views`
  - `Summary Views`
- `hang-crash-report.html` / `performance-report.html` 作为浏览器友好的摘要页与 Markdown/JSON 产物并存
- multi-model / multi-model-instance 场景会生成 run-level portal 和 per-model report bundle

## 当前不做

- 不在第一阶段改 `super_kernel`
- 不在第一阶段承诺 profiling 自动归因
- 不在第一阶段承诺实时联动的完整前端平台
- 不把 `--with-ai` 当作基础产物生成前提

## 后续扩展

- 多报告之间更强的对象级联动
- profiling 数据自动归因
- `performance-report` 的事件优先重构：
  - `sk_event_dev_device_*.json` 作为主资产
  - scope / task 作为结构补充和深链层
- 结构级分析细节增强
- 真实 AI 增强分析
