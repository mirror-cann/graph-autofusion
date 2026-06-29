# HTML Design Workflow

## 目的

`sk-network-analysis` 的 HTML 优化采用“可选外部设计指导 + 本 skill 实现”模式。

这套模式的目标是：

- 保持 `sk-network-analysis` 自己掌握诊断语义和 JSON 契约
- 用目标环境可选设计评审能力收口页面层级、导航、视觉和交互质量
- 避免把第三方设计建议直接混进业务判断逻辑

## 可选外部设计能力

发布包不强依赖任何外部设计 skill。目标环境如果提供设计评审能力，可以覆盖：

- 页面层级
- 导航组织
- 阅读顺序
- 大数据下的 `Top-N + fold + aggregate`
- HTML/CSS/JS 的视觉系统
- 组件层级
- 页面风格
- 密集信息页面的可读性和交互细节

## 边界

外部设计建议可以改进：

- 页面分区
- 首屏主次
- 卡片、折叠、导航、筛选、视觉样式
- graph 页的上下文栏和页面组织

外部设计建议不得改动：

- 日志解析逻辑
- `evidence_tier` / `capability_mode` / `diagnostic_completeness`
- 任何诊断结论的定义

## 推荐调用顺序

每次做 HTML 优化，统一按这个顺序：

1. `sk-network-analysis` 先明确页面目标和结构化字段来源
2. 可选外部设计评审先给出页面层级、入口组织和阅读路径
3. 可选外部设计评审再给出视觉系统、组件样式和 HTML/CSS/JS 改造建议
4. 最后在本 skill 代码中实现

## 第一阶段目标页面

默认只对这 3 个页面启用这套工作流：

- `run-portal.html`
- `scope-graph.html`
- `task-queue-graph.html`

次级同步页面：

- `hang-crash-report.html`
- `performance-report.html`

## 性能页面的额外前提

- `performance-report.html` 后续进入主改造前，要先完成一轮“事件优先”语义收口：
  - `sk_event_dev_device_*.json` 作为性能主资产
  - scope / task 只作为结构锚点和深链补充
- 在这条语义收口完成前，第三方设计 skill 可以优化性能页层级和风格，但不应把当前结构固化成长期页面合同

## 设计输入包模板

每次调用外部设计 skill 前，先准备一份最小输入：

- 页面名
- 页面目标
- 当前用户的首要任务
- 当前首屏问题
- 当前结构化字段来源
- 当前可直接点击的对象
- 当前大数据下的主要认知负载问题
- 当前不能改动的语义边界

推荐最小模板：

```md
Page: run-portal.html
Goal: make the portal a decision cockpit instead of a flat report directory
Primary user task: decide what to read next in <30s
Current issues:
- first screen still has too many equal-weight blocks
- summary-only vs graph-capable is visible but not dominant enough
- primary vs secondary entry is still not obvious enough
Structured sources:
- scope-library.json
- graph-library.json
Important objects:
- scopeId
- nodeId
- taskIndex
Do not change:
- capability_mode semantics
- current_capabilities semantics
- evidence tier semantics
```

## 默认落地原则

- 优先改信息层级，再改视觉风格
- 优先减少“首屏平铺”，再加新装饰
- graph 页仍是一等能力，不能被摘要表格替代
- summary 页只做判断和导航，不做全量浏览
- structured JSON 继续作为锚点和稳定索引，不被第三方设计 skill 重定义
- 第一波页面设计 brief 固定沉淀为：
  - `portal-ia-brief.md`
  - `portal-frontend-brief.md`
- 第二波 graph 页面设计 brief 固定沉淀为：
  - `scope-graph-ia-brief.md`
  - `task-queue-graph-ia-brief.md`
  - `scope-graph-frontend-brief.md`
  - `task-queue-graph-frontend-brief.md`
- 第三波故障与性能主页面 brief 固定沉淀为：
  - `hang-ia-brief.md`
  - `hang-frontend-brief.md`
  - `performance-ia-brief.md`
  - `performance-frontend-brief.md`
