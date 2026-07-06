---
name: sk-model-analysis
description: 面向 SuperKernel 场景的整网诊断 skill，覆盖结果目录资产发现、hang/coredump 定位、性能分析、scope/task 图可视化、节点追踪和 AOT task queue 兼容解析。
---

# SK 模型分析

当任务涉及整网级 SK 诊断时使用这个 skill，典型场景包括：

- 分析用户提供的 SK 运行结果目录
- 定位 hang、coredump、异常退出
- 分析性能问题
- 生成 scope/task 图和交互式 HTML 视图
- 追踪 node、task、scope 之间的关系

入口：

```bash
python3 <skills_root>/sk-model-analysis/scripts/model_analysis.py <subcommand> ...
```

## 模式

- base mode：默认模式，不依赖网络服务或大模型，生成机器可消费的 HTML / Markdown / JSON 报告。
- ai mode：可选模式，通过 `--with-ai` 开启，只能基于 base mode 产物补充建议；即使 AI 不可用，base artifacts 仍然有效。

## 主要命令

- `analyze`：分析一个用户结果目录，识别资产、解释缺失证据影响，并给出下一步应收集的信息。
- `diagnose-hang-crash`：关联 `sk_meta`、plog、device log，定位 hang/coredump/异常。
- `diagnose-performance`：关联 `sk_event_dev_device_*.json`、`sk_prof_device_*.json`、task 结构、fused nodes 和时间事件。大 JSON 是性能分析的一等输入，支持懒加载、`ijson` 流式解析和多进程批处理。
- `trace-nodes`：导出 `edge://tracing` / `chrome://tracing` 兼容的 node trace，并输出跨报告链接元数据。

## 输出契约

base artifacts 包括：

- `run-portal.html`
- HTML / Markdown / JSON 报告
- 每个报告的生成状态和诊断完整度
- 资产 guidance、收集提示、下一步所需信息

AI artifacts 只能作为可选建议，不能替代 base artifacts。

## 推荐先读

- `references/workflow.md`
- `references/dependencies.md`
- `references/artifact-contract.md`
- `references/script-index.md`
- `references/diagnosis-matrix.md`
- `references/update-view-registry-guide.md`

## 诊断原则

- 资产存在时，说明它能提供什么诊断价值。
- 资产缺失时，说明它会阻塞或降级哪些报告，以及下次应保留什么。
- 证据不足时，明确列出下一步需要的信息，不只返回泛化失败。
- `references/` 和生成的 JSON artifacts 是可迁移事实源；目标仓额外设计说明只能作为可选上下文。

## 视图要求

- `run-portal.html` 是默认入口，应展示当前证据、诊断完整度、推荐下一步、报告入口和对象入口。
- `scope-graph.html` 是一等交互图视图，必须保留图探索能力，支持 `scopeId` / `nodeId` 等对象上下文。
- `task-queue-graph.html` 是一等 queue 视图，应支持 `taskIndex` / `nodeId` 上下文和稳定 task anchor。
- `hang-crash-report.html` / `performance-report.html` 是 Markdown 摘要的浏览器友好版本，需提供稳定 section anchor。
- performance 报告中，event/prof 数据优先；task/scope 是二级 drill-down。
- 页面视觉和信息架构设计说明属于维护侧设计材料，不随 skill 发布包作为运行事实源。

## AOT / task queue 兼容

- 过渡期内，log 解析仍是 canonical 兼容路径。
- task queue JSON 用于可安全对齐时的 shadow validation。
- 支持 `scopes[].taskQueues` 按 `scopeId` / `skId` 对齐。
- root-level `taskQueues` 只在能唯一匹配 section 时参与校验。
- JSON-only 字段若 log 侧没有可比值，不报告 mismatch。
- sync/event/custom task 在 graph identity 无效时，不参与 graph-bound duplicate identity 检查。

## 性能和进度要求

- 完整性能分析显示 `Stage 0/4` 到 `Stage 4/4`。
- event/prof 解析是可见的 `Stage 2/4`。
- TTY 使用 `tqdm`；非 TTY 使用紧凑 start/done 行。
- 抑制 worker parser/render 噪声。
- `--profile` 开启时写入 `reports/data/diagnose-profile.json`。

## 修改后的验证

- 仅文档修改：至少验证 help 和 stale-reference 搜索。
- parser/report 修改：运行相关 `sk_model_analysis` 单元测试。
- 改动真实样例路径时，优先使用 `regression_runner.py` 做临时工作区回归。
