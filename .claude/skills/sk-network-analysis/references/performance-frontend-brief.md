# Performance Frontend Brief

## 视觉目标

把 `performance-report.html` 做成“事件主视角主页面”，不是长矩阵工具页。

## 视觉原则

- 首屏先看焦点和事件主矩阵
- 热点事件退到次级卡片
- 表格和折叠区都保持简洁，不堆长说明

## 组件约束

- `当前焦点` 用主卡片
- `事件主视角矩阵` 作为主块展示
- `热点事件` 用次级卡片
- `更多矩阵行` 继续用折叠区

## 风格约束

- 与 `run-portal`、`scope-graph`、`task-queue-graph` 保持一致的卡片和摘要语言
- 页面说明必须是短句中文
- 不新增独立 `performance-graph.html`
