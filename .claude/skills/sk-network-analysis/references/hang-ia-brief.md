# Hang / Crash IA Brief

## 页面

`hang-crash-report.html`

## 页面目标

把当前 `hang / crash` 页面收成客户主页面，先回答：

- 当前最可能卡在哪个阶段
- 当前最值得先看的主信号是什么
- 下一步先看哪块，不要先看 warning 海

## 首要用户任务

用户应在 15 秒内知道：

- 当前故障优先落在哪个阶段
- 当前最高优先级信号是什么
- 是不是该继续看 Queue / DFX

## 默认信息层级

1. `当前判断`
2. `主信号`
3. `阶段关联信号`
4. `Queue / DFX 摘要`
5. `更多明细`

## 大数据默认策略

- 首屏只展示最高优先级信号
- 次级信号和噪声默认折叠
- Queue / DFX 只保留摘要，不和主信号同权

## 不允许改动

- `signal_confidence` 语义
- `likely_failure_stage` 语义
- 现有 markdown companion 产物
- query / 锚点 / 文件名
