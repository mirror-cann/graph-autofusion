# 产物契约

## 输入合同

统一入口支持三类输入：

- result root
  - 包含 `sk_meta/`、`log/`、`kernel_meta/` 等运行产物的顶层目录
- model asset root
  - 例如 `sk_meta/<pid>/`
- model directory
  - 例如 `sk_meta/<pid>/model_<model_ri>/`

脚本会从输入路径向下或向上推断实际 `model_*` 目录和 result root。

## 资产到能力映射

| 资产 | 主要能力 | 缺失时的降级 |
| --- | --- | --- |
| `super_kernel.log` | AOT 阶段、update、DFX 高层异常 | DFX 和阶段诊断降级 |
| `sk_scope_split.log` | scope / stream / node 结构 | Scope View 降级 |
| `sk_node_detail.log` | node tracing、node 级映射 | tracing 不生成或降级 |
| `sk_fused_nodes.log` | fused function 与节点成员 | 融合解释降级 |
| `sk_device_args.log` | task queue、entry、args、DFX payload | TaskQue View 与 queue 关联降级 |
| `sk_task_queue.json` | task queue 结构化旁路校验 | 不替代 `sk_device_args.log`，仅缺少一致性诊断 |
| `sk_event_dev_device_*.json` / `sk_prof_device_*.json` | time event 与性能矩阵 | 性能诊断缺少时间证据 |

## 当前兼容口径

- `sk_scope_split.log`
  - 支持旧格式 `Scope <n>: ...`
  - 支持新版显示序号与真实 id 分离的 `Scope <display> (scopeId=<id>): ...`
  - 节点段和 stream 段仍按显示序号输出时，解析器会映射回真实 `scopeId`
- `sk_device_args.log`
  - `SkHeaderInfo` 中 `wsOff` 可缺失；缺失时导出为 `null`
  - task 行可携带 `debugOptions`
  - task 行可携带 `relatedType` 与 `extraInfo`
  - DFX 行可携带 `numBlocks`、`cubeNum`、`vecNum`
  - DFX entry 可按单行多个 `entryAic[]` / `entryAiv[]` 输出
- `sk_task_queue.json`
  - 兼容期只用于校验 `sk_device_args.log` 解析结果
  - 支持 `scopes[].taskQueues`，按 `scopeId` / `skId` 对齐，不依赖 JSON `scopes` 数组顺序
  - 支持真实 AOT 输出中的 root-level `taskQueues`，在没有 `scopeId` 时仅对唯一候选 section 做旁路校验
  - 校验共同字段：task 数、`idx/nodeIndex`、`type`、`blk/numBlocks`、`entries/entryCnt`、`args`、`debugOptions`、`entry[]`
  - 不校验 JSON 缺失字段：`nodeId`、`relatedType`、`extraInfo`、header offset、DFX payload
- event JSON
  - 支持旧名 `sk_event_dev_device_*.json`
  - 支持当前 AOT profiling 输出 `sk_prof_device_*.json`

## 输出合同

统一入口默认在 result root 下生成：

```text
reports/
├── run-portal.html
├── data/
│   ├── scope-library.json
│   ├── graph-library.json
│   ├── dfx-library.json
│   ├── node-trace.json
│   └── node-trace_meta.json
└── views/
    ├── scope-graph.html
    ├── task-queue-graph.html
    ├── hang-crash-report.html
    └── performance-report.html
```

multi-model 或 multi-model-instance 运行可能在 `reports/<ri>/<model-instance>/` 下创建 per-model report bundle。

## 诊断完整度口径

报告生成成功不等于诊断完整。

页面和 JSON 应区分：

- report generation status
  - 是否成功产出页面或结构化文件
- diagnostic completeness
  - 当前证据是否足够支持诊断结论
- capability mode
  - graph-capable、summary-only、insufficient 等能力状态

缺失资产时，报告必须解释：

- 缺什么
- 为什么重要
- 影响哪些视图或结论
- 下次应该保留什么

## 命令合同

首选命令：

```bash
python3 <skills_root>/sk-model-analysis/scripts/model_analysis.py analyze <input>
python3 <skills_root>/sk-model-analysis/scripts/model_analysis.py diagnose-hang-crash <input>
python3 <skills_root>/sk-model-analysis/scripts/model_analysis.py diagnose-performance <input>
python3 <skills_root>/sk-model-analysis/scripts/model_analysis.py trace-nodes <input>
```

底层脚本只在需要调试单一阶段时直接使用。

## 验证合同

发布包本身不携带维护侧测试目录。维护脚本或文档时，至少确认 CLI help 可用：

```bash
python3 <skills_root>/sk-model-analysis/scripts/model_analysis.py --help
python3 <skills_root>/sk-model-analysis/scripts/model_analysis.py analyze --help
```
