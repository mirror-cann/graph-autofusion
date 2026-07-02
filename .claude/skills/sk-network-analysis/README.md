# SK 网络分析

`sk-network-analysis` 是面向 `SuperKernel` 结果目录的诊断与可视化工具集。

它有两种用法：

- 作为仓内 skill，被上层 agent 调用
- 直接当脚本工具使用

对同事来说，第二种更重要：**不需要 AI，也可以直接用脚本产出 HTML / JSON / tracing 结果。**

## 解决什么问题

这个工具主要覆盖 4 类场景：

- `Scope View`
  看 scope、stream、node 结构，以及 update 后的 scope 图
- `TaskQue View`
  看 task queue、entry、event、sync、memory wait/write 关系
- `Dfx View`
  看 hang / coredump / exception / `op_trace` / counter / PC 定位
- `Analysis View`
  看性能相关事件和结构化分析

## 当前定位

当前最稳定的定位是：**无 AI 依赖的 SK 结果目录分析工具**。

也就是说：

- `base mode` 必须始终可用
- HTML / JSON / tracing 是基础产物
- `--with-ai` 只能作为可选增强，不是报告生成前提
- 报告能生成不代表诊断完整，页面会按资产充足度降级

## 输入目录

下面几种输入都支持：

- `result root`
- `sk_meta` 目录
- 单个 `model_*` 目录

脚本会自动向下解析到实际的 `model_*` 目录。

常见真实结构也会自动识别：

- 单模型：`<result_root>/runs/sk_meta/<ri>/model_*`
- 多模型/多卡：`<result_root>/runs/sk_meta/<process_or_rank>/model_*`
- 单个模型目录直接输入：`.../model_48_1`

如果输入目录是只读的，报告写入会失败。做真实样例回归时建议先复制到可写目录，
例如 `build/examples/sk-na-sample`，再对副本运行分析。

## 最常用入口

统一入口脚本：

```bash
python <skills_root>/sk-network-analysis/scripts/network_analysis.py <subcommand> <input>
```

支持的子命令：

- `analyze`
- `diagnose-hang-crash`
- `diagnose-performance`
- `trace-nodes`

查看帮助：

```bash
python <skills_root>/sk-network-analysis/scripts/network_analysis.py --help
python <skills_root>/sk-network-analysis/scripts/network_analysis.py analyze --help
python <skills_root>/sk-network-analysis/scripts/network_analysis.py diagnose-hang-crash --help
python <skills_root>/sk-network-analysis/scripts/network_analysis.py diagnose-performance --help
python <skills_root>/sk-network-analysis/scripts/network_analysis.py trace-nodes --help
```

## 推荐用法

### 1. 直接生成完整报告

```bash
python <skills_root>/sk-network-analysis/scripts/network_analysis.py analyze <result_root_or_model_dir>
```

适合第一次看一个样例。它会统一生成报告入口和子页面。

常用参数：

```bash
# 输出阶段耗时 profile，便于定位卡在哪个阶段
python <skills_root>/sk-network-analysis/scripts/network_analysis.py analyze --profile <input>

# 指定并行 worker 数。默认会按 model 数和 CPU 数自动扩展
python <skills_root>/sk-network-analysis/scripts/network_analysis.py analyze --jobs 96 <input>

# 禁用 parser cache，用于确认真实解析路径
python <skills_root>/sk-network-analysis/scripts/network_analysis.py analyze --no-cache <input>

# 禁用多进程，排查并发相关问题时使用
python <skills_root>/sk-network-analysis/scripts/network_analysis.py analyze --no-parallel <input>
```

常见输出目录：

- `reports/run-portal.html`
- `reports/views/scope-graph.html`
- `reports/views/task-queue-graph.html`
- `reports/views/hang-crash-report.html`
- `reports/views/performance-report.html`
- `reports/data/scope-library.json`
- `reports/data/graph-library.json`
- `reports/data/dfx-library.json`
- `reports/data/node-trace.json`
- `reports/data/node-trace_meta.json`

multi-model 或多次 `aclskOptimize` 的结果可能会在 `reports/<ri>/<model-instance>/`
下生成子报告包，顶层 `run-portal.html` 负责汇总入口。

### 1.1 终端阶段和进度

`analyze` / `diagnose-performance` 会把主流程拆成可观察阶段：

- `Stage 0/4: 初始化输入`
- `Stage 1/4: 收集 model 信息库`
- `Stage 2/4: 解析 event/prof 数据`
- `Stage 3/4: 生成 model 视图`
- `Stage 4/4: 汇总运行结果`

如果安装了 `tqdm` 且当前是 TTY，会显示动态进度条；非 TTY 或日志重定向时，
降级为紧凑的 start/done 行，避免刷屏。

`Stage 2/4` 主要处理 `sk_event_dev_device_*.json` / `sk_prof_device_*.json`。
这类文件可能很大；当前实现优先用 `ijson` 流式解析，避免一次性把几个 GB 的 JSON
整体读入内存。

打开 `--profile` 后，详细耗时会写到：

- `reports/data/diagnose-profile.json`

重点看这些 section：

- `collect_model_libraries`
- `build_model_update_report`
- `register_event_stats_provider`
- `collect_process_event_stats`
- `collect_process_event_stats_batch`
- `render_model_views`
- `summarize_run_event_stats`

### 2. 只看 DFX

```bash
python <skills_root>/sk-network-analysis/scripts/network_analysis.py diagnose-hang-crash <result_root_or_model_dir>
```

适合 hang / coredump / exception 排查。

重点页面：

- `reports/views/hang-crash-report.html`

### 3. 只看性能

```bash
python <skills_root>/sk-network-analysis/scripts/network_analysis.py diagnose-performance <result_root_or_model_dir>
```

重点页面：

- `reports/views/performance-report.html`

### 4. 只导出 node tracing

```bash
python <skills_root>/sk-network-analysis/scripts/network_analysis.py trace-nodes <result_root_or_model_dir>
```

适合导入 `edge://tracing` 或 `chrome://tracing`。

## 纯脚本模式

如果你不想走统一入口，也可以直接调用底层脚本。

### 提取基础信息库

```bash
python <skills_root>/sk-network-analysis/scripts/sk_library_extractor.py <result_root_or_model_dir>
```

默认会生成：

- `scope-library.json`
- `graph-library.json`
- `dfx-library.json`

也可以显式指定输出目录：

```bash
python <skills_root>/sk-network-analysis/scripts/sk_library_extractor.py <input> -o <output_dir>
```

### 只生成 Scope View

```bash
python <skills_root>/sk-network-analysis/scripts/sk_scope_visualizer.py <result_root_or_model_dir> -o scope-graph.html
```

或者直接指定库文件：

```bash
python <skills_root>/sk-network-analysis/scripts/sk_scope_visualizer.py \
  --scope-library <scope-library.json> \
  --graph-library <graph-library.json> \
  -o scope-graph.html
```

### 只生成 TaskQue View

```bash
python <skills_root>/sk-network-analysis/scripts/sk_task_queue_visualizer.py <result_root_or_model_dir> -o task-queue-graph.html
```

或者直接指定库文件：

```bash
python <skills_root>/sk-network-analysis/scripts/sk_task_queue_visualizer.py \
  --scope-library <scope-library.json> \
  --graph-library <graph-library.json> \
  -o task-queue-graph.html
```

## 可选 AI 说明

统一入口保留了 `--with-ai` 参数：

```bash
python <skills_root>/sk-network-analysis/scripts/network_analysis.py analyze <input> --with-ai
```

但**当前这套工具对外推荐的仍然是 base mode**，也就是：

- 不依赖 AI
- 直接生成 HTML / JSON / tracing 产物
- 直接给脚本使用者和人工分析使用

目前 README 里提到的能力，默认都按 **无 AI、纯脚本** 方式保证可用。

换句话说：

- 不加 `--with-ai` 可以正常工作
- 当前也**不要把这套工具理解成“已经使能了稳定的 AI 自动分析”**
- 对同事推广时，建议直接按脚本工具来使用，不把 AI 作为前提

基础产物始终应该可直接使用，包括：

- HTML 报告
- JSON 信息库
- tracing 产物

## DFX 当前依赖什么

`Dfx View` 当前主链只依赖 `sk_meta` 相关资产，重点包括：

- `super_kernel.log`
- `sk_scope_split.log`
- `sk_node_detail.log`
- `sk_fused_nodes.log`
- `sk_device_args.log`

其中：

- `dfx-library.json` 是 DFX 页面主底座
- `scope-library.json` / `graph-library.json` 是结构视图底座

更完整的输入、输出和降级合同见：

- [references/artifact-contract.md](./references/artifact-contract.md)
- [references/diagnosis-matrix.md](./references/diagnosis-matrix.md)

## AOT / Task Queue 兼容说明

当前解析器仍以现有 log 为主，不会立刻切换成只读新 JSON 格式。
新的 AOT task queue JSON 会作为 shadow validation 输入参与校验：

- 支持 `scopes[].taskQueues`，按 `scopeId` / `skId` 对齐到已有 section。
- 支持真实 AOT 输出里的 root-level `taskQueues`，没有 `scopeId` 时只对唯一候选 section 做旁路校验。
- JSON 字段只在 log 侧有对应可比较字段时参与校验；log 缺字段时不会因为 JSON 多字段直接报 mismatch。
- event / sync / custom task 这类不能稳定绑定 graph node 的任务，不参与 graph-bound duplicate identity 校验。

这段兼容期的目标是：

1. 保持老 log 解析结果稳定。
2. 用新 JSON 捕获明显解析偏差。
3. 给新 AOT 输出格式留出过渡时间。

相关合同见：

- [references/artifact-contract.md](./references/artifact-contract.md)
- [scripts/sk_library_extractor.py](./scripts/sk_library_extractor.py)

## 性能与大文件策略

当前主流程已经覆盖这些性能路径：

- 多 model 目录并行收集。
- 单 model 内多次 `aclskOptimize` / 多 model instance 并行解析。
- parser cache：同一 model 源日志未变化时复用已生成的信息库。
- event/prof JSON 懒加载：只有 performance 相关报告需要时才解析。
- event/prof JSON batch 并行：多 process 时按 process 并行统计。
- `ijson` 流式事件解析：避免大 JSON 一次性加载。
- worker 内部静默 `Generated HTML` 和 parser detail 输出，终端只保留阶段进度和最终摘要。

如果发现慢：

1. 先加 `--profile` 看 `diagnose-profile.json`。
2. 看是否卡在 `collect_model_libraries`、`collect_process_event_stats_batch`、还是 `render_model_views`。
3. 多模型场景优先检查 `--jobs` 和机器 CPU 数；默认会自动扩展，但也可以手动指定。
4. 大 JSON 场景优先确认环境里安装了 `ijson`。

## 依赖安装

仓库根目录提供了 `requirements.txt`。最小运行依赖包括：

- `ijson`：大 event/prof JSON 流式解析。
- `tqdm`：TTY 下显示进度条。

安装：

```bash
python -m pip install -r requirements.txt
```

如果没有 `tqdm`，分析仍可运行，只会使用紧凑文本进度。
如果没有 `ijson`，会回退到标准 `json` 解析，大 JSON 场景可能明显更慢、占用更多内存。

## 维护入口

如果你在维护这个 skill，优先按下面顺序看：

1. [SKILL.md](./SKILL.md)
2. [references/workflow.md](./references/workflow.md)
3. [references/artifact-contract.md](./references/artifact-contract.md)
4. [references/dependencies.md](./references/dependencies.md)
5. [references/script-index.md](./references/script-index.md)

发布后的 skill 不要求目标仓提供额外设计文档。优先使用本 skill 的
`references/` 和运行产物作为事实源；目标仓如果提供额外设计说明，只能作为可选上下文。

## 结果怎么看

推荐从下面顺序进入：

1. `reports/run-portal.html`
2. `reports/views/scope-graph.html`
3. `reports/views/task-queue-graph.html`
4. `reports/views/hang-crash-report.html`
5. `reports/views/performance-report.html`

如果只关心异常定位：

1. 先看 `hang-crash-report.html`
2. 再跳 `scope-graph.html`
3. 再跳 `task-queue-graph.html`

## 固定 ST 样例和回归

这里更推荐关注 **固定 ST 样例** 和 **结果回归**，而不是把 README 写成开发态的 unit/system 测试说明。

### 测试资产放在哪里

发布包不携带维护侧测试目录或固定样例。真实回归样例由维护流程或目标仓自行提供；
如果需要回归某个结果目录，优先把样例复制到可写临时目录后运行。

生成产物不进入发布包：

- sample runs
- generated `sk_meta`
- generated `reports`
- manually rendered HTML snapshots
- `__pycache__/` / `.pytest_cache/`

真实样例不作为发布包的前置 fixture，发布包只假设用户提供一个可读写的结果目录。

### 跑 skill 回归

```bash
python <skills_root>/sk-network-analysis/scripts/regression_runner.py <sample_root>
```

它会在临时目录里跑，不污染原始样例。

### 跑真实样例

真实样例可能会在输入目录下写 `reports/`。如果源目录只读或不希望污染样例，
先复制到临时目录：

```bash
tmp_parent="${TMPDIR:-build/tmp}"
mkdir -p "$tmp_parent"
tmp_dir="$(mktemp -d "$tmp_parent/sk-na-example.XXXXXX")"
cp -a <sample_root>/. "$tmp_dir/"
python <skills_root>/sk-network-analysis/scripts/network_analysis.py analyze --profile --no-cache "$tmp_dir"
```

真实回归样例由维护流程或目标仓自行提供，发布包不假设固定样例路径。

预期结果：

- 命令退出码为 0。
- 生成 `reports/run-portal.html`。
- 生成 `reports/data/diagnose-profile.json`。
- 终端只保留 Stage 进度和最终摘要，不应再被 `[PARSE]` 或 `Generated HTML:` 刷屏。

## 常见问题

### 1. 为什么没有图或报告没生成

先看输入目录里是否真的有对应资产，例如：

- `Scope View` 需要结构类日志
- `TaskQue View` 需要 `sk_device_args.log`
- `Dfx View` 需要 `super_kernel.log`
- `Analysis View` 依赖事件类资产

### 2. `op_trace=false` 怎么办

如果 `Dfx View` 明确提示 `op_trace=false`，说明当前不能使用 counter 侧子核诊断。

需要重新复现并打开：

```bash
export ASCEND_SK_OP_TRACE_ON=1
```

### 3. 想直接看 JSON 信息库

直接看：

- `reports/data/scope-library.json`
- `reports/data/graph-library.json`
- `reports/data/dfx-library.json`

multi-model 时，每个 model / model instance 会在自己的报告包下生成独立 `data/`：

- `reports/<process>/<model-or-instance>/data/scope-library.json`
- `reports/<process>/<model-or-instance>/data/graph-library.json`
- `reports/<process>/<model-or-instance>/data/dfx-library.json`

## 代码索引

核心脚本：

- [scripts/network_analysis.py](./scripts/network_analysis.py)
- [scripts/sk_library_extractor.py](./scripts/sk_library_extractor.py)
- [scripts/sk_scope_visualizer.py](./scripts/sk_scope_visualizer.py)
- [scripts/sk_task_queue_visualizer.py](./scripts/sk_task_queue_visualizer.py)
- [scripts/diagnose_run.py](./scripts/diagnose_run.py)
- [scripts/regression_runner.py](./scripts/regression_runner.py)

补充参考：

- [SKILL.md](./SKILL.md)
- [references/workflow.md](./references/workflow.md)
- [references/dependencies.md](./references/dependencies.md)
- [references/artifact-contract.md](./references/artifact-contract.md)
- [references/script-index.md](./references/script-index.md)
- [references/diagnosis-matrix.md](./references/diagnosis-matrix.md)
