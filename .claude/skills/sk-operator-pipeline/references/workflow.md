# workflow

## 统一运行合同

- 顶层入口固定为：
  - `python3 <skills_root>/sk-operator-pipeline/scripts/operator_pipeline.py <subcommand> ...`
- 主流程优先使用 `run-sk-pipeline`
  - 串起资产接入、源码适配、校验、样例生成、构建打包
  - 按 stage 落盘输入、输出、状态和交付物索引
- `route` / `index` 默认跑 `base mode`
  - 不依赖联网
  - 不依赖大模型
  - 负责本地能力索引和基础路由
- `route` / `index` 的 `ai mode` 通过 `--with-ai` 触发
  - 只做增强提示
  - 不替代基础索引和路由结果

## 当前第一版范围

- `run-sk-pipeline`
  - 编排 `sk-operator-*` 子能力完成算子 SK 适配闭环
- `index`
  - 默认只扫描 `skills_root`
  - 如需扫描用户仓内容，显式传 `--index-root`
- `route`
  - 根据问题和关键词路由到对应 built-in skill

## 当前不做

- 不把这个 skill 设计成只能依赖大模型的统一入口
- 不在第一版做复杂 issue 平台集成
