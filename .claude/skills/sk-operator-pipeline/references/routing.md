# 路由规则

## 主入口职责

- 承接 SK 算子适配主流水线
- 调度资产接入、源码适配、校验、样例生成、构建打包等子 skill
- 在用户只需要定位能力时，把任务路由到其它 built-in skill
- 在需要本地资料导航时，生成能力索引

## 路由规则

- 单算子源码到 SK 源码生成、scaffold 适配
  - 去 `sk-operator-codegen`
- SK 算子编码规范、风险代码扫描
  - 去 `sk-operator-validate`
- SK 算子版本 / 芯片 / SDK / 驱动 / ACL 兼容性
  - 去 `sk-operator-validate`
- SK 算子运行样例、运行时输出比对、scoped correctness verdict
  - 去 `sk-operator-sample-gen`
- SK 算子编译验证、pybind 打包、wheel 构建
  - 去 `sk-operator-build-package`
- 融合分析、scope、task queue、性能定位
  - 去 `sk-network-analysis`
- 仓内能力导航、依赖判断、流水线入口定位
  - 当前 skill 自身处理

## 当前缺口

- 还没有 issue 索引
- 还没有方案回顾索引
- 还没有统一的问题模板
- route / index 仍是辅助能力，不替代 `run-sk-pipeline`
