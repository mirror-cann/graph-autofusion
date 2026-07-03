# 依赖边界

## Skill 包边界

- `sk-operator-pipeline` 通过 `skills_root` 定位 sibling skills。
- 用户资产、输出目录和构建缓存通过显式 CLI 参数加 `repo_root` 解释。
- `index` 默认只扫描 `skills_root`；如果要索引用户工作区，必须显式传 `--index-root`。

## Skill 链接

- `sk-operator-asset-adapter`
- `sk-operator-codegen`
- `sk-operator-validate`
- `sk-operator-sample-gen`
- `sk-operator-build-package`
- `sk-network-analysis`

## 算子构建依赖

- CANN 标准依赖由 pipeline 根据 `--target-cann`、`ASCEND_HOME_PATH` 或 `ASCEND_TOOLKIT_HOME` 推导。
- 用户私有依赖必须通过 `operator-build-config.json` 或 CLI 显式声明。
- 调试覆盖统一使用 `--operator-build-config-set FIELD=JSON_VALUE`，不为每个字段增加独立 CLI 参数。
- 相对路径在 `operator-build-config.json` 中按 JSON 所在目录解析；只使用 CLI 覆盖时按当前执行目录解析。
- 绝对路径按用户声明直接使用；路径不存在时报错，repo/CANN 外部路径只在 resolved 记录中标记为 `external-explicit`。
- 核心脚本不全仓搜索 include/lib，不根据样例目录名硬编码依赖路径。
