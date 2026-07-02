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
