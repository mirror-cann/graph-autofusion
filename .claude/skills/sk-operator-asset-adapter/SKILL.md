---
name: sk-operator-asset-adapter
description: 将用户提供的算子仓、源码树或构建资产转换为 SK 核心流水线需要的稳定 JSON contract；适合处理用户侧目录结构、命名规则、构建命令和验证环境差异。
---

# SK 算子资产适配

当用户提供的算子资产不能直接被 SK 核心能力消费时，使用这个 skill。它的定位是“可定制适配层”：用户可以复制或修改这里的适配范式，去匹配自己的仓库布局、命名约定、构建命令和验证方式。

入口：

```bash
python3 <skills_root>/sk-operator-asset-adapter/scripts/operator_asset_adapter.py <subcommand> ...
```

## 命令

- `adapt-asset <asset> --output-dir DIR [--target-chip CHIP] [--target-arch ARCH]`
  扫描资产，应用默认适配规则，并输出：
  - `operator-asset-inventory.json`
  - `operator-asset-layout.json`
  - `operator-build-context.json`
  - `operator-verify-context.json`
  - `adapter-report.json`
- `validate-contracts --layout LAYOUT --build-context BUILD --verify-context VERIFY`
  只校验适配产物，不执行 SK 核心流水线。

## 边界

这个 skill 负责解释用户资产：目录遍历、host/kernel 匹配、默认 include 发现、构建/验证上下文整理。

它不负责：

- 生成 `SK_BIND`
- 内部编译打包实现
- 正确性验证 verdict 的最终语义

核心 skill 只消费这里输出的稳定 JSON contract。如果适配层无法产出足够信息，应显式失败或要求用户补充，而不是猜测。

## 契约规则

适配层可以使用仓库约定、用户输入或项目定制规则生成 contract，但必须把不确定性暴露出来：

- 运行时参数角色必须显式声明为 `input`、`output`、`workspace`、`tiling`、`scalar` 或 `descriptor`。
- 可比较输出和 comparator 策略必须显式声明。若资产没有声明，不要根据 `out`、`y`、`workspace` 这类名字猜测，应标记为需要用户确认。
- graph capture 前需要准备的状态必须显式声明，例如 TensorList descriptor、持久 workspace 状态、静态 cache 初始化。
- 整网验证 contract 需要包含 package、runner adapter、nodes、edges、inputs、outputs、prepare 步骤和期望融合 scope。拓扑或输出语义未知时，应输出 partial contract 并要求用户或 adapter 补充。
- 用户仓特有规则只能留在 adapter 层。核心 skill 只依赖稳定 JSON contract，不学习某个仓的目录或命名习惯。
