# SK Operator Asset Adapter

This skill is the customizable intake layer for SK operator assets. It converts
user-owned source trees into stable contracts for the core SK skills.

Users may customize recipes in this layer when their repository layout differs
from the bundled defaults. Core skills should not be modified for repository
layout differences.

## 需要沉淀的资产语义

adapter 的目标不是让核心脚本猜测用户仓库，而是把用户资产里的语义转成稳定
JSON contract：

- 算子参数角色：`input`、`output`、`workspace`、`tiling`、`scalar`、`descriptor`。
- 可比较输出：输出名、比较方式、容差。
- capture 前准备步骤：descriptor 准备、workspace tail、静态缓存初始化等。
- 单算子运行 fixture：同输入下非 SK 入口和 SK 入口如何运行、哪些输出可比较。
- 整网验证 contract：package/wheel、runner adapter、nodes、edges、inputs、outputs、
  prepare 步骤和预期融合范围。

如果这些信息无法从用户仓库可靠获得，adapter 应产出 `needs-user-confirmation` 或
部分 contract，让用户补充；不要把变量名、目录名或参数顺序的猜测写成确定事实。
