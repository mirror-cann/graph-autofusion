# SK 算子资产适配

这个 skill 是 SK 算子资产的可定制接入层，负责把用户自己的源码树转换成核心 SK skills 可消费的稳定 contract。

当用户仓库布局和内置默认规则不一致时，应优先定制这一层的 recipe。核心 skills 不应该因为某个仓库的目录差异而被修改。

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
