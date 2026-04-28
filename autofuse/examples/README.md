# autofuse 样例使用指导

## 功能描述

使用 autofuse 完成各种类型的算子融合

## 目录结构
```
├── af_pointwise                  # pointwise 类型算子融合的样例
│  └── af_eq_where.py             # 通过 autofuse 完成 equal 和 where 两个 pointwise 类型算子的融合
```
## 前置说明
请务必参考[《Autofuse 简介与快速上手》](../README.md)完成前置环境准备。

## 用例演示

[用例1](af_pointwise/README.md)

## 参考

请参考[inductor-npu-ext使用手册](https://gitcode.com/Ascend/torchair/blob/master/experimental/_inductor_npu_ext/docs/manuals.md)的相关内容。