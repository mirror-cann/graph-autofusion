/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __EXPRESSION_CONVERT_STRUCT_H__
#define __EXPRESSION_CONVERT_STRUCT_H__

#include "expression_struct.h"

namespace codegen {
std::string TilerSize(const Tiler &tiler, const ge::Expression &expr);
std::string TilerActualSize(const Tiler &tiler, const ge::Expression &expr);

// ============================================================================
// 辅助工厂函数 - 快速创建常用 ExpressionItem
// ============================================================================
namespace ExprItemFactory {

// 创建 Size 转换项
inline ExpressionItem Size(const ge::Expression &expr) {
  return ExpressionItem(expr, ExpressionConvertMethod::kSize);
}

// 创建 ActualSize 转换项
inline ExpressionItem ActualSize(const ge::Expression &expr) {
  return ExpressionItem(expr, ExpressionConvertMethod::kActualSize);
}

// 创建带乘数的 Size 转换项（stride转字节）
inline ExpressionItem SizeWithMultiplier(const ge::Expression &expr, int64_t dtype_size) {
  return ExpressionItem(expr, ExpressionConvertMethod::kSizeWithMultiplier, dtype_size);
}

// 创建带乘数的 ActualSize 转换项
inline ExpressionItem ActualSizeWithMultiplier(const ge::Expression &expr, int64_t dtype_size) {
  return ExpressionItem(expr, ExpressionConvertMethod::kActualSizeWithMultiplier, dtype_size);
}

// 创建带 cast 的 Size 转换项
inline ExpressionItem SizeWithCast(const ge::Expression &expr, const std::string &cast) {
  return ExpressionItem(expr, ExpressionConvertMethod::kSize, 0, cast);
}

// 创建带 cast 和乘数的 Size 转换项
inline ExpressionItem SizeWithCastAndMultiplier(const ge::Expression &expr, int64_t dtype_size,
                                                const std::string &cast) {
  return ExpressionItem(expr, ExpressionConvertMethod::kSizeWithMultiplier, dtype_size, cast);
}

// 创建带 cast 的 ActualSize 转换项
inline ExpressionItem ActualSizeWithCast(const ge::Expression &expr, const std::string &cast) {
  return ExpressionItem(expr, ExpressionConvertMethod::kActualSize, 0, cast);
}

// 创建带 cast 和乘数的 ActualSize 转换项
inline ExpressionItem ActualSizeWithCastAndMultiplier(const ge::Expression &expr, int64_t dtype_size,
                                                      const std::string &cast) {
  return ExpressionItem(expr, ExpressionConvertMethod::kActualSizeWithMultiplier, dtype_size, cast);
}

// 创建不转换的项（直接输出常量或符号名）
inline ExpressionItem Direct(const ge::Expression &expr) {
  return ExpressionItem(expr, ExpressionConvertMethod::kNone);
}

// 创建常量项
inline ExpressionItem Constant(int64_t value) {
  return ExpressionItem(ge::Symbol(value), ExpressionConvertMethod::kNone);
}

// 创建常量项（带 cast）
inline ExpressionItem ConstantWithCast(int64_t value, const std::string &cast) {
  return ExpressionItem(ge::Symbol(value), ExpressionConvertMethod::kNone, 0, cast);
}

// 创建符号变量项（如 "curAivM"）
inline ExpressionItem SymbolVar(const std::string &name) {
  return ExpressionItem(ge::Symbol(name.c_str()), ExpressionConvertMethod::kNone);
}

// 创建符号变量项（带 cast）
inline ExpressionItem SymbolVarWithCast(const std::string &name, const std::string &cast) {
  return ExpressionItem(ge::Symbol(name.c_str()), ExpressionConvertMethod::kNone, 0, cast);
}

// 创建循环变量乘以 Size 的项：输出为 "outer_for_${loop_idx} * Size(expr)"
inline ExpressionItem LoopVarTimesSize(const ge::Expression &expr, int64_t loop_idx) {
  ExpressionItem item(expr, ExpressionConvertMethod::kLoopVarTimesSize);
  item.loop_idx = loop_idx;
  return item;
}

// 创建循环变量乘以 ActualSize 的项：输出为 "outer_for_${loop_idx} * ActualSize(expr)"
inline ExpressionItem LoopVarTimesActualSize(const ge::Expression &expr, int64_t loop_idx) {
  ExpressionItem item(expr, ExpressionConvertMethod::kLoopVarTimesActualSize);
  item.loop_idx = loop_idx;
  return item;
}

}  // namespace ExprItemFactory

// ============================================================================
// CombinedExpression 辅助工厂函数
// ============================================================================
namespace CombinedExprFactory {

// 创建 Size - ActualSize 的组合（api_call_utils.cpp 187-188 行场景）
inline CombinedExpression SizeMinusActualSize(const ge::Expression &size_expr, const ge::Expression &actual_size_expr) {
  return CombinedExpression(ExprItemFactory::Size(size_expr), ExprItemFactory::ActualSize(actual_size_expr), "-");
}

// 创建 ActualSize(expr1 - expr2) 的组合（api_call_utils.cpp 185-186 行场景）
inline CombinedExpression ActualSizeOfDiff(const ge::Expression &expr1, const ge::Expression &expr2) {
  ge::Expression diff_expr = expr1 - expr2;
  return CombinedExpression(ExprItemFactory::ActualSize(diff_expr));
}

// 创建 Size(expr) + Size(expr) 的组合
inline CombinedExpression SizePlusSize(const ge::Expression &expr1, const ge::Expression &expr2) {
  return CombinedExpression(ExprItemFactory::Size(expr1), ExprItemFactory::Size(expr2), "+");
}

// 创建 ActualSize(expr) + ActualSize(expr) 的组合
inline CombinedExpression ActualSizePlusActualSize(const ge::Expression &expr1, const ge::Expression &expr2) {
  return CombinedExpression(ExprItemFactory::ActualSize(expr1), ExprItemFactory::ActualSize(expr2), "+");
}

// 创建常量表达式
inline CombinedExpression Constant(int64_t value) {
  return CombinedExpression(ExprItemFactory::Constant(value));
}

// 创建符号变量表达式
inline CombinedExpression SymbolVar(const std::string &name) {
  return CombinedExpression(ExprItemFactory::SymbolVar(name));
}

// 创建带 cast 的 ActualSize 表达式（用于 loop_sizes）
inline CombinedExpression ActualSizeWithCast(const ge::Expression &expr,
                                             const std::string &cast = "static_cast<uint32_t>") {
  return CombinedExpression(ExprItemFactory::ActualSizeWithCast(expr, cast));
}

// 创建带 cast 和乘数的 Size 表达式（用于 loop_strides）
inline CombinedExpression SizeWithCastAndMultiplier(const ge::Expression &expr, int64_t dtype_size,
                                                    const std::string &cast = "static_cast<uint64_t>") {
  return CombinedExpression(ExprItemFactory::SizeWithCastAndMultiplier(expr, dtype_size, cast));
}
}  // namespace CombinedExprFactory

// ============================================================================
// 类型别名，简化使用
// ============================================================================
using ExprItem = ExpressionItem;
using CombinedExpr = CombinedExpression;

}  // namespace codegen

#endif  // __EXPRESSION_CONVERT_STRUCT_H__
