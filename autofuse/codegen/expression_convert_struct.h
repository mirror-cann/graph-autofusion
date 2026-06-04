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

#include <string>
#include <vector>
#include <sstream>
#include "graph/symbolizer/symbolic.h"

namespace codegen {
class Tiler;
std::string TilerSize(const Tiler& tiler, const ge::Expression& expr);
std::string TilerActualSize(const Tiler& tiler, const ge::Expression& expr);
}

namespace codegen {

// ============================================================================
// 表达式转换方式枚举
// ============================================================================
enum class ExpressionConvertMethod : uint8_t {
  kNone = 0,                     // 不转换，直接用 Expression.Str() 输出（用于常量/符号名）
  kSize = 1,                     // 使用 tiler.Size() 转换
  kActualSize = 2,               // 使用 tiler.ActualSize() 转换
  kSizeWithMultiplier = 3,       // 使用 tiler.Size() 并乘以 dtype_size（stride转字节）
  kActualSizeWithMultiplier = 4, // 使用 tiler.ActualSize() 并乘以 dtype_size
  kLoopVarTimesSize = 5,         // 循环变量乘以 Size(expr): "outer_for_${loop_idx} * Size(expr)"
  kLoopVarTimesActualSize = 6,   // 循环变量乘以 ActualSize(expr): "outer_for_${loop_idx} * ActualSize(expr)"
};

// ============================================================================
// 单个表达式项及其转换方式
// ============================================================================
struct ExpressionItem {
  ge::Expression expr;                    // 表达式本身
  ExpressionConvertMethod method;         // 转换方式

  // 可选参数 - 注意成员初始化顺序必须与声明顺序一致
  int64_t dtype_size_multiplier = 0;      // 数据类型大小乘数（stride转字节时使用）
  int64_t loop_idx = 0;                   // 循环变量索引（用于 kLoopVarTimesSize/kLoopVarTimesActualSize）
  std::string cast_prefix = "";           // 类型转换前缀（如 "static_cast<uint64_t>"）

  // 默认构造
  ExpressionItem() : method(ExpressionConvertMethod::kNone) {}

  // 常用构造函数
  ExpressionItem(const ge::Expression& e, ExpressionConvertMethod m)
      : expr(e), method(m) {}

  // 带乘数的构造函数（用于 kSizeWithMultiplier/kActualSizeWithMultiplier）
  ExpressionItem(const ge::Expression& e, ExpressionConvertMethod m, int64_t multiplier)
      : expr(e), method(m), dtype_size_multiplier(multiplier) {}

  // 带乘数和 cast 的构造函数
  ExpressionItem(const ge::Expression& e, ExpressionConvertMethod m,
                 int64_t multiplier, const std::string& cast)
      : expr(e), method(m), dtype_size_multiplier(multiplier), loop_idx(0), cast_prefix(cast) {}

  // 转换为字符串（需要 tiler）
  std::string ToStr(const Tiler& tiler) const;
};

// ============================================================================
// 组合表达式 - 支持加减法组合多个表达式项
// ============================================================================
struct CombinedExpression {
  ge::Expression combined_expr;           // 合并后的完整表达式（用于性能建模）
  std::vector<ExpressionItem> items;      // 分解后的各部分及其转换方式
  std::vector<std::string> operators;     // 连接操作符（"+" 或 "-"）

  // 默认构造（简单表达式）
  CombinedExpression() {}

  // 单项表达式构造
  explicit CombinedExpression(const ExpressionItem& item)
      : combined_expr(item.expr), items{item} {}

  // 两项加减构造
  CombinedExpression(const ExpressionItem& item1, const ExpressionItem& item2,
                     const std::string& op)
      : items{item1, item2}, operators{op} {
    // 构建 combined_expr
    if (op == "+") {
      combined_expr = item1.expr + item2.expr;
    } else if (op == "-") {
      combined_expr = item1.expr - item2.expr;
    }
  }

  // 多项构造
  CombinedExpression(const std::vector<ExpressionItem>& its,
                     const std::vector<std::string>& ops)
      : items(its), operators(ops) {
    // 构建 combined_expr（从第一项开始依次加减）
    if (!items.empty()) {
      combined_expr = items[0].expr;
      for (size_t i = 1; i < items.size() && i <= operators.size(); i++) {
        if (operators[i-1] == "+") {
          combined_expr = combined_expr + items[i].expr;
        } else if (operators[i-1] == "-") {
          combined_expr = combined_expr - items[i].expr;
        }
      }
    }
  }

  // 添加一项（带操作符）
  void AddItem(const ExpressionItem& item, const std::string& op) {
    if (items.empty()) {
      // 第一项不需要操作符
      items.push_back(item);
      combined_expr = item.expr;
    } else {
      items.push_back(item);
      operators.push_back(op);
      // 更新 combined_expr
      if (op == "+") {
        combined_expr = combined_expr + item.expr;
      } else if (op == "-") {
        combined_expr = combined_expr - item.expr;
      }
    }
  }

  // 添加一个表达式（带操作符）- 简化合并表达式操作
  void AddExpression(const CombinedExpression& expr, const std::string& op) {
    if (expr.items.empty()) {
      return;
    }
    for (size_t i = 0; i < expr.items.size(); i++) {
      if (i == 0) {
        AddItem(expr.items[i], op);
      } else {
        AddItem(expr.items[i], expr.operators[i-1]);
      }
    }
  }

  // 转换为字符串
  std::string ToStr(const Tiler& tiler) const;

  // 判断是否为简单表达式（单项）
  bool IsSimple() const { return items.size() == 1; }

  // 判断是否为空
  bool IsEmpty() const { return items.empty(); }
};

// ============================================================================
// 辅助工厂函数 - 快速创建常用 ExpressionItem
// ============================================================================
namespace ExprItemFactory {

  // 创建 Size 转换项
  inline ExpressionItem Size(const ge::Expression& expr) {
    return ExpressionItem(expr, ExpressionConvertMethod::kSize);
  }

  // 创建 ActualSize 转换项
  inline ExpressionItem ActualSize(const ge::Expression& expr) {
    return ExpressionItem(expr, ExpressionConvertMethod::kActualSize);
  }

  // 创建带乘数的 Size 转换项（stride转字节）
  inline ExpressionItem SizeWithMultiplier(const ge::Expression& expr, int64_t dtype_size) {
    return ExpressionItem(expr, ExpressionConvertMethod::kSizeWithMultiplier, dtype_size);
  }

  // 创建带乘数的 ActualSize 转换项
  inline ExpressionItem ActualSizeWithMultiplier(const ge::Expression& expr, int64_t dtype_size) {
    return ExpressionItem(expr, ExpressionConvertMethod::kActualSizeWithMultiplier, dtype_size);
  }

  // 创建带 cast 的 Size 转换项
  inline ExpressionItem SizeWithCast(const ge::Expression& expr, const std::string& cast) {
    return ExpressionItem(expr, ExpressionConvertMethod::kSize, 0, cast);
  }

  // 创建带 cast 和乘数的 Size 转换项
  inline ExpressionItem SizeWithCastAndMultiplier(const ge::Expression& expr,
                                                    int64_t dtype_size,
                                                    const std::string& cast) {
    return ExpressionItem(expr, ExpressionConvertMethod::kSizeWithMultiplier, dtype_size, cast);
  }

  // 创建带 cast 的 ActualSize 转换项
  inline ExpressionItem ActualSizeWithCast(const ge::Expression& expr, const std::string& cast) {
    return ExpressionItem(expr, ExpressionConvertMethod::kActualSize, 0, cast);
  }

  // 创建带 cast 和乘数的 ActualSize 转换项
  inline ExpressionItem ActualSizeWithCastAndMultiplier(const ge::Expression& expr,
                                                          int64_t dtype_size,
                                                          const std::string& cast) {
    return ExpressionItem(expr, ExpressionConvertMethod::kActualSizeWithMultiplier, dtype_size, cast);
  }

  // 创建不转换的项（直接输出常量或符号名）
  inline ExpressionItem Direct(const ge::Expression& expr) {
    return ExpressionItem(expr, ExpressionConvertMethod::kNone);
  }

  // 创建常量项
  inline ExpressionItem Constant(int64_t value) {
    return ExpressionItem(ge::Symbol(value), ExpressionConvertMethod::kNone);
  }

  // 创建常量项（带 cast）
  inline ExpressionItem ConstantWithCast(int64_t value, const std::string& cast) {
    return ExpressionItem(ge::Symbol(value), ExpressionConvertMethod::kNone, 0, cast);
  }

  // 创建符号变量项（如 "curAivM"）
  inline ExpressionItem SymbolVar(const std::string& name) {
    return ExpressionItem(ge::Symbol(name.c_str()), ExpressionConvertMethod::kNone);
  }

  // 创建符号变量项（带 cast）
  inline ExpressionItem SymbolVarWithCast(const std::string& name, const std::string& cast) {
    return ExpressionItem(ge::Symbol(name.c_str()), ExpressionConvertMethod::kNone, 0, cast);
  }

  // 创建循环变量乘以 Size 的项：输出为 "outer_for_${loop_idx} * Size(expr)"
  inline ExpressionItem LoopVarTimesSize(const ge::Expression& expr, int64_t loop_idx) {
    ExpressionItem item(expr, ExpressionConvertMethod::kLoopVarTimesSize);
    item.loop_idx = loop_idx;
    return item;
  }

  // 创建循环变量乘以 ActualSize 的项：输出为 "outer_for_${loop_idx} * ActualSize(expr)"
  inline ExpressionItem LoopVarTimesActualSize(const ge::Expression& expr, int64_t loop_idx) {
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
  inline CombinedExpression SizeMinusActualSize(
      const ge::Expression& size_expr,
      const ge::Expression& actual_size_expr) {
    return CombinedExpression(
        ExprItemFactory::Size(size_expr),
        ExprItemFactory::ActualSize(actual_size_expr),
        "-"
    );
  }

  // 创建 ActualSize(expr1 - expr2) 的组合（api_call_utils.cpp 185-186 行场景）
  inline CombinedExpression ActualSizeOfDiff(
      const ge::Expression& expr1,
      const ge::Expression& expr2) {
    ge::Expression diff_expr = expr1 - expr2;
    return CombinedExpression(ExprItemFactory::ActualSize(diff_expr));
  }

  // 创建 Size(expr) + Size(expr) 的组合
  inline CombinedExpression SizePlusSize(
      const ge::Expression& expr1,
      const ge::Expression& expr2) {
    return CombinedExpression(
        ExprItemFactory::Size(expr1),
        ExprItemFactory::Size(expr2),
        "+"
    );
  }

  // 创建 ActualSize(expr) + ActualSize(expr) 的组合
  inline CombinedExpression ActualSizePlusActualSize(
      const ge::Expression& expr1,
      const ge::Expression& expr2) {
    return CombinedExpression(
        ExprItemFactory::ActualSize(expr1),
        ExprItemFactory::ActualSize(expr2),
        "+"
    );
  }

  // 创建常量表达式
  inline CombinedExpression Constant(int64_t value) {
    return CombinedExpression(ExprItemFactory::Constant(value));
  }

  // 创建符号变量表达式
  inline CombinedExpression SymbolVar(const std::string& name) {
    return CombinedExpression(ExprItemFactory::SymbolVar(name));
  }

  // 创建带 cast 的 ActualSize 表达式（用于 loop_sizes）
  inline CombinedExpression ActualSizeWithCast(
      const ge::Expression& expr,
      const std::string& cast = "static_cast<uint32_t>") {
    return CombinedExpression(ExprItemFactory::ActualSizeWithCast(expr, cast));
  }

  // 创建带 cast 和乘数的 Size 表达式（用于 loop_strides）
  inline CombinedExpression SizeWithCastAndMultiplier(
      const ge::Expression& expr,
      int64_t dtype_size,
      const std::string& cast = "static_cast<uint64_t>") {
    return CombinedExpression(ExprItemFactory::SizeWithCastAndMultiplier(expr, dtype_size, cast));
  }
}  // namespace CombinedExprFactory

// ============================================================================
// 类型别名，简化使用
// ============================================================================
using ExprItem = ExpressionItem;
using CombinedExpr = CombinedExpression;

// ============================================================================
// ExpressionItem::ToStr 实现（使用前向声明的函数）
// ============================================================================
inline std::string ExpressionItem::ToStr(const Tiler& tiler) const {
  std::string result;

  switch (method) {
    case ExpressionConvertMethod::kNone:
      // 直接输出表达式字符串（空指针保护）
      {
        auto str_ptr = expr.Str(af::StrType::kStrCpp);
        if (str_ptr) {
          result = str_ptr.get();
        } else {
          result = "";  // 空指针时返回空字符串作为默认值
        }
      }
      break;

    case ExpressionConvertMethod::kSize:
      result = TilerSize(tiler, expr);
      break;

    case ExpressionConvertMethod::kActualSize:
      result = TilerActualSize(tiler, expr);
      break;

    case ExpressionConvertMethod::kSizeWithMultiplier:
      result = TilerSize(tiler, expr) + " * " + std::to_string(dtype_size_multiplier);
      break;

    case ExpressionConvertMethod::kActualSizeWithMultiplier:
      result = TilerActualSize(tiler, expr) + " * " + std::to_string(dtype_size_multiplier);
      break;

    case ExpressionConvertMethod::kLoopVarTimesSize:
      // 输出: "outer_for_${loop_idx} * Size(expr)"
      result = "outer_for_" + std::to_string(loop_idx) + " * " + TilerSize(tiler, expr);
      break;

    case ExpressionConvertMethod::kLoopVarTimesActualSize:
      // 输出: "outer_for_${loop_idx} * ActualSize(expr)"
      result = "outer_for_" + std::to_string(loop_idx) + " * " + TilerActualSize(tiler, expr);
      break;
  }

  // 添加 cast 前缀
  if (!cast_prefix.empty()) {
    return cast_prefix + "(" + result + ")";
  }
  return result;
}

// ============================================================================
// CombinedExpression::ToStr 实现
// ============================================================================
inline std::string CombinedExpression::ToStr(const Tiler& tiler) const {
  if (items.empty()) {
    return "0";  // 默认值
  }

  // 简单表达式（单项）
  if (IsSimple()) {
    return items[0].ToStr(tiler);
  }

  // 组合表达式
  std::stringstream ss;
  ss << "(";
  for (size_t i = 0; i < items.size(); i++) {
    ss << items[i].ToStr(tiler);
    if (i < operators.size()) {
      ss << " " << operators[i] << " ";
    }
  }
  ss << ")";
  return ss.str();
}

}  // namespace codegen

#endif  // __EXPRESSION_CONVERT_STRUCT_H__