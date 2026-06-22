/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "expression_convert_struct.h"
#include "codegen/codegen_kernel.h"

#include <sstream>

namespace codegen {

// 辅助函数实现，调用 Tiler 的 Size 和 ActualSize 方法
std::string TilerSize(const Tiler &tiler, const ge::Expression &expr) {
  // 空指针保护：检查 Expression 的 Str() 返回值
  auto str_ptr = expr.Str(af::StrType::kStrCpp);
  if (!str_ptr) {
    return "";  // 空指针时返回空字符串作为默认值
  }
  return tiler.Size(expr);
}

std::string TilerActualSize(const Tiler &tiler, const ge::Expression &expr) {
  // 空指针保护：检查 Expression 的 Str() 返回值
  auto str_ptr = expr.Str(af::StrType::kStrCpp);
  if (!str_ptr) {
    return "";  // 空指针时返回空字符串作为默认值
  }
  return tiler.ActualSize(expr);
}

// ============================================================================
// ExpressionItem::ToStr 实现
// ============================================================================
std::string ExpressionItem::ToStr(const Tiler &tiler) const {
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
std::string CombinedExpression::ToStr(const Tiler &tiler) const {
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
