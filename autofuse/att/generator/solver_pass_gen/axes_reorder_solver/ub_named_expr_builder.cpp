/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_named_expr_builder.h"

#include "common/ub_expr/ub_expr_utils.h"
#include "named_origin_buf_expr_generator.h"

namespace att {
namespace {

ExprExprMap ToAttExprMap(const ascir::UbExprExprMap &expr_map) {
  ExprExprMap result;
  for (const auto &item : expr_map) {
    result[item.first] = item.second;
  }
  return result;
}

std::map<Expr, std::string, ExprCmp> ToAttExprNameMap(
    const std::map<ascir::UbExpr, std::string, ascir::UbExprCmp> &expr_names) {
  std::map<Expr, std::string, ExprCmp> result;
  for (const auto &item : expr_names) {
    result[item.first] = item.second;
  }
  return result;
}

}  // namespace

std::pair<std::string, std::string> BuildNamedUbExpr(const ascir::UbExprContext &context, const std::string &indent) {
  const auto ub_expr_result = ascir::UbExprUtils::BuildUbExpr(context);
  if (!ub_expr_result.has_ub_expr) {
    return {"", ""};
  }
  const auto container_expr = ToAttExprMap(context.container_expr);
  const auto container_names = ToAttExprNameMap(context.container_names);
  return NamedOriginBufExprGenerator(container_expr, container_names).Generate(ub_expr_result.ub_expr, indent);
}

}  // namespace att
