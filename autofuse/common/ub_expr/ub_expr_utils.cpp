/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/ub_expr/ub_expr_utils.h"

namespace ascir {

UbExprBuildResult UbExprUtils::BuildUbExpr(const UbExprContext &context) {
  UbExprBuildResult result;
  if (!context.ub_expr.IsValid() || context.ub_expr.Str() == nullptr) {
    return result;
  }
  result.has_ub_expr = true;
  result.ub_expr = context.ub_expr;
  result.origin_expr = context.ub_expr.Str().get();
  return result;
}

}  // namespace ascir
