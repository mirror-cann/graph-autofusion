/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_NAMED_ORIGIN_BUF_EXPR_GENERATOR_H_
#define ATT_NAMED_ORIGIN_BUF_EXPR_GENERATOR_H_

#include <map>
#include <string>
#include <utility>

#include "base/base_types.h"

namespace att {
class NamedOriginBufExprGenerator {
 public:
  NamedOriginBufExprGenerator(const ExprExprMap &container_expr,
                              const std::map<Expr, std::string, ExprCmp> &container_names)
      : container_expr_(container_expr), container_names_(container_names) {}
  ~NamedOriginBufExprGenerator() = default;

  std::pair<std::string, std::string> Generate(const Expr &expr, const std::string &indent) const;

 private:
  const ExprExprMap &container_expr_;
  const std::map<Expr, std::string, ExprCmp> &container_names_;
};
}  // namespace att

#endif  // ATT_NAMED_ORIGIN_BUF_EXPR_GENERATOR_H_
