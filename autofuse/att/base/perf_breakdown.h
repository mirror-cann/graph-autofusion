/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_BASE_PERF_BREAKDOWN_H_
#define ATT_BASE_PERF_BREAKDOWN_H_

#include <string>
#include <vector>

#include "base/base_types.h"

namespace att {
struct PerfBreakdownItem {
  std::string name;
  Expr expr;
  std::string desc;
  uint32_t indent{0U};
};

struct PerfBreakdownGroup {
  std::string title;
  std::vector<PerfBreakdownItem> items;
};
}  // namespace att

#endif  // ATT_BASE_PERF_BREAKDOWN_H_
