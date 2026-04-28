/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_FRAME_AUTOFUSE_FRAMES_H_
#define AUTOFUSE_FRAME_AUTOFUSE_FRAMES_H_

#include <cstdint>

#include "ge_common/ge_api_types.h"
#include "graph/gnode.h"

namespace af {
class Counter {
 public:
  Counter() = default;
  virtual ~Counter() = default;
  virtual int64_t NextId() = 0;
};
using CounterPtr = Counter *;
}  // namespace af

namespace ge {
using af::Counter;
using af::CounterPtr;

struct GraphPasses {
  std::function<Status(const ComputeGraphPtr &)> prune_graph_func;
  std::function<Status(NodePtr &)> constant_folding_func;
};
}  // namespace ge

#endif  // AUTOFUSE_FRAME_AUTOFUSE_FRAMES_H_
