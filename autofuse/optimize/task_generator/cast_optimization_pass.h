/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_TASK_GENERATOR_CAST_OPTIMIZATION_PASS_H
#define OPTIMIZE_TASK_GENERATOR_CAST_OPTIMIZATION_PASS_H
#include "ascendc_ir.h"

namespace af::optimize {
class CastOptimizationPass {
 public:
  static Status Run(AscGraph &graph, int32_t concat_alg = 0);

 private:
  static bool NeedOptimize(const AscNodePtr &node, DataType src_dtype, DataType dst_dtype, int32_t concat_alg);
  static bool MayCauseDegradation(const AscNodePtr &concat_node, int32_t src_dtype_size, int32_t dst_dtype_size);
  static Status DoOptimize(AscGraph &graph,
                           const AscNodePtr &node,
                           const AscNodePtr &out_cast_node,
                           DataType src_dtype,
                           DataType dst_dtype);
};
}  // namespace af::optimize

#endif  // OPTIMIZE_TASK_GENERATOR_CAST_OPTIMIZATION_PASS_H
