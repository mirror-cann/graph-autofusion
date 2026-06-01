/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_ASCENDC_REDUCE_API_PERF_V2_H_
#define AUTOFUSE_ASCENDC_REDUCE_API_PERF_V2_H_

#include "api_perf_register/utils/api_perf_utils.h"

namespace att {
namespace ascendcapi_v2 {

// Reduce API perf is exposed as a typed call because the model depends on structured codegen context.
// Plain API formulas without such context should use ApiPerfFactory registration instead.
enum class ReducePattern {
  kAR,
  kRA,
};

enum class ReduceMergeMode {
  kNone,
  kCopy,
  kMergeByElementwise,
};

struct ReduceApiPerfContext {
  NodeDetail node_detail;
  ReducePattern pattern{ReducePattern::kAR};
  bool is_reuse_source{false};
  ReduceMergeMode merge_mode{ReduceMergeMode::kNone};
  Expr merge_size{CreateExpr(0)};
  Expr merge_times{CreateExpr(1)};
};

ge::Status ReduceMinPerf(const ReduceApiPerfContext &context, PerfOutputInfo &perf);
ge::Status ReduceMaxPerf(const ReduceApiPerfContext &context, PerfOutputInfo &perf);
ge::Status ReduceAnyPerf(const ReduceApiPerfContext &context, PerfOutputInfo &perf);
ge::Status ReduceAllPerf(const ReduceApiPerfContext &context, PerfOutputInfo &perf);
ge::Status ReduceSumPerf(const ReduceApiPerfContext &context, PerfOutputInfo &perf);
ge::Status ReduceProdPerf(const ReduceApiPerfContext &context, PerfOutputInfo &perf);
ge::Status ReduceMeanPerf(const ReduceApiPerfContext &context, PerfOutputInfo &perf);

}  // namespace ascendcapi_v2
}  // namespace att

#endif  // AUTOFUSE_ASCENDC_REDUCE_API_PERF_V2_H_
