/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_ASCENDC_DATA_MOVE_API_PERF_V2_H_
#define AUTOFUSE_ASCENDC_DATA_MOVE_API_PERF_V2_H_

#include "api_perf_register/utils/api_perf_utils.h"

namespace att {
namespace ascendcapi_v2 {

af::Status LoadPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status StorePerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status NddmaPerf(const NodeDetail &node_info, PerfOutputInfo &perf);

}  // namespace ascendcapi_v2
}  // namespace att

#endif  // AUTOFUSE_ASCENDC_DATA_MOVE_API_PERF_V2_H_
