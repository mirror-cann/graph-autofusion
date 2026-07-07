/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_BASE_OPTIMIZE_STATIC_UB_TEMPLATE_FILTER_H_
#define ASCGEN_DEV_BASE_OPTIMIZE_STATIC_UB_TEMPLATE_FILTER_H_

#include "schedule_result.h"

namespace optimize {

class StaticUbTemplateFilter {
 public:
  af::Status Filter(ascir::FusedScheduledResult &fused_scheduled_result) const;
};

}  // namespace optimize

#endif  // ASCGEN_DEV_BASE_OPTIMIZE_STATIC_UB_TEMPLATE_FILTER_H_
