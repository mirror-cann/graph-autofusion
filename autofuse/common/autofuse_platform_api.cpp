/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/autofuse_platform_api.h"
#include "common/platform_context.h"

namespace ge {
ge::Status SetAutofusePlatform(const std::string &platform_name) {
  PlatformContext::GetInstance().SetPlatform(platform_name);
  return ge::SUCCESS;
}

ge::Status GetAutofusePlatform(std::string &platform_name) {
  return PlatformContext::GetInstance().GetCurrentPlatformString(platform_name);
}

void ResetAutofusePlatform() {
  PlatformContext::GetInstance().Reset();
}
}  // namespace ge
