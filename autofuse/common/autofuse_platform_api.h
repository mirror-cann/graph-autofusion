/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_BASE_COMMON_AUTOFUSE_PLATFORM_API_H_
#define ASCGEN_DEV_BASE_COMMON_AUTOFUSE_PLATFORM_API_H_

#include <string>
#include "ge_common/ge_api_error_codes.h"

namespace ge {
ge::Status SetAutofusePlatform(const std::string &platform_name);
ge::Status GetAutofusePlatform(std::string &platform_name);
void ResetAutofusePlatform();
}  // namespace ge

#endif  // ASCGEN_DEV_BASE_COMMON_AUTOFUSE_PLATFORM_API_H_
