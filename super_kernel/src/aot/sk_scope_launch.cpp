/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file sk_scope_launch.cpp
 * \brief
 */

#include "sk_scope_launch.h"

#include <cstring>
#include <cstdlib>

#include "securec.h"
#include "sk_log.h"

extern "C" void sk_scope_kernel_begin_do(void* stream, ScopeKernelArgs args);
extern "C" void sk_scope_kernel_end_do(void* stream, ScopeKernelArgs args);

typedef void (*ScopeFuncImpl)(void*, ScopeKernelArgs);

static aclError LaunchScopeKernelImpl(const char* scopeName, aclrtStream stream, ScopeFuncImpl scopeKernelImpl) {
    ScopeKernelArgs args;
    if (scopeName == nullptr) {
        scopeName = "default_sk_scope_name";
    }
    size_t len = strlen(scopeName);
    size_t maxLen = MAX_SCOPE_NAME_LENN - 1;
    if (len > maxLen) {
        SK_LOGE("scope name length: %zu exceeds maximum allowed size: %zu", len, maxLen);
        return ACL_ERROR_INVALID_PARAM;
    }
    errno_t ret = memcpy_s(args.name, maxLen, scopeName, len);
    if (ret != EOK) {
        SK_LOGE("memcpy_s failed, ret: %d", ret);
        return ACL_ERROR_INVALID_PARAM;
    }
    args.name[len] = '\0';
    scopeKernelImpl(stream, args);
    return ACL_SUCCESS;
}

aclError LaunchScopeKernel(const char* scopeName, aclrtStream stream, bool isBegin)
{   
    if (isBegin) {
        return LaunchScopeKernelImpl(scopeName, stream, sk_scope_kernel_begin_do);
    } else {
        return LaunchScopeKernelImpl(scopeName, stream, sk_scope_kernel_end_do);
    }
}