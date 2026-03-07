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

bool IsScopeKernel(aclrtTaskKernelParams params, JudgeTaskKernelInfo* info) {
    if (params.type != ACL_RT_TASK_KERNEL) {
        SK_LOGE("current task is not kernel task");
        return false;
    }
    const char* defaultScopeName = "default_sk_scope_name";
    const char* targetBeginName = "sk_scope_kernel_begin";
    const char* targetEndName = "sk_scope_kernel_end";
    char kernelName[MAX_SCOPE_NAME_LENN] = {0};
    int ret = aclrtGetFunctionName(params.funcHandle, sizeof(kernelName), kernelName);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("get kernel name failed, ret: %d", ret);
        return false;
    }
    bool isBegin = (strcmp(kernelName, targetBeginName) == 0);
    bool isEnd = (strcmp(kernelName, targetEndName) == 0);
    if (!isBegin && !isEnd) {
        SK_LOGD("current kernel is not scope kernel, current kernel name is: %s", kernelName);
        return false;
    }
    auto parseArgsAddr = std::make_unique<ScopeKernelArgs>();
    ret = aclrtMemcpy((void*)parseArgsAddr.get(), sizeof(ScopeKernelArgs), params.devArgs, sizeof(ScopeKernelArgs), 
        ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("aclrtMemcpy failed, ret: %d", ret);
        return false;
    }
    parseArgsAddr->name[MAX_SCOPE_NAME_LENN - 1] = '\0';
    size_t nameLen = strlen(parseArgsAddr->name);
    info->scopeName = std::make_unique<char[]>(nameLen + 1);
    errno_t res = memcpy_s(info->scopeName.get(), nameLen + 1, parseArgsAddr->name, nameLen + 1);
    if (res != EOK) {
        SK_LOGE("memcpy_s failed, ret: %d", res);
        return false;
    }
    info->isBegin = isBegin;
    if (strcmp(info->scopeName.get(), defaultScopeName) == 0) {
        info->isFuseEnable = false;
    }
    SK_LOGD("Success parse scope kernel task, kernelName: %s, scopeName: %s, isBegin: %d, isFuseEnable: %d", kernelName, 
        info->scopeName.get(), info->isBegin, info->isFuseEnable);
    return true;
}

aclError aclskScopeBegin(const char* scopeName, aclrtStream stream) {
    if (scopeName[0] == '\0') {
        SK_LOGE("Invalid scopeName: name is empty.");
        return ACL_ERROR_INVALID_PARAM;
    }
    return LaunchScopeKernelImpl(scopeName, stream, sk_scope_kernel_begin_do);
}

aclError aclskScopeEnd(const char* scopeName, aclrtStream stream) {
    if (scopeName[0] == '\0') {
        SK_LOGE("Invalid scopeName: name is empty.");
        return ACL_ERROR_INVALID_PARAM;
    }
    return LaunchScopeKernelImpl(scopeName, stream, sk_scope_kernel_end_do);
}