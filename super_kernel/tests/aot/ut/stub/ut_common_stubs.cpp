/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/**
* @file ut_common_stubs.cpp
* @brief Common stub implementations used by aot unit tests
*/

#include "acl/acl.h"
#include "sk_scope_kernel_types.h"

namespace {

int g_entryBinHandleNull = 0;
int g_securecMemcpyFailOnCall = -1;
int g_securecMemcpySeen = 0;
int g_securecMemsetFailOnCall = -1;
int g_securecMemsetSeen = 0;

} // namespace

extern "C" {

void SkUtResetCommonStubControls()
{
    g_entryBinHandleNull = 0;
    g_securecMemcpyFailOnCall = -1;
    g_securecMemcpySeen = 0;
    g_securecMemsetFailOnCall = -1;
    g_securecMemsetSeen = 0;
}

void SkUtSetEntryBinHandleNull(int enable)
{
    g_entryBinHandleNull = enable;
}

void SkUtSetSecurecMemcpyFailOnCall(int hitOnCall)
{
    g_securecMemcpyFailOnCall = hitOnCall;
    g_securecMemcpySeen = 0;
}

void SkUtSetSecurecMemsetFailOnCall(int hitOnCall)
{
    g_securecMemsetFailOnCall = hitOnCall;
    g_securecMemsetSeen = 0;
}

int SkUtSecurecShouldFailMemcpy()
{
    if (g_securecMemcpyFailOnCall <= 0) {
        return 0;
    }
    ++g_securecMemcpySeen;
    return g_securecMemcpySeen == g_securecMemcpyFailOnCall ? 1 : 0;
}

int SkUtSecurecShouldFailMemset()
{
    if (g_securecMemsetFailOnCall <= 0) {
        return 0;
    }
    ++g_securecMemsetSeen;
    return g_securecMemsetSeen == g_securecMemsetFailOnCall ? 1 : 0;
}

aclrtBinHandle AscendGetEntryBinHandle()
{
    if (g_entryBinHandleNull != 0) {
        return nullptr;
    }
    return reinterpret_cast<aclrtBinHandle>(0x1234);
}

void sk_scope_kernel_begin_do(void* stream, ScopeKernelArgs args)
{
    (void)stream;
    (void)args;
}

void sk_scope_kernel_end_do(void* stream, ScopeKernelArgs args)
{
    (void)stream;
    (void)args;
}

void sk_placeholder_kernel_do(void* stream, ScopeKernelArgs args)
{
    (void)stream;
    (void)args;
}

} // extern "C"
