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
* @file rt_stub.cpp
* @brief Stub implementations for runtime functions used in unit tests
*        Only contains runtime-specific functions (rt*), not ACL functions (aclrt*)
*        For UT stub mode, all functions return success (0) by default
*/

#include "acl/acl.h"
#include "runtime/kernel.h"
#include "runtime/base.h"
#include <cstring>
#include <cstdio>
extern "C" {

// Error codes
#define RT_SUCCESS 0
#define RT_ERROR -1

// rtBinaryGetMetaNum - stub implementation for unit tests
// Always returns success for UT stub mode
int rtBinaryGetMetaNum(void* binHdl, int type_enum, size_t* metaNum) {
    if (metaNum != nullptr) {
        *metaNum = 0;
    }
    return RT_SUCCESS;
}

// rtBinaryGetMetaInfo - stub implementation for unit tests
// Always returns success for UT stub mode
int rtBinaryGetMetaInfo(void* binHdl, int type_enum, size_t metaNum, void** data_list, size_t* size_list) {
    (void)binHdl;
    (void)type_enum;
    (void)metaNum;
    (void)data_list;
    (void)size_list;
    return RT_SUCCESS;
}

// rtGetBinBuffer - stub implementation for unit tests
int rtGetBinBuffer(void* binHdl, int addrType, void** buffer, uint32_t* size) {
    if (buffer != nullptr) {
        *buffer = nullptr;
    }
    if (size != nullptr) {
        *size = 0;
    }
    return RT_SUCCESS;
}

// rtBinaryGetMetaData - stub implementation for unit tests
// Always returns success for UT stub mode
int rtBinaryGetMetaData(void* binHdl, int type_enum, size_t metaNum, void** data_list, size_t* size_list) {
    (void)binHdl;
    (void)type_enum;
    (void)metaNum;
    (void)data_list;
    (void)size_list;
    return RT_SUCCESS;
}

// Super kernel capture/replay API stubs
void rt_sk_start_capture(void) {
    // Stub implementation
}

void rt_sk_stop_capture(void) {
    // Stub implementation
}

void rt_sk_capture_snapshot(void) {
    // Stub implementation
}

void rt_sk_replay(void) {
    // Stub implementation
}

// rtGetExceptionRegInfo - stub implementation for unit tests
// Returns success with no cores in error
rtError_t rtGetExceptionRegInfo(const void* exception, rtExceptionErrRegInfo_t** errRegInfo, uint32_t* coreNum)
{
    (void)exception;
    (void)errRegInfo;
    *coreNum = 0; // No cores with errors by default
    return RT_SUCCESS;
}

// aclmdlRIDebugJsonPrint - stub implementation for unit tests
// Always returns success for UT stub mode
aclError aclmdlRIDebugJsonPrint(aclmdlRI model, const char* path, uint32_t flag) {
    (void)model;
    (void)path;
    (void)flag;
    return ACL_SUCCESS;
}

} // extern "C"
