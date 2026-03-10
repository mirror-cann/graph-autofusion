/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef __RUNTIME_KERNEL_H_STUB__
#define __RUNTIME_KERNEL_H_STUB__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Binary metadata type enum - 这些类型定义来自真实的runtime/kernel.h
// 用于unit test stub
typedef enum {
    RT_BIN_HOST_ADDR = 0,
    RT_BIN_DEVICE_ADDR = 1,
} rtBinaryAddrType;

// Binary metadata type enum
typedef enum {
    RT_BINARY_TYPE_BIN_VERSION = 0U,
    RT_BINARY_TYPE_DEBUG_INFO = 1U,
    RT_BINARY_TYPE_DYNAMIC_PARAM = 2U,
    RT_BINARY_TYPE_OPTIONAL_PARAM = 3U,
    RT_BINARY_TYPE_RUNTIME_IMPLICIT_INFO = 4U,
    RT_BINARY_TYPE_SK_INFO = 5U,
    RT_BINARY_TYPE_MAX
} rtBinaryMetaType;

#define RT_BIN_HOST_ADDR 0
#define RT_BIN_DEVICE_ADDR 1

// Runtime binary metadata API
int rtBinaryGetMetaNum(void* binHdl, int type_enum, size_t* metaNum);
int rtBinaryGetMetaData(void* binHdl, int type_enum, size_t metaNum, void** data_list, size_t* size_list);
int rtBinaryGetMetaInfo(void* binHdl, int type_enum, size_t metaNum, void** data_list, size_t* size_list);
int rtGetBinBuffer(void* binHdl, int addrType, void** buffer, uint32_t* size);

#ifdef __cplusplus
}
#endif

#endif // __RUNTIME_KERNEL_H_STUB__
