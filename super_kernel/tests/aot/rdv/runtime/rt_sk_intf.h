/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#include <stddef.h>
#include <stdint.h>
#include "rt_intf.h"

// If real runtime headers are available, define ACLRT_USE_REAL_RUNTIME_HEADER
// and ensure the headers are discoverable in include paths.
#if __has_include("acl/acl.h")
#include "acl/acl.h"
#define ACLRT_STUB_TYPES_DEFINED
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Base runtime types (avoid redefining if real headers already provide them)
#ifndef ACLRT_STUB_TYPES_DEFINED
#define ACLRT_STUB_TYPES_DEFINED
typedef void* aclrtFuncHandle;
typedef void* aclrtBinHandle;
typedef void* aclrtLaunchKernelCfg;
typedef void* aclrtArgsHandle;
typedef void* aclrtTaskGrp;
typedef int aclError;
typedef void* aclmdlRI;
typedef void* aclrtStream;
#endif  // ACLRT_STUB_TYPES_DEFINED

typedef enum aclrtTaskType {
    ACL_RT_TASK_DEFAULT,
    ACL_RT_TASK_KERNEL,
    ACL_RT_TASK_EVENT_RECORD,
    ACL_RT_TASK_EVENT_WAIT,
    ACL_RT_TASK_EVENT_RESET,
    ACL_RT_TASK_VALUE_WRITE,
    ACL_RT_TASK_VALUE_WAIT,
} aclrtTaskType;

typedef enum aclrtTaskFlag{
    ACL_RT_TASK_INVALID,
    ACL_RT_TASK_VALID,
} aclrtTaskFlag;

typedef enum aclrtEventType {
    ACL_RT_EVENT_NORMAL,
    ACL_RT_EVENT_MEMORY,
} aclrtEventType;

// typedef enum {
//     ACL_FUNC_ATTR_KERNEL_TYPE = 1,
//     ACL_FUNC_ATTR_KERNEL_RATIO = 2,
// }aclrtFuncAttribute;

// typedef enum {
//     ACL_KERNEL_TYPE_AICORE = 0,
//     ACL_KERNEL_TYPE_CUBE = 1,
//     ACL_KERNEL_TYPE_VERTOR = 2,
//     ACL_KERNEL_TYPE_MIX = 3,
//     ACL_KERNEL_TYPE_AICPU = 100,
// } aclrtKernelType;

// typedef enum aclmdlRIKernelType {
//     K_TYPE_AICORE,
//     K_TYPE_AIC,
//     K_TYPE_AIV,
//     K_TYPE_MIX_AIC_MAIN,
//     K_TYPE_MIX_AIV_MAIN,
// } aclmdlRIKernelType;

typedef struct aclrtTaskKernelParams{
    aclrtTaskType type;
    aclrtTaskFlag flag;
    aclrtFuncHandle funcHandle;
    aclrtBinHandle binHandle;
    aclrtLaunchKernelCfg* cfg;
    aclrtArgsHandle argsHandle;
    aclrtTaskGrp taskGrp;
    void* devArgs;
    size_t argsSize;
    void* opInfoPtr;
    size_t opInfoSize;
    uint32_t numBlocks;
    // Temporary: embed sk info here instead of aclrtTaskSkInfo.
    const char* func_name;
    uint32_t sk_kernel_type;
    uint32_t sk_task_ratio[2];
    uint32_t reserve1[8];
    void* pExtend;
} aclrtTaskKernelParams;


typedef struct aclrtTaskEventParams{
    aclrtTaskType type;
    aclrtTaskFlag flag;
    aclrtEventType eventType;
    uint32_t eventId;
    void *eventAddr;
    uint64_t value;
    uint32_t valueSize;
    uint32_t waitFlag;
    uint32_t reserve1[8];
    void* pExtend;
} aclrtTaskEventParams;

typedef struct aclrtTaskMemValueParams{
    aclrtTaskType type;
    aclrtTaskFlag flag;
    void *valueAddr;
    uint64_t value;
    uint32_t valueSize;
    uint32_t waitFlag;
    uint32_t reserve1[8];
    void* pExtend;
} aclrtMemValueParams;

typedef struct aclrtTaskDefaultParams{
    aclrtTaskType type;
    aclrtTaskFlag flag;
    uint32_t reserve1[8];
    void* pExtend;
} aclrtTaskDefaultParams;

// Binary metadata type enum (stub for UT)
typedef enum {
    RT_BINARY_TYPE_BIN_VERSION = 0U,
    RT_BINARY_TYPE_DEBUG_INFO = 1U,
    RT_BINARY_TYPE_DYNAMIC_PARAM = 2U,
    RT_BINARY_TYPE_OPTIONAL_PARAM = 3U,
    RT_BINARY_TYPE_RUNTIME_IMPLICIT_INFO = 4U,
    RT_BINARY_TYPE_SK_INFO = 5U,
    RT_BINARY_TYPE_MAX
} rtBinaryMetaType;

typedef struct aclrtTask{
    uint32_t task_id;
    aclrtTaskType type;
    union {
        aclrtTaskKernelParams kernel;
        aclrtTaskEventParams event;
        aclrtTaskDefaultParams def;
        aclrtMemValueParams memValue;
    };
}aclrtTask;

aclError aclmdlRIGetStreams(aclmdlRI modelRI, aclrtStream *streams, uint32_t *numStreams);

aclError aclrtStreamGetTasks(aclrtStream stream, aclrtTask *tasks, uint32_t *numTasks);

aclError aclrtTaskGetType(aclrtTask task, aclrtTaskType *type); // 类型错了，我做了修正

aclError aclrtTaskGetKernelParams(aclrtTask task, aclrtTaskKernelParams *params);
aclError aclrtTaskSetKernelParams(aclrtTask task, aclrtTaskKernelParams *params);

aclError aclrtTaskGetEventParams(aclrtTask task, aclrtTaskEventParams *params);
aclError aclrtTaskSetEventParams(aclrtTask task, aclrtTaskEventParams *params);

aclError aclrtTaskGetMemValueParams(aclrtTask task, aclrtMemValueParams *params);
aclError aclrtTaskSetMemValueParams(aclrtTask task, aclrtMemValueParams *params);
aclError aclmdlRIUpdate(aclmdlRI modelRI);

void rt_sk_start_capture(void);
void rt_sk_stop_capture(void);
void rt_sk_capture_snapshot(void);
void rt_sk_replay(void);

// aclError aclrtGetFunctionSize(aclrtFuncHandle funcHandle, size_t *aicSize, size_t *aivSize);
// aclError aclrtGetFunctionAttribute(aclrtFuncHandle funcHandle, aclrtFuncAttribute attrType, int64_t *attrValue);

__attribute__((visibility("default"))) int rtBinaryGetMetaNum(void* binHdl, int type_enum, size_t* metaNum);
__attribute__((visibility("default"))) int rtBinaryGetMetaData(void* binHdl, int type_enum, size_t metaNum, void** data_list, size_t *size_list);
__attribute__((visibility("default"))) int rtBinaryGetMetaInfo(void* binHdl, int type_enum, size_t metaNum, void** data_list, size_t *size_list);

#ifdef __cplusplus
}
#endif