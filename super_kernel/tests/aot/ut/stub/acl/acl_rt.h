/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef INC_EXTERNAL_ACL_ACL_RT_H_
#define INC_EXTERNAL_ACL_ACL_RT_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
static const int ACL_SUCCESS = 0;
static const int ACL_ERROR_NONE = 0;
static const int ACL_ERROR_INVALID_PARAM = 100000;

// EOK for secure functions
#ifndef EOK
#define EOK 0
#endif

// Basic types
typedef void *aclrtStream;
typedef void *aclrtEvent;
typedef void *aclrtContext;
typedef void *aclrtNotify;
typedef void *aclrtTaskGrp;
typedef void *aclrtFuncHandle;
typedef void *aclrtBinHandle;
typedef void *aclrtLaunchKernelCfg;
typedef void *aclrtArgsHandle;
typedef void *aclrtParamHandle;
typedef void *aclmdlRI;
typedef void *aclrtTask;  // void* as per user request
typedef int aclError;

// Memory copy kind
typedef enum aclrtMemcpyKind {
    ACL_MEMCPY_HOST_TO_HOST,
    ACL_MEMCPY_HOST_TO_DEVICE,
    ACL_MEMCPY_DEVICE_TO_HOST,
    ACL_MEMCPY_DEVICE_TO_DEVICE,
} aclrtMemcpyKind;

// Device attributes
typedef enum aclrtDevAttr {
    ACL_DEV_ATTR_CUBE_CORE_NUM,
    ACL_DEV_ATTR_VECTOR_CORE_NUM,
} aclrtDevAttr;

// Task types
typedef enum aclrtTaskType {
    ACL_RT_TASK_DEFAULT,
    ACL_RT_TASK_KERNEL,
    ACL_RT_TASK_EVENT_RECORD,
    ACL_RT_TASK_EVENT_WAIT,
    ACL_RT_TASK_EVENT_RESET,
    ACL_RT_TASK_VALUE_WRITE,
    ACL_RT_TASK_VALUE_WAIT,
} aclrtTaskType;

typedef enum aclrtTaskFlag {
    ACL_RT_TASK_INVALID,
    ACL_RT_TASK_VALID,
} aclrtTaskFlag;

typedef enum aclrtEventType {
    ACL_RT_EVENT_NORMAL,
    ACL_RT_EVENT_MEMORY,
} aclrtEventType;

// Task parameter structures
typedef struct aclrtTaskKernelParams {
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
    const char* func_name;
    uint32_t sk_kernel_type;
    uint32_t sk_task_ratio[2];
    uint32_t reserve1[8];
    void* pExtend;
} aclrtTaskKernelParams;

typedef struct aclrtNormalEventInfo {
    uint32_t eventId;
} aclrtNormalEventInfo;

typedef struct aclrtMemoryEventInfo {
    void *eventAddr;            // 内存语义的device地址
    uint64_t value;             // 当前默认是写1
    uint32_t valueSize;         // 当前默认是1字节
    uint32_t waitFlag;          // 使用captureWait的默认比较策略
} aclrtMemoryEventInfo;

typedef struct aclrtTaskEventParams {
    aclrtTaskType type;         // task类型
    aclrtTaskFlag flag;         // 当前Task融合时是否删除
    uint32_t sequenceId;        // Task的序号
    aclrtEvent event;           // 其他类型Task转Event时，需要先申请event句柄并传递下来，get的时候返回nullptr
    aclrtEventType eventType;   // 区分是内存语义还是普通event
    union {
        aclrtNormalEventInfo normalEventInfo;
        aclrtMemoryEventInfo memoryEventInfo;
    } u;
    uint32_t reserve[8];        // 预留字段
    void *pExtend;              // 兼容扩展字段
} aclrtTaskEventParams;

typedef struct aclrtTaskMemValueParams {
    aclrtTaskType type;
    aclrtTaskFlag flag;
    void *valueAddr;
    uint64_t value;
    uint32_t valueSize;
    uint32_t waitFlag;
    uint32_t reserve1[8];
    void* pExtend;
} aclrtTaskMemValueParams;

typedef struct aclrtTaskDefaultParams {
    aclrtTaskType type;
    aclrtTaskFlag flag;
    uint32_t reserve1[8];
    void* pExtend;
} aclrtTaskDefaultParams;



typedef enum {
    ACL_KERNEL_TYPE_AICORE = 0, // MIX KERNEL
    ACL_KERNEL_TYPE_CUBE = 1,   // AI CUBE CORE
    ACL_KERNEL_TYPE_VECTOR = 2, // AI VECTOR CORE
    ACL_KERNEL_TYPE_MIX = 3,
    ACL_KERNEL_TYPE_AICPU = 100,
} aclrtKernelType;

typedef enum {
    ACL_FUNC_ATTR_KERNEL_TYPE = 1,
    ACL_FUNC_ATTR_KERNEL_RATIO = 2,
} aclrtFuncAttribute;

// Function declarations
aclError aclmdlRIGetStreams(aclmdlRI modelRI, aclrtStream *streams, uint32_t *numStreams);
aclError aclrtStreamGetTasks(aclrtStream stream, aclrtTask *tasks, uint32_t *numTasks);
aclError aclrtTaskGetType(aclrtTask task, aclrtTaskType *type);
aclError aclrtTaskGetKernelParams(aclrtTask task, aclrtTaskKernelParams *params);
aclError aclrtTaskSetKernelParams(aclrtTask task, aclrtTaskKernelParams *params);
aclError aclrtTaskGetEventParams(aclrtTask task, aclrtTaskEventParams *params);
aclError aclrtTaskSetEventParams(aclrtTask task, aclrtTaskEventParams *params);
aclError aclrtTaskGetMemValueParams(aclrtTask task, aclrtTaskMemValueParams *params);
aclError aclrtTaskSetMemValueParams(aclrtTask task, aclrtTaskMemValueParams *params);
aclError aclmdlRIUpdate(aclmdlRI modelRI);
aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t *value);
aclError aclrtGetDevice(int32_t *deviceId);

// Binary API
aclError aclrtBinaryGetFunction(aclrtBinHandle binHdl, const char* funcName, aclrtFuncHandle* funcHdl);
aclError aclrtGetFunctionAddr(aclrtFuncHandle funcHdl, void** addrAicore, void** addrAiv);
aclError aclrtKernelArgsGetHandleMemSize(aclrtFuncHandle funcHdl, size_t* memSize);
aclError aclrtKernelArgsGetMemSize(aclrtFuncHandle funcHdl, size_t argsSize, size_t* devArgsSize);
aclError aclrtKernelArgsInitByUserMem(aclrtFuncHandle funcHdl, aclrtArgsHandle argsHdl, void* devArgs, size_t devArgsSize);
aclError aclrtKernelArgsAppendPlaceHolder(aclrtArgsHandle argsHdl, aclrtParamHandle* phdl);
aclError aclrtKernelArgsGetPlaceHolderBuffer(aclrtArgsHandle argsHdl, aclrtParamHandle phdl, size_t bufferSize, void** buffer);
aclError aclrtKernelArgsFinalize(aclrtArgsHandle argsHdl);
aclError aclrtGetFunctionName(aclrtFuncHandle funcHandle, uint32_t maxLen, char* name);
aclError aclrtMemcpy(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind);
aclError aclrtBinaryGetDevAddress(aclrtBinHandle binHdl, void** devAddr, size_t* devSize);
aclError aclrtGetFunctionAttribute(aclrtFuncHandle funcHandle, aclrtFuncAttribute attrType, int64_t *attrValue);


#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_RT_H_
