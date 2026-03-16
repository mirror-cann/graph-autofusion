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
* @file aclrt_stubs.cpp
* @brief Stub implementations for aclrt runtime interfaces used in unit tests
*/

#include "acl/acl.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {

aclError g_aclmdlRIGetStreamsRet[2] = {ACL_SUCCESS, ACL_SUCCESS};
aclError g_aclrtStreamGetTasksRet[2] = {ACL_SUCCESS, ACL_SUCCESS};
aclError g_aclrtTaskGetTypeRet = ACL_SUCCESS;
aclError g_aclrtGetDeviceRet = ACL_SUCCESS;
aclError g_aclrtGetDeviceInfoRet = ACL_SUCCESS;
aclError g_aclmdlRIUpdateRet = ACL_SUCCESS;
int g_throwOnAclmdlRIGetStreams = 0;
int g_binaryGetFunctionNullHandle = 0;

uint32_t g_streamNum = 0;
std::vector<uint32_t> g_streamTaskNums;
std::vector<std::vector<aclrtTaskType>> g_taskTypes;

void EnsureStreamStorage(uint32_t streamIdx)
{
    if (g_streamTaskNums.size() <= streamIdx) {
        g_streamTaskNums.resize(streamIdx + 1, 0);
    }
    if (g_taskTypes.size() <= streamIdx) {
        g_taskTypes.resize(streamIdx + 1);
    }
}

void EnsureTaskStorage(uint32_t streamIdx, uint32_t taskIdx)
{
    EnsureStreamStorage(streamIdx);
    if (g_taskTypes[streamIdx].size() <= taskIdx) {
        g_taskTypes[streamIdx].resize(taskIdx + 1, ACL_RT_TASK_KERNEL);
    }
}

}

extern "C" {

void SkUtResetCommonStubControls();

void SkUtResetTestControls()
{
    g_aclmdlRIGetStreamsRet[0] = ACL_SUCCESS;
    g_aclmdlRIGetStreamsRet[1] = ACL_SUCCESS;
    g_aclrtStreamGetTasksRet[0] = ACL_SUCCESS;
    g_aclrtStreamGetTasksRet[1] = ACL_SUCCESS;
    g_aclrtTaskGetTypeRet = ACL_SUCCESS;
    g_aclrtGetDeviceRet = ACL_SUCCESS;
    g_aclrtGetDeviceInfoRet = ACL_SUCCESS;
    g_aclmdlRIUpdateRet = ACL_SUCCESS;
    g_throwOnAclmdlRIGetStreams = 0;
    g_binaryGetFunctionNullHandle = 0;
    SkUtResetCommonStubControls();

    g_streamNum = 0;
    g_streamTaskNums.clear();
    g_taskTypes.clear();
}

void SkUtSetAclmdlRIGetStreamsRet(int phase, aclError ret)
{
    if (phase < 0 || phase > 1) {
        return;
    }
    g_aclmdlRIGetStreamsRet[phase] = ret;
}

void SkUtSetAclrtStreamGetTasksRet(int phase, aclError ret)
{
    if (phase < 0 || phase > 1) {
        return;
    }
    g_aclrtStreamGetTasksRet[phase] = ret;
}

void SkUtSetAclrtTaskGetTypeRet(aclError ret)
{
    g_aclrtTaskGetTypeRet = ret;
}

void SkUtSetAclrtGetDeviceRet(aclError ret)
{
    g_aclrtGetDeviceRet = ret;
}

void SkUtSetAclrtGetDeviceInfoRet(aclError ret)
{
    g_aclrtGetDeviceInfoRet = ret;
}

void SkUtSetAclmdlRIUpdateRet(aclError ret)
{
    g_aclmdlRIUpdateRet = ret;
}

void SkUtSetThrowOnAclmdlRIGetStreams(int enable)
{
    g_throwOnAclmdlRIGetStreams = enable;
}

void SkUtSetBinaryGetFunctionNullHandle(int enable)
{
    g_binaryGetFunctionNullHandle = enable;
}

aclError SkUtGetAclmdlRIGetStreamsRet(int phase)
{
    if (phase < 0 || phase > 1) {
        return ACL_SUCCESS;
    }
    return g_aclmdlRIGetStreamsRet[phase];
}

aclError SkUtGetAclrtStreamGetTasksRet(int phase)
{
    if (phase < 0 || phase > 1) {
        return ACL_SUCCESS;
    }
    return g_aclrtStreamGetTasksRet[phase];
}

aclError SkUtGetAclrtTaskGetTypeRet()
{
    return g_aclrtTaskGetTypeRet;
}

aclError SkUtGetAclrtGetDeviceRet()
{
    return g_aclrtGetDeviceRet;
}

aclError SkUtGetAclrtGetDeviceInfoRet()
{
    return g_aclrtGetDeviceInfoRet;
}

aclError SkUtGetAclmdlRIUpdateRet()
{
    return g_aclmdlRIUpdateRet;
}

int SkUtGetThrowOnAclmdlRIGetStreams()
{
    return g_throwOnAclmdlRIGetStreams;
}

void SkUtSetModelStreamNum(uint32_t streamNum)
{
    g_streamNum = streamNum;
    if (g_streamTaskNums.size() < streamNum) {
        g_streamTaskNums.resize(streamNum, 0);
    }
    if (g_taskTypes.size() < streamNum) {
        g_taskTypes.resize(streamNum);
    }
}

uint32_t SkUtGetModelStreamNum()
{
    return g_streamNum;
}

void SkUtSetStreamTaskNum(uint32_t streamIdx, uint32_t taskNum)
{
    EnsureStreamStorage(streamIdx);
    g_streamTaskNums[streamIdx] = taskNum;
    g_taskTypes[streamIdx].resize(taskNum, ACL_RT_TASK_KERNEL);
}

uint32_t SkUtGetStreamTaskNum(uint32_t streamIdx)
{
    if (streamIdx >= g_streamTaskNums.size()) {
        return 0;
    }
    return g_streamTaskNums[streamIdx];
}

void SkUtSetTaskType(uint32_t streamIdx, uint32_t taskIdx, aclrtTaskType type)
{
    EnsureTaskStorage(streamIdx, taskIdx);
    g_taskTypes[streamIdx][taskIdx] = type;
}

aclrtTaskType SkUtGetTaskType(uint32_t streamIdx, uint32_t taskIdx)
{
    if (streamIdx >= g_taskTypes.size() || taskIdx >= g_taskTypes[streamIdx].size()) {
        return ACL_RT_TASK_KERNEL;
    }
    return g_taskTypes[streamIdx][taskIdx];
}

// Error codes
#ifndef ACL_ERROR_NONE
#define ACL_ERROR_NONE 0
#endif
#define ACL_ERROR_INVALID_PARAM 100001

// Internal task structure for stub implementation
typedef struct AclrtTaskInternal {
    uint32_t task_id;
    aclrtTaskType type;
    union {
        aclrtTaskKernelParams kernel;
        aclrtTaskEventParams event;
        aclrtTaskDefaultParams def;
        aclrtTaskMemValueParams memValue;
    };
} AclrtTaskInternal;

// Internal RI task structure for stub implementation
typedef struct AclmdlRITaskInternal {
    uint32_t task_id;
    aclmdlRITaskType type;
    aclmdlRITaskParams params;
} AclmdlRITaskInternal;

// Helper to convert void* to internal structure
static inline AclrtTaskInternal* TaskToInternal(aclrtTask task) {
    return reinterpret_cast<AclrtTaskInternal*>(task);
}

static inline AclmdlRITaskInternal* RITaskToInternal(aclmdlRITask task) {
    return reinterpret_cast<AclmdlRITaskInternal*>(task);
}

// 获取流
aclError aclmdlRIGetStreams(aclmdlRI modelRI, aclrtStream *streams, uint32_t *numStreams) {
    (void)modelRI;
    if (SkUtGetThrowOnAclmdlRIGetStreams() != 0) {
        throw std::runtime_error("ut-injected aclmdlRIGetStreams exception");
    }
    const int phase = (streams == nullptr ? 0 : 1);
    aclError forcedRet = SkUtGetAclmdlRIGetStreamsRet(phase);
    if (forcedRet != ACL_SUCCESS) {
        return forcedRet;
    }
    if (numStreams == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    uint32_t streamNum = SkUtGetModelStreamNum();
    if (streams == nullptr) {
        *numStreams = streamNum;
        return ACL_ERROR_NONE;
    }

    for (uint32_t i = 0; i < streamNum; ++i) {
        streams[i] = reinterpret_cast<aclrtStream>(static_cast<uintptr_t>(i + 1));
    }
    *numStreams = streamNum;
    return ACL_ERROR_NONE;
}

// 获取任务
aclError aclrtStreamGetTasks(aclrtStream stream, aclrtTask *tasks, uint32_t *numTasks) {
    const int phase = (tasks == nullptr ? 0 : 1);
    aclError forcedRet = SkUtGetAclrtStreamGetTasksRet(phase);
    if (forcedRet != ACL_SUCCESS) {
        return forcedRet;
    }
    if (numTasks == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    uint32_t streamIdx = 0;
    if (stream != nullptr) {
        streamIdx = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(stream) - 1U);
    }
    uint32_t taskNum = SkUtGetStreamTaskNum(streamIdx);
    if (tasks == nullptr) {
        *numTasks = taskNum;
        return ACL_ERROR_NONE;
    }

    for (uint32_t i = 0; i < taskNum; ++i) {
        uintptr_t taskHandle = (static_cast<uintptr_t>(streamIdx + 1) << 32U) | static_cast<uintptr_t>(i + 1);
        tasks[i] = reinterpret_cast<aclrtTask>(taskHandle);
    }
    *numTasks = taskNum;
    return ACL_ERROR_NONE;
}

// 获取任务类型
aclError aclrtTaskGetType(aclrtTask task, aclrtTaskType *type) {
    aclError forcedRet = SkUtGetAclrtTaskGetTypeRet();
    if (forcedRet != ACL_SUCCESS) {
        return forcedRet;
    }
    if (type == nullptr || task == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    uintptr_t taskHandle = reinterpret_cast<uintptr_t>(task);
    uint32_t streamIdx = static_cast<uint32_t>((taskHandle >> 32U) & 0xFFFFFFFFU);
    uint32_t taskIdx = static_cast<uint32_t>(taskHandle & 0xFFFFFFFFU);
    if (streamIdx > 0) {
        streamIdx -= 1U;
    }
    if (taskIdx > 0) {
        taskIdx -= 1U;
    }
    *type = SkUtGetTaskType(streamIdx, taskIdx);
    return ACL_ERROR_NONE;
}

// 获取内核参数
aclError aclrtTaskGetKernelParams(aclrtTask task, aclrtTaskKernelParams *params) {
    if (params == nullptr || task == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    AclrtTaskInternal* internal = TaskToInternal(task);
    *params = internal->kernel;
    return ACL_ERROR_NONE;
}

// 设置内核参数
aclError aclrtTaskSetKernelParams(aclrtTask task, aclrtTaskKernelParams *params) {
    if (params == nullptr || task == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    AclrtTaskInternal* internal = TaskToInternal(task);
    internal->kernel = *params;
    return ACL_ERROR_NONE;
}

// 获取事件参数
aclError aclrtTaskGetEventParams(aclrtTask task, aclrtTaskEventParams *params) {
    if (params == nullptr || task == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    AclrtTaskInternal* internal = TaskToInternal(task);
    *params = internal->event;
    return ACL_ERROR_NONE;
}

// 设置事件参数
aclError aclrtTaskSetEventParams(aclrtTask task, aclrtTaskEventParams *params) {
    if (params == nullptr || task == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    AclrtTaskInternal* internal = TaskToInternal(task);
    internal->event = *params;
    return ACL_ERROR_NONE;
}

// 获取内存参数
aclError aclrtTaskGetMemValueParams(aclrtTask task, aclrtTaskMemValueParams *params) {
    if (params == nullptr || task == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    AclrtTaskInternal* internal = TaskToInternal(task);
    *params = internal->memValue;
    return ACL_ERROR_NONE;
}

// 设置内存参数
aclError aclrtTaskSetMemValueParams(aclrtTask task, aclrtTaskMemValueParams *params) {
    if (params == nullptr || task == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    AclrtTaskInternal* internal = TaskToInternal(task);
    internal->memValue = *params;
    return ACL_ERROR_NONE;
}

// 更新模型资源信息
aclError aclmdlRIUpdate(aclmdlRI modelRI) {
    (void)modelRI;
    aclError forcedRet = SkUtGetAclmdlRIUpdateRet();
    if (forcedRet != ACL_SUCCESS) {
        return forcedRet;
    }
    return ACL_ERROR_NONE;
}

// 获取设备
aclError aclrtGetDevice(int32_t *deviceId) {
    aclError forcedRet = SkUtGetAclrtGetDeviceRet();
    if (forcedRet != ACL_SUCCESS) {
        return forcedRet;
    }
    if (deviceId == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *deviceId = 0;
    return ACL_ERROR_NONE;
}

// 获取设备信息
aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t *value) {
    (void)deviceId;
    aclError forcedRet = SkUtGetAclrtGetDeviceInfoRet();
    if (forcedRet != ACL_SUCCESS) {
        return forcedRet;
    }
    if (value == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (attr == ACL_DEV_ATTR_CUBE_CORE_NUM) {
        *value = 32;
    } else if (attr == ACL_DEV_ATTR_VECTOR_CORE_NUM) {
        *value = 32;
    }
    return ACL_ERROR_NONE;
}

// 从二进制获取函数
aclError aclrtBinaryGetFunction(aclrtBinHandle binHdl, const char *funcName, aclrtFuncHandle *funcHdl) {
    if (funcName == nullptr || funcHdl == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (g_binaryGetFunctionNullHandle != 0) {
        *funcHdl = nullptr;
        return ACL_ERROR_NONE;
    }
    *funcHdl = reinterpret_cast<aclrtFuncHandle>(0x1000);
    return ACL_ERROR_NONE;
}

// 获取函数地址
aclError aclrtGetFunctionAddr(aclrtFuncHandle funcHdl, void **addrAicore, void **addrAiv) {
    if (addrAicore == nullptr || addrAiv == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_ERROR_NONE;
}

// 获取 kernel 参数句柄的内存大小
aclError aclrtKernelArgsGetHandleMemSize(aclrtFuncHandle funcHdl, size_t *memSize) {
    if (memSize == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *memSize = 1024;
    return ACL_ERROR_NONE;
}

// 获取 kernel 参数的内存大小
aclError aclrtKernelArgsGetMemSize(aclrtFuncHandle funcHdl, size_t argsSize, size_t *devArgsSize) {
    if (devArgsSize == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *devArgsSize = 1024;
    return ACL_ERROR_NONE;
}

// 通过用户内存初始化 kernel 参数
aclError aclrtKernelArgsInitByUserMem(aclrtFuncHandle funcHdl, aclrtArgsHandle argsHdl, void *devArgs, size_t devArgsSize) {
    if (argsHdl == nullptr || devArgs == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_ERROR_NONE;
}

// 添加占位符
aclError aclrtKernelArgsAppendPlaceHolder(aclrtArgsHandle argsHdl, aclrtParamHandle *phdl) {
    if (argsHdl == nullptr || phdl == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *phdl = reinterpret_cast<aclrtParamHandle>(0x2000);
    return ACL_ERROR_NONE;
}

// 获取占位符缓冲区
aclError aclrtKernelArgsGetPlaceHolderBuffer(aclrtArgsHandle argsHdl, aclrtParamHandle phdl,
                                              size_t bufferSize, void **buffer) {
    if (argsHdl == nullptr || buffer == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    static char placeholderBuffer[4096];
    *buffer = placeholderBuffer;
    return ACL_ERROR_NONE;
}

// 完成 kernel 参数设置
aclError aclrtKernelArgsFinalize(aclrtArgsHandle argsHdl) {
    if (argsHdl == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_ERROR_NONE;
}

// 获取函数名称
aclError aclrtGetFunctionName(aclrtFuncHandle funcHandle, uint32_t maxLen, char *name) {
    if (name == nullptr || maxLen == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    const char *funcName = "test_function";
    return ACL_ERROR_NONE;
}

// 内存复制
aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) {
    if (dst == nullptr || src == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (count > destMax) {
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_ERROR_NONE;
}

// 获取二进制设备地址
aclError aclrtBinaryGetDevAddress(aclrtBinHandle binHdl, void **devAddr, size_t *devSize) {
    if (devAddr == nullptr || devSize == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *devAddr = nullptr;
    *devSize = 0;
    return ACL_ERROR_NONE;
}

aclError aclrtGetFunctionAttribute(aclrtFuncHandle funcHandle, aclrtFuncAttribute attrType, int64_t *attrValue)
{
    if (attrType == ACL_FUNC_ATTR_KERNEL_TYPE) {
        *attrValue = 0; // 默认内核类型
    } else {
        *attrValue = 0;
    }
    return ACL_ERROR_NONE;
}

// Memory management
aclError aclrtMallocHost(void** hostPtr, size_t size) {
    if (hostPtr == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    // G.RES.02-CPP: 内存申请前，必须对申请内存大小进行合法性校验
    if (size == 0 || size > 1024 * 1024 * 1024) { // Limit to 1GB
        return ACL_ERROR_INVALID_PARAM;
    }
    *hostPtr = malloc(size);
    if (*hostPtr == nullptr) {
        return ACL_ERROR_FAILURE;
    }
    return ACL_ERROR_NONE;
}

aclError aclrtFreeHost(void* hostPtr) {
    if (hostPtr != nullptr) {
        free(hostPtr);
    }
    return ACL_ERROR_NONE;
}

// Exception handling
aclError aclrtSetExceptionInfoCallback(aclrtExceptionInfoCallbackFunc callback) {
    if (callback == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    // Stub: just return success, don't actually register the callback
    return ACL_ERROR_NONE;
}

aclError aclrtGetFuncHandleFromExceptionInfo(const aclrtExceptionInfo* exceptionInfo, aclrtFuncHandle* funcHandle) {
    if (exceptionInfo == nullptr || funcHandle == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    // Stub: return a fake function handle
    *funcHandle = reinterpret_cast<aclrtFuncHandle>(0x3000);
    return ACL_ERROR_NONE;
}

aclError aclrtGetArgsFromExceptionInfo(const aclrtExceptionInfo* exceptionInfo, void** args, uint32_t* argsLen) {
    if (exceptionInfo == nullptr || args == nullptr || argsLen == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    // Stub: return fake args pointer and length
    *args = nullptr;
    *argsLen = 0;
    return ACL_ERROR_NONE;
}

// ==================== RI Task API Stubs ====================

aclError aclmdlRITaskGetType(aclmdlRITask task, aclmdlRITaskType *type) {
    if (type == nullptr || task == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    AclmdlRITaskInternal* internal = RITaskToInternal(task);
    *type = internal->type;
    return ACL_ERROR_NONE;
}

aclError aclmdlRITaskGetParams(aclmdlRITask task, aclmdlRITaskParams* params) {
    if (params == nullptr || task == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    AclmdlRITaskInternal* internal = RITaskToInternal(task);
    *params = internal->params;
    return ACL_ERROR_NONE;
}

aclError aclmdlRITaskSetParams(aclmdlRITask task, aclmdlRITaskParams* params) {
    if (params == nullptr || task == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    AclmdlRITaskInternal* internal = RITaskToInternal(task);
    internal->params = *params;
    return ACL_ERROR_NONE;
}

aclError aclmdlRITaskDisable(aclmdlRITask task) {
    if (task == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_ERROR_NONE;
}

aclError aclmdlRITaskGetSeqId(aclmdlRITask task, uint32_t *id) {
    if (id == nullptr || task == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    AclmdlRITaskInternal* internal = RITaskToInternal(task);
    *id = internal->task_id;
    return ACL_ERROR_NONE;
}

aclError aclmdlRIGetTasksByStream(aclrtStream stream, aclmdlRITask *tasks, uint32_t *numTasks) {
    if (numTasks == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (tasks != nullptr) {
        *numTasks = 0;
    }
    return ACL_ERROR_NONE;
}

aclError aclrtFunctionGetBinary(aclrtFuncHandle funcHandle, aclrtBinHandle *binHandle) {
    if (binHandle == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *binHandle = nullptr;
    return ACL_ERROR_NONE;
}

} // extern "C"
