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
#include "sk_scope_kernel_types.h"
#include <cstring>
#include <cstdio>

extern "C" {

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

// Helper to convert void* to internal structure
static inline AclrtTaskInternal* TaskToInternal(aclrtTask task) {
    return reinterpret_cast<AclrtTaskInternal*>(task);
}

// 获取流
aclError aclmdlRIGetStreams(aclmdlRI modelRI, aclrtStream *streams, uint32_t *numStreams) {
    if (numStreams == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (streams != nullptr) {
        *numStreams = 0;
    }
    return ACL_ERROR_NONE;
}

// 获取任务
aclError aclrtStreamGetTasks(aclrtStream stream, aclrtTask *tasks, uint32_t *numTasks) {
    if (numTasks == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (tasks != nullptr) {
        *numTasks = 0;
    }
    return ACL_ERROR_NONE;
}

// 获取任务类型
aclError aclrtTaskGetType(aclrtTask task, aclrtTaskType *type) {
    if (type == nullptr || task == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    AclrtTaskInternal* internal = TaskToInternal(task);
    *type = internal->type;
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
    return ACL_ERROR_NONE;
}

// 获取设备
aclError aclrtGetDevice(int32_t *deviceId) {
    if (deviceId == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *deviceId = 0;
    return ACL_ERROR_NONE;
}

// 获取设备信息
aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t *value) {
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

// Scope begin stub - signature matches sk_scope_launch.cpp declaration
void sk_scope_kernel_begin_do(void* stream, ScopeKernelArgs args) {
    (void)stream;
    (void)args;
}

// Scope end stub - signature matches sk_scope_launch.cpp declaration
void sk_scope_kernel_end_do(void* stream, ScopeKernelArgs args) {
    (void)stream;
    (void)args;
}

} // extern "C"
