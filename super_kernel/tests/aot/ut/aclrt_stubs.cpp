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

#include "rt_sk_intf.h"
#include <cstring>
#include <cstdio>

extern "C" {

// Error codes - 与 CHECK_ACL 宏中使用的保持一致
#ifndef ACL_ERROR_NONE
#define ACL_ERROR_NONE 0
#endif
#define ACL_ERROR_INVALID_PARAM 100001

// 获取流
aclError aclmdlRIGetStreams(aclmdlRI modelRI, aclrtStream *streams, uint32_t *numStreams) {
    if (numStreams == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    // 在单元测试中,这个函数应该由测试用例设置预期行为
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
    // 在单元测试中,这个函数应该由测试用例设置预期行为
    if (tasks != nullptr) {
        *numTasks = 0;
    }
    return ACL_ERROR_NONE;
}

// 获取任务类型
aclError aclrtTaskGetType(aclrtTask task, aclrtTaskType *type) {
    if (type == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *type = task.type;
    return ACL_ERROR_NONE;
}

// 获取内核参数
aclError aclrtTaskGetKernelParams(aclrtTask task, aclrtTaskKernelParams *params) {
    if (params == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *params = task.kernel;
    return ACL_ERROR_NONE;
}

// 设置内核参数
aclError aclrtTaskSetKernelParams(aclrtTask task, aclrtTaskKernelParams *params) {
    if (params == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    // 注意:这里需要修改 task 的内容,但由于 task 是按值传递的,
    // 在真实实现中,task 应该是指针类型
    // 当前实现中只是返回成功,实际修改需要在调用处处理
    return ACL_ERROR_NONE;
}

// 获取事件参数
aclError aclrtTaskGetEventParams(aclrtTask task, aclrtTaskEventParams *params) {
    if (params == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *params = task.event;
    return ACL_ERROR_NONE;
}

// 设置事件参数
aclError aclrtTaskSetEventParams(aclrtTask task, aclrtTaskEventParams *params) {
    if (params == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    // 注意:这里需要修改 task 的内容
    return ACL_ERROR_NONE;
}

// 获取内存参数
aclError aclrtTaskGetMemValueParams(aclrtTask task, aclrtMemValueParams *params) {
    if (params == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *params = task.memValue;
    return ACL_ERROR_NONE;
}

// 设置内存参数
aclError aclrtTaskSetMemValueParams(aclrtTask task, aclrtMemValueParams *params) {
    if (params == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    // 注意:这里需要修改 task 的内容
    return ACL_ERROR_NONE;
}

// 更新模型资源信息
aclError aclmdlRIUpdate(aclmdlRI modelRI) {
    // 在单元测试中,这个函数应该由测试用例设置预期行为
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
        *value = 32; // 默认值
    } else if (attr == ACL_DEV_ATTR_VECTOR_CORE_NUM) {
        *value = 32; // 默认值
    }
    return ACL_ERROR_NONE;
}

// 从二进制获取函数
aclError aclrtBinaryGetFunction(aclrtBinHandle binHdl, const char *funcName, aclrtFuncHandle *funcHdl) {
    if (funcName == nullptr || funcHdl == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *funcHdl = reinterpret_cast<aclrtFuncHandle>(0x1000); // 返回一个假的句柄
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
    *memSize = 1024; // 默认大小
    return ACL_ERROR_NONE;
}

// 获取 kernel 参数的内存大小
aclError aclrtKernelArgsGetMemSize(aclrtFuncHandle funcHdl, size_t argsSize, size_t *devArgsSize) {
    if (devArgsSize == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *devArgsSize = 1024; // 默认大小
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
aclError aclrtKernelArgsAppendPlaceHolder(aclrtArgsHandle argsHdl, void **phdl) {
    if (argsHdl == nullptr || phdl == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
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

// Scope begin stub
void sk_scope_kernel_begin_do(const char *scopeName, void *args) {
    // Stub implementation for unit tests
}

// Scope end stub
void sk_scope_kernel_end_do(const char *scopeName, void *args) {
    // Stub implementation for unit tests
}

} // extern "C"
