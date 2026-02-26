/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "kernel_launcher.h"

void KernelLauncher::Launch(const SkHostEntryInfo &entryInfo, SkDeviceEntryArgs *skEntryArgs,
                            aclrtStream stream, aclrtFuncHandle fhdl) {
    if (fhdl == nullptr) {
        printf("[sk error] entry func handle is null\n");
        return;
    }
    aclrtArgsHandle ahdl;
    aclrtParamHandle phdl;
    void *args_ptr = nullptr;

    // *************************** version 3 : 不可用，debug中
    const size_t MAX_HANDLE_MEM_SIZE = 1024 * 1024;  // 1MB
    const size_t MAX_ARGS_MEM_SIZE = 256 * 1024 * 1024;  // 64MB
    size_t memSize;
    CHECK_ACL(aclrtKernelArgsGetHandleMemSize(fhdl, &memSize));
    if (memSize == 0 || memSize > MAX_HANDLE_MEM_SIZE) {
        printf("[sk error] invalid memSize: %zu\n", memSize);
        return;
    }
    ahdl = (aclrtArgsHandle)malloc(memSize);
    if (ahdl == nullptr) {
        printf("[sk error] malloc memSize failed\n");
        return;
    }

    size_t devArgsSize;
    CHECK_ACL(aclrtKernelArgsGetMemSize(fhdl, skEntryArgs->skHeader.totalSize, &devArgsSize));
    if (devArgsSize == 0 || devArgsSize > MAX_ARGS_MEM_SIZE) {
        printf("[sk error] invalid devArgsSize: %zu\n", devArgsSize);
        free(ahdl);
        return;
    }
    void *devArgs = nullptr;
    devArgs = malloc(devArgsSize);
    if (devArgs == nullptr) {
        printf("[sk error] malloc devArgsSize failed\n");
        free(ahdl);
        return;
    }
    CHECK_ACL(aclrtKernelArgsInitByUserMem(fhdl, ahdl, devArgs, devArgsSize));

    CHECK_ACL(aclrtKernelArgsAppendPlaceHolder(ahdl, &phdl));
    CHECK_ACL(aclrtKernelArgsGetPlaceHolderBuffer(ahdl, phdl, skEntryArgs->skHeader.totalSize, (void**)&args_ptr));
    errno_t err = memcpy_s(args_ptr, skEntryArgs->skHeader.totalSize, skEntryArgs, skEntryArgs->skHeader.totalSize);
    if (err != 0) {
        printf("[sk error] memcpy_s failed\n");
        free(devArgs);
        free(ahdl);
        return;
    }
    CHECK_ACL(aclrtKernelArgsFinalize(ahdl));
    // 后续调用aclrtTaskSetXXXXXXXX方法之后，将host内存转移至rts buffer当中即可进行释放，目前在launcher之后释放


    // *************************** version 2 : 可用，但限制hostargs最多占用64KB
    // CHECK_ACL(aclrtKernelArgsInit(fhdl, &ahdl));

    // CHECK_ACL(aclrtKernelArgsAppendPlaceHolder(ahdl, &phdl));
    // CHECK_ACL(aclrtKernelArgsGetPlaceHolderBuffer(ahdl, phdl, skEntryArgs->skHeader.totalSize, (void**)&args_ptr));
    // memcpy_s(args_ptr, skEntryArgs->skHeader.totalSize, skEntryArgs, skEntryArgs->skHeader.totalSize);
    // CHECK_ACL(aclrtKernelArgsFinalize(ahdl));
    

    // *************************** version 1 : 不可用，会申请device内存
    // CHECK_ACL(aclrtKernelArgsInit(fhdl, &ahdl));
    // CHECK_ACL(aclrtMalloc(&args_ptr, skEntryArgs->skHeader.totalSize, ACL_MEM_MALLOC_HUGE_FIRST));
    // CHECK_ACL(aclrtmemcpy_s(args_ptr, skEntryArgs->skHeader.totalSize, skEntryArgs, skEntryArgs->skHeader.totalSize, ACL_memcpy_s_HOST_TO_DEVICE));
    // CHECK_ACL(aclrtKernelArgsAppend(ahdl, &args_ptr, sizeof(void *), &phdl));

    // CHECK_ACL(aclrtKernelArgsFinalize(ahdl));
    // printf("[sk info] skEntryArgs addr on device : %p \n", args_ptr);
    
    CHECK_ACL(aclrtLaunchKernelWithConfig(fhdl, entryInfo.blockDim, stream, nullptr, ahdl, nullptr));
    // CHECK_ACL(aclrtFree(args_ptr)); // version 1
    free(devArgs); // version 3
    free(ahdl); // version 3
}

