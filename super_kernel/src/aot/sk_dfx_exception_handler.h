/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file sk_dfx_exception_handler.h
 * \brief
 */

#ifndef __SK_DFX_EXCEPTION_HANDLER_H__
#define __SK_DFX_EXCEPTION_HANDLER_H__

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string.h>
#include "acl/acl.h"
#include "sk_common.h"
#include "runtime/base.h"

struct ExceptionRegInfo {
    uint32_t coreNum;
    rtExceptionErrRegInfo_t *errRegInfo;
};

struct KernelFuncName {
    std::string name;
};

class SuperKernelExceptionHandler {
public:
    SuperKernelExceptionHandler();
    ~SuperKernelExceptionHandler() = default;

    void HandleException(aclrtExceptionInfo *exceptionInfo);

private:
    bool IsSuperKernelException(aclrtExceptionInfo *exceptionInfo);
    bool ExtractSkEntryArgs(aclrtExceptionInfo *exceptionInfo);
    bool ExtractSkDeviceEntryArgsPtr(aclrtExceptionInfo *exceptionInfo);
    bool CopySkDeviceEntryArgsToHost();
    bool ExtractSkHeaderInfo();
    bool ExtractTaskQueue();
    void ExtractAndPrintSkInfo();
    void PrintSkHeaderInfo() const;
    void PrintTaskQueue() const;
    void PrintCounterInfo() const;
    void PrintDfxInfo() const;
    bool ParseAndPrintSubKernelSymbols(aclrtExceptionInfo *exceptionInfo);
    KernelFuncName GetOrLoadKernelSymbols(uint32_t opId);
    void IdentifyErrorNodeByPC(uint32_t coreId, rtCoreType_t coreType, uint64_t startPC, uint64_t currentPC);
    void PrintCoreSymbols(uint32_t coreId, rtCoreType_t coreType, uint64_t startPC, uint64_t currentPC);
    void PrintSymbolByCoreId(uint32_t coreId, rtCoreType_t coreType, uint64_t startPC, uint64_t currentPC,
                            const KernelFuncName &kernelFuncName);
    void PrintAllCoreSymbols();
    void FreeResources();

    bool StartsWith(const char* source, const char* prefix);
    bool GetExceptionRegInfo(const aclrtExceptionInfo &exception, ExceptionRegInfo &exceptionRegInfo);

    uint32_t aicoreNums;
    std::map<uint32_t, KernelFuncName> opSymbolCache;  // Operator function name cache: opId -> KernelFuncName

    void *skDeviceEntryArgsDev;      // GM address directly obtained from aclrtGetArgsFromExceptionInfo
    uint32_t skDeviceEntryArgsPtrLen;

    SkDeviceEntryArgs *skDeviceEntryArgsHost;  // Host side, complete SkDeviceEntryArgs data copied at once

    SkHeaderInfo *skHeaderInfoHost;  // Points to skHeaderInfo in skDeviceEntryArgsHost

    void *aicTaskQueDevPtr;          // Reserved for recording device address (for debugging)
    void *aivTaskQueDevPtr;          // Reserved for recording device address (for debugging)
    uint32_t aicTaskCnt;
    uint32_t aivTaskCnt;
    bool hasOpTrace_;                // Whether sk_entry name contains "op_trace"

    aclError CheckError(aclError ret, const char *errorMsg);
};

void SuperKernelExceptionCallBackFunc(aclrtExceptionInfo *exceptionInfo);

#endif // __SK_DFX_EXCEPTION_HANDLER_H__
