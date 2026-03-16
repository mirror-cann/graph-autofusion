/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include "mockcpp/mockcpp.hpp"
#include <memory>
#include <cstring>
#include <map>
#include <unordered_map>
#include <cstdlib>
#include <cstdio>
#include "securec.h"

#define private public
#define protected public
#include "sk_dfx_exception_handler.h"
#include "sk_common.h"

class SkDfxExceptionHandlerTest : public testing::Test {
protected:
    void SetUp() override {
        handler = std::make_unique<SuperKernelExceptionHandler>();
    }

    void TearDown() override {
        GlobalMockObject::verify();
        handler.reset();
    }

    std::unique_ptr<SuperKernelExceptionHandler> handler;
};

// ==================== Helper Functions Tests ====================

TEST_F(SkDfxExceptionHandlerTest, StartsWith_ValidPrefix)
{
    EXPECT_TRUE(handler->StartsWith("sk_entry_test", "sk_entry"));
    EXPECT_TRUE(handler->StartsWith("sk_entry_test", ""));
    EXPECT_TRUE(handler->StartsWith("test", ""));
}

TEST_F(SkDfxExceptionHandlerTest, StartsWith_InvalidPrefix)
{
    EXPECT_FALSE(handler->StartsWith("test_function", "sk_entry"));
    EXPECT_FALSE(handler->StartsWith("SK_ENTRY", "sk_entry"));
    EXPECT_FALSE(handler->StartsWith("", "prefix"));
}

TEST_F(SkDfxExceptionHandlerTest, StartsWith_NullPointers)
{
    EXPECT_FALSE(handler->StartsWith(nullptr, "prefix"));
    EXPECT_FALSE(handler->StartsWith("source", nullptr));
    EXPECT_FALSE(handler->StartsWith(nullptr, nullptr));
}

TEST_F(SkDfxExceptionHandlerTest, CheckError_Success)
{
    aclError ret = handler->CheckError(ACL_SUCCESS, "Test operation");
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(SkDfxExceptionHandlerTest, CheckError_Failure)
{
    aclError ret = handler->CheckError(ACL_ERROR_FAILURE, "Test operation");
    EXPECT_EQ(ret, ACL_ERROR_FAILURE);
}

// ==================== HandleException Tests ====================

TEST_F(SkDfxExceptionHandlerTest, HandleException_NullExceptionInfo)
{
    handler->HandleException(nullptr);
    // Should not crash, just return early
    SUCCEED();
}

// Mock for aclrtGetFuncHandleFromExceptionInfo
aclError Fake_aclrtGetFuncHandleFromExceptionInfo_Success(const aclrtExceptionInfo* exceptionInfo, aclrtFuncHandle* funcHandle)
{
    (void)exceptionInfo;
    *funcHandle = reinterpret_cast<aclrtFuncHandle>(0x1000);
    return ACL_SUCCESS;
}

// Mock for aclrtGetFunctionName - returns sk_entry function
aclError Fake_aclrtGetFunctionName_sk_entry(void* funcHandle, uint32_t maxLen, char* name)
{
    (void)funcHandle;
    snprintf_s(name, maxLen, maxLen, "%s", "sk_entry");
    return ACL_SUCCESS;
}

// Mock for aclrtGetFunctionName - returns non-sk_entry function
aclError Fake_aclrtGetFunctionName_other(void* funcHandle, uint32_t maxLen, char* name)
{
    (void)funcHandle;
    snprintf_s(name, maxLen, maxLen, "%s", "some_other_function");
    return ACL_SUCCESS;
}

// Mock for aclrtGetArgsFromExceptionInfo
aclError Fake_aclrtGetArgsFromExceptionInfo_Success(const aclrtExceptionInfo* exceptionInfo, void** args, uint32_t* argsLen)
{
    (void)exceptionInfo;
    *args = reinterpret_cast<void*>(0x3000);  // Directly return GM address
    *argsLen = 8;
    return ACL_SUCCESS;
}

// Mock for aclrtMallocHost
aclError Fake_aclrtMallocHost_Success(void** hostPtr, size_t size)
{
    if (hostPtr == nullptr || size == 0 || size > 1024 * 1024) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *hostPtr = malloc(size);
    return (*hostPtr != nullptr) ? ACL_SUCCESS : ACL_ERROR_FAILURE;
}

// Mock for aclrtMemcpy
aclError Fake_aclrtMemcpy_Success(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind)
{
    (void)destMax;
    (void)src;
    (void)count;
    (void)kind;
    if (dst != nullptr && src != nullptr && count > 0 && count <= destMax) {
        memcpy_s(dst, destMax, src, count);
    }
    return ACL_SUCCESS;
}

// Mock for aclrtFreeHost
aclError Fake_aclrtFreeHost_Success(void* hostPtr)
{
    if (hostPtr != nullptr) {
        free(hostPtr);
    }
    return ACL_SUCCESS;
}

// Mock for rtGetExceptionRegInfo
int Fake_rtGetExceptionRegInfo_Success(const void* exception, void** errRegInfo, uint32_t* coreNum)
{
    (void)exception;
    (void)errRegInfo;
    *coreNum = 0; // No cores with errors
    return 0;
}

// Mock for aclrtGetArgsFromExceptionInfo - returns short length
aclError Fake_aclrtGetArgsFromExceptionInfo_Short(const aclrtExceptionInfo* exceptionInfo, void** args, uint32_t* argsLen)
{
    (void)exceptionInfo;
    *args = reinterpret_cast<void*>(0x2000);
    *argsLen = 4; // Less than 8
    return ACL_SUCCESS;
}

TEST_F(SkDfxExceptionHandlerTest, IsSuperKernelException_Success)
{
    aclrtExceptionInfo* exceptionInfo = reinterpret_cast<aclrtExceptionInfo*>(0x500);

    MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
    MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));

    bool result = handler->IsSuperKernelException(exceptionInfo);
    EXPECT_TRUE(result);
}

TEST_F(SkDfxExceptionHandlerTest, IsSuperKernelException_NotSkEntry)
{
    aclrtExceptionInfo* exceptionInfo = reinterpret_cast<aclrtExceptionInfo*>(0x500);

    MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
    MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_other));

    bool result = handler->IsSuperKernelException(exceptionInfo);
    EXPECT_FALSE(result);
}

TEST_F(SkDfxExceptionHandlerTest, ExtractTaskQueue_EmptyOffsets)
{
    // Setup minimal SkHeaderInfo
    uint32_t skDeviceEntryArgsSize = sizeof(SkHeaderInfo);
    SkHeaderInfo headerInfo{0, 0, 0, 0, 0, 0, skDeviceEntryArgsSize};
    handler->skHeaderInfoHost = &headerInfo;

    bool result = handler->ExtractTaskQueue();
    EXPECT_TRUE(result);
    EXPECT_EQ(handler->aicTaskCnt, 0);
    EXPECT_EQ(handler->aivTaskCnt, 0);
}

TEST_F(SkDfxExceptionHandlerTest, ExtractTaskQueue_WithTaskCounts)
{
    // Setup test data
    uint32_t skDeviceEntryArgsSize = sizeof(SkHeaderInfo);
    SkHeaderInfo headerInfo;
    headerInfo.aicQueOffset = skDeviceEntryArgsSize;
    headerInfo.aivQueOffset = skDeviceEntryArgsSize + sizeof(TaskQue) + sizeof(TaskInfo);
    headerInfo.counterOffset = 0;
    headerInfo.wsOffset = 0;
    headerInfo.dfxOffset = 0;
    headerInfo.nodeCnt = 0;
    headerInfo.totalSize = 1024;

    uint8_t buffer[1024] = {0};
    SkDeviceEntryArgs* deviceArgs = reinterpret_cast<SkDeviceEntryArgs*>(buffer);
    deviceArgs->skHeader = headerInfo;

    TaskQue* aicQueue = reinterpret_cast<TaskQue*>(buffer + headerInfo.aicQueOffset);
    aicQueue->taskCnt = 2;
    aicQueue->cap = 10;

    TaskQue* aivQueue = reinterpret_cast<TaskQue*>(buffer + headerInfo.aivQueOffset);
    aivQueue->taskCnt = 3;
    aivQueue->cap = 10;

    handler->skDeviceEntryArgsHost = deviceArgs;
    handler->skHeaderInfoHost = &headerInfo;

    bool result = handler->ExtractTaskQueue();
    EXPECT_TRUE(result);
    EXPECT_EQ(handler->aicTaskCnt, 2);
    EXPECT_EQ(handler->aivTaskCnt, 3);
}

TEST_F(SkDfxExceptionHandlerTest, FreeResources_AllNull)
{
    handler->FreeResources();
    // Should not crash with all null pointers
    SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, FreeResources_WithAllocatedMemory)
{
    // Allocate some memory to free
    const size_t allocSize = 1024;

    if (allocSize == 0 || allocSize > 1024 * 1024) {
        return;
    }
    handler->skDeviceEntryArgsHost = reinterpret_cast<SkDeviceEntryArgs*>(malloc(allocSize));

    EXPECT_NE(handler->skDeviceEntryArgsHost, nullptr);

    // Set skHeaderInfoHost to point inside skDeviceEntryArgsHost (simulating normal flow)
    handler->skHeaderInfoHost = &(handler->skDeviceEntryArgsHost->skHeader);

    // Mock aclrtFreeHost to track calls
    MOCKER(aclrtFreeHost).stubs().will(invoke(Fake_aclrtFreeHost_Success));

    handler->FreeResources();

    EXPECT_EQ(handler->skDeviceEntryArgsHost, nullptr);
    EXPECT_EQ(handler->skHeaderInfoHost, nullptr);
}

TEST_F(SkDfxExceptionHandlerTest, GetOrLoadKernelSymbols_CacheHit)
{
    // Setup cache
    KernelFuncName kernelFuncName{"test_function"};
    handler->opSymbolCache[1] = kernelFuncName;

    KernelFuncName result = handler->GetOrLoadKernelSymbols(1);

    EXPECT_EQ(result.name, "test_function");
}

TEST_F(SkDfxExceptionHandlerTest, PrintCoreSymbols_CoreIdExceedsAicoreNums)
{
    uint32_t coreId = 100; // Exceeds default aicoreNums=75

    handler->PrintCoreSymbols(coreId, RT_CORE_TYPE_AIC, 0, 0);

    // Should not crash, just log error and return
    SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintSymbolByCoreId_EmptySymbols)
{
    KernelFuncName emptyKernelFuncName{""};

    handler->PrintSymbolByCoreId(0, RT_CORE_TYPE_AIC, 0, 0, emptyKernelFuncName);

    // Should not crash, just log that no function name is available
    SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintSymbolByCoreId_WithSymbols)
{
    KernelFuncName kernelFuncName{"test_kernel"};

    handler->PrintSymbolByCoreId(0, RT_CORE_TYPE_AIC, 0x1000, 0x2000, kernelFuncName);

    // Should not crash and print function name
    SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintAllCoreSymbols_DefaultAicoreNums)
{
    // Setup minimal required structures
    uint32_t skDeviceEntryArgsSize = sizeof(SkHeaderInfo);
    SkHeaderInfo headerInfo{0, 0, 0, 0, 0, 0, skDeviceEntryArgsSize};
    handler->skHeaderInfoHost = &headerInfo;

    const size_t allocSize = 1024;
    if (allocSize == 0 || allocSize > 1024 * 1024) {
        return;
    }
    handler->skDeviceEntryArgsHost = reinterpret_cast<SkDeviceEntryArgs*>(malloc(allocSize));

    handler->PrintAllCoreSymbols();

    // Should iterate through all 75 cores without crashing
    free(handler->skDeviceEntryArgsHost);
    SUCCEED();
}

// ==================== ExtractSkDeviceEntryArgsPtr Tests ====================

TEST_F(SkDfxExceptionHandlerTest, ExtractSkDeviceEntryArgsPtr_Success)
{
    aclrtExceptionInfo* exceptionInfo = reinterpret_cast<aclrtExceptionInfo*>(0x500);

    MOCKER(aclrtGetArgsFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetArgsFromExceptionInfo_Success));

    bool result = handler->ExtractSkDeviceEntryArgsPtr(exceptionInfo);
    EXPECT_TRUE(result);
    EXPECT_EQ(handler->skDeviceEntryArgsDev, reinterpret_cast<void*>(0x3000));
}

TEST_F(SkDfxExceptionHandlerTest, ExtractSkDeviceEntryArgsPtr_TooShort)
{
    aclrtExceptionInfo* exceptionInfo = reinterpret_cast<aclrtExceptionInfo*>(0x500);

    MOCKER(aclrtGetArgsFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetArgsFromExceptionInfo_Short));

    bool result = handler->ExtractSkDeviceEntryArgsPtr(exceptionInfo);
    EXPECT_FALSE(result);
}

// ==================== SuperKernelExceptionCallBackFunc Tests ====================

TEST_F(SkDfxExceptionHandlerTest, SuperKernelExceptionCallBackFunc_Null)
{
    SuperKernelExceptionCallBackFunc(nullptr);
    // Should not crash
    SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, SuperKernelExceptionCallBackFunc_WithValidInfo)
{
    aclrtExceptionInfo* exceptionInfo = reinterpret_cast<aclrtExceptionInfo*>(0x500);

    MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
    MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_other)); // Not sk_entry, should skip

    SuperKernelExceptionCallBackFunc(exceptionInfo);

    // Should not crash
    SUCCEED();
}
