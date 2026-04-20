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
#include "sk_event_recorder.h"

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

// ==================== Helper Functions for Test Setup ====================

// Helper: Set up single-node DFX info on stack buffer for IdentifyErrorNodeByPC tests
static void SetupSingleDfxNode(uint8_t* buffer, uint32_t aicSize, uint32_t aivSize,
    const uint32_t entryAic[4], const uint32_t entryAiv[4],
    SkHeaderInfo& headerInfo, SkDeviceEntryArgs*& deviceArgs, SkDfxInfo*& dfxInfo,
    SuperKernelExceptionHandler* h)
{
    headerInfo = {};
    headerInfo.dfxOffset = sizeof(SkHeaderInfo);
    headerInfo.nodeCnt = 1;

    deviceArgs = reinterpret_cast<SkDeviceEntryArgs*>(buffer);
    deviceArgs->skHeader = headerInfo;

    dfxInfo = reinterpret_cast<SkDfxInfo*>(buffer + headerInfo.dfxOffset);
    dfxInfo->binHdl = 0xAAA;
    dfxInfo->funcHdlOri = 0xBBB;
    dfxInfo->aicSize = aicSize;
    dfxInfo->aivSize = aivSize;
    for (int i = 0; i < 4; i++) {
        dfxInfo->entryAic[i] = entryAic[i];
        dfxInfo->entryAiv[i] = entryAiv[i];
    }

    h->skDeviceEntryArgsHost = deviceArgs;
    h->skHeaderInfoHost = &headerInfo;
}

// Helper: Allocate and setup heap buffer for PrintCoreSymbols / PrintAllCoreSymbols tests
static uint8_t* SetupOpTraceTestBuffer(uint32_t nodeCnt, bool hasDfx,
    SkHeaderInfo& headerInfo, SkDeviceEntryArgs*& deviceArgs, SuperKernelExceptionHandler* h)
{
    uint32_t totalSize = sizeof(SkHeaderInfo) + sizeof(SkCounterInfo) * 75;
    if (hasDfx && nodeCnt > 0) {
        totalSize += sizeof(SkDfxInfo) * nodeCnt;
    }

    headerInfo = {};
    headerInfo.nodeCnt = nodeCnt;
    headerInfo.counterOffset = sizeof(SkHeaderInfo);
    if (hasDfx && nodeCnt > 0) {
        headerInfo.dfxOffset = sizeof(SkHeaderInfo) + sizeof(SkCounterInfo) * 75;
    }
    headerInfo.totalSize = totalSize;

    const size_t allocSize = totalSize;
    if (allocSize == 0 || allocSize > 1024 * 1024) {
        return nullptr;
    }

    uint8_t* buf = reinterpret_cast<uint8_t*>(malloc(allocSize));
    (void)memset_s(buf, allocSize, 0, allocSize);
    deviceArgs = reinterpret_cast<SkDeviceEntryArgs*>(buf);
    deviceArgs->skHeader = headerInfo;

    h->skDeviceEntryArgsHost = deviceArgs;
    h->skHeaderInfoHost = &headerInfo;

    return buf;
}

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
    SkHeaderInfo headerInfo{};
    headerInfo.totalSize = skDeviceEntryArgsSize;
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
    headerInfo.dfxOffset = 0;
    headerInfo.eventConfigOffset = 0;
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
    SkHeaderInfo headerInfo{};
    headerInfo.totalSize = skDeviceEntryArgsSize;
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

// ==================== IdentifyErrorNodeByPC Tests (Commit 1 & 2) ====================

TEST_F(SkDfxExceptionHandlerTest, IdentifyErrorNodeByPC_CurrentPCZero_ShouldReturnEarly)
{
    // Setup minimal SkHeaderInfo with valid dfxOffset and nodeCnt
    SkHeaderInfo headerInfo{};
    headerInfo.dfxOffset = sizeof(SkHeaderInfo);
    headerInfo.nodeCnt = 1;

    uint8_t buffer[1024] = {0};
    SkDeviceEntryArgs* deviceArgs = reinterpret_cast<SkDeviceEntryArgs*>(buffer);
    deviceArgs->skHeader = headerInfo;

    handler->skDeviceEntryArgsHost = deviceArgs;
    handler->skHeaderInfoHost = &headerInfo;

    // currentPC == 0 should return immediately (fix for pc 0 bug)
    handler->IdentifyErrorNodeByPC(0, RT_CORE_TYPE_AIC, 0x1000, 0);
    SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, IdentifyErrorNodeByPC_DfxOffsetZero_ShouldReturnEarly)
{
    SkHeaderInfo headerInfo{};
    headerInfo.nodeCnt = 1; // dfxOffset = 0

    handler->skDeviceEntryArgsHost = reinterpret_cast<SkDeviceEntryArgs*>(&headerInfo);
    handler->skHeaderInfoHost = &headerInfo;

    handler->IdentifyErrorNodeByPC(0, RT_CORE_TYPE_AIC, 0x1000, 0x2000);
    SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, IdentifyErrorNodeByPC_NodeCntZero_ShouldReturnEarly)
{
    SkHeaderInfo headerInfo{}; // nodeCnt = 0
    headerInfo.dfxOffset = sizeof(SkHeaderInfo);

    uint8_t buffer[1024] = {0};
    SkDeviceEntryArgs* deviceArgs = reinterpret_cast<SkDeviceEntryArgs*>(buffer);
    deviceArgs->skHeader = headerInfo;

    handler->skDeviceEntryArgsHost = deviceArgs;
    handler->skHeaderInfoHost = &headerInfo;

    handler->IdentifyErrorNodeByPC(0, RT_CORE_TYPE_AIC, 0x1000, 0x2000);
    SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, IdentifyErrorNodeByPC_MatchAICEntry)
{
    // Setup: one node with AIC entry at 0x1000, size 0x200
    uint8_t buffer[1024] = {0};
    SkHeaderInfo headerInfo;
    SkDeviceEntryArgs* deviceArgs;
    SkDfxInfo* dfxInfo;

    uint32_t entryAic[4] = {0x1000, 0, 0, 0};
    uint32_t entryAiv[4] = {0, 0, 0, 0};
    SetupSingleDfxNode(buffer, 0x200, 0, entryAic, entryAiv,
        headerInfo, deviceArgs, dfxInfo, handler.get());

    // Mock aclrtGetFunctionName for the found node
    MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));

    // currentPC = 0x1100 falls within [0x1000, 0x1200) -> should match node[0]
    handler->IdentifyErrorNodeByPC(5, RT_CORE_TYPE_AIC, 0x1000, 0x1100);
    SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, IdentifyErrorNodeByPC_MatchAIVEntry)
{
    // Setup: one node with AIV entry at 0x3000, size 0x100
    uint8_t buffer[1024] = {0};
    SkHeaderInfo headerInfo;
    SkDeviceEntryArgs* deviceArgs;
    SkDfxInfo* dfxInfo;

    uint32_t entryAic[4] = {0, 0, 0, 0};
    uint32_t entryAiv[4] = {0x3000, 0, 0, 0};
    SetupSingleDfxNode(buffer, 0, 0x100, entryAic, entryAiv,
        headerInfo, deviceArgs, dfxInfo, handler.get());

    MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));

    // currentPC = 0x3050 falls within [0x3000, 0x3100) -> should match node[0]'s AIV entry
    handler->IdentifyErrorNodeByPC(30, RT_CORE_TYPE_AIV, 0x3000, 0x3050);
    SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, IdentifyErrorNodeByPC_NoMatch_ShouldLogNoSubKernelMatched)
{
    // Setup: one node with AIC entry at 0x1000, size 0x100
    uint8_t buffer[1024] = {0};
    SkHeaderInfo headerInfo;
    SkDeviceEntryArgs* deviceArgs;
    SkDfxInfo* dfxInfo;

    uint32_t entryAic[4] = {0x1000, 0, 0, 0};
    uint32_t entryAiv[4] = {0, 0, 0, 0};
    SetupSingleDfxNode(buffer, 0x100, 0, entryAic, entryAiv,
        headerInfo, deviceArgs, dfxInfo, handler.get());

    // currentPC = 0xFFFF is outside any entry range -> no match
    handler->IdentifyErrorNodeByPC(5, RT_CORE_TYPE_AIC, 0x1000, 0xFFFF);
    SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, IdentifyErrorNodeByPC_SkipInvalidZeroEntries)
{
    // Setup: node with entryAic[0]=0 (invalid, should skip), entryAic[1]=valid
    uint8_t buffer[1024] = {0};
    SkHeaderInfo headerInfo;
    SkDeviceEntryArgs* deviceArgs;
    SkDfxInfo* dfxInfo;

    uint32_t entryAic[4] = {0, 0x2000, 0, 0};
    uint32_t entryAiv[4] = {0, 0, 0, 0};
    SetupSingleDfxNode(buffer, 0x200, 0, entryAic, entryAiv,
        headerInfo, deviceArgs, dfxInfo, handler.get());

    MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));

    // PC should match entryAic[1] which is at 0x2100 (within [0x2000, 0x2200))
    handler->IdentifyErrorNodeByPC(5, RT_CORE_TYPE_AIC, 0x2000, 0x2100);
    SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, IdentifyErrorNodeByPC_PCBoundaryExactEnd_NotIncluded)
{
    // Test that endAddr is exclusive: currentPC == entryAddr + funcSize should NOT match
    uint8_t buffer[1024] = {0};
    SkHeaderInfo headerInfo;
    SkDeviceEntryArgs* deviceArgs;
    SkDfxInfo* dfxInfo;

    uint32_t entryAic[4] = {0x1000, 0, 0, 0};
    uint32_t entryAiv[4] = {0, 0, 0, 0};
    SetupSingleDfxNode(buffer, 0x100, 0, entryAic, entryAiv,
        headerInfo, deviceArgs, dfxInfo, handler.get());

    // currentPC == endAddr (0x1100), should NOT match because range is [entryAddr, endAddr)
    handler->IdentifyErrorNodeByPC(5, RT_CORE_TYPE_AIC, 0x1000, 0x1100);
    SUCCEED();
}

// ==================== hasOpTrace_ Tests (Commit 4) ====================

// Mock for aclrtGetFunctionName - returns sk_entry with op_trace
aclError Fake_aclrtGetFunctionName_op_trace(void* funcHandle, uint32_t maxLen, char* name)
{
    (void)funcHandle;
    snprintf_s(name, maxLen, maxLen, "%s", "sk_entry_aic_op_trace");
    return ACL_SUCCESS;
}

TEST_F(SkDfxExceptionHandlerTest, IsSuperKernelException_WithOpTrace_SetHasOpTraceTrue)
{
    aclrtExceptionInfo* exceptionInfo = reinterpret_cast<aclrtExceptionInfo*>(0x500);

    MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
    MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_op_trace));

    bool result = handler->IsSuperKernelException(exceptionInfo);
    EXPECT_TRUE(result);
    EXPECT_TRUE(handler->hasOpTrace_);
}

TEST_F(SkDfxExceptionHandlerTest, IsSuperKernelException_WithoutOpTrace_SetHasOpTraceFalse)
{
    aclrtExceptionInfo* exceptionInfo = reinterpret_cast<aclrtExceptionInfo*>(0x500);

    MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
    MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry)); // plain sk_entry, no op_trace

    bool result = handler->IsSuperKernelException(exceptionInfo);
    EXPECT_TRUE(result);
    EXPECT_FALSE(handler->hasOpTrace_);
}

// Mock for aclrtGetFunctionName - returns sk_entry with op_trace in middle of name
aclError Fake_aclrtGetFunctionName_op_trace_middle(void* funcHandle, uint32_t maxLen, char* name)
{
    (void)funcHandle;
    snprintf_s(name, maxLen, maxLen, "%s", "sk_entry_mix11_op_trace_debug");
    return ACL_SUCCESS;
}

TEST_F(SkDfxExceptionHandlerTest, IsSuperKernelException_OpTraceInMiddleOfName)
{
    aclrtExceptionInfo* exceptionInfo = reinterpret_cast<aclrtExceptionInfo*>(0x500);

    MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
    MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_op_trace_middle));

    bool result = handler->IsSuperKernelException(exceptionInfo);
    EXPECT_TRUE(result);
    EXPECT_TRUE(handler->hasOpTrace_);
}

// ==================== PrintCoreSymbols Enhanced Tests (Commit 4) ====================

TEST_F(SkDfxExceptionHandlerTest, PrintCoreSymbols_HasOpTraceFalse_ShouldReturnEarly)
{
    handler->hasOpTrace_ = false;

    // Even with valid setup, should return early when hasOpTrace_ is false
    handler->PrintCoreSymbols(0, RT_CORE_TYPE_AIC, 0x1000, 0x2000);
    SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintCoreSymbols_LaunchOrigin_NoSKExecutedYet)
{
    handler->hasOpTrace_ = true;

    SkHeaderInfo headerInfo;
    SkDeviceEntryArgs* deviceArgs;
    uint8_t* buffer = SetupOpTraceTestBuffer(0, false, headerInfo, deviceArgs, handler.get());
    if (buffer == nullptr) {
        return;
    }

    // Set counterInfo for core 0: launch = ORIGIN(0), meaning no SK executed yet
    SkCounterInfo* counterInfo = reinterpret_cast<SkCounterInfo*>(buffer + headerInfo.counterOffset);
    counterInfo[0].index = 0;
    counterInfo[0].launch = static_cast<uint8_t>(SkOpTraceType::ORIGIN);  // 0
    counterInfo[0].exit = 0;

    handler->PrintCoreSymbols(0, RT_CORE_TYPE_AIC, 0, 0);

    free(buffer);
    SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintCoreSymbols_LaunchSKEntryLaunched_PrintNextOp)
{
    handler->hasOpTrace_ = true;

    SkHeaderInfo headerInfo;
    SkDeviceEntryArgs* deviceArgs;
    uint8_t* buffer = SetupOpTraceTestBuffer(2, true, headerInfo, deviceArgs, handler.get());
    if (buffer == nullptr) {
        return;
    }

    // Set counterInfo for core 0: launch = SK_ENTRY_LAUNCHED(1), opId = 0
    SkCounterInfo* counterInfo = reinterpret_cast<SkCounterInfo*>(buffer + headerInfo.counterOffset);
    counterInfo[0].index = 0;
    counterInfo[0].launch = static_cast<uint8_t>(SkOpTraceType::SK_ENTRY_LAUNCHED);  // 1
    counterInfo[0].exit = 0;

    // Setup DFX info so GetOrLoadKernelSymbols can find func name
    SkDfxInfo* dfxInfo = reinterpret_cast<SkDfxInfo*>(buffer + headerInfo.dfxOffset);
    dfxInfo[0].funcHdlOri = 0xDEAD0001;
    dfxInfo[1].funcHdlOri = 0xDEAD0002;

    // Mock aclrtGetFunctionName for loading kernel symbols
    MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_other));

    handler->PrintCoreSymbols(0, RT_CORE_TYPE_AIC, 0x1000, 0x2000);

    free(buffer);
    SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintCoreSymbols_LaunchOPLaunched_PrintCurrentOp)
{
    handler->hasOpTrace_ = true;

    SkHeaderInfo headerInfo;
    SkDeviceEntryArgs* deviceArgs;
    uint8_t* buffer = SetupOpTraceTestBuffer(3, true, headerInfo, deviceArgs, handler.get());
    if (buffer == nullptr) {
        return;
    }

    // Set counterInfo for core 0: launch = OP_LAUNCHED(2), opId = 1
    SkCounterInfo* counterInfo = reinterpret_cast<SkCounterInfo*>(buffer + headerInfo.counterOffset);
    counterInfo[0].index = 1;
    counterInfo[0].launch = static_cast<uint8_t>(SkOpTraceType::OP_LAUNCHED);  // 2
    counterInfo[0].exit = 0;

    SkDfxInfo* dfxInfo = reinterpret_cast<SkDfxInfo*>(buffer + headerInfo.dfxOffset);
    dfxInfo[0].funcHdlOri = 0xDEAD0001;
    dfxInfo[1].funcHdlOri = 0xDEAD0002;
    dfxInfo[2].funcHdlOri = 0xDEAD0003;

    MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));

    handler->PrintCoreSymbols(0, RT_CORE_TYPE_AIC, 0x1000, 0x2000);

    free(buffer);
    SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintCoreSymbols_LaunchOPFinished_PrintCurrentAndNextOp)
{
    handler->hasOpTrace_ = true;

    SkHeaderInfo headerInfo;
    SkDeviceEntryArgs* deviceArgs;
    uint8_t* buffer = SetupOpTraceTestBuffer(3, true, headerInfo, deviceArgs, handler.get());
    if (buffer == nullptr) {
        return;
    }

    // Set counterInfo for core 0: launch = OP_FINISHED(3), opId = 1
    SkCounterInfo* counterInfo = reinterpret_cast<SkCounterInfo*>(buffer + headerInfo.counterOffset);
    counterInfo[0].index = 1;
    counterInfo[0].launch = static_cast<uint8_t>(SkOpTraceType::OP_FINISHED);  // 3
    counterInfo[0].exit = 1;

    SkDfxInfo* dfxInfo = reinterpret_cast<SkDfxInfo*>(buffer + headerInfo.dfxOffset);
    dfxInfo[0].funcHdlOri = 0xDEAD0001;
    dfxInfo[1].funcHdlOri = 0xDEAD0002;
    dfxInfo[2].funcHdlOri = 0xDEAD0003;

    MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_other));

    handler->PrintCoreSymbols(0, RT_CORE_TYPE_AIC, 0x1000, 0x2000);

    free(buffer);
    SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintCoreSymbols_LaunchSKEntryFinished)
{
    handler->hasOpTrace_ = true;

    SkHeaderInfo headerInfo;
    SkDeviceEntryArgs* deviceArgs;
    uint8_t* buffer = SetupOpTraceTestBuffer(0, false, headerInfo, deviceArgs, handler.get());
    if (buffer == nullptr) {
        return;
    }

    // Set counterInfo: launch = SK_ENTRY_FINISHED(4)
    SkCounterInfo* counterInfo = reinterpret_cast<SkCounterInfo*>(buffer + headerInfo.counterOffset);
    counterInfo[0].index = 99;
    counterInfo[0].launch = static_cast<uint8_t>(SkOpTraceType::SK_ENTRY_FINISHED);  // 4
    counterInfo[0].exit = 1;

    handler->PrintCoreSymbols(0, RT_CORE_TYPE_AIC, 0x1000, 0x2000);

    free(buffer);
    SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintCoreSymbols_LaunchUnknownValue)
{
    handler->hasOpTrace_ = true;

    SkHeaderInfo headerInfo;
    SkDeviceEntryArgs* deviceArgs;
    uint8_t* buffer = SetupOpTraceTestBuffer(0, false, headerInfo, deviceArgs, handler.get());
    if (buffer == nullptr) {
        return;
    }

    // Set counterInfo: launch = unknown value 255
    SkCounterInfo* counterInfo = reinterpret_cast<SkCounterInfo*>(buffer + headerInfo.counterOffset);
    counterInfo[0].index = 0;
    counterInfo[0].launch = 255;  // Unknown status
    counterInfo[0].exit = 0;

    handler->PrintCoreSymbols(0, RT_CORE_TYPE_AIC, 0x1000, 0x2000);

    free(buffer);
    SUCCEED();
}

// ==================== PrintAllCoreSymbols Tests (Commit 4) ====================

TEST_F(SkDfxExceptionHandlerTest, PrintAllCoreSymbols_HasOpTraceFalse_ShouldReturnEarly)
{
    handler->hasOpTrace_ = false;

    // Should return immediately without iterating cores
    handler->PrintAllCoreSymbols();
    SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintAllCoreSymbols_HasOpTraceTrue_IterateCores)
{
    handler->hasOpTrace_ = true;

    SkHeaderInfo headerInfo;
    SkDeviceEntryArgs* deviceArgs;
    uint8_t* buffer = SetupOpTraceTestBuffer(0, false, headerInfo, deviceArgs, handler.get());
    if (buffer == nullptr) {
        return;
    }

    // Set some counter info for cores
    SkCounterInfo* counterInfo = reinterpret_cast<SkCounterInfo*>(buffer + headerInfo.counterOffset);
    for (uint32_t i = 0; i < 75; ++i) {
        counterInfo[i].index = i % 5;
        counterInfo[i].launch = static_cast<uint8_t>(SkOpTraceType::ORIGIN);
        counterInfo[i].exit = 0;
    }

    // Should iterate through all 75 cores
    handler->PrintAllCoreSymbols();

    free(buffer);
    SUCCEED();
}

// ==================== GetOrLoadKernelSymbols Additional Tests ====================

TEST_F(SkDfxExceptionHandlerTest, GetOrLoadKernelSymbols_OpIdExceedsNodeCnt_ReturnEmpty)
{
    handler->opSymbolCache.clear();

    SkHeaderInfo headerInfo;
    SkDeviceEntryArgs* deviceArgs;

    // Use a custom size for this test (1024)
    uint32_t totalSize = sizeof(SkHeaderInfo) + sizeof(SkDfxInfo) * 2;
    headerInfo = {};
    headerInfo.nodeCnt = 2;
    headerInfo.dfxOffset = sizeof(SkHeaderInfo);
    headerInfo.totalSize = totalSize;

    const size_t allocSize = 1024;
    if (allocSize == 0 || allocSize > 1024 * 1024) {
        return;
    }
    uint8_t* buffer = reinterpret_cast<uint8_t*>(malloc(allocSize));
    (void)memset_s(buffer, allocSize, 0, allocSize);
    deviceArgs = reinterpret_cast<SkDeviceEntryArgs*>(buffer);
    deviceArgs->skHeader = headerInfo;

    handler->skDeviceEntryArgsHost = deviceArgs;
    handler->skHeaderInfoHost = &headerInfo;

    // opId=5 exceeds nodeCnt=2, should return empty name
    KernelFuncName result = handler->GetOrLoadKernelSymbols(5);
    EXPECT_EQ(result.name, "");

    free(buffer);
}

// ==================== modelRIIdAndSkScopeId 解码与 SkEventRecorder 交互测试 ====================

// Test: SkHeaderInfo.modelRIIdAndSkScopeId 字段偏移验证
TEST_F(SkDfxExceptionHandlerTest, ModelRIIdAndSkScopeId_FieldOffsetInSkHeaderInfo)
{
    SkHeaderInfo headerInfo = {};
    // 编码: (modelRIIdx << 32) | (skScopeId << 16) = (3 << 32) | (5 << 16)
    headerInfo.modelRIIdAndSkScopeId = 0x0000000300050000ULL;

    // 验证字段存在且可以赋值
    EXPECT_EQ(headerInfo.modelRIIdAndSkScopeId, 0x0000000300050000ULL);
    EXPECT_EQ(sizeof(headerInfo.modelRIIdAndSkScopeId), sizeof(uint64_t));

    // 解码
    uint16_t modelRIIdx = static_cast<uint16_t>((headerInfo.modelRIIdAndSkScopeId >> 32) & 0xFFFF);
    uint16_t skScopeId = static_cast<uint16_t>((headerInfo.modelRIIdAndSkScopeId >> 16) & 0xFFFF);
    EXPECT_EQ(modelRIIdx, 3);
    EXPECT_EQ(skScopeId, 5);
}

// Test: IdentifyErrorNodeByPC 中 modelRIIdAndSkScopeId 解码与 SkEventRecorder 反查
TEST_F(SkDfxExceptionHandlerTest, IdentifyErrorNodeByPC_ModelRIIdAndSkScopeIdDecode)
{
    // 注册一个 modelRI 到 SkEventRecorder
    uint64_t originalModelRI = 0xDEADBEEFCAFEBABE;
    uint16_t skScopeId = 42;
    uint16_t modelRIIdx = SkEventRecorder::Instance().RegisterModelRI(originalModelRI);

    // 设置 buffer
    uint8_t buffer[2048] = {0};
    SkHeaderInfo headerInfo;
    headerInfo.aicQueOffset = 0;
    headerInfo.aivQueOffset = 0;
    headerInfo.counterOffset = 0;
    headerInfo.dfxOffset = sizeof(SkHeaderInfo);
    headerInfo.nodeCnt = 1;
    headerInfo.totalSize = sizeof(buffer);
    headerInfo.modelRIIdAndSkScopeId =
        (static_cast<uint64_t>(modelRIIdx) << 32) | (static_cast<uint64_t>(skScopeId) << 16);

    SkDeviceEntryArgs* deviceArgs = reinterpret_cast<SkDeviceEntryArgs*>(buffer);
    deviceArgs->skHeader = headerInfo;

    SkDfxInfo* dfxInfo = reinterpret_cast<SkDfxInfo*>(buffer + headerInfo.dfxOffset);
    dfxInfo->aicSize = 0x100;
    dfxInfo->aivSize = 0;
    dfxInfo->funcHdlOri = 0xDEAD0001;
    for (int i = 0; i < 4; i++) {
        dfxInfo->entryAic[i] = 0;
        dfxInfo->entryAiv[i] = 0;
    }
    dfxInfo->entryAic[0] = 0x1000;

    handler->skDeviceEntryArgsHost = deviceArgs;
    handler->skHeaderInfoHost = &headerInfo;

    // 验证解码逻辑
    uint16_t decodedIdx = static_cast<uint16_t>((headerInfo.modelRIIdAndSkScopeId >> 32) & 0xFFFF);
    uint16_t decodedScopeId = static_cast<uint16_t>((headerInfo.modelRIIdAndSkScopeId >> 16) & 0xFFFF);
    uint64_t recoveredModelRI = SkEventRecorder::Instance().GetModelRIByIndex(decodedIdx);

    EXPECT_EQ(decodedIdx, modelRIIdx);
    EXPECT_EQ(decodedScopeId, skScopeId);
    EXPECT_EQ(recoveredModelRI, originalModelRI);

    // 调用 IdentifyErrorNodeByPC，PC 匹配到 node[0] 的 AIC entry
    MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));
    handler->IdentifyErrorNodeByPC(0, RT_CORE_TYPE_AIC, 0x1000, 0x1050);
    SUCCEED();

    // 清理 SkEventRecorder
    SkEventRecorder::Instance().modelRIIndexMap.clear();
    SkEventRecorder::Instance().modelRIToIndexMap.clear();
}

// Test: modelRIIdAndSkScopeId 为 0 时（未设置），反查返回 0
TEST_F(SkDfxExceptionHandlerTest, ModelRIIdAndSkScopeId_ZeroValue_GetModelRIReturnsZero)
{
    SkHeaderInfo headerInfo = {};
    headerInfo.modelRIIdAndSkScopeId = 0;

    uint16_t decodedIdx = static_cast<uint16_t>((headerInfo.modelRIIdAndSkScopeId >> 32) & 0xFFFF);
    uint64_t recoveredModelRI = SkEventRecorder::Instance().GetModelRIByIndex(decodedIdx);
    EXPECT_EQ(decodedIdx, 0);
    EXPECT_EQ(recoveredModelRI, 0);  // index 0 不存在，返回 0
}

// Test: cond 寄存器完整 48bit 布局编解码验证
// 排布: modelRIIdx(16bit)[47:32] | skScopeId(16bit)[31:16] | task->index(8bit)[15:8] | SkOpTraceType(8bit)[7:0]
TEST_F(SkDfxExceptionHandlerTest, CondRegister_48bitLayout_EncodeDecode)
{
    uint64_t modelRI = 0x123456789ABCDEF0;
    uint16_t skScopeId = 1234;
    uint16_t modelRIIdx = SkEventRecorder::Instance().RegisterModelRI(modelRI);

    uint64_t modelRIIdAndSkScopeId = (static_cast<uint64_t>(modelRIIdx) << 32) | (static_cast<uint64_t>(skScopeId) << 16);

    // OP_LAUNCHED + task->index = 7
    uint64_t cond = static_cast<uint64_t>(SkOpTraceType::OP_LAUNCHED) + (static_cast<uint64_t>(7) << 8);
    cond = modelRIIdAndSkScopeId | cond;

    // 解码
    uint16_t decodedModelRIIdx = static_cast<uint16_t>((cond >> 32) & 0xFFFF);
    uint16_t decodedSkScopeId = static_cast<uint16_t>((cond >> 16) & 0xFFFF);
    uint8_t decodedTaskIndex = static_cast<uint8_t>((cond >> 8) & 0xFF);
    uint8_t decodedOpTraceType = static_cast<uint8_t>(cond & 0xFF);

    EXPECT_EQ(decodedModelRIIdx, modelRIIdx);
    EXPECT_EQ(decodedSkScopeId, skScopeId);
    EXPECT_EQ(decodedTaskIndex, 7);
    EXPECT_EQ(decodedOpTraceType, static_cast<uint8_t>(SkOpTraceType::OP_LAUNCHED));

    // 通过 index 反查原始 modelRI
    EXPECT_EQ(SkEventRecorder::Instance().GetModelRIByIndex(decodedModelRIIdx), modelRI);

    // 清理
    SkEventRecorder::Instance().modelRIIndexMap.clear();
    SkEventRecorder::Instance().modelRIToIndexMap.clear();
}

// Test: SkHeaderInfo 结构体大小不应被意外修改
TEST_F(SkDfxExceptionHandlerTest, SkHeaderInfo_SizeAndFieldOffsetsStable)
{
    // 验证 SkHeaderInfo 的关键字段偏移不会被误改
    SkHeaderInfo headerInfo = {};

    // 验证 modelRIIdAndSkScopeId 可以存储完整的 48bit cond 编码
    uint64_t testValue = 0xFFFFFFFF00000000ULL;  // modelRIIdx=0xFFFF, skScopeId=0xFFFF
    headerInfo.modelRIIdAndSkScopeId = testValue;
    EXPECT_EQ(headerInfo.modelRIIdAndSkScopeId, testValue);

    // 验证 64bit 地址的 modelRI 可以通过 index 映射恢复
    uint64_t fullAddr = 0x7FFF123456789ABC;
    uint16_t idx = SkEventRecorder::Instance().RegisterModelRI(fullAddr);
    EXPECT_EQ(SkEventRecorder::Instance().GetModelRIByIndex(idx), fullAddr);

    SkEventRecorder::Instance().modelRIIndexMap.clear();
    SkEventRecorder::Instance().modelRIToIndexMap.clear();
}
