/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include "mockcpp/mockcpp.hpp"
#include <memory>
#include <cstring>
#include <cstdio>
#include <map>
#include <string>
#include <unordered_map>
#include <sys/stat.h>
#include <unistd.h>
#include <bitset>
#include <limits>

#define private public
#define protected public
#include "sk_graph.h"
#include "sk_node.h"
#include "sk_scope_launch.h"
#include "sk_common.h"
#include "ut_common_stubs.h"
#include "runtime/kernel.h"
#include "securec.h"

class SkNodeTest : public testing::Test {
protected:
    void SetUp() override {
        SkUtResetTestControls();
        sk::logger::FileLogger::Instance().SetEnabled(false);
    }

    void TearDown() override {
        GlobalMockObject::verify();
        sk::logger::FileLogger::Instance().SetEnabled(false);
        SkUtResetTestControls();
    }
};

struct JudgeTaskKernelInfo {
    bool isBegin = true;
    bool isFuseEnable = true;
    std::unique_ptr<char[]> scopeName;
};

extern bool IsScopeKernel(aclmdlRIKernelTaskParams params, JudgeTaskKernelInfo* info);
extern bool DumpSingleKernelBinary(const KernelInfos& kernelInfo, const std::string& kernelBinsDir);

TEST_F(SkNodeTest, IsScopeKernel_GetFunctionName_Failed)
{
    aclmdlRIKernelTaskParams params{};
    params.funcHandle = nullptr;
    JudgeTaskKernelInfo info;
    bool ret = IsScopeKernel(params, &info);
    EXPECT_EQ(ret, false);
}

int Fake_aclrtGetFunctionNameBegin(void* funcHandle, size_t size, char* name) 
{
    const char* src = "sk_scope_kernel_begin";
    snprintf_s(name, size, size, "%s", src);
    return 0;
}

int Fake_aclrtMemcpy(void* dst, size_t dstSize, const void* src, size_t count, aclrtMemcpyKind kind) 
{
    ScopeKernelArgs fakeArgs;
    const char* defaultName = "default_sk_scope_name";
    snprintf_s(fakeArgs.name, sizeof(fakeArgs.name), sizeof(fakeArgs.name), "%s", defaultName);
    fakeArgs.name[MAX_SCOPE_NAME_LEN - 1] = '\0';
    memcpy_s(dst, sizeof(ScopeKernelArgs), &fakeArgs, sizeof(ScopeKernelArgs));
    return 0;
}

TEST_F(SkNodeTest, IsScopeKernel_Normal_ScopeName)
{
    aclmdlRIKernelTaskParams params{};
    params.funcHandle = nullptr;
    JudgeTaskKernelInfo info;
    MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionNameBegin));
    MOCKER(aclrtMemcpy).stubs().will(invoke(Fake_aclrtMemcpy));
    bool ret = IsScopeKernel(params, &info);
    EXPECT_EQ(ret, true);
}

TEST_F(SkNodeTest, IsScopeKernel_ScopeName_Isnullptr)
{
    aclmdlRIKernelTaskParams params{};
    params.funcHandle = nullptr;
    JudgeTaskKernelInfo info;
    MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionNameBegin));
    MOCKER(aclrtMemcpy).stubs().will(invoke(Fake_aclrtMemcpy));
    bool ret = IsScopeKernel(params, &info);
    EXPECT_EQ(ret, true);
}

namespace {

struct UtSkNodeRITaskInternal {
    uint32_t taskId;
    aclmdlRITaskType type;
    aclmdlRITaskParams params;
};

std::unique_ptr<aclmdlRITask> MakeOriginTask(UtSkNodeRITaskInternal& task)
{
    return std::make_unique<aclmdlRITask>(reinterpret_cast<aclmdlRITask>(&task));
}

aclError FakeAclrtGetFunctionNameRegular(aclrtFuncHandle funcHandle, uint32_t maxLen, char* name)
{
    (void)funcHandle;
    const char* src = "regular_kernel";
    return snprintf_s(name, maxLen, maxLen, "%s", src) < 0 ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}

aclError FakeAclrtFunctionGetBinaryNonNull(aclrtFuncHandle funcHandle, aclrtBinHandle* binHandle)
{
    (void)funcHandle;
    if (binHandle == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *binHandle = reinterpret_cast<aclrtBinHandle>(0x5000);
    return ACL_SUCCESS;
}

int FakeRtGetBinBufferForNodeDump(void* binHdl, int addrType, void** buffer, uint32_t* size)
{
    (void)binHdl;
    (void)addrType;
    static const char kBinData[] = "node-binary";
    if (buffer == nullptr || size == nullptr) {
        return -1;
    }
    *buffer = const_cast<char*>(kBinData);
    *size = sizeof(kBinData);
    return 0;
}

int FakeRtGetBinBufferFailureForNodeDump(void* binHdl, int addrType, void** buffer, uint32_t* size)
{
    (void)binHdl;
    (void)addrType;
    if (buffer != nullptr) {
        *buffer = nullptr;
    }
    if (size != nullptr) {
        *size = 0;
    }
    return -1;
}

aclError FakeAclrtBinaryGetDevAddressForNodeDump(aclrtBinHandle binHdl, void** devAddr, size_t* devSize)
{
    (void)binHdl;
    if (devAddr == nullptr || devSize == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *devAddr = reinterpret_cast<void*>(0xABC000);
    *devSize = 0x1000;
    return ACL_SUCCESS;
}

aclError FakeAclrtBinaryGetDevAddressFailureForNodeDump(aclrtBinHandle binHdl, void** devAddr, size_t* devSize)
{
    (void)binHdl;
    (void)devAddr;
    (void)devSize;
    return ACL_ERROR_FAILURE;
}

std::string MakeTmpDir(const std::string& suffix)
{
    std::string dir = "/tmp/sk_node_ut_" + std::to_string(getpid()) + "_" + suffix;
    mkdir(dir.c_str(), 0755);
    return dir;
}

} // namespace

TEST_F(SkNodeTest, FusionFailReasonInfo_BindmapDetailsAndStrings)
{
    FusionFailReasonInfo noneInfo(FusionFailReason::BINDMAP_IS_EMPTY, BindmapFailReason::NONE);
    EXPECT_EQ(noneInfo.GetBindmapDetail(), BindmapFailReason::NONE);
    EXPECT_EQ(FusionFailReasonToStr(noneInfo),
        std::string(FusionFailReasonToStr(FusionFailReason::BINDMAP_IS_EMPTY)));

    FusionFailReasonInfo info(FusionFailReason::BINDMAP_IS_EMPTY, BindmapFailReason::FUNCHDL_NULL);
    EXPECT_EQ(info.GetBindmapDetail(), BindmapFailReason::FUNCHDL_NULL);
    EXPECT_NE(FusionFailReasonToStr(info).find("funcHdl is null"), std::string::npos);

    info.SetBindmapDetail(BindmapFailReason::FUNC_NOT_FOUND);
    EXPECT_EQ(info.GetBindmapDetail(), BindmapFailReason::FUNC_NOT_FOUND);
    EXPECT_NE(FusionFailReasonToStr(info).find("initialize kernel function failed"), std::string::npos);

    EXPECT_STREQ(BindmapFailReasonToStr(BindmapFailReason::BINDMAP_INIT_EMPTY), "bindmap init empty");
    EXPECT_STREQ(BindmapFailReasonToStr(BindmapFailReason::BINHDL_NULL), "binHdl is null");
    EXPECT_STREQ(BindmapFailReasonToStr(static_cast<BindmapFailReason>(255)), "UNKNOWN_BINDMAP_REASON");
}

TEST_F(SkNodeTest, FusionFailReasonInfo_ScopeAndDeadlockDetails)
{
    FusionFailReasonInfo scopeInfo(FusionFailReason::SCOPE_FUSE_PART, static_cast<ScopeFailReason>(1));
    EXPECT_EQ(scopeInfo.GetScopeDetail(), static_cast<ScopeFailReason>(1));
    EXPECT_NE(FusionFailReasonToStr(scopeInfo).find("["), std::string::npos);

    FusionFailReasonInfo deadlockInfo(FusionFailReason::EXIST_DEADLOCK, static_cast<DeadlockFailReason>(1));
    EXPECT_EQ(deadlockInfo.GetDeadlockDetail(), static_cast<DeadlockFailReason>(1));
    EXPECT_NE(FusionFailReasonToStr(deadlockInfo).find("["), std::string::npos);

    scopeInfo.SetScopeDetail(static_cast<ScopeFailReason>(0));
    deadlockInfo.SetDeadlockDetail(static_cast<DeadlockFailReason>(0));
    EXPECT_EQ(scopeInfo.GetScopeDetail(), static_cast<ScopeFailReason>(0));
    EXPECT_EQ(deadlockInfo.GetDeadlockDetail(), static_cast<DeadlockFailReason>(0));
}

TEST_F(SkNodeTest, AlignUpAndClamp_CoversAlignmentAndCoreClamp)
{
    EXPECT_EQ(AlignUpAndClamp(1, 0), 1);
    EXPECT_EQ(AlignUpAndClamp(0x801, 0), 2);
    EXPECT_EQ(AlignUpAndClamp(0x800 * 20, 0), 16);
    EXPECT_EQ(AlignUpAndClamp(0x800 * 20, 1), 8);
    EXPECT_EQ(AlignUpAndClamp(0x800 * 20, 2), 20);
}

TEST_F(SkNodeTest, InValidateNode_SetsInvalidatedFlag)
{
    UtSkNodeRITaskInternal task{};
    task.taskId = 7;
    task.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
    task.params.type = ACL_MODEL_RI_TASK_VALUE_WRITE;

    SuperKernelMemoryNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_VALUE_WRITE, 0, 0, 0, INVALID_TASK_ID);
    node.SetNodeId(7);

    EXPECT_FALSE(node.IsInvalidated());
    EXPECT_EQ(node.InValidateNode(), ACL_SUCCESS);
    EXPECT_TRUE(node.IsInvalidated());
}

TEST_F(SkNodeTest, KernelUpdate_CustomParamsSyncTaskParamsForDump)
{
    UtSkNodeRITaskInternal task{};
    task.taskId = 8;
    task.type = ACL_MODEL_RI_TASK_KERNEL;
    task.params.type = ACL_MODEL_RI_TASK_KERNEL;

    SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
    node.SetNodeId(8);

    int value = 0;
    aclmdlRITaskParams custom{};
    custom.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
    custom.valueWriteTaskParams.devAddr = &value;
    custom.valueWriteTaskParams.value = 0x1234;

    UpdateContext ctx{};
    ctx.customParams = &custom;
    EXPECT_TRUE(node.Update(ctx));
    EXPECT_TRUE(node.IsUpdated());
    EXPECT_EQ(node.GetTaskParams().type, ACL_MODEL_RI_TASK_VALUE_WRITE);
    EXPECT_EQ(node.GetTaskParams().valueWriteTaskParams.devAddr, &value);
    EXPECT_EQ(node.GetTaskParams().valueWriteTaskParams.value, 0x1234U);
}

TEST_F(SkNodeTest, MemoryUpdate_CustomParamsSyncTaskParamsForDump)
{
    UtSkNodeRITaskInternal task{};
    task.taskId = 9;
    task.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
    task.params.type = ACL_MODEL_RI_TASK_VALUE_WAIT;

    SuperKernelMemoryNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_VALUE_WAIT, 0, 0, 0, INVALID_TASK_ID);
    node.SetNodeId(9);

    int value = 0;
    aclmdlRITaskParams custom{};
    custom.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
    custom.valueWaitTaskParams.devAddr = &value;
    custom.valueWaitTaskParams.value = 0x5678;
    custom.valueWaitTaskParams.flag = 0x9;

    UpdateContext ctx{};
    ctx.customParams = &custom;
    EXPECT_TRUE(node.Update(ctx));
    EXPECT_TRUE(node.IsUpdated());
    EXPECT_EQ(node.GetTaskParams().type, ACL_MODEL_RI_TASK_VALUE_WAIT);
    EXPECT_EQ(node.GetTaskParams().valueWaitTaskParams.devAddr, &value);
    EXPECT_EQ(node.GetTaskParams().valueWaitTaskParams.value, 0x5678U);
    EXPECT_EQ(node.GetTaskParams().valueWaitTaskParams.flag, 0x9U);
}

TEST_F(SkNodeTest, KernelInitNode_BindmapEmptyReasonIsRecorded)
{
    UtSkNodeRITaskInternal task{};
    task.taskId = 10;
    task.type = ACL_MODEL_RI_TASK_KERNEL;
    task.params.type = ACL_MODEL_RI_TASK_KERNEL;
    task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x2000);
    task.params.kernelTaskParams.numBlocks = 4;

    MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
    MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryNonNull));

    SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
    EXPECT_TRUE(node.InitNode());
    EXPECT_FALSE(node.IsFusible());
    EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::BINDMAP_IS_EMPTY);
    EXPECT_EQ(node.GetFusionFailReasonInfo().GetBindmapDetail(), BindmapFailReason::BINDMAP_INIT_EMPTY);
}

TEST_F(SkNodeTest, KernelInitNode_NullFuncHandleRecordsBindmapReason)
{
    UtSkNodeRITaskInternal task{};
    task.taskId = 11;
    task.type = ACL_MODEL_RI_TASK_KERNEL;
    task.params.type = ACL_MODEL_RI_TASK_KERNEL;
    task.params.kernelTaskParams.funcHandle = nullptr;
    task.params.kernelTaskParams.numBlocks = 4;

    MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
    MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryNonNull));

    SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
    EXPECT_TRUE(node.InitNode());
    EXPECT_FALSE(node.IsFusible());
    EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::BINDMAP_IS_EMPTY);
    EXPECT_EQ(node.GetFusionFailReasonInfo().GetBindmapDetail(), BindmapFailReason::FUNCHDL_NULL);
}

TEST_F(SkNodeTest, KernelInitNode_TaskGroupNotEmptyRecordsReason)
{
    UtSkNodeRITaskInternal task{};
    task.taskId = 12;
    task.type = ACL_MODEL_RI_TASK_KERNEL;
    task.params.type = ACL_MODEL_RI_TASK_KERNEL;
    task.params.taskGrp = reinterpret_cast<aclrtTaskGrp>(0x6000);
    task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x2000);
    task.params.kernelTaskParams.numBlocks = 4;

    MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));

    SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
    EXPECT_TRUE(node.InitNode());
    EXPECT_FALSE(node.IsFusible());
    EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::TASK_GROUP_NOT_EMPTY);
}

TEST_F(SkNodeTest, DumpKernelBinaries_DisabledLoggerSkipsWork)
{
    SuperKernelGraph graph;
    EXPECT_TRUE(DumpKernelBinaries(graph, MakeTmpDir("disabled")));
}

TEST_F(SkNodeTest, DumpSingleKernelBinary_CoversSuccessAndFailurePaths)
{
    KernelInfos kernelInfo;
    kernelInfo.binHdl = reinterpret_cast<aclrtBinHandle>(0x7000);
    kernelInfo.funcName = "kernel/name:with*chars";

    MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferFailureForNodeDump));
    EXPECT_FALSE(DumpSingleKernelBinary(kernelInfo, MakeTmpDir("bin_fail")));
    GlobalMockObject::verify();

    MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferForNodeDump));
    MOCKER(aclrtBinaryGetDevAddress).stubs().will(invoke(FakeAclrtBinaryGetDevAddressFailureForNodeDump));
    EXPECT_FALSE(DumpSingleKernelBinary(kernelInfo, MakeTmpDir("dev_fail")));
    GlobalMockObject::verify();

    std::string outDir = MakeTmpDir("bin_success");
    MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferForNodeDump));
    MOCKER(aclrtBinaryGetDevAddress).stubs().will(invoke(FakeAclrtBinaryGetDevAddressForNodeDump));
    EXPECT_TRUE(DumpSingleKernelBinary(kernelInfo, outDir));
}

TEST_F(SkNodeTest, DumpKernelBinaries_EnabledDumpsKernelAndEntryBinary)
{
    SuperKernelGraph graph;
    auto node = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
    node->SetNodeId(13);
    node->SetNodeType(SkNodeType::NODE_KERNEL);
    node->nodeInfos.kernelInfos.binHdl = reinterpret_cast<aclrtBinHandle>(0x8000);
    node->nodeInfos.kernelInfos.funcName = "regular_kernel";
    graph.graphMap[13] = std::move(node);

    sk::logger::FileLogger::Instance().SetEnabled(true);
    MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferForNodeDump));
    MOCKER(aclrtBinaryGetDevAddress).stubs().will(invoke(FakeAclrtBinaryGetDevAddressForNodeDump));

    EXPECT_TRUE(DumpKernelBinaries(graph, MakeTmpDir("dump_graph")));
}
