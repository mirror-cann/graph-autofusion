/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for the details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNEsS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file test_sk_dump_json.cpp
 * \brief Unit tests for SuperKernel graph JSON dump utilities
 */

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include "mockcpp/mockcpp.hpp"
#include <fstream>
#include <cstring>
#include <memory>
#include <vector>

#define private public
#define protected public

#include "sk_dump_json.h"
#include "sk_common.h"
#include "sk_graph.h"
#include "sk_node.h"
#include "sk_scope_info.h"
#include "sk_log.h"
#include "sk_options_manager.h"
#include "ut_common_stubs.h"

#include <nlohmann/json.hpp>

#include "sk_dump_json.cpp"

// ==================== Test Fixture ====================

class SkDumpJsonTest : public ::testing::Test {
protected:
    void SetUp() override {
        sk::logger::FileLogger::Instance().SetEnabled(false);
    }

    void TearDown() override {
        GlobalMockObject::verify();
        sk::logger::FileLogger::Instance().SetEnabled(true);
    }

    std::unique_ptr<SuperKernelKernelNode> CreateKernelNode(uint64_t nodeId) {
        auto node = std::make_unique<SuperKernelKernelNode>(
            nullptr, ACL_MODEL_RI_TASK_KERNEL, nodeId, 0, 0, INVALID_TASK_ID);
        node->SetNodeId(nodeId);
        return node;
    }

    std::unique_ptr<SuperKernelMemoryNode> CreateMemoryNode(uint64_t nodeId, aclmdlRITaskType taskType) {
        auto node = std::make_unique<SuperKernelMemoryNode>(
            nullptr, taskType, nodeId, 0, 0, INVALID_TASK_ID);
        node->SetNodeId(nodeId);
        return node;
    }

    void AddKernelInfoToNode(SuperKernelKernelNode* node) {
        node->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
        node->nodeInfos.kernelInfos.numBlocks = 8;
        node->nodeInfos.kernelInfos.funcName = "test_kernel";
        node->nodeInfos.kernelInfos.vecNum = 4;
        node->nodeInfos.kernelInfos.cubeNum = 2;
    }
};

// ==================== DumpGraphNodesToJson Tests ====================

class DumpGraphNodesToJsonTest : public SkDumpJsonTest {};

TEST_F(DumpGraphNodesToJsonTest, EmptyGraph)
{
    SuperKernelGraph graph(nullptr);
    bool result = DumpGraphNodesToJson(graph);
    EXPECT_TRUE(result);
}

TEST_F(DumpGraphNodesToJsonTest, GraphWithKernelNodes)
{
    SuperKernelGraph graph(nullptr);
    auto node1 = CreateKernelNode(10);
    AddKernelInfoToNode(node1.get());
    graph.graphMap[10] = std::move(node1);

    auto node2 = CreateKernelNode(11);
    AddKernelInfoToNode(node2.get());
    graph.graphMap[11] = std::move(node2);

    bool result = DumpGraphNodesToJson(graph);
    EXPECT_TRUE(result);
}

TEST_F(DumpGraphNodesToJsonTest, GraphWithMixedNodes)
{
    SuperKernelGraph graph(nullptr);
    auto kernelNode = CreateKernelNode(10);
    AddKernelInfoToNode(kernelNode.get());
    graph.graphMap[10] = std::move(kernelNode);

    auto memoryNode = CreateMemoryNode(11, ACL_MODEL_RI_TASK_EVENT_WAIT);
    graph.graphMap[11] = std::move(memoryNode);

    bool result = DumpGraphNodesToJson(graph);
    EXPECT_TRUE(result);
}

TEST_F(DumpGraphNodesToJsonTest, DisabledLoggerSkipDump)
{
    sk::logger::FileLogger::Instance().SetEnabled(false);

    SuperKernelGraph graph(nullptr);
    auto node = CreateKernelNode(10);
    AddKernelInfoToNode(node.get());
    graph.graphMap[10] = std::move(node);

    bool result = DumpGraphNodesToJson(graph);
    EXPECT_TRUE(result);

    sk::logger::FileLogger::Instance().SetEnabled(true);
}

// ==================== DumpFusedGraphToJson Tests ====================

class DumpFusedGraphToJsonTest : public SkDumpJsonTest {};

TEST_F(DumpFusedGraphToJsonTest, EmptyGraphAndScopes)
{
    SuperKernelGraph graph(nullptr);
    std::vector<SuperKernelScopeInfo> emptyScopes;

    bool result = DumpFusedGraphToJson(graph, emptyScopes);
    EXPECT_TRUE(result);
}

TEST_F(DumpFusedGraphToJsonTest, GraphWithKernelNodes)
{
    SuperKernelGraph graph(nullptr);
    auto node1 = CreateKernelNode(10);
    AddKernelInfoToNode(node1.get());
    graph.graphMap[10] = std::move(node1);

    auto node2 = CreateKernelNode(11);
    AddKernelInfoToNode(node2.get());
    graph.graphMap[11] = std::move(node2);

    std::vector<SuperKernelScopeInfo> emptyScopes;

    bool result = DumpFusedGraphToJson(graph, emptyScopes);
    EXPECT_TRUE(result);
}

TEST_F(DumpFusedGraphToJsonTest, GraphWithKernelNodesAndEmptyScopeInfo)
{
    SuperKernelGraph graph(nullptr);
    auto node1 = CreateKernelNode(10);
    AddKernelInfoToNode(node1.get());
    graph.graphMap[10] = std::move(node1);

    auto node2 = CreateKernelNode(11);
    AddKernelInfoToNode(node2.get());
    graph.graphMap[11] = std::move(node2);

    SuperKernelScopeInfo scopeInfo;
    std::vector<SuperKernelScopeInfo> scopeInfos;
    scopeInfos.push_back(std::move(scopeInfo));

    bool result = DumpFusedGraphToJson(graph, scopeInfos);
    EXPECT_TRUE(result);
}

TEST_F(DumpFusedGraphToJsonTest, DisabledLoggerSkipDump)
{
    sk::logger::FileLogger::Instance().SetEnabled(false);

    SuperKernelGraph graph(nullptr);
    auto node = CreateKernelNode(10);
    AddKernelInfoToNode(node.get());
    graph.graphMap[10] = std::move(node);

    std::vector<SuperKernelScopeInfo> emptyScopes;

    bool result = DumpFusedGraphToJson(graph, emptyScopes);
    EXPECT_TRUE(result);

    sk::logger::FileLogger::Instance().SetEnabled(true);
}

// ==================== DumpSkTaskQueueToJson Tests ====================

class DumpSkTaskQueueToJsonTest : public SkDumpJsonTest {};

TEST_F(DumpSkTaskQueueToJsonTest, EmptyTasks)
{
    SuperKernelGraph graph(nullptr);
    SkTask aicTask;
    SkTask aivTask;

    bool result = DumpSkTaskQueueToJson(graph, aicTask, aivTask);
    EXPECT_TRUE(result);
}

TEST_F(DumpSkTaskQueueToJsonTest, TasksWithInit)
{
    SuperKernelGraph graph(nullptr);
    SkTask aicTask;
    SkTask aivTask;

    ASSERT_TRUE(aicTask.Init(4));
    ASSERT_TRUE(aivTask.Init(4));

    bool result = DumpSkTaskQueueToJson(graph, aicTask, aivTask);
    EXPECT_TRUE(result);
}

TEST_F(DumpSkTaskQueueToJsonTest, DisabledLogger)
{
    sk::logger::FileLogger::Instance().SetEnabled(false);

    SuperKernelGraph graph(nullptr);
    SkTask aicTask;
    SkTask aivTask;

    bool result = DumpSkTaskQueueToJson(graph, aicTask, aivTask);
    EXPECT_TRUE(result);

    sk::logger::FileLogger::Instance().SetEnabled(true);
}

// ==================== PrintOriginalScopes Tests ====================

class PrintOriginalScopesTest : public SkDumpJsonTest {};

TEST_F(PrintOriginalScopesTest, EmptyGraph)
{
    SuperKernelGraph graph(nullptr);
    PrintOriginalScopes(graph);
    SUCCEED();
}

TEST_F(PrintOriginalScopesTest, GraphWithKernelNodes)
{
    SuperKernelGraph graph(nullptr);
    auto node = CreateKernelNode(10);
    AddKernelInfoToNode(node.get());
    graph.graphMap[10] = std::move(node);

    PrintOriginalScopes(graph);
    SUCCEED();
}

// ==================== PrintFusedScopes Tests ====================

class PrintFusedScopesTest : public SkDumpJsonTest {};

TEST_F(PrintFusedScopesTest, EmptyGraphAndScopes)
{
    SuperKernelGraph graph(nullptr);
    std::vector<SuperKernelScopeInfo> emptyScopes;

    PrintFusedScopes(graph, emptyScopes);
    SUCCEED();
}

TEST_F(PrintFusedScopesTest, ScopeWithKernelNodes)
{
    SuperKernelGraph graph(nullptr);
    auto node1 = CreateKernelNode(10);
    AddKernelInfoToNode(node1.get());
    graph.graphMap[10] = std::move(node1);

    auto node2 = CreateKernelNode(11);
    AddKernelInfoToNode(node2.get());
    graph.graphMap[11] = std::move(node2);

    SuperKernelScopeInfo scopeInfo;
    scopeInfo.SetNodes({graph.graphMap[10].get(), graph.graphMap[11].get()});
    scopeInfo.MutableExtInfo().filteredNodes = {graph.graphMap[10].get(), graph.graphMap[11].get()};

    std::vector<SuperKernelScopeInfo> scopeInfos;
    scopeInfos.push_back(std::move(scopeInfo));

    PrintFusedScopes(graph, scopeInfos);
    SUCCEED();
}

TEST_F(PrintFusedScopesTest, ScopeWithMemoryNode)
{
    SuperKernelGraph graph(nullptr);
    auto memoryNode = CreateMemoryNode(10, ACL_MODEL_RI_TASK_EVENT_RECORD);
    graph.graphMap[10] = std::move(memoryNode);

    SuperKernelScopeInfo scopeInfo;
    scopeInfo.SetNodes({graph.graphMap[10].get()});

    std::vector<SuperKernelScopeInfo> scopeInfos;
    scopeInfos.push_back(std::move(scopeInfo));

    PrintFusedScopes(graph, scopeInfos);
    SUCCEED();
}

// ==================== SuperKernelScopeInfo Tests ====================

class SuperKernelScopeInfoTest : public SkDumpJsonTest {};

TEST_F(SuperKernelScopeInfoTest, EmptyScope)
{
    SuperKernelScopeInfo scopeInfo;
    EXPECT_GE(scopeInfo.GetScopeId(), 0);
    EXPECT_EQ(scopeInfo.GetNodes().size(), 0);
}

TEST_F(SuperKernelScopeInfoTest, ScopeWithNodes)
{
    auto node1 = CreateKernelNode(10);
    AddKernelInfoToNode(node1.get());
    auto node2 = CreateKernelNode(11);
    AddKernelInfoToNode(node2.get());

    std::vector<SuperKernelBaseNode*> nodes = {node1.get(), node2.get()};

    SuperKernelScopeInfo scopeInfo;
    scopeInfo.SetNodes(nodes);

    EXPECT_GE(scopeInfo.GetScopeId(), 0);
    EXPECT_EQ(scopeInfo.GetNodes().size(), 2);
}

TEST_F(SuperKernelScopeInfoTest, ScopeSetNodes)
{
    auto node = CreateKernelNode(10);
    AddKernelInfoToNode(node.get());

    SuperKernelScopeInfo scopeInfo;
    scopeInfo.SetNodes({node.get()});

    auto retrievedNodes = scopeInfo.GetNodes();
    EXPECT_EQ(retrievedNodes.size(), 1);
    EXPECT_EQ(retrievedNodes[0]->GetNodeId(), 10);
}

TEST_F(SuperKernelScopeInfoTest, ScopeMutableExtInfo)
{
    SuperKernelScopeInfo scopeInfo;
    auto& extInfo = scopeInfo.MutableExtInfo();

    extInfo.fusionStatus = ScopeFusionStatus::SUCCESS;
    extInfo.filteredNodes = {};

    EXPECT_EQ(extInfo.fusionStatus, ScopeFusionStatus::SUCCESS);
    EXPECT_EQ(extInfo.filteredNodes.size(), 0);
}

// ==================== SkTask Tests ====================

class SkTaskTest : public SkDumpJsonTest {};

TEST_F(SkTaskTest, EmptyTask)
{
    SkTask task;
    EXPECT_EQ(task.numBlocks, 0);
    EXPECT_EQ(task.funcCnt, 0);
}

TEST_F(SkTaskTest, TaskInit)
{
    SkTask task;
    ASSERT_TRUE(task.Init(4));
    EXPECT_TRUE(task.GetTaskQue() != nullptr);
}

TEST_F(SkTaskTest, TaskWithData)
{
    SkTask task;
    task.numBlocks = 10;
    task.funcCnt = 2;
    task.nodeType = SkKernelType::AIC_ONLY;

    ASSERT_TRUE(task.Init(4));
    EXPECT_EQ(task.numBlocks, 10);
    EXPECT_EQ(task.funcCnt, 2);
    EXPECT_EQ(task.nodeType, SkKernelType::AIC_ONLY);
}

// ==================== SuperKernelGraph Tests ====================

class DumpJsonSuperKernelGraphTest : public SkDumpJsonTest {};

TEST_F(DumpJsonSuperKernelGraphTest, EmptyGraph)
{
    SuperKernelGraph graph(nullptr);
    EXPECT_EQ(graph.GetSortedNodeIds().size(), 0);
}

TEST_F(DumpJsonSuperKernelGraphTest, GraphWithSingleNode)
{
    SuperKernelGraph graph(nullptr);
    auto node = CreateKernelNode(10);
    AddKernelInfoToNode(node.get());
    graph.graphMap[10] = std::move(node);

    auto sortedIds = graph.GetSortedNodeIds();
    EXPECT_EQ(sortedIds.size(), 1);
    EXPECT_EQ(sortedIds[0], 10);
}

TEST_F(DumpJsonSuperKernelGraphTest, GraphWithMultipleNodes)
{
    SuperKernelGraph graph(nullptr);

    auto node1 = CreateKernelNode(30);
    AddKernelInfoToNode(node1.get());
    graph.graphMap[30] = std::move(node1);

    auto node2 = CreateKernelNode(10);
    AddKernelInfoToNode(node2.get());
    graph.graphMap[10] = std::move(node2);

    auto node3 = CreateKernelNode(20);
    AddKernelInfoToNode(node3.get());
    graph.graphMap[20] = std::move(node3);

    auto sortedIds = graph.GetSortedNodeIds();
    EXPECT_EQ(sortedIds.size(), 3);
    EXPECT_EQ(sortedIds[0], 10);
    EXPECT_EQ(sortedIds[1], 20);
    EXPECT_EQ(sortedIds[2], 30);
}

// ==================== DumpModelRITasksToJson Tests ====================

class DumpModelRITasksToJsonTest : public SkDumpJsonTest {};

TEST_F(DumpModelRITasksToJsonTest, NullModelRI)
{
    SuperKernelOptionsManager optsMgr;
    bool result = DumpModelRITasksToJson(nullptr, 0, &optsMgr, "test_dump_opts");
    EXPECT_FALSE(result);
}

TEST_F(DumpModelRITasksToJsonTest, DisabledLogger)
{
    sk::logger::FileLogger::Instance().SetEnabled(false);
    SuperKernelOptionsManager optsMgr;
    bool result = DumpModelRITasksToJson(nullptr, 0, &optsMgr, "test_dump_opts_disabled");
    EXPECT_FALSE(result);
    sk::logger::FileLogger::Instance().SetEnabled(true);
}

// ==================== Direct JSON Helper Tests ====================

class SkDumpJsonDirectHelperTest : public SkDumpJsonTest {};

namespace {

struct __attribute__((packed)) UtDumpSknlMapInfo {
    uint64_t cap;
    void* globalFunc;
    void* sknlFunc[4];
};

struct __attribute__((packed)) UtDumpSknlValuePayload {
    uint32_t res;
    UtDumpSknlMapInfo info;
};

struct UtRITaskInternal {
    uint32_t taskId;
    aclmdlRITaskType type;
    aclmdlRITaskParams params;
};

UtRITaskInternal g_utStreamTask;

int FakeRtBinaryGetMetaNumSuccess(void* binHdl, int typeEnum, size_t* metaNum)
{
    (void)binHdl;
    (void)typeEnum;
    *metaNum = 2;
    return 0;
}

int FakeRtBinaryGetMetaNumFailure(void* binHdl, int typeEnum, size_t* metaNum)
{
    (void)binHdl;
    (void)typeEnum;
    (void)metaNum;
    return -1;
}

int FakeRtBinaryGetMetaInfoSuccess(void* binHdl, int typeEnum, size_t metaNum, void** dataList, size_t* sizeList)
{
    (void)binHdl;
    (void)typeEnum;
    UtDumpSknlValuePayload payloads[2]{};
    payloads[0].info.globalFunc = reinterpret_cast<void*>(0x100);
    payloads[0].info.sknlFunc[0] = reinterpret_cast<void*>(0x1000);
    payloads[0].info.sknlFunc[1] = reinterpret_cast<void*>(0x1100);
    payloads[0].info.sknlFunc[2] = reinterpret_cast<void*>(0x1200);
    payloads[0].info.sknlFunc[3] = reinterpret_cast<void*>(0x1300);
    payloads[1].info.globalFunc = reinterpret_cast<void*>(0x200);
    payloads[1].info.sknlFunc[0] = reinterpret_cast<void*>(0x2000);
    payloads[1].info.sknlFunc[1] = reinterpret_cast<void*>(0x2100);
    payloads[1].info.sknlFunc[2] = reinterpret_cast<void*>(0x2200);
    payloads[1].info.sknlFunc[3] = reinterpret_cast<void*>(0x2300);

    for (size_t i = 0; i < metaNum && i < 2; ++i) {
        *static_cast<UtDumpSknlValuePayload*>(dataList[i]) = payloads[i];
        sizeList[i] = sizeof(UtDumpSknlValuePayload);
    }
    return 0;
}

int FakeRtBinaryGetMetaInfoFailure(void* binHdl, int typeEnum, size_t metaNum, void** dataList, size_t* sizeList)
{
    (void)binHdl;
    (void)typeEnum;
    (void)metaNum;
    (void)dataList;
    (void)sizeList;
    return -1;
}

int FakeRtGetBinBufferEmpty(void* binHdl, int addrType, void** buffer, uint32_t* size)
{
    (void)binHdl;
    (void)addrType;
    *buffer = nullptr;
    *size = 0;
    return 0;
}

aclError FakeAclrtGetFunctionAddrForDump(aclrtFuncHandle funcHdl, void** addrAicore, void** addrAiv)
{
    (void)funcHdl;
    *addrAicore = reinterpret_cast<void*>(0x1100);
    *addrAiv = reinterpret_cast<void*>(0x1200);
    return ACL_SUCCESS;
}

aclError FakeAclrtBinaryGetDevAddressForDump(aclrtBinHandle binHdl, void** devAddr, size_t* devSize)
{
    (void)binHdl;
    *devAddr = reinterpret_cast<void*>(0x1000);
    *devSize = 0x10000;
    return ACL_SUCCESS;
}

aclError FakeAclmdlRIGetTasksByStreamOneTask(aclrtStream stream, aclmdlRITask* tasks, uint32_t* numTasks)
{
    (void)stream;
    if (numTasks == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (tasks == nullptr) {
        *numTasks = 1;
        return ACL_SUCCESS;
    }
    tasks[0] = reinterpret_cast<aclmdlRITask>(&g_utStreamTask);
    *numTasks = 1;
    return ACL_SUCCESS;
}

aclError FakeAclmdlRIGetTasksByStreamCountFailure(aclrtStream stream, aclmdlRITask* tasks, uint32_t* numTasks)
{
    (void)stream;
    (void)tasks;
    (void)numTasks;
    return ACL_ERROR_FAILURE;
}

aclError FakeAclmdlRIGetTasksByStreamFetchFailure(aclrtStream stream, aclmdlRITask* tasks, uint32_t* numTasks)
{
    (void)stream;
    if (numTasks == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (tasks == nullptr) {
        *numTasks = 1;
        return ACL_SUCCESS;
    }
    return ACL_ERROR_FAILURE;
}

aclError FakeAclmdlRITaskGetParamsFailure(aclmdlRITask task, aclmdlRITaskParams* params)
{
    (void)task;
    (void)params;
    return ACL_ERROR_FAILURE;
}

} // namespace

TEST_F(SkDumpJsonDirectHelperTest, TaskAndKernelTypeStringHelpers)
{
    EXPECT_STREQ(TaskTypeToString(ACL_MODEL_RI_TASK_KERNEL), "KERNEL");
    EXPECT_STREQ(TaskTypeToString(ACL_MODEL_RI_TASK_VALUE_WRITE), "VALUE_WRITE");
    EXPECT_STREQ(TaskTypeToString(ACL_MODEL_RI_TASK_VALUE_WAIT), "VALUE_WAIT");
    EXPECT_STREQ(TaskTypeToString(ACL_MODEL_RI_TASK_EVENT_RECORD), "NOTIFY");
    EXPECT_STREQ(TaskTypeToString(ACL_MODEL_RI_TASK_EVENT_WAIT), "WAIT");
    EXPECT_STREQ(TaskTypeToString(ACL_MODEL_RI_TASK_EVENT_RESET), "RESET");
    EXPECT_STREQ(TaskTypeToString(ACL_MODEL_RI_TASK_DEFAULT), "UNKNOWN");

    uint32_t taskRatio[2] = {1, 0};
    EXPECT_STREQ(GetKernelTypeString(ACL_KERNEL_TYPE_CUBE, taskRatio), "AIC_ONLY");
    EXPECT_STREQ(GetKernelTypeString(ACL_KERNEL_TYPE_VECTOR, taskRatio), "AIV_ONLY");
    EXPECT_STREQ(GetKernelTypeString(ACL_KERNEL_TYPE_MIX, taskRatio), "AIC_ONLY");
    taskRatio[1] = 1;
    EXPECT_STREQ(GetKernelTypeString(ACL_KERNEL_TYPE_MIX, taskRatio), "MIX_AIC_1_1");
    taskRatio[1] = 2;
    EXPECT_STREQ(GetKernelTypeString(ACL_KERNEL_TYPE_MIX, taskRatio), "MIX_AIC_1_2");

    EXPECT_STREQ(GetKernelTypeString(ACL_KERNEL_TYPE_AICPU, taskRatio), "DEFAULT");

    EXPECT_EQ(PtrToHexString(reinterpret_cast<void*>(0x1234)), "0x1234");
    EXPECT_EQ(Uint64ToHexString(0xabcd), "0xabcd");
}

TEST_F(SkDumpJsonDirectHelperTest, KernelAttrToJsonCoversKnownAndRawAttributes)
{
    aclrtLaunchKernelAttr attr{};
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attr.value.schemMode = 1;
    EXPECT_EQ(KernelAttrToJson(attr)["schemMode"], 1);

    attr = {};
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
    attr.value.dynUBufSize = 4096;
    EXPECT_EQ(KernelAttrToJson(attr)["dynUBufSize"], 4096);

    attr = {};
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
    attr.value.engineType = 2;
    EXPECT_EQ(KernelAttrToJson(attr)["engineType"], 2);

    attr = {};
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_BLOCK_TASK_PREFETCH;
    attr.value.isBlockTaskPrefetch = 1;
    EXPECT_EQ(KernelAttrToJson(attr)["blockTaskPrefetch"], 1);

    attr = {};
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_DATA_DUMP;
    attr.value.isDataDump = 1;
    EXPECT_EQ(KernelAttrToJson(attr)["dataDump"], 1);

    attr = {};
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = 77;
    EXPECT_EQ(KernelAttrToJson(attr)["timeout"], 77);

    attr = {};
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US;
    attr.value.timeout = 88;
    EXPECT_EQ(KernelAttrToJson(attr)["timeoutUs"], 88);

    attr = {};
    attr.id = static_cast<aclrtLaunchKernelAttrId>(99);
    attr.value.rsv[0] = 12345;
    EXPECT_EQ(KernelAttrToJson(attr)["rawValue"], 12345);
}

TEST_F(SkDumpJsonDirectHelperTest, BinaryBindMapAndResolvedFuncsCoverSuccessAndFailurePaths)
{
    aclrtBinHandle binHdl = reinterpret_cast<aclrtBinHandle>(0xaaaa);
    EXPECT_TRUE(InitSuperKernelBindMap(binHdl).empty());

    MOCKER(rtBinaryGetMetaNum).stubs().will(invoke(FakeRtBinaryGetMetaNumFailure));
    EXPECT_TRUE(InitSuperKernelBindMap(binHdl).empty());
}

TEST_F(SkDumpJsonDirectHelperTest, BinaryBindMapAndResolvedFuncsSerializeResolvedEntries)
{
    MOCKER(rtBinaryGetMetaNum).stubs().will(invoke(FakeRtBinaryGetMetaNumSuccess));
    MOCKER(rtBinaryGetMetaInfo).stubs().will(invoke(FakeRtBinaryGetMetaInfoSuccess));
    MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferEmpty));
    MOCKER(aclrtGetFunctionAddr).stubs().will(invoke(FakeAclrtGetFunctionAddrForDump));
    MOCKER(aclrtBinaryGetDevAddress).stubs().will(invoke(FakeAclrtBinaryGetDevAddressForDump));

    aclrtBinHandle binHdl = reinterpret_cast<aclrtBinHandle>(0xbbbb);
    SkBindMap bindMap = InitSuperKernelBindMap(binHdl);
    EXPECT_EQ(bindMap.size(), 2);
    EXPECT_EQ(bindMap[0x100][0], 0x1000);
    EXPECT_EQ(bindMap[0x200][3], 0x2300);

    ResolvedFunctionInfo resolvedFuncs[K_MAX_SPLIT_BIN_COUNT];
    uint32_t resolvedNum = 0;
    GetResolvedFuncsForDump(reinterpret_cast<aclrtFuncHandle>(0xcccc), binHdl, resolvedFuncs, resolvedNum);
    EXPECT_EQ(resolvedNum, K_MAX_SPLIT_BIN_COUNT);
    EXPECT_EQ(resolvedFuncs[0].funcAddr[0], 0x2000);
    EXPECT_EQ(resolvedFuncs[0].funcAddr[1], 0x3000);

    Json resolvedJson;
    AddResolvedFuncsToJson(resolvedJson, reinterpret_cast<aclrtFuncHandle>(0xcccc), binHdl);
    EXPECT_EQ(resolvedJson["resolvedNum"], K_MAX_SPLIT_BIN_COUNT);
    EXPECT_EQ(resolvedJson["resolvedFuncs"].size(), K_MAX_SPLIT_BIN_COUNT);
}

TEST_F(SkDumpJsonDirectHelperTest, BinaryBindMapMetaInfoFailureReturnsEmpty)
{
    MOCKER(rtBinaryGetMetaNum).stubs().will(invoke(FakeRtBinaryGetMetaNumSuccess));
    MOCKER(rtBinaryGetMetaInfo).stubs().will(invoke(FakeRtBinaryGetMetaInfoFailure));
    EXPECT_TRUE(InitSuperKernelBindMap(reinterpret_cast<aclrtBinHandle>(0xdddd)).empty());
}

TEST_F(SkDumpJsonDirectHelperTest, KernelNodeToJsonIncludesAttrsResolvedFuncsAndUpdatedParams)
{
    SuperKernelGraph graph(nullptr);
    graph.scopeIdxToName[0] = "(none)";
    graph.scopeIdxToName[1] = "scope_a";

    auto node = CreateKernelNode(10);
    node->SetNodeType(SkNodeType::NODE_KERNEL);
    node->SetIsFusible(true);
    node->SetInvalidated(true);
    node->SetPreNodeId(7);
    node->SetNextNodeId(11);
    node->SetScopeStreamIds({3, 5});
    node->scopeName = "scope_a";
    node->isScopeBegin = true;
    node->isScopeEnd = false;
    node->isPlaceholder = false;
    std::bitset<MAX_SCOPE_NUM> flags;
    flags.set(1);
    node->SetScopeBitFlags(flags);

    auto& kernelInfo = node->nodeInfos.kernelInfos;
    kernelInfo.kernelType = SkKernelType::MIX_AIC_1_1;
    kernelInfo.taskRatio[0] = 3;
    kernelInfo.taskRatio[1] = 4;
    kernelInfo.numBlocks = 32;
    kernelInfo.vecNum = 8;
    kernelInfo.cubeNum = 16;
    kernelInfo.devArgs = reinterpret_cast<void*>(0x1010);
    kernelInfo.opInfoPtr = reinterpret_cast<void*>(0x2020);
    kernelInfo.opInfoSize = 64;
    kernelInfo.funcName = "origin_kernel";
    kernelInfo.binHdl = reinterpret_cast<aclrtBinHandle>(0x3030);
    kernelInfo.funcHdl = reinterpret_cast<aclrtFuncHandle>(0x4040);
    kernelInfo.isScheModeOn = true;
    kernelInfo.resolvedNum = 2;
    kernelInfo.resolvedFuncs[0].funcAddr[0] = 0x1000;
    kernelInfo.resolvedFuncs[0].funcAddr[1] = 0x2000;
    kernelInfo.resolvedFuncs[0].funcOffset[0] = 0x10;
    kernelInfo.resolvedFuncs[0].funcOffset[1] = 0x20;
    kernelInfo.resolvedFuncs[0].prefetchCnt[0] = 1;
    kernelInfo.resolvedFuncs[0].prefetchCnt[1] = 2;
    kernelInfo.resolvedFuncs[0].symbolBind[0] = "GLOBAL";
    kernelInfo.resolvedFuncs[0].symbolBind[1] = "WEAK";

    aclrtLaunchKernelAttr attrs[2]{};
    attrs[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attrs[0].value.schemMode = 1;
    attrs[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_DATA_DUMP;
    attrs[1].value.isDataDump = 1;
    aclrtLaunchKernelCfg cfg{attrs, 2};
    kernelInfo.launchKernelCfg = &cfg;

    node->SetUpdate(true);
    node->taskParams.type = ACL_MODEL_RI_TASK_KERNEL;
    node->taskParams.kernelTaskParams.numBlocks = 64;
    node->taskParams.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x5050);
    node->taskParams.kernelTaskParams.args = reinterpret_cast<void*>(0x6060);
    node->taskParams.kernelTaskParams.argsSize = 128;
    node->taskParams.kernelTaskParams.isHostArgs = 1;
    node->taskParams.opInfoPtr = reinterpret_cast<void*>(0x7070);
    node->taskParams.opInfoSize = 256;

    Json nodeJson = NodeToJson(node.get(), graph);
    EXPECT_EQ(nodeJson["taskId"], 10);
    EXPECT_EQ(nodeJson["scopeName"], "scope_a");
    EXPECT_TRUE(nodeJson["isFusible"]);
    EXPECT_TRUE(nodeJson["isUpdated"]);
    EXPECT_TRUE(nodeJson["isInvalidated"]);
    EXPECT_EQ(nodeJson["preNodeId"], 7);
    EXPECT_EQ(nodeJson["nextNodeId"], 11);
    EXPECT_EQ(nodeJson["scopeStreamIds"].size(), 2);
    EXPECT_EQ(nodeJson["nodeType"], "KERNEL");
    EXPECT_EQ(nodeJson["kernelInfos"]["numBlocks"], 64);
    EXPECT_EQ(nodeJson["kernelInfos"]["funcHandle"], "0x5050");
    EXPECT_EQ(nodeJson["kernelInfos"]["devArgs"], "0x6060");
    EXPECT_EQ(nodeJson["kernelInfos"]["launchKernelCfgAttrs"].size(), 2);
    EXPECT_EQ(nodeJson["kernelInfos"]["resolvedFuncs"].size(), 2);

    Json kernelInfos;
    KernelInfos emptyKernelInfo;
    AddLaunchKernelCfgAttrs(kernelInfos, emptyKernelInfo);
    EXPECT_FALSE(kernelInfos.contains("launchKernelCfgAttrs"));
}

TEST_F(SkDumpJsonDirectHelperTest, SyncNodeToJsonCoversNotifyWaitAndUpdatedMemoryParams)
{
    SuperKernelGraph graph(nullptr);

    auto notifyNode = CreateMemoryNode(20, ACL_MODEL_RI_TASK_VALUE_WRITE);
    notifyNode->SetNodeType(SkNodeType::NODE_MEMORY_WRITE);
    notifyNode->SetNotifyExpandVecNum(6);
    notifyNode->SetNotifyExpandCubeNum(7);
    notifyNode->nodeInfos.syncInfos.eventId = 0x1111;
    notifyNode->nodeInfos.syncInfos.addrValue = reinterpret_cast<void*>(0x2222);
    notifyNode->nodeInfos.syncInfos.correspondingWaitNodeIds = {21, 22};
    notifyNode->nodeInfos.syncInfos.correspondingResetNodeIds = {23};
    notifyNode->nodeInfos.syncInfos.correspondingMemoryWriteNodeIds = {24};
    notifyNode->nodeInfos.syncInfos.memoryValue = 0x3333;
    notifyNode->nodeInfos.syncInfos.memoryWaitFlag = 9;
    notifyNode->nodeInfos.syncInfos.eventFlag = 0x44;
    notifyNode->SetUpdate(true);
    notifyNode->taskParams.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
    notifyNode->taskParams.valueWriteTaskParams.devAddr = reinterpret_cast<void*>(0x5555);
    notifyNode->taskParams.valueWriteTaskParams.value = 0x6666;

    Json notifyJson = NodeToJson(notifyNode.get(), graph);
    EXPECT_EQ(notifyJson["nodeType"], "VALUE_WRITE");
    EXPECT_EQ(notifyJson["syncInfos"]["addrValue"], "0x5555");
    EXPECT_EQ(notifyJson["syncInfos"]["memoryValue"], 0x6666);
    EXPECT_EQ(notifyJson["syncInfos"]["vecNum"], 6);
    EXPECT_EQ(notifyJson["syncInfos"]["cubeNum"], 7);
    EXPECT_EQ(notifyJson["syncInfos"]["correspondingWaitNodeIds"].size(), 2);
    EXPECT_EQ(notifyJson["syncInfos"]["correspondingResetNodeIds"].size(), 1);
    EXPECT_EQ(notifyJson["syncInfos"]["correspondingMemoryWriteNodeIds"].size(), 1);
    EXPECT_EQ(notifyJson["syncInfos"]["memoryWaitFlag"], 9);
    EXPECT_EQ(notifyJson["syncInfos"]["eventFlag"], "0x68");

    auto waitNode = CreateMemoryNode(21, ACL_MODEL_RI_TASK_VALUE_WAIT);
    waitNode->SetNodeType(SkNodeType::NODE_MEMORY_WAIT);
    waitNode->nodeInfos.syncInfos.eventId = 0x7777;
    waitNode->nodeInfos.syncInfos.correspondingNotifyNodeId = 20;
    waitNode->SetUpdate(true);
    waitNode->taskParams.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
    waitNode->taskParams.valueWaitTaskParams.devAddr = reinterpret_cast<void*>(0x8888);
    waitNode->taskParams.valueWaitTaskParams.value = 0x9999;
    waitNode->taskParams.valueWaitTaskParams.flag = 12;

    Json waitJson = NodeToJson(waitNode.get(), graph);
    EXPECT_EQ(waitJson["nodeType"], "VALUE_WAIT");
    EXPECT_EQ(waitJson["syncInfos"]["eventId"], "0x7777");
    EXPECT_EQ(waitJson["syncInfos"]["correspondingNotifyNodeId"], 20);
    EXPECT_EQ(waitJson["syncInfos"]["addrValue"], "0x8888");
    EXPECT_EQ(waitJson["syncInfos"]["memoryValue"], 0x9999);
    EXPECT_EQ(waitJson["syncInfos"]["memoryWaitFlag"], 12);

    EXPECT_TRUE(NodeToJson(nullptr, graph).empty());
}

TEST_F(SkDumpJsonDirectHelperTest, GraphJsonBuildersSerializeGraphAndFusedScopes)
{
    SuperKernelGraph graph(nullptr);
    graph.streams = {
        reinterpret_cast<aclrtStream>(0x1),
        reinterpret_cast<aclrtStream>(0x2)
    };
    graph.headNodes = {10, 30};
    graph.nodeSizeInStream = {2, 1};
    graph.scopeIdxToName[0] = "(none)";
    graph.scopeIdxToName[1] = "scope_a";

    auto node1 = CreateKernelNode(10);
    AddKernelInfoToNode(node1.get());
    auto node2 = CreateKernelNode(20);
    AddKernelInfoToNode(node2.get());
    auto node3 = CreateMemoryNode(30, ACL_MODEL_RI_TASK_EVENT_RECORD);
    node3->SetNodeType(SkNodeType::NODE_NOTIFY);

    SuperKernelBaseNode* node1Ptr = node1.get();
    SuperKernelBaseNode* node2Ptr = node2.get();
    SuperKernelBaseNode* node3Ptr = node3.get();
    graph.graphMap[10] = std::move(node1);
    graph.graphMap[20] = std::move(node2);
    graph.graphMap[30] = std::move(node3);

    Json graphJson = BuildGraphNodesJson(graph);
    EXPECT_EQ(graphJson["graph"]["totalNodes"], 3);
    EXPECT_EQ(graphJson["graph"]["totalStreams"], 2);
    EXPECT_EQ(graphJson["graph"]["streams"].size(), 2);
    EXPECT_EQ(graphJson["graph"]["scopeNames"].size(), 2);
    EXPECT_EQ(graphJson["nodes"].size(), 3);

    SuperKernelScopeInfo scopeInfo;
    scopeInfo.SetNodes({node1Ptr, node2Ptr, nullptr});
    std::bitset<MAX_SCOPE_NUM> flags;
    flags.set(1);
    scopeInfo.SetScopeBitFlags(flags);
    scopeInfo.AddScopeStreamInfo({0, 0, 1, 2});
    scopeInfo.MutableExtInfo().scopeName = "scope_a";
    scopeInfo.MutableExtInfo().failReason = ScopeFailReason::STREAM_SYNC_FAIL;

    std::vector<SuperKernelScopeInfo> mappingScopes;
    mappingScopes.push_back(std::move(scopeInfo));
    NodeScopeMapping mapping = BuildNodeScopeMapping(mappingScopes);
    EXPECT_NE(FindScopeByNodeId(10, mapping), nullptr);
    EXPECT_EQ(FindScopeByNodeId(20, mapping), nullptr);
    EXPECT_EQ(FindScopeByNodeId(30, mapping), nullptr);

    std::vector<SuperKernelScopeInfo> scopeInfos;
    SuperKernelScopeInfo fusedScope;
    fusedScope.SetNodes({node1Ptr, node2Ptr});
    fusedScope.SetScopeBitFlags(flags);
    fusedScope.AddScopeStreamInfo({0, 0, 1, 2});
    fusedScope.MutableExtInfo().scopeName = "scope_a";
    fusedScope.MutableExtInfo().fusionStatus = ScopeFusionStatus::FAILED;
    fusedScope.MutableExtInfo().failReason = ScopeFailReason::STREAM_SYNC_FAIL;
    scopeInfos.push_back(std::move(fusedScope));

    Json fusedJson = BuildFusedGraphJson(graph, scopeInfos);
    EXPECT_EQ(fusedJson["graph"]["totalNodes"], 3);
    EXPECT_EQ(fusedJson["graph"]["totalScopes"], 1);
    EXPECT_EQ(fusedJson["nodes"].size(), 2);
    EXPECT_TRUE(fusedJson["nodes"][0].contains("nodeIds"));
    EXPECT_TRUE(fusedJson["nodes"][1].contains("taskId"));

    Json nodesArray = Json::array();
    std::unordered_set<uint64_t> processedNodes;
    ProcessNodeEntry(nodesArray, 10, BuildNodeScopeMapping(scopeInfos), graph, processedNodes);
    ProcessNodeEntry(nodesArray, 20, BuildNodeScopeMapping(scopeInfos), graph, processedNodes);
    ProcessNodeEntry(nodesArray, 30, BuildNodeScopeMapping(scopeInfos), graph, processedNodes);
    EXPECT_EQ(nodesArray.size(), 2);
}

TEST_F(SkDumpJsonDirectHelperTest, ParseOriginalScopesGroupsNamedScopeNodesByBitFlags)
{
    SuperKernelGraph graph(nullptr);
    std::bitset<MAX_SCOPE_NUM> flags;
    flags.set(3);

    auto marker = CreateKernelNode(1);
    marker->SetIsScopeNode(true);
    marker->isScopeBegin = true;
    marker->scopeName = "named_scope";
    marker->SetScopeBitFlags(flags);

    auto scopedNode = CreateKernelNode(2);
    scopedNode->SetNodeType(SkNodeType::NODE_KERNEL);
    scopedNode->SetScopeBitFlags(flags);

    auto emptyFlagsNode = CreateKernelNode(3);
    emptyFlagsNode->SetNodeType(SkNodeType::NODE_KERNEL);

    graph.graphMap[1] = std::move(marker);
    graph.graphMap[2] = std::move(scopedNode);
    graph.graphMap[3] = std::move(emptyFlagsNode);

    graph.ParseOriginalScopes();
    ASSERT_EQ(graph.GetOriginalScopeInfos().size(), 1);
    EXPECT_EQ(graph.GetOriginalScopeInfos()[0].nodeIds, std::vector<uint64_t>({2}));
    EXPECT_EQ(graph.GetOriginalScopeInfos()[0].GetScopeBitFlags(), flags);
}

TEST_F(SkDumpJsonDirectHelperTest, ParseOriginalScopesCreatesSingleScopeWhenNoScopeNodesExist)
{
    SuperKernelGraph graph(nullptr);
    std::bitset<MAX_SCOPE_NUM> flags;
    flags.set(4);

    auto node1 = CreateKernelNode(10);
    node1->SetNodeType(SkNodeType::NODE_KERNEL);
    node1->SetScopeBitFlags(flags);
    auto node2 = CreateMemoryNode(11, ACL_MODEL_RI_TASK_EVENT_WAIT);
    node2->SetNodeType(SkNodeType::NODE_WAIT);
    node2->SetScopeBitFlags(flags);

    graph.graphMap[10] = std::move(node1);
    graph.graphMap[11] = std::move(node2);

    graph.ParseOriginalScopes();
    ASSERT_EQ(graph.GetOriginalScopeInfos().size(), 1);
    EXPECT_EQ(graph.GetOriginalScopeInfos()[0].nodeIds, std::vector<uint64_t>({10, 11}));
}

TEST_F(SkDumpJsonDirectHelperTest, ParseOriginalScopesExcludesUnnamedScopeBody)
{
    SuperKernelGraph graph(nullptr);
    std::bitset<MAX_SCOPE_NUM> flags;
    flags.set(5);

    auto unnamedBegin = CreateKernelNode(20);
    unnamedBegin->SetIsScopeNode(true);
    unnamedBegin->isScopeBegin = true;
    unnamedBegin->SetScopeBitFlags(flags);

    auto insideNode = CreateKernelNode(21);
    insideNode->SetNodeType(SkNodeType::NODE_KERNEL);
    insideNode->SetScopeBitFlags(flags);

    auto unnamedEnd = CreateKernelNode(22);
    unnamedEnd->SetIsScopeNode(true);
    unnamedEnd->isScopeEnd = true;
    unnamedEnd->SetScopeBitFlags(flags);

    auto placeholder = CreateKernelNode(23);
    placeholder->SetIsScopeNode(true);
    placeholder->isPlaceholder = true;
    placeholder->SetScopeBitFlags(flags);

    auto outsideNode = CreateKernelNode(24);
    outsideNode->SetNodeType(SkNodeType::NODE_KERNEL);
    outsideNode->SetScopeBitFlags(flags);

    graph.graphMap[20] = std::move(unnamedBegin);
    graph.graphMap[21] = std::move(insideNode);
    graph.graphMap[22] = std::move(unnamedEnd);
    graph.graphMap[23] = std::move(placeholder);
    graph.graphMap[24] = std::move(outsideNode);

    graph.ParseOriginalScopes();
    ASSERT_EQ(graph.GetOriginalScopeInfos().size(), 1);
    EXPECT_EQ(graph.GetOriginalScopeInfos()[0].nodeIds, std::vector<uint64_t>({24}));
}

TEST_F(SkDumpJsonDirectHelperTest, SkTaskQueueAndFileWritingHelpers)
{
    SkTask task;
    task.numBlocks = 16;
    task.funcCnt = 2;
    task.nodeType = SkKernelType::AIC_ONLY;
    ASSERT_TRUE(task.Init(2));

    TaskQue* queue = task.GetTaskQue();
    ASSERT_NE(queue, nullptr);
    queue->taskCnt = 2;
    queue->taskInfos[0].index = 10;
    queue->taskInfos[0].type = SkTaskType::TYPE_FUNC;
    queue->taskInfos[0].numBlocks = 8;
    queue->taskInfos[0].entryCnt = 2;
    queue->taskInfos[0].args = 0x1000;
    queue->taskInfos[0].entry[0] = 0x2000;
    queue->taskInfos[0].entry[1] = 0x3000;
    queue->taskInfos[0].debugOptions = 3;
    queue->taskInfos[1].index = 11;
    queue->taskInfos[1].type = SkTaskType::TYPE_EVENT_WAIT;
    queue->taskInfos[1].numBlocks = 1;
    queue->taskInfos[1].entryCnt = 5;
    queue->taskInfos[1].args = 0x4000;
    queue->taskInfos[1].entry[0] = 0x5000;
    queue->taskInfos[1].entry[1] = 0x6000;
    queue->taskInfos[1].entry[2] = 0x7000;
    queue->taskInfos[1].entry[3] = 0x8000;
    queue->taskInfos[1].debugOptions = 16;

    Json taskJson = SkTaskToJson(task, "AIC");
    EXPECT_EQ(taskJson["queueName"], "AIC");
    EXPECT_EQ(taskJson["taskQue"]["taskCnt"], 2);
    EXPECT_EQ(taskJson["taskQue"]["taskInfos"][0]["entries"].size(), 2);
    EXPECT_EQ(taskJson["taskQue"]["taskInfos"][1]["entries"].size(), 4);

    Json noQueueJson = SkTaskToJson(SkTask(), "EMPTY");
    EXPECT_FALSE(noQueueJson.contains("taskQue"));

    std::string outputPath = CreateSkMetaDirectory(nullptr) + "/ut_write_json.json";
    ASSERT_TRUE(WriteJsonToFile(taskJson, outputPath));
    std::ifstream inFile(outputPath);
    EXPECT_TRUE(inFile.good());
    EXPECT_FALSE(WriteJsonToFile(taskJson, "missing_dir/ut_write_json.json"));

    SuperKernelGraph graph(nullptr);
    sk::logger::FileLogger::Instance().SetEnabled(true);
    EXPECT_TRUE(DumpSkTaskQueueToJson(graph, task, task));
    sk::logger::FileLogger::Instance().SetEnabled(false);
}

TEST_F(SkDumpJsonDirectHelperTest, RawTaskParamJsonCoversAllTaskTypes)
{
    aclmdlRITaskParams params{};
    params.type = ACL_MODEL_RI_TASK_KERNEL;
    params.taskGrp = reinterpret_cast<aclrtTaskGrp>(0x100);
    params.opInfoPtr = reinterpret_cast<void*>(0x200);
    params.opInfoSize = 16;
    params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x300);
    params.kernelTaskParams.args = reinterpret_cast<void*>(0x400);
    params.kernelTaskParams.argsSize = 32;
    params.kernelTaskParams.isHostArgs = 1;
    params.kernelTaskParams.numBlocks = 64;
    aclrtLaunchKernelAttr attr{};
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attr.value.schemMode = 1;
    aclrtLaunchKernelCfg cfg{&attr, 1};
    params.kernelTaskParams.cfg = &cfg;

    Json kernelTaskJson = TaskToJson(0, 7, ACL_MODEL_RI_TASK_KERNEL, &params, nullptr);
    EXPECT_EQ(kernelTaskJson["streamId"], 7);
    EXPECT_EQ(kernelTaskJson["taskType"], "KERNEL");
    EXPECT_EQ(kernelTaskJson["kernelParams"]["numBlocks"], 64);
    EXPECT_EQ(kernelTaskJson["kernelParams"]["taskGrp"], "0x100");
    EXPECT_EQ(kernelTaskJson["kernelParams"]["kernelType"], "DEFAULT");
    EXPECT_EQ(kernelTaskJson["kernelParams"]["taskRatio"].size(), 2);

    params = {};
    params.type = ACL_MODEL_RI_TASK_EVENT_RECORD;
    params.eventRecordTaskParams.event = reinterpret_cast<aclrtEvent>(0x500);
    params.eventRecordTaskParams.eventFlag = 9;
    Json recordJson;
    AddTaskParamsToJson(recordJson, params, nullptr);
    EXPECT_EQ(recordJson["eventRecordParams"]["eventId"], "0x500");
    EXPECT_EQ(recordJson["eventRecordParams"]["eventFlag"], 9);

    params = {};
    params.type = ACL_MODEL_RI_TASK_EVENT_WAIT;
    params.eventWaitTaskParams.event = reinterpret_cast<aclrtEvent>(0x600);
    params.eventWaitTaskParams.eventFlag = 10;
    Json waitJson = TaskToJson(1, 8, ACL_MODEL_RI_TASK_EVENT_WAIT, &params, nullptr);
    EXPECT_EQ(waitJson["eventWaitParams"]["event"], "0x600");
    EXPECT_EQ(waitJson["eventWaitParams"]["eventFlag"], 10);

    params = {};
    params.type = ACL_MODEL_RI_TASK_EVENT_RESET;
    params.eventResetTaskParams.event = reinterpret_cast<aclrtEvent>(0x700);
    params.eventResetTaskParams.eventFlag = 11;
    Json resetJson;
    AddTaskParamsToJson(resetJson, params, nullptr);
    EXPECT_EQ(resetJson["eventResetParams"]["event"], "0x700");
    EXPECT_EQ(resetJson["eventResetParams"]["eventFlag"], 11);

    params = {};
    params.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
    params.valueWriteTaskParams.devAddr = reinterpret_cast<void*>(0x800);
    params.valueWriteTaskParams.value = 0x900;
    Json valueWriteJson;
    AddTaskParamsToJson(valueWriteJson, params, nullptr);
    EXPECT_EQ(valueWriteJson["valueWriteParams"]["devAddr"], "0x800");
    EXPECT_EQ(valueWriteJson["valueWriteParams"]["value"], "0x900");

    params = {};
    params.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
    params.valueWaitTaskParams.devAddr = reinterpret_cast<void*>(0xa00);
    params.valueWaitTaskParams.value = 0xb00;
    params.valueWaitTaskParams.flag = 13;
    Json valueWaitJson;
    AddTaskParamsToJson(valueWaitJson, params, nullptr);
    EXPECT_EQ(valueWaitJson["valueWaitParams"]["devAddr"], "0xa00");
    EXPECT_EQ(valueWaitJson["valueWaitParams"]["value"], "0xb00");
    EXPECT_EQ(valueWaitJson["valueWaitParams"]["flag"], 13);

    params = {};
    params.type = ACL_MODEL_RI_TASK_DEFAULT;
    Json defaultJson;
    AddTaskParamsToJson(defaultJson, params, nullptr);
    EXPECT_TRUE(defaultJson.empty());
    EXPECT_EQ(TaskToJson(2, 9, ACL_MODEL_RI_TASK_DEFAULT, nullptr, nullptr)["taskType"], "UNKNOWN");
}

TEST_F(SkDumpJsonDirectHelperTest, RawTaskProcessingCoversSeqIdParamsAndStreamBranches)
{
    UtRITaskInternal task{};
    task.taskId = 123;
    task.type = ACL_MODEL_RI_TASK_EVENT_RECORD;
    task.params.type = ACL_MODEL_RI_TASK_EVENT_RECORD;
    task.params.eventRecordTaskParams.event = reinterpret_cast<aclrtEvent>(0x1230);
    task.params.eventRecordTaskParams.eventFlag = 77;

    Json tasksJson = Json::array();
    size_t totalTasks = 0;
    ProcessTaskToJson(tasksJson, 0, 42, reinterpret_cast<aclmdlRITask>(&task), totalTasks);
    EXPECT_EQ(totalTasks, 1);
    EXPECT_EQ(tasksJson[0]["taskId"], 123);
    EXPECT_EQ(tasksJson[0]["eventRecordParams"]["eventFlag"], 77);

    MOCKER(aclmdlRITaskGetParams).stubs().will(invoke(FakeAclmdlRITaskGetParamsFailure));
    ProcessTaskToJson(tasksJson, 1, 42, reinterpret_cast<aclmdlRITask>(&task), totalTasks);
    EXPECT_EQ(totalTasks, 2);
    EXPECT_FALSE(tasksJson[1].contains("eventRecordParams"));
}

TEST_F(SkDumpJsonDirectHelperTest, RawStreamProcessingCoversTaskFetchSuccessAndFailures)
{
    g_utStreamTask = {};
    g_utStreamTask.taskId = 456;
    g_utStreamTask.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
    g_utStreamTask.params.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
    g_utStreamTask.params.valueWaitTaskParams.devAddr = reinterpret_cast<void*>(0x4560);
    g_utStreamTask.params.valueWaitTaskParams.value = 0x4561;
    g_utStreamTask.params.valueWaitTaskParams.flag = 3;

    SkUtResetTestControls();
    SkUtSetStreamId(0, 900);
    MOCKER(aclmdlRIGetTasksByStream).stubs().will(invoke(FakeAclmdlRIGetTasksByStreamOneTask));

    size_t totalTasks = 0;
    Json streamJson = ProcessStreamToJson(reinterpret_cast<aclrtStream>(0x1), 0, totalTasks);
    EXPECT_EQ(totalTasks, 1);
    EXPECT_EQ(streamJson["streamId"], 900);
    EXPECT_EQ(streamJson["taskCount"], 1);
    EXPECT_EQ(streamJson["tasks"][0]["taskId"], 456);
    EXPECT_EQ(streamJson["tasks"][0]["valueWaitParams"]["flag"], 3);
}

TEST_F(SkDumpJsonDirectHelperTest, RawStreamProcessingCoversCountFailure)
{
    MOCKER(aclmdlRIGetTasksByStream).stubs().will(invoke(FakeAclmdlRIGetTasksByStreamCountFailure));
    size_t totalTasks = 0;
    Json streamJson = ProcessStreamToJson(reinterpret_cast<aclrtStream>(0x1), 0, totalTasks);
    EXPECT_TRUE(streamJson.empty());
    EXPECT_EQ(totalTasks, 0);
}

TEST_F(SkDumpJsonDirectHelperTest, RawStreamProcessingCoversFetchFailure)
{
    MOCKER(aclmdlRIGetTasksByStream).stubs().will(invoke(FakeAclmdlRIGetTasksByStreamFetchFailure));
    size_t totalTasks = 0;
    Json streamJson = ProcessStreamToJson(reinterpret_cast<aclrtStream>(0x1), 0, totalTasks);
    EXPECT_TRUE(streamJson.empty());
    EXPECT_EQ(totalTasks, 0);
}

TEST_F(SkDumpJsonDirectHelperTest, RawTaskDumpCoversStreamCollectionAndErrorPaths)
{
    aclmdlRI modelRI = reinterpret_cast<aclmdlRI>(0x12345678);
    SkUtResetTestControls();
    SkUtSetModelStreamNum(2);
    SkUtSetStreamId(0, 100);
    SkUtSetStreamId(1, 101);

    sk::logger::FileLogger::Instance().SetEnabled(true);
    EXPECT_TRUE(DumpModelRITasksToJson(modelRI, 0, nullptr, "ut_raw_tasks"));

    SuperKernelOptionsManager optsMgr;
    EXPECT_TRUE(DumpModelRITasksToJson(modelRI, 1, &optsMgr, "ut_raw_tasks_opts"));

    SkUtSetAclrtStreamGetIdRet(ACL_ERROR_FAILURE);
    EXPECT_TRUE(DumpModelRITasksToJson(modelRI, 2, nullptr, "ut_raw_tasks_stream_id_fail"));

    SkUtResetTestControls();
    SkUtSetAclmdlRIGetStreamsRet(0, ACL_ERROR_FAILURE);
    EXPECT_FALSE(DumpModelRITasksToJson(modelRI, 0, nullptr, "ut_raw_tasks_stream_count_fail"));

    SkUtResetTestControls();
    SkUtSetModelStreamNum(1);
    SkUtSetAclmdlRIGetStreamsRet(1, ACL_ERROR_FAILURE);
    EXPECT_FALSE(DumpModelRITasksToJson(modelRI, 0, nullptr, "ut_raw_tasks_stream_fetch_fail"));

    sk::logger::FileLogger::Instance().SetEnabled(false);
    EXPECT_TRUE(DumpModelRITasksToJson(modelRI, 0, nullptr, "ut_raw_tasks_disabled"));
    SkUtResetTestControls();
}

TEST_F(SkDumpJsonDirectHelperTest, ScopePrintingHelpersCoverKernelSetAndBatchBranches)
{
    SuperKernelGraph graph(nullptr);
    graph.scopeIdxToName[1] = "scope_one";
    std::bitset<MAX_SCOPE_NUM> flags;
    flags.set(1);

    auto node1 = CreateKernelNode(201);
    node1->SetNodeType(SkNodeType::NODE_KERNEL);
    node1->nodeInfos.kernelInfos.funcName = "kernel_201";
    node1->SetScopeBitFlags(flags);
    auto node2 = CreateKernelNode(202);
    node2->SetNodeType(SkNodeType::NODE_KERNEL);
    node2->nodeInfos.kernelInfos.funcName = "kernel_202";
    node2->SetScopeBitFlags(flags);
    auto scopeMarker = CreateKernelNode(203);
    scopeMarker->SetNodeType(SkNodeType::NODE_KERNEL);
    scopeMarker->isScopeBegin = true;
    scopeMarker->SetScopeBitFlags(flags);

    SuperKernelBaseNode* node1Ptr = node1.get();
    SuperKernelBaseNode* node2Ptr = node2.get();
    SuperKernelBaseNode* markerPtr = scopeMarker.get();
    graph.graphMap[201] = std::move(node1);
    graph.graphMap[202] = std::move(node2);
    graph.graphMap[203] = std::move(scopeMarker);
    graph.originalScopeInfos_.push_back({1, flags, {201, 203, 999}});

    std::vector<uint64_t> longNodeIds;
    for (uint64_t i = 0; i < 700; ++i) {
        longNodeIds.push_back(1000000000000ULL + i);
    }
    PrintNodeIdsInBatches(longNodeIds, "    hugeNodeIds: [");
    PrintFusedNodeIds(longNodeIds);
    PrintOriginalScopes(graph);

    const SuperKernelBaseNode* nodes[] = {nullptr, node1Ptr, markerPtr, node2Ptr};
    auto kernelIds = CollectKernelIdsFromNodes(nodes, 4);
    EXPECT_EQ(kernelIds.size(), 2);
    EXPECT_EQ(kernelIds.count(201), 1);
    EXPECT_EQ(kernelIds.count(202), 1);

    SuperKernelScopeInfo scopeInfo;
    scopeInfo.SetScopeBitFlags(flags);
    scopeInfo.SetNodes({node1Ptr, node2Ptr, markerPtr});
    scopeInfo.MutableExtInfo().filteredNodes = {nullptr, markerPtr, node1Ptr, node2Ptr};
    scopeInfo.MutableExtInfo().fusionStatus = ScopeFusionStatus::FAILED;
    scopeInfo.MutableExtInfo().failReason = ScopeFailReason::STREAM_SYNC_FAIL;
    scopeInfo.SetBreakInfo(ScopeBreakInfo()
        .SetReason(ScopeBreakReason::UNFUSIBLE_NODE)
        .SetTriggerNode(202, 0)
        .SetFusionFailReason(FusionFailReason::IN_UNFUSIBLE_SCOPE));

    auto originalSets = BuildOriginalKernelSets(graph, graph.GetOriginalScopeInfos());
    EXPECT_FALSE(IsKernelSetMatch(scopeInfo, originalSets, graph));

    std::vector<uint64_t> fusedNodeIds;
    bool hasKernel = false;
    CollectFusedNodes(scopeInfo.GetExtInfo(), fusedNodeIds, hasKernel);
    EXPECT_TRUE(hasKernel);
    EXPECT_EQ(fusedNodeIds.size(), 2);

    std::vector<SuperKernelScopeInfo> scopeInfos;
    scopeInfos.push_back(std::move(scopeInfo));
    PrintFusedScopes(graph, scopeInfos);
}

TEST_F(SkDumpJsonDirectHelperTest, ScopePrintingHelpersCoverSuccessFusionStatus)
{
    SuperKernelGraph graph(nullptr);
    std::bitset<MAX_SCOPE_NUM> flags;
    flags.set(2);

    auto node = CreateKernelNode(301);
    node->SetNodeType(SkNodeType::NODE_KERNEL);
    node->SetScopeBitFlags(flags);
    SuperKernelBaseNode* nodePtr = node.get();
    graph.graphMap[301] = std::move(node);
    graph.originalScopeInfos_.push_back({2, flags, {301}});

    SuperKernelScopeInfo scopeInfo;
    scopeInfo.SetScopeBitFlags(flags);
    scopeInfo.SetNodes({nodePtr});
    scopeInfo.MutableExtInfo().filteredNodes = {nodePtr};
    scopeInfo.MutableExtInfo().fusionStatus = ScopeFusionStatus::SUCCESS;

    std::vector<SuperKernelScopeInfo> scopeInfos;
    scopeInfos.push_back(std::move(scopeInfo));
    PrintFusedScopes(graph, scopeInfos);
}

TEST_F(SkDumpJsonDirectHelperTest, EnabledDumpApisWriteGraphFiles)
{
    SuperKernelGraph graph(nullptr);
    graph.scopeIdxToName[0] = "scope_zero";

    auto node1 = CreateKernelNode(100);
    AddKernelInfoToNode(node1.get());
    auto node2 = CreateMemoryNode(101, ACL_MODEL_RI_TASK_EVENT_WAIT);
    node2->SetNodeType(SkNodeType::NODE_WAIT);
    node2->nodeInfos.syncInfos.eventId = 0xabc;
    node2->nodeInfos.syncInfos.correspondingNotifyNodeId = 100;

    SuperKernelBaseNode* node1Ptr = node1.get();
    SuperKernelBaseNode* node2Ptr = node2.get();
    graph.graphMap[100] = std::move(node1);
    graph.graphMap[101] = std::move(node2);

    SuperKernelScopeInfo scopeInfo;
    scopeInfo.SetNodes({node1Ptr, node2Ptr});
    scopeInfo.MutableExtInfo().scopeName = "scope_zero";
    scopeInfo.MutableExtInfo().fusionStatus = ScopeFusionStatus::SUCCESS;
    scopeInfo.MutableExtInfo().filteredNodes = {node1Ptr, node2Ptr};
    std::vector<SuperKernelScopeInfo> scopeInfos;
    scopeInfos.push_back(std::move(scopeInfo));

    sk::logger::FileLogger::Instance().SetEnabled(true);
    EXPECT_TRUE(DumpGraphNodesToJson(graph));
    EXPECT_TRUE(DumpFusedGraphToJson(graph, scopeInfos));
    sk::logger::FileLogger::Instance().SetEnabled(false);
}

// ==================== OptionsManagerToJson Tests ====================

class OptionsManagerToJsonTest : public SkDumpJsonTest {};

TEST_F(OptionsManagerToJsonTest, EmptyOptionsManager)
{
    SuperKernelOptionsManager optsMgr;
    Json optionsJson = OptionsManagerToJson(optsMgr);
    EXPECT_EQ(optionsJson["numOptions"], 0);
    EXPECT_TRUE(optionsJson["options"].is_array());
    EXPECT_EQ(optionsJson["options"].size(), 0);
}

TEST_F(OptionsManagerToJsonTest, OptionsManagerWithNumberOption)
{
    SuperKernelOptionsManager optsMgr;
    optsMgr.AddOption(std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 1, 0, 2));
    auto* opt = optsMgr.GetOption(aclskOptionType::PRELOAD_CODE);
    if (opt != nullptr) {
        opt->SetValue(1);
    }

    Json optionsJson = OptionsManagerToJson(optsMgr);
    EXPECT_EQ(optionsJson["numOptions"], 1);
    EXPECT_EQ(optionsJson["options"].size(), 1);
    EXPECT_EQ(optionsJson["options"][0]["name"], "preload_code");
    EXPECT_EQ(optionsJson["options"][0]["type"], static_cast<int>(aclskOptionType::PRELOAD_CODE));
    EXPECT_EQ(optionsJson["options"][0]["value"], 1);
}

TEST_F(OptionsManagerToJsonTest, OptionsManagerWithMultipleNumberOptions)
{
    SuperKernelOptionsManager optsMgr;
    optsMgr.AddOption(std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 1, 0, 2));
    optsMgr.AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 4, 1, 4));
    optsMgr.AddOption(std::make_unique<NumberOptOption>("stream_fusion", aclskOptionType::STREAM_FUSION, 1, 0, 1));

    auto* preloadOpt = optsMgr.GetOption(aclskOptionType::PRELOAD_CODE);
    if (preloadOpt != nullptr) preloadOpt->SetValue(1);
    auto* splitOpt = optsMgr.GetOption(aclskOptionType::SPLIT_MODE);
    if (splitOpt != nullptr) splitOpt->SetValue(2);
    auto* fusionOpt = optsMgr.GetOption(aclskOptionType::STREAM_FUSION);
    if (fusionOpt != nullptr) fusionOpt->SetValue(0);

    Json optionsJson = OptionsManagerToJson(optsMgr);
    EXPECT_EQ(optionsJson["numOptions"], 3);
    EXPECT_EQ(optionsJson["options"].size(), 3);
}

TEST_F(OptionsManagerToJsonTest, OptionsManagerWithStringListOption)
{
    SuperKernelOptionsManager optsMgr;
    optsMgr.AddOption(std::make_unique<StringListOptOption>("dcci_disable_on_kernel", aclskOptionType::DCCI_DISABLE_ON_KERNEL));
    auto* opt = optsMgr.GetOption(aclskOptionType::DCCI_DISABLE_ON_KERNEL);
    if (opt != nullptr) {
        std::vector<std::string> kernelNames = {"kernel1", "kernel2", "kernel3"};
        opt->SetValue(kernelNames);
    }

    Json optionsJson = OptionsManagerToJson(optsMgr);
    EXPECT_EQ(optionsJson["numOptions"], 1);
    EXPECT_EQ(optionsJson["options"][0]["name"], "dcci_disable_on_kernel");
    EXPECT_TRUE(optionsJson["options"][0]["value"].is_array());
    EXPECT_EQ(optionsJson["options"][0]["value"].size(), 3);
    EXPECT_EQ(optionsJson["options"][0]["value"][0], "kernel1");
    EXPECT_EQ(optionsJson["options"][0]["value"][1], "kernel2");
    EXPECT_EQ(optionsJson["options"][0]["value"][2], "kernel3");
}

TEST_F(OptionsManagerToJsonTest, OptionsManagerWithMapOption)
{
    SuperKernelOptionsManager optsMgr;
    optsMgr.AddOption(std::make_unique<MapOptOption>("opt_extend_option", aclskOptionType::OPT_EXTEND_OPTION));
    auto* opt = optsMgr.GetOption(aclskOptionType::OPT_EXTEND_OPTION);
    if (opt != nullptr) {
        std::unordered_map<std::string, std::vector<std::string>> mapValue;
        mapValue["key1"] = {"value1", "value2"};
        mapValue["key2"] = {"value3"};
        opt->SetValue(mapValue);
    }

    Json optionsJson = OptionsManagerToJson(optsMgr);
    EXPECT_EQ(optionsJson["numOptions"], 1);
    EXPECT_EQ(optionsJson["options"][0]["name"], "opt_extend_option");
    EXPECT_TRUE(optionsJson["options"][0]["value"].is_object());
    EXPECT_EQ(optionsJson["options"][0]["value"]["key1"].size(), 2);
    EXPECT_EQ(optionsJson["options"][0]["value"]["key2"].size(), 1);
}

TEST_F(OptionsManagerToJsonTest, OptionsManagerWithMixedOptions)
{
    SuperKernelOptionsManager optsMgr;
    optsMgr.AddOption(std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 1, 0, 2));
    optsMgr.AddOption(std::make_unique<StringListOptOption>("dcci_disable_on_kernel", aclskOptionType::DCCI_DISABLE_ON_KERNEL));
    optsMgr.AddOption(std::make_unique<MapOptOption>("opt_extend_option", aclskOptionType::OPT_EXTEND_OPTION));

    auto* preloadOpt = optsMgr.GetOption(aclskOptionType::PRELOAD_CODE);
    if (preloadOpt != nullptr) preloadOpt->SetValue(1);

    auto* dcciOpt = optsMgr.GetOption(aclskOptionType::DCCI_DISABLE_ON_KERNEL);
    if (dcciOpt != nullptr) {
        std::vector<std::string> kernels = {"test_kernel"};
        dcciOpt->SetValue(kernels);
    }

    auto* extendOpt = optsMgr.GetOption(aclskOptionType::OPT_EXTEND_OPTION);
    if (extendOpt != nullptr) {
        std::unordered_map<std::string, std::vector<std::string>> extendValue;
        extendValue["param1"] = {"v1"};
        extendOpt->SetValue(extendValue);
    }

    Json optionsJson = OptionsManagerToJson(optsMgr);
    EXPECT_EQ(optionsJson["numOptions"], 3);
    EXPECT_EQ(optionsJson["options"].size(), 3);
}

TEST_F(OptionsManagerToJsonTest, OptionsManagerWithConstantCodegenOption)
{
    SuperKernelOptionsManager optsMgr;
    optsMgr.AddOption(std::make_unique<NumberOptOption>("constant_codegen", aclskOptionType::CONSTANT_CODEGEN, 0, 0, 1));
    auto* opt = optsMgr.GetOption(aclskOptionType::CONSTANT_CODEGEN);
    if (opt != nullptr) {
        opt->SetValue(1);
    }

    Json optionsJson = OptionsManagerToJson(optsMgr);
    EXPECT_EQ(optionsJson["numOptions"], 1);
    EXPECT_EQ(optionsJson["options"][0]["name"], "constant_codegen");
    EXPECT_EQ(optionsJson["options"][0]["value"], 1);
}

TEST_F(OptionsManagerToJsonTest, OptionsManagerWithAutoOpParallelOption)
{
    SuperKernelOptionsManager optsMgr;
    optsMgr.AddOption(std::make_unique<NumberOptOption>("auto_op_parallel", aclskOptionType::AUTO_OP_PARALLEL, 0, 0, 1));
    auto* opt = optsMgr.GetOption(aclskOptionType::AUTO_OP_PARALLEL);
    if (opt != nullptr) {
        opt->SetValue(0);
    }

    Json optionsJson = OptionsManagerToJson(optsMgr);
    EXPECT_EQ(optionsJson["numOptions"], 1);
    EXPECT_EQ(optionsJson["options"][0]["name"], "auto_op_parallel");
    EXPECT_EQ(optionsJson["options"][0]["value"], 0);
}

TEST_F(OptionsManagerToJsonTest, OptionsManagerWithDebugCrossCoreSyncCheckOption)
{
    SuperKernelOptionsManager optsMgr;
    optsMgr.AddOption(std::make_unique<NumberOptOption>("debug_cross_core_sync_check", aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK, 0, 0, 1));
    auto* opt = optsMgr.GetOption(aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK);
    if (opt != nullptr) {
        opt->SetValue(1);
    }

    Json optionsJson = OptionsManagerToJson(optsMgr);
    EXPECT_EQ(optionsJson["numOptions"], 1);
    EXPECT_EQ(optionsJson["options"][0]["name"], "debug_cross_core_sync_check");
    EXPECT_EQ(optionsJson["options"][0]["value"], 1);
}

TEST_F(OptionsManagerToJsonTest, OptionsManagerWithDebugDcciBeforeKernelStart)
{
    SuperKernelOptionsManager optsMgr;
    optsMgr.AddOption(std::make_unique<StringListOptOption>("dcci_before_kernel_start", aclskOptionType::DCCI_BEFORE_KERNEL_START));
    auto* opt = optsMgr.GetOption(aclskOptionType::DCCI_BEFORE_KERNEL_START);
    if (opt != nullptr) {
        std::vector<std::string> kernels = {"kernel_a", "kernel_b"};
        opt->SetValue(kernels);
    }

    Json optionsJson = OptionsManagerToJson(optsMgr);
    EXPECT_EQ(optionsJson["numOptions"], 1);
    EXPECT_EQ(optionsJson["options"][0]["name"], "dcci_before_kernel_start");
    EXPECT_EQ(optionsJson["options"][0]["value"].size(), 2);
}

TEST_F(OptionsManagerToJsonTest, OptionsManagerWithDebugDcciAfterKernelEnd)
{
    SuperKernelOptionsManager optsMgr;
    optsMgr.AddOption(std::make_unique<StringListOptOption>("dcci_after_kernel_end", aclskOptionType::DCCI_AFTER_KERNEL_END));
    auto* opt = optsMgr.GetOption(aclskOptionType::DCCI_AFTER_KERNEL_END);
    if (opt != nullptr) {
        std::vector<std::string> kernels = {"kernel_end"};
        opt->SetValue(kernels);
    }

    Json optionsJson = OptionsManagerToJson(optsMgr);
    EXPECT_EQ(optionsJson["numOptions"], 1);
    EXPECT_EQ(optionsJson["options"][0]["name"], "dcci_after_kernel_end");
    EXPECT_EQ(optionsJson["options"][0]["value"].size(), 1);
}

TEST_F(OptionsManagerToJsonTest, OptionsManagerWithDebugSyncAll)
{
    SuperKernelOptionsManager optsMgr;
    optsMgr.AddOption(std::make_unique<NumberOptOption>("debug_sync_all", aclskOptionType::DEBUG_SYNC_ALL, 0, 0, 1));
    auto* opt = optsMgr.GetOption(aclskOptionType::DEBUG_SYNC_ALL);
    if (opt != nullptr) {
        opt->SetValue(1);
    }

    Json optionsJson = OptionsManagerToJson(optsMgr);
    EXPECT_EQ(optionsJson["numOptions"], 1);
    EXPECT_EQ(optionsJson["options"][0]["name"], "debug_sync_all");
    EXPECT_EQ(optionsJson["options"][0]["value"], 1);
}

TEST_F(OptionsManagerToJsonTest, OptionsManagerWithDebugExtendOption)
{
    SuperKernelOptionsManager optsMgr;
    optsMgr.AddOption(std::make_unique<MapOptOption>("debug_extend_option", aclskOptionType::DEBUG_EXTEND_OPTION));
    auto* opt = optsMgr.GetOption(aclskOptionType::DEBUG_EXTEND_OPTION);
    if (opt != nullptr) {
        std::unordered_map<std::string, std::vector<std::string>> debugExtend;
        debugExtend["debug_flag"] = {"enabled"};
        opt->SetValue(debugExtend);
    }

    Json optionsJson = OptionsManagerToJson(optsMgr);
    EXPECT_EQ(optionsJson["numOptions"], 1);
    EXPECT_EQ(optionsJson["options"][0]["name"], "debug_extend_option");
    EXPECT_TRUE(optionsJson["options"][0]["value"].contains("debug_flag"));
}

TEST_F(OptionsManagerToJsonTest, OptionsManagerWithAllOptionTypes)
{
    SuperKernelOptionsManager optsMgr;
    optsMgr.AddOption(std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 1, 0, 2));
    optsMgr.AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 4, 1, 4));
    optsMgr.AddOption(std::make_unique<StringListOptOption>("dcci_disable_on_kernel", aclskOptionType::DCCI_DISABLE_ON_KERNEL));
    optsMgr.AddOption(std::make_unique<StringListOptOption>("dcci_before_kernel_start", aclskOptionType::DCCI_BEFORE_KERNEL_START));
    optsMgr.AddOption(std::make_unique<StringListOptOption>("dcci_after_kernel_end", aclskOptionType::DCCI_AFTER_KERNEL_END));
    optsMgr.AddOption(std::make_unique<NumberOptOption>("debug_sync_all", aclskOptionType::DEBUG_SYNC_ALL, 0, 0, 1));
    optsMgr.AddOption(std::make_unique<MapOptOption>("opt_extend_option", aclskOptionType::OPT_EXTEND_OPTION));
    optsMgr.AddOption(std::make_unique<MapOptOption>("debug_extend_option", aclskOptionType::DEBUG_EXTEND_OPTION));
    optsMgr.AddOption(std::make_unique<NumberOptOption>("stream_fusion", aclskOptionType::STREAM_FUSION, 1, 0, 1));
    optsMgr.AddOption(std::make_unique<NumberOptOption>("constant_codegen", aclskOptionType::CONSTANT_CODEGEN, 0, 0, 1));
    optsMgr.AddOption(std::make_unique<NumberOptOption>("auto_op_parallel", aclskOptionType::AUTO_OP_PARALLEL, 0, 0, 1));
    optsMgr.AddOption(std::make_unique<NumberOptOption>("debug_cross_core_sync_check", aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK, 0, 0, 1));

    Json optionsJson = OptionsManagerToJson(optsMgr);
    EXPECT_EQ(optionsJson["numOptions"], 12);
    EXPECT_EQ(optionsJson["options"].size(), 12);
}

// ==================== DumpModelRITasksToJson Comprehensive Tests ====================

TEST_F(DumpModelRITasksToJsonTest, WithMultipleOptions)
{
    sk::logger::FileLogger::Instance().SetEnabled(true);
    
    aclmdlRI modelRI = reinterpret_cast<aclmdlRI>(0x12345678);
    SkUtResetTestControls();
    SkUtSetModelStreamNum(1);
    SkUtSetStreamId(0, 100);
    SkUtSetStreamTaskNum(0, 0);

    SuperKernelOptionsManager optsMgr;
    optsMgr.AddOption(std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 1, 0, 2));
    optsMgr.AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 4, 1, 4));
    auto* preloadOpt = optsMgr.GetOption(aclskOptionType::PRELOAD_CODE);
    if (preloadOpt != nullptr) preloadOpt->SetValue(2);
    auto* splitOpt = optsMgr.GetOption(aclskOptionType::SPLIT_MODE);
    if (splitOpt != nullptr) splitOpt->SetValue(3);

    EXPECT_TRUE(DumpModelRITasksToJson(modelRI, 0, &optsMgr, "ut_raw_tasks_with_multi_opts"));
    
    sk::logger::FileLogger::Instance().SetEnabled(false);
    SkUtResetTestControls();
}

TEST_F(DumpModelRITasksToJsonTest, WithStringListOptions)
{
    sk::logger::FileLogger::Instance().SetEnabled(true);
    
    aclmdlRI modelRI = reinterpret_cast<aclmdlRI>(0x12345678);
    SkUtResetTestControls();
    SkUtSetModelStreamNum(1);
    SkUtSetStreamId(0, 100);
    SkUtSetStreamTaskNum(0, 0);

    SuperKernelOptionsManager optsMgr;
    optsMgr.AddOption(std::make_unique<StringListOptOption>("dcci_disable_on_kernel", aclskOptionType::DCCI_DISABLE_ON_KERNEL));
    auto* opt = optsMgr.GetOption(aclskOptionType::DCCI_DISABLE_ON_KERNEL);
    if (opt != nullptr) {
        std::vector<std::string> kernels = {"kernel_alpha", "kernel_beta"};
        opt->SetValue(kernels);
    }

    EXPECT_TRUE(DumpModelRITasksToJson(modelRI, 0, &optsMgr, "ut_raw_tasks_with_string_list_opts"));
    
    sk::logger::FileLogger::Instance().SetEnabled(false);
    SkUtResetTestControls();
}

TEST_F(DumpModelRITasksToJsonTest, WithMapOptions)
{
    sk::logger::FileLogger::Instance().SetEnabled(true);
    
    aclmdlRI modelRI = reinterpret_cast<aclmdlRI>(0x12345678);
    SkUtResetTestControls();
    SkUtSetModelStreamNum(1);
    SkUtSetStreamId(0, 100);
    SkUtSetStreamTaskNum(0, 0);

    SuperKernelOptionsManager optsMgr;
    optsMgr.AddOption(std::make_unique<MapOptOption>("opt_extend_option", aclskOptionType::OPT_EXTEND_OPTION));
    auto* opt = optsMgr.GetOption(aclskOptionType::OPT_EXTEND_OPTION);
    if (opt != nullptr) {
        std::unordered_map<std::string, std::vector<std::string>> extendValue;
        extendValue["config1"] = {"param1", "param2"};
        extendValue["config2"] = {"param3"};
        opt->SetValue(extendValue);
    }

    EXPECT_TRUE(DumpModelRITasksToJson(modelRI, 0, &optsMgr, "ut_raw_tasks_with_map_opts"));
    
    sk::logger::FileLogger::Instance().SetEnabled(false);
    SkUtResetTestControls();
}
