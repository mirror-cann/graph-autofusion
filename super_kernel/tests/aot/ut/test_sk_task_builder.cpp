/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include <memory>
#include <map>
#include <string>
#include <vector>

#define private public
#define protected public
#include "sk_graph.h"
#include "sk_task_builder.h"
#include "sk_options_manager.h"
#include "sk_node.h"
#include "stub/ut_common_stubs.h"

class SkTaskBuilderTest : public testing::Test {
protected:
    void SetUp() override
    {
        SkUtResetTestControls();
        graph = std::make_unique<SuperKernelGraph>();
        opts = std::make_unique<SuperKernelOptionsManager>();
        builder = std::make_unique<SkTaskBuilder>(*opts, *graph);
    }

    void TearDown() override
    {
        SkUtResetTestControls();
    }

    SuperKernelBaseNode* CreateKernelNode(uint64_t nodeId)
    {
        auto node = std::make_unique<SuperKernelKernelNode>(
            nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
        node->SetNodeType(SkNodeType::NODE_KERNEL);
        node->SetNodeId(nodeId);
        node->SetPreNodeId(INVALID_TASK_ID);
        node->SetNextNodeId(INVALID_TASK_ID);
        auto* ptr = node.get();
        graph->graphMap[nodeId] = std::move(node);
        return ptr;
    }

    SuperKernelBaseNode* CreateKernelNodeEx(uint64_t nodeId, uint32_t streamIdx, uint64_t preNodeId,
                                            uint64_t nextNodeId, SkKernelType kernelType = SkKernelType::AIC_ONLY)
    {
        auto node = std::make_unique<SuperKernelKernelNode>(
            nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, streamIdx, INVALID_STREAM_ID, preNodeId);
        node->SetNodeType(SkNodeType::NODE_KERNEL);
        node->SetNodeId(nodeId);
        node->SetPreNodeId(preNodeId);
        node->SetNextNodeId(nextNodeId);
        node->nodeInfos.kernelInfos.kernelType = kernelType;
        node->nodeInfos.kernelInfos.numBlocks = 1;
        node->nodeInfos.kernelInfos.funcName = "k";
        node->nodeInfos.kernelInfos.devArgs = reinterpret_cast<void*>(0x1);
        node->nodeInfos.kernelInfos.binHdl = reinterpret_cast<aclrtBinHandle>(0x2);
        node->nodeInfos.kernelInfos.funcHdl = reinterpret_cast<aclrtFuncHandle>(0x3);
        node->nodeInfos.kernelInfos.resolvedFuncs[0].funcAddr[0] = 0x1000;
        node->nodeInfos.kernelInfos.resolvedFuncs[0].funcAddr[1] = 0x2000;
        node->nodeInfos.kernelInfos.resolvedFuncs[0].prefetchCnt[0] = 2;
        node->nodeInfos.kernelInfos.resolvedFuncs[0].prefetchCnt[1] = 1;
        auto* ptr = node.get();
        graph->graphMap[nodeId] = std::move(node);
        return ptr;
    }

    SuperKernelBaseNode* CreateNotifyNodeEx(uint64_t nodeId, uint32_t streamIdx, uint64_t preNodeId,
                                            uint64_t nextNodeId, uint64_t eventId)
    {
        auto node = std::make_unique<SuperKernelMemoryNode>(
            nullptr, ACL_MODEL_RI_TASK_EVENT_RECORD, 0, streamIdx, INVALID_STREAM_ID, preNodeId);
        node->SetNodeType(SkNodeType::NODE_NOTIFY);
        node->SetNodeId(nodeId);
        node->SetPreNodeId(preNodeId);
        node->SetNextNodeId(nextNodeId);
        node->nodeInfos.syncInfos.eventId = eventId;
        node->nodeInfos.syncInfos.addrValue = reinterpret_cast<void*>(0xABC);
        auto* ptr = node.get();
        graph->graphMap[nodeId] = std::move(node);
        return ptr;
    }

    SuperKernelBaseNode* CreateWaitNodeEx(uint64_t nodeId, uint32_t streamIdx, uint64_t preNodeId,
                                          uint64_t nextNodeId, uint64_t eventId)
    {
        auto node = std::make_unique<SuperKernelMemoryNode>(
            nullptr, ACL_MODEL_RI_TASK_EVENT_WAIT, 0, streamIdx, INVALID_STREAM_ID, preNodeId);
        node->SetNodeType(SkNodeType::NODE_WAIT);
        node->SetNodeId(nodeId);
        node->SetPreNodeId(preNodeId);
        node->SetNextNodeId(nextNodeId);
        node->nodeInfos.syncInfos.eventId = eventId;
        node->nodeInfos.syncInfos.addrValue = reinterpret_cast<void*>(0xDEF);
        auto* ptr = node.get();
        graph->graphMap[nodeId] = std::move(node);
        return ptr;
    }

    SuperKernelBaseNode* CreateResetNodeEx(uint64_t nodeId, uint32_t streamIdx, uint64_t preNodeId,
                                           uint64_t nextNodeId)
    {
        auto node = std::make_unique<SuperKernelMemoryNode>(
            nullptr, ACL_MODEL_RI_TASK_EVENT_RESET, 0, streamIdx, INVALID_STREAM_ID, preNodeId);
        node->SetNodeType(SkNodeType::NODE_RESET);
        node->SetNodeId(nodeId);
        node->SetPreNodeId(preNodeId);
        node->SetNextNodeId(nextNodeId);
        auto* ptr = node.get();
        graph->graphMap[nodeId] = std::move(node);
        return ptr;
    }

    std::unique_ptr<SuperKernelGraph> graph;
    std::unique_ptr<SuperKernelOptionsManager> opts;
    std::unique_ptr<SkTaskBuilder> builder;
};

TEST_F(SkTaskBuilderTest, Build_EmptyTasks_ReturnEmptyLaunchInfo)
{
    std::vector<SuperKernelBaseNode*> tasks;
    std::vector<SuperKernelBaseNode*> customTasks;

    SkLaunchInfo launchInfo = builder->Build("Unknown", tasks, customTasks);

    EXPECT_EQ(launchInfo.entryInfo.skEntryFunc, nullptr);
    EXPECT_EQ(launchInfo.entryInfo.nodeCnt, 0U);
    EXPECT_EQ(launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, InitTaskSyncInfos_UnsupportedNodeType_ReturnFalse)
{
    auto resetNode = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_EVENT_RESET, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    resetNode->SetNodeType(SkNodeType::NODE_RESET);
    resetNode->SetNodeId(21);
    resetNode->SetPreNodeId(INVALID_TASK_ID);
    resetNode->SetNextNodeId(INVALID_TASK_ID);

    std::vector<SuperKernelBaseNode*> tasks;
    tasks.push_back(resetNode.get());

    bool ok = builder->InitTaskSyncInfos(tasks);

    EXPECT_FALSE(ok);
    graph->graphMap[21] = std::move(resetNode);
}

TEST_F(SkTaskBuilderTest, Build_UnsupportedNodeType_ReturnEmptyLaunchInfo)
{
    auto resetNode = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_EVENT_RESET, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    resetNode->SetNodeType(SkNodeType::NODE_RESET);
    resetNode->SetNodeId(22);
    resetNode->SetPreNodeId(INVALID_TASK_ID);
    resetNode->SetNextNodeId(INVALID_TASK_ID);

    std::vector<SuperKernelBaseNode*> tasks;
    tasks.push_back(resetNode.get());
    std::vector<SuperKernelBaseNode*> customTasks;

    graph->graphMap[22] = std::move(resetNode);

    SkLaunchInfo launchInfo = builder->Build("Unknown", tasks, customTasks);

    EXPECT_EQ(launchInfo.entryInfo.skEntryFunc, nullptr);
    EXPECT_EQ(launchInfo.entryInfo.nodeCnt, 0U);
    EXPECT_EQ(launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, InitTaskSyncInfos_KernelNotifyWait_Success)
{
    auto* k1 = CreateKernelNodeEx(1, 0, INVALID_TASK_ID, 2, SkKernelType::AIC_ONLY);
    auto* n2 = CreateNotifyNodeEx(2, 0, 1, 3, 99);
    auto* w3 = CreateWaitNodeEx(3, 1, INVALID_TASK_ID, 4, 99);
    auto* k4 = CreateKernelNodeEx(4, 1, 3, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
    (void)k1;
    (void)n2;
    (void)w3;
    (void)k4;

    std::vector<SuperKernelBaseNode*> tasks = {
        graph->GetNodeById(1), graph->GetNodeById(2), graph->GetNodeById(3), graph->GetNodeById(4)};

    ASSERT_TRUE(builder->InitTaskSyncInfos(tasks));
    ASSERT_EQ(builder->taskSyncInfos_.size(), tasks.size());
}

TEST_F(SkTaskBuilderTest, PrecomputeAndOptimizeSyncRelations_Smoke)
{
    auto* k1 = CreateKernelNodeEx(10, 0, INVALID_TASK_ID, 11, SkKernelType::AIC_ONLY);
    auto* n2 = CreateNotifyNodeEx(11, 0, 10, INVALID_TASK_ID, 77);
    auto* w3 = CreateWaitNodeEx(20, 1, INVALID_TASK_ID, 21, 77);
    auto* k4 = CreateKernelNodeEx(21, 1, 20, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
    (void)k1;
    (void)n2;
    (void)w3;
    (void)k4;

    std::vector<SuperKernelBaseNode*> tasks = {
        graph->GetNodeById(10), graph->GetNodeById(11), graph->GetNodeById(20), graph->GetNodeById(21)};

    ASSERT_TRUE(builder->PrecomputeSyncRelationsFromGraph(tasks));
    builder->PrintSyncInfo("ut-before-opt");
    builder->OptimizeSyncRelations();
    builder->PrintSyncInfo("ut-after-opt");
}

TEST_F(SkTaskBuilderTest, InsertAndRemoveSyncInfo_Works)
{
    builder->taskSyncInfos_.resize(3);
    builder->taskSyncInfos_[0].queueType = SkQueueType::AIC;
    builder->taskSyncInfos_[1].queueType = SkQueueType::AIV;
    builder->taskSyncInfos_[2].queueType = SkQueueType::AIV;

    builder->InsertSyncEvent(0, 1);
    builder->InsertSyncEvent(0, 2);

    EXPECT_FALSE(builder->taskSyncInfos_[0].cubSendInfo.empty());
    builder->RemoveSyncInfo(0, 1, false, SyncDirection::CUB_TO_VEC);
    EXPECT_TRUE(builder->taskSyncInfos_[0].cubSendInfo.find(1) == builder->taskSyncInfos_[0].cubSendInfo.end());
}

TEST_F(SkTaskBuilderTest, JudgeRemoveCrossSync_CanDetectCrossing)
{
    builder->taskSyncInfos_.resize(4);
    builder->taskSyncInfos_[1].vecRecvInfo[2] = SyncDirection::CUB_TO_VEC;

    EXPECT_TRUE(builder->JudgeRemoveCrossSync(0, 2, true));
}

TEST_F(SkTaskBuilderTest, AddAndDispatchTasks_Smoke)
{
    SkTask aic;
    SkTask aiv;
    ASSERT_TRUE(aic.taskQue.Init(16));
    ASSERT_TRUE(aiv.taskQue.Init(16));

    auto* kernel = CreateKernelNodeEx(1001, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
    auto* notify = CreateNotifyNodeEx(1002, 0, 1001, INVALID_TASK_ID, 55);

    SkDfxInfo dfx{};
    ASSERT_TRUE(builder->AddSyncTask(aic, 0, SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC));
    ASSERT_TRUE(builder->AddEventTask(aiv, notify, 1, SkTaskType::TYPE_EVENT_NOTIFY));
    ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 2, 0, 1, SkTaskType::TYPE_FUNC, 1));

    ASSERT_TRUE(builder->DispatchEventTask(aic, aiv, notify, 3, SkTaskType::TYPE_EVENT_NOTIFY, SkQueueType::AIV));
    ASSERT_TRUE(builder->DispatchFuncTask(aic, aiv, kernel, &dfx, 4, 1, SkTaskType::TYPE_FUNC, SkQueueType::AIC));

    builder->taskSyncInfos_.resize(2);
    builder->taskSyncInfos_[0].queueType = SkQueueType::AIC;
    builder->taskSyncInfos_[1].queueType = SkQueueType::AIV;
    std::map<size_t, SyncDirection> syncInfo{{1, SyncDirection::CUB_TO_VEC}};
    ASSERT_TRUE(builder->DispatchSyncTasks(aic, aiv, 5, syncInfo, true, SkQueueType::AIC));
}

TEST_F(SkTaskBuilderTest, EntryInfoAndArgs_GenerateSuccessfully)
{
    SkTask aic;
    SkTask aiv;
    ASSERT_TRUE(aic.taskQue.Init(8));
    ASSERT_TRUE(aiv.taskQue.Init(8));

    auto* kernel = CreateKernelNodeEx(2001, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
    SkDfxInfo dfx{};
    ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

    SkHostEntryInfo entryInfo = builder->GenEntryInfo(aic, aiv);
    EXPECT_NE(entryInfo.skEntryFunc, nullptr);

    DeviceArgsPtr devArgs = builder->GenEntryArgs(aic, aiv, &dfx, 1);
    EXPECT_NE(devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, GetPreFetchCnt_OptionBranches)
{
    opts->AddOption(std::make_unique<NumberOptOption>("preload", aclskOtionType::PRELOAD_CODE, 1, 0, 2));
    ResolvedFunctionInfo resolved{};
    resolved.prefetchCnt[0] = 3;
    resolved.prefetchCnt[1] = 4;

    auto p1 = builder->GetPreFetchCnt(resolved);
    EXPECT_EQ(p1.first, 3);
    EXPECT_EQ(p1.second, 4);

    opts->GetOption(aclskOtionType::PRELOAD_CODE)->SetValue(0);
    auto p2 = builder->GetPreFetchCnt(resolved);
    EXPECT_EQ(p2.first, 16);
    EXPECT_EQ(p2.second, 8);

    opts->GetOption(aclskOtionType::PRELOAD_CODE)->SetValue(2);
    auto p3 = builder->GetPreFetchCnt(resolved);
    EXPECT_EQ(p3.first, 0);
    EXPECT_EQ(p3.second, 0);
}

TEST_F(SkTaskBuilderTest, Build_DebugMode_TriggerDumpHelpers)
{
    opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOtionType::SPLIT_MODE, 1, 1, 4));
    opts->AddOption(std::make_unique<NumberOptOption>("debug_sync_all", aclskOtionType::DEBUG_SYNC_ALL, 1, 0, 1));

    std::vector<SuperKernelBaseNode*> tasks;
    tasks.push_back(CreateKernelNodeEx(3001, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY));

    SkLaunchInfo launchInfo = builder->Build("Unknown", tasks, {});
    EXPECT_NE(launchInfo.entryInfo.skEntryFunc, nullptr);
    EXPECT_NE(launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, InitTaskSyncInfos_NotifyMissingKernel_TriggerSearchFailureLogPath)
{
    auto* notify = CreateNotifyNodeEx(4001, 0, INVALID_TASK_ID, INVALID_TASK_ID, 123);
    std::vector<SuperKernelBaseNode*> tasks = {notify};

    EXPECT_FALSE(builder->InitTaskSyncInfos(tasks));
}

TEST_F(SkTaskBuilderTest, AddSyncTask_WhenQueueFull_TriggerExpand)
{
    SkTask task;
    ASSERT_TRUE(task.taskQue.Init(1));
    task.taskQue.get()->taskCnt = 1;

    EXPECT_TRUE(builder->AddSyncTask(task, 0, SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC));
    EXPECT_GE(task.taskQue.get()->cap, 2U);
    EXPECT_EQ(task.taskQue.get()->taskCnt, 2U);
}

TEST_F(SkTaskBuilderTest, DispatchFuncTask_UnknownQueueType_ReturnFalse)
{
    SkTask aic;
    SkTask aiv;
    ASSERT_TRUE(aic.taskQue.Init(4));
    ASSERT_TRUE(aiv.taskQue.Init(4));
    auto* kernel = CreateKernelNodeEx(5001, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
    SkDfxInfo dfx{};

    EXPECT_FALSE(builder->DispatchFuncTask(aic, aiv, kernel, &dfx, 0, 1, SkTaskType::TYPE_FUNC, SkQueueType::UNKNOWN));
}

TEST_F(SkTaskBuilderTest, DispatchEventTask_UnsupportedNodeType_ReturnFalse)
{
    SkTask aic;
    SkTask aiv;
    ASSERT_TRUE(aic.taskQue.Init(4));
    ASSERT_TRUE(aiv.taskQue.Init(4));
    auto* kernel = CreateKernelNodeEx(5002, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);

    EXPECT_FALSE(builder->DispatchEventTask(aic, aiv, kernel, 0, SkTaskType::TYPE_EVENT_NOTIFY, SkQueueType::AIC));
}

TEST_F(SkTaskBuilderTest, AddFuncTask_UnresolvedAddress_ReturnFalse)
{
    SkTask aic;
    ASSERT_TRUE(aic.taskQue.Init(4));
    auto* kernel = CreateKernelNodeEx(5003, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
    auto* kernelNode = dynamic_cast<SuperKernelKernelNode*>(kernel);
    ASSERT_NE(kernelNode, nullptr);
    kernelNode->nodeInfos.kernelInfos.resolvedFuncs[0].funcAddr[0] = 0;

    SkDfxInfo dfx{};
    EXPECT_FALSE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));
}

TEST_F(SkTaskBuilderTest, DispatchSyncTasks_UnknownSyncType_SkipsAndReturnsTrue)
{
    SkTask aic;
    SkTask aiv;
    ASSERT_TRUE(aic.taskQue.Init(8));
    ASSERT_TRUE(aiv.taskQue.Init(8));

    builder->taskSyncInfos_.resize(2);
    builder->taskSyncInfos_[0].queueType = SkQueueType::AIC;
    builder->taskSyncInfos_[1].queueType = SkQueueType::AIV;

    std::map<size_t, SyncDirection> syncInfo{{1, SyncDirection::MIX_TO_MIX}};
    EXPECT_FALSE(builder->DispatchSyncTasks(aic, aiv, 0, syncInfo, true, SkQueueType::AIC));
    EXPECT_EQ(aic.taskQue.get()->taskCnt, 0U);
    EXPECT_EQ(aiv.taskQue.get()->taskCnt, 0U);
}

TEST_F(SkTaskBuilderTest, Build_CustomTaskMissingPreviousKernel_FallbackAndSucceed)
{
    opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOtionType::SPLIT_MODE, 1, 1, 4));

    auto* kernel = CreateKernelNodeEx(5004, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
    auto* customWait = CreateWaitNodeEx(5005, 0, INVALID_TASK_ID, INVALID_TASK_ID, 100);
    (void)kernel;

    std::vector<SuperKernelBaseNode*> tasks = {graph->GetNodeById(5004)};
    std::vector<SuperKernelBaseNode*> customTasks = {customWait};
    SkLaunchInfo launchInfo = builder->Build("Unknown", tasks, customTasks);

    // Missing previous kernel for custom WAIT now falls back to AIV queue instead of hard-fail.
    EXPECT_NE(launchInfo.entryInfo.skEntryFunc, nullptr);
    EXPECT_NE(launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, Build_CustomTaskUnsupportedNodeType_ReturnEmpty)
{
    opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOtionType::SPLIT_MODE, 1, 1, 4));

    auto* kernel = CreateKernelNodeEx(5006, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
    auto* customKernel = CreateKernelNodeEx(5007, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
    (void)kernel;

    std::vector<SuperKernelBaseNode*> tasks = {graph->GetNodeById(5006)};
    std::vector<SuperKernelBaseNode*> customTasks = {customKernel};
    SkLaunchInfo launchInfo = builder->Build("Unknown", tasks, customTasks);

    EXPECT_EQ(launchInfo.entryInfo.skEntryFunc, nullptr);
    EXPECT_EQ(launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, GenEntryInfo_Mix12Branch_Selected)
{
    SkTask aic;
    SkTask aiv;
    aic.funcCnt = 1;
    aiv.funcCnt = 1;
    aic.nodeType = SkKernelType::MIX_AIC_1_2;
    aiv.nodeType = SkKernelType::MIX_AIC_1_2;
    aic.numBlocks = 4;
    aiv.numBlocks = 10;
    ASSERT_TRUE(aiv.taskQue.Init(2));

    SkHostEntryInfo entryInfo = builder->GenEntryInfo(aic, aiv);
    ASSERT_NE(entryInfo.skEntryFunc, nullptr);
    EXPECT_EQ(entryInfo.entryType, SkKernelType::MIX_AIC_1_2);
}

TEST_F(SkTaskBuilderTest, InitTaskSyncInfos_SearchDirectionFailureBranches)
{
    auto* resetNoNext = CreateResetNodeEx(6001, 0, INVALID_TASK_ID, INVALID_TASK_ID);
    auto* notifyNoNext = CreateNotifyNodeEx(6002, 0, 6001, INVALID_TASK_ID, 1);
    (void)resetNoNext;

    std::vector<SuperKernelBaseNode*> tasksNoNext = {notifyNoNext};
    EXPECT_FALSE(builder->InitTaskSyncInfos(tasksNoNext));

    auto* resetLoopA = CreateResetNodeEx(6101, 0, 6102, INVALID_TASK_ID);
    auto* resetLoopB = CreateResetNodeEx(6102, 0, 6101, INVALID_TASK_ID);
    auto* notifyLoop = CreateNotifyNodeEx(6103, 0, 6101, INVALID_TASK_ID, 2);
    (void)resetLoopA;
    (void)resetLoopB;

    std::vector<SuperKernelBaseNode*> tasksLoop = {notifyLoop};
    EXPECT_FALSE(builder->InitTaskSyncInfos(tasksLoop));

    const uint64_t chainStart = 6201;
    const uint64_t chainLen = 105;
    for (uint64_t i = 0; i < chainLen; ++i) {
        uint64_t nodeId = chainStart + i;
        uint64_t preNodeId = (i + 1 < chainLen) ? (nodeId + 1) : (nodeId + 1);
        CreateResetNodeEx(nodeId, 0, preNodeId, INVALID_TASK_ID);
    }
    auto* tailReset = CreateResetNodeEx(chainStart + chainLen, 0, chainStart + chainLen, INVALID_TASK_ID);
    (void)tailReset;
    auto* notifyMaxHop = CreateNotifyNodeEx(6300, 0, chainStart, INVALID_TASK_ID, 3);

    std::vector<SuperKernelBaseNode*> tasksMaxHop = {notifyMaxHop};
    EXPECT_FALSE(builder->InitTaskSyncInfos(tasksMaxHop));
}

TEST_F(SkTaskBuilderTest, SyncOptimizationAndDispatchSyncBranches)
{
    builder->taskSyncInfos_.clear();
    builder->taskSyncInfos_.resize(5);
    for (auto& info : builder->taskSyncInfos_) {
        info.queueType = SkQueueType::AIC;
    }
    builder->taskSyncInfos_[1].queueType = SkQueueType::AIV;
    builder->taskSyncInfos_[2].queueType = SkQueueType::AIV;
    builder->taskSyncInfos_[3].queueType = SkQueueType::AIV;
    builder->taskSyncInfos_[4].queueType = SkQueueType::AIV;

    builder->taskSyncInfos_[0].cubSendInfo[3] = SyncDirection::CUB_TO_VEC;
    builder->taskSyncInfos_[1].vecRecvInfo[2] = SyncDirection::CUB_TO_VEC;
    builder->taskSyncInfos_[2].cubSendInfo[4] = SyncDirection::CUB_TO_VEC;
    builder->taskSyncInfos_[2].cubSendInfo[3] = SyncDirection::CUB_TO_VEC;
    builder->taskSyncInfos_[3].vecRecvInfo[2] = SyncDirection::CUB_TO_VEC;
    builder->taskSyncInfos_[4].vecRecvInfo[2] = SyncDirection::CUB_TO_VEC;

    builder->taskSyncInfos_[2].vecSendInfo[4] = SyncDirection::VEC_TO_CUB;
    builder->taskSyncInfos_[1].cubRecvInfo[3] = SyncDirection::VEC_TO_CUB;
    builder->taskSyncInfos_[0].vecSendInfo[4] = SyncDirection::VEC_TO_CUB;
    builder->taskSyncInfos_[0].vecSendInfo[3] = SyncDirection::VEC_TO_CUB;
    builder->taskSyncInfos_[3].cubRecvInfo[0] = SyncDirection::VEC_TO_CUB;
    builder->taskSyncInfos_[4].cubRecvInfo[0] = SyncDirection::VEC_TO_CUB;

    builder->taskSyncInfos_[4].vecRecvInfo[1] = SyncDirection::CUB_TO_VEC;
    builder->taskSyncInfos_[4].vecRecvInfo[0] = SyncDirection::CUB_TO_VEC;
    builder->taskSyncInfos_[4].cubRecvInfo[2] = SyncDirection::VEC_TO_CUB;
    builder->taskSyncInfos_[4].cubRecvInfo[1] = SyncDirection::VEC_TO_CUB;

    builder->OptimizeSyncRelations();

    SkTask aic;
    SkTask aiv;
    ASSERT_TRUE(aic.taskQue.Init(32));
    ASSERT_TRUE(aiv.taskQue.Init(32));

    std::map<size_t, SyncDirection> recvCubToVec{{1, SyncDirection::CUB_TO_VEC}};
    EXPECT_TRUE(builder->DispatchSyncTasks(aic, aiv, 0, recvCubToVec, false, SkQueueType::AIV));

    std::map<size_t, SyncDirection> sendVecToCub{{1, SyncDirection::VEC_TO_CUB}};
    EXPECT_TRUE(builder->DispatchSyncTasks(aic, aiv, 0, sendVecToCub, true, SkQueueType::AIV));

    std::map<size_t, SyncDirection> sendCubToCub{{1, SyncDirection::CUB_TO_CUB}};
    EXPECT_TRUE(builder->DispatchSyncTasks(aic, aiv, 0, sendCubToCub, true, SkQueueType::AIC));

    std::map<size_t, SyncDirection> sendVecToVec{{1, SyncDirection::VEC_TO_VEC}};
    EXPECT_TRUE(builder->DispatchSyncTasks(aic, aiv, 0, sendVecToVec, true, SkQueueType::AIV));

    EXPECT_GT(aic.taskQue.get()->taskCnt + aiv.taskQue.get()->taskCnt, 0U);
}

TEST_F(SkTaskBuilderTest, Build_WithCustomNotifyWaitReset_Success)
{
    opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOtionType::SPLIT_MODE, 1, 1, 4));

    auto* k0 = CreateKernelNodeEx(7001, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_1);
    auto* k1 = CreateKernelNodeEx(7002, 1, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_2);
    (void)k0;
    (void)k1;

    auto* cNotify = CreateNotifyNodeEx(7010, 0, 7001, INVALID_TASK_ID, 91);
    auto* cWait = CreateWaitNodeEx(7011, 0, 7001, INVALID_TASK_ID, 91);
    auto* cReset = CreateResetNodeEx(7012, 0, 7002, INVALID_TASK_ID);

    std::vector<SuperKernelBaseNode*> tasks = {graph->GetNodeById(7001), graph->GetNodeById(7002)};
    std::vector<SuperKernelBaseNode*> customTasks = {cNotify, cWait, cReset};

    SkLaunchInfo launchInfo = builder->Build("Unknown", tasks, customTasks);
    EXPECT_NE(launchInfo.entryInfo.skEntryFunc, nullptr);
    EXPECT_NE(launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, AddTasks_NullQueueAndDispatchQueueSelection)
{
    SkTask emptyTask;
    auto* notify = CreateNotifyNodeEx(8001, 0, INVALID_TASK_ID, INVALID_TASK_ID, 11);
    auto* kernel = CreateKernelNodeEx(8002, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
    SkDfxInfo dfx{};

    EXPECT_FALSE(builder->AddSyncTask(emptyTask, 0, SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC));
    EXPECT_FALSE(builder->AddEventTask(emptyTask, notify, 0, SkTaskType::TYPE_EVENT_NOTIFY));
    EXPECT_FALSE(builder->AddFuncTask(emptyTask, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

    SkTask aic;
    SkTask aiv;
    ASSERT_TRUE(aic.taskQue.Init(8));
    ASSERT_TRUE(aiv.taskQue.Init(8));

    EXPECT_TRUE(builder->DispatchEventTask(aic, aiv, notify, 1, SkTaskType::TYPE_EVENT_NOTIFY, SkQueueType::AIC));
    EXPECT_EQ(aic.taskQue.get()->taskCnt, 1U);
    EXPECT_TRUE(builder->DispatchEventTask(aic, aiv, notify, 2, SkTaskType::TYPE_EVENT_WAIT, SkQueueType::MIX_1_1));
    EXPECT_EQ(aiv.taskQue.get()->taskCnt, 1U);
}

TEST_F(SkTaskBuilderTest, DispatchFuncTask_MixFailureAndDcciBranch)
{
    opts->AddOption(std::make_unique<StringListOptOption>(
        "dcci_disable", aclskOtionType::DEBUG_DCCI_DISABLE_ON_KERNEL, std::vector<std::string>{"k"}));

    SkTask aic;
    SkTask aiv;
    ASSERT_TRUE(aic.taskQue.Init(16));
    ASSERT_TRUE(aiv.taskQue.Init(16));

    auto* mix11 = CreateKernelNodeEx(8101, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_1);
    auto* mix12 = CreateKernelNodeEx(8102, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_2);
    auto* n1 = dynamic_cast<SuperKernelKernelNode*>(mix11);
    auto* n2 = dynamic_cast<SuperKernelKernelNode*>(mix12);
    ASSERT_NE(n1, nullptr);
    ASSERT_NE(n2, nullptr);

    SkDfxInfo dfx{};
    ASSERT_TRUE(builder->DispatchFuncTask(aic, aiv, mix11, &dfx, 0, 1, SkTaskType::TYPE_FUNC, SkQueueType::MIX_1_1));
    TaskInfo& firstFunc = aic.taskQue.get()->taskInfos[aic.taskQue.get()->taskCnt - 1];
    EXPECT_NE((firstFunc.debugOptions & 0x1U), 0U);

    n1->nodeInfos.kernelInfos.resolvedFuncs[0].funcAddr[1] = 0;
    EXPECT_FALSE(builder->DispatchFuncTask(aic, aiv, mix11, &dfx, 1, 1, SkTaskType::TYPE_FUNC, SkQueueType::MIX_1_1));

    n2->nodeInfos.kernelInfos.resolvedFuncs[0].funcAddr[1] = 0;
    EXPECT_FALSE(builder->DispatchFuncTask(aic, aiv, mix12, &dfx, 2, 1, SkTaskType::TYPE_FUNC, SkQueueType::MIX_1_2));
}

TEST_F(SkTaskBuilderTest, GenEntryInfo_AllModeBranches)
{
    SkTask aic;
    SkTask aiv;
    ASSERT_TRUE(aiv.taskQue.Init(8));

    aic.funcCnt = 0;
    aiv.funcCnt = 1;
    aiv.numBlocks = 5;
    SkHostEntryInfo onlyAiv = builder->GenEntryInfo(aic, aiv);
    ASSERT_NE(onlyAiv.skEntryFunc, nullptr);
    EXPECT_EQ(onlyAiv.entryType, SkKernelType::AIV_ONLY);

    aic.funcCnt = 1;
    aiv.funcCnt = 0;
    aic.numBlocks = 6;
    SkHostEntryInfo onlyAic = builder->GenEntryInfo(aic, aiv);
    ASSERT_NE(onlyAic.skEntryFunc, nullptr);
    EXPECT_EQ(onlyAic.entryType, SkKernelType::AIC_ONLY);

    aic.funcCnt = 1;
    aiv.funcCnt = 1;
    aic.nodeType = SkKernelType::AIC_ONLY;
    aiv.nodeType = SkKernelType::AIV_ONLY;
    aic.numBlocks = 8;
    aiv.numBlocks = 7;
    SkHostEntryInfo mix11 = builder->GenEntryInfo(aic, aiv);
    ASSERT_NE(mix11.skEntryFunc, nullptr);
    EXPECT_EQ(mix11.entryType, SkKernelType::MIX_AIC_1_1);

    aic.funcCnt = 0;
    aiv.funcCnt = 0;
    SkHostEntryInfo invalid = builder->GenEntryInfo(aic, aiv);
    EXPECT_EQ(invalid.skEntryFunc, nullptr);
    EXPECT_EQ(invalid.entryType, SkKernelType::DEFAULT);
}

TEST_F(SkTaskBuilderTest, Build_WithNotifyAndWaitNodes_Success)
{
    opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOtionType::SPLIT_MODE, 1, 1, 4));

    auto* k0 = CreateKernelNodeEx(8201, 0, INVALID_TASK_ID, 8202, SkKernelType::AIC_ONLY);
    auto* n1 = CreateNotifyNodeEx(8202, 0, 8201, 8203, 101);
    auto* w2 = CreateWaitNodeEx(8203, 1, INVALID_TASK_ID, 8204, 101);
    auto* k3 = CreateKernelNodeEx(8204, 1, 8203, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
    (void)k0;
    (void)n1;
    (void)w2;
    (void)k3;

    std::vector<SuperKernelBaseNode*> tasks = {
        graph->GetNodeById(8201), graph->GetNodeById(8202), graph->GetNodeById(8203), graph->GetNodeById(8204)};

    SkLaunchInfo launchInfo = builder->Build("Unknown", tasks, {});
    EXPECT_NE(launchInfo.entryInfo.skEntryFunc, nullptr);
    EXPECT_NE(launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, InitTaskSyncInfos_UnsupportedKernelType_FallbackAic)
{
    auto* kernel = CreateKernelNodeEx(8301, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::DEFAULT);
    std::vector<SuperKernelBaseNode*> tasks = {kernel};

    ASSERT_TRUE(builder->InitTaskSyncInfos(tasks));
    ASSERT_EQ(builder->taskSyncInfos_.size(), 1U);
    EXPECT_EQ(builder->taskSyncInfos_[0].queueType, SkQueueType::AIC);
}

TEST_F(SkTaskBuilderTest, RemoveMultiSendRecv_DenseBranches)
{
    builder->taskSyncInfos_.clear();
    builder->taskSyncInfos_.resize(5);
    for (auto& info : builder->taskSyncInfos_) {
        info.queueType = SkQueueType::AIV;
    }

    builder->taskSyncInfos_[0].vecSendInfo[2] = SyncDirection::VEC_TO_CUB;
    builder->taskSyncInfos_[0].vecSendInfo[3] = SyncDirection::VEC_TO_CUB;
    builder->taskSyncInfos_[2].cubRecvInfo[0] = SyncDirection::VEC_TO_CUB;
    builder->taskSyncInfos_[3].cubRecvInfo[0] = SyncDirection::VEC_TO_CUB;

    builder->taskSyncInfos_[1].cubSendInfo[2] = SyncDirection::CUB_TO_VEC;
    builder->taskSyncInfos_[1].cubSendInfo[4] = SyncDirection::CUB_TO_VEC;
    builder->taskSyncInfos_[2].vecRecvInfo[1] = SyncDirection::CUB_TO_VEC;
    builder->taskSyncInfos_[4].vecRecvInfo[1] = SyncDirection::CUB_TO_VEC;

    builder->taskSyncInfos_[4].vecRecvInfo[0] = SyncDirection::CUB_TO_VEC;
    builder->taskSyncInfos_[4].vecRecvInfo[2] = SyncDirection::CUB_TO_VEC;
    builder->taskSyncInfos_[4].cubRecvInfo[0] = SyncDirection::VEC_TO_CUB;
    builder->taskSyncInfos_[4].cubRecvInfo[2] = SyncDirection::VEC_TO_CUB;

    builder->RemoveMultiSendSync();
    builder->RemoveMultiRecvSync();

    EXPECT_LE(builder->taskSyncInfos_[0].vecSendInfo.size(), 1U);
    EXPECT_LE(builder->taskSyncInfos_[1].cubSendInfo.size(), 1U);
    EXPECT_LE(builder->taskSyncInfos_[4].vecRecvInfo.size(), 1U);
    EXPECT_LE(builder->taskSyncInfos_[4].cubRecvInfo.size(), 1U);
}

TEST_F(SkTaskBuilderTest, DispatchSyncTasks_NoneAndUnknownDirection_SkipAndSucceed)
{
    SkTask aic;
    SkTask aiv;
    ASSERT_TRUE(aic.taskQue.Init(4));
    ASSERT_TRUE(aiv.taskQue.Init(4));

    builder->taskSyncInfos_.resize(2);
    builder->taskSyncInfos_[0].queueType = SkQueueType::AIC;
    builder->taskSyncInfos_[1].queueType = SkQueueType::AIV;

    std::map<size_t, SyncDirection> noneDir{{1, SyncDirection::NONE}};
    EXPECT_FALSE(builder->DispatchSyncTasks(aic, aiv, 0, noneDir, true, SkQueueType::AIC));

    std::map<size_t, SyncDirection> unknownDir{{1, static_cast<SyncDirection>(999)}};
    EXPECT_FALSE(builder->DispatchSyncTasks(aic, aiv, 0, unknownDir, false, SkQueueType::AIC));

    EXPECT_EQ(aic.taskQue.get()->taskCnt, 0U);
    EXPECT_EQ(aiv.taskQue.get()->taskCnt, 0U);
}

TEST_F(SkTaskBuilderTest, AddTasks_RemainsFullAfterExpand_AndUnsupportedNodeType)
{
    auto* notify = CreateNotifyNodeEx(8601, 0, INVALID_TASK_ID, INVALID_TASK_ID, 66);
    auto* kernel = CreateKernelNodeEx(8602, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
    SkDfxInfo dfx{};

    SkTask syncTask;
    ASSERT_TRUE(syncTask.taskQue.Init(1));
    syncTask.taskQue.get()->taskCnt = std::numeric_limits<uint32_t>::max();
    EXPECT_FALSE(builder->AddSyncTask(syncTask, 0, SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC));

    SkTask eventTask;
    ASSERT_TRUE(eventTask.taskQue.Init(1));
    eventTask.taskQue.get()->taskCnt = std::numeric_limits<uint32_t>::max();
    EXPECT_FALSE(builder->AddEventTask(eventTask, notify, 0, SkTaskType::TYPE_EVENT_NOTIFY));

    SkTask funcTask;
    ASSERT_TRUE(funcTask.taskQue.Init(1));
    funcTask.taskQue.get()->taskCnt = std::numeric_limits<uint32_t>::max();
    EXPECT_FALSE(builder->AddFuncTask(funcTask, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

    SkTask normalTask;
    ASSERT_TRUE(normalTask.taskQue.Init(4));
    EXPECT_FALSE(builder->AddFuncTask(normalTask, notify, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));
}

TEST_F(SkTaskBuilderTest, DispatchFuncTask_UnsupportedNodeTypeAndQueueFailure)
{
    auto* notify = CreateNotifyNodeEx(8701, 0, INVALID_TASK_ID, INVALID_TASK_ID, 77);
    SkDfxInfo dfx{};

    SkTask aic;
    SkTask aiv;
    ASSERT_TRUE(aic.taskQue.Init(1));
    ASSERT_TRUE(aiv.taskQue.Init(1));

    EXPECT_FALSE(builder->DispatchFuncTask(aic, aiv, notify, &dfx, 0, 1, SkTaskType::TYPE_FUNC, SkQueueType::AIC));
}

TEST_F(SkTaskBuilderTest, GenEntryInfo_Mix12ElseBranch_Selected)
{
    SkTask aic;
    SkTask aiv;
    aic.funcCnt = 1;
    aiv.funcCnt = 1;
    aic.nodeType = SkKernelType::AIC_ONLY;
    aiv.nodeType = SkKernelType::AIV_ONLY;
    aic.numBlocks = 2;
    aiv.numBlocks = 9;
    ASSERT_TRUE(aiv.taskQue.Init(2));

    SkHostEntryInfo entryInfo = builder->GenEntryInfo(aic, aiv);
    ASSERT_NE(entryInfo.skEntryFunc, nullptr);
    EXPECT_EQ(entryInfo.entryType, SkKernelType::MIX_AIC_1_2);
}

TEST_F(SkTaskBuilderTest, Build_OnlyEventTask_GenEntryInfoFail)
{
    auto* notify = CreateNotifyNodeEx(8801, 0, INVALID_TASK_ID, INVALID_TASK_ID, 201);
    std::vector<SuperKernelBaseNode*> tasks = {notify};

    SkLaunchInfo launchInfo = builder->Build("Unknown", tasks, {});
    EXPECT_EQ(launchInfo.entryInfo.skEntryFunc, nullptr);
    EXPECT_EQ(launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, Build_ResolveEntryHandleAtBuildStage_WhenBinNotReady_ReturnEmpty)
{
    opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOtionType::SPLIT_MODE, 1, 1, 4));
    SkUtSetEntryBinHandleNull(1);

    auto* kernel = CreateKernelNodeEx(8911, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
    std::vector<SuperKernelBaseNode*> tasks = {kernel};

    SkLaunchInfo launchInfo = builder->Build("Unknown", tasks, {});
    EXPECT_EQ(launchInfo.entryInfo.skEntryFunc, nullptr);
    EXPECT_EQ(launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, PrecomputeSyncRelations_WaitWithoutNotify_Success)
{
    auto* wait = CreateWaitNodeEx(9101, 0, INVALID_TASK_ID, INVALID_TASK_ID, 333);
    std::vector<SuperKernelBaseNode*> tasks = {wait};

    EXPECT_FALSE(builder->PrecomputeSyncRelationsFromGraph(tasks));
}

TEST_F(SkTaskBuilderTest, Build_RollingPreloadDispatchFail_ReturnEmpty)
{
    opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOtionType::SPLIT_MODE, 1, 1, 4));

    auto* notify = CreateNotifyNodeEx(9501, 0, INVALID_TASK_ID, 9502, 5001);
    auto* kernel = CreateKernelNodeEx(9502, 0, 9501, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
    auto* k = dynamic_cast<SuperKernelKernelNode*>(kernel);
    ASSERT_NE(k, nullptr);
    k->nodeInfos.kernelInfos.resolvedFuncs[0].funcAddr[0] = 0;

    std::vector<SuperKernelBaseNode*> tasks = {notify, kernel};
    SkLaunchInfo launchInfo = builder->Build("Unknown", tasks, {});

    EXPECT_EQ(launchInfo.entryInfo.skEntryFunc, nullptr);
    EXPECT_EQ(launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, Build_CustomTaskUnsupportedType_ReturnEmpty)
{
    opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOtionType::SPLIT_MODE, 1, 1, 4));

    auto* kernel = CreateKernelNodeEx(9001, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
    auto* customKernel = CreateKernelNodeEx(9002, 0, 9001, INVALID_TASK_ID, SkKernelType::AIC_ONLY);

    std::vector<SuperKernelBaseNode*> tasks = {kernel};
    std::vector<SuperKernelBaseNode*> customTasks = {customKernel};

    SkLaunchInfo launchInfo = builder->Build("Unknown", tasks, customTasks);
    EXPECT_EQ(launchInfo.entryInfo.skEntryFunc, nullptr);
    EXPECT_EQ(launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, GenEntryInfo_ResolveFailWhenEntryBinNull_ReturnEmpty)
{
    SkUtSetEntryBinHandleNull(1);

    SkTask aic;
    SkTask aiv;
    aic.funcCnt = 1;
    aic.numBlocks = 1;

    SkHostEntryInfo entryInfo = builder->GenEntryInfo(aic, aiv);
    EXPECT_EQ(entryInfo.skEntryFunc, nullptr);
    EXPECT_EQ(entryInfo.entryType, SkKernelType::DEFAULT);
}

TEST_F(SkTaskBuilderTest, GenEntryInfo_ResolveFailWhenBinaryGetNullHandle_ReturnEmpty)
{
    SkUtSetBinaryGetFunctionNullHandle(1);

    SkTask aic;
    SkTask aiv;
    aiv.funcCnt = 1;
    aiv.numBlocks = 2;

    SkHostEntryInfo entryInfo = builder->GenEntryInfo(aic, aiv);
    EXPECT_EQ(entryInfo.skEntryFunc, nullptr);
    EXPECT_EQ(entryInfo.entryType, SkKernelType::DEFAULT);
}

TEST_F(SkTaskBuilderTest, InitTaskSyncInfos_MaxHopsExceededBranch_ReturnsTrueWithFallback)
{
    const uint64_t chainStart = 10000;
    const uint64_t chainLen = 120;
    for (uint64_t i = 0; i < chainLen; ++i) {
        uint64_t nodeId = chainStart + i;
        uint64_t preNodeId = nodeId + 1;
        CreateResetNodeEx(nodeId, 0, preNodeId, INVALID_TASK_ID);
    }

    auto* notify = CreateNotifyNodeEx(20000, 0, chainStart, INVALID_TASK_ID, 77);
    std::vector<SuperKernelBaseNode*> tasks = {notify};

    EXPECT_FALSE(builder->InitTaskSyncInfos(tasks));
}

TEST_F(SkTaskBuilderTest, PrintSyncInfo_CoversVecAndCubFormattingPaths)
{
    builder->taskSyncInfos_.clear();
    builder->taskSyncInfos_.resize(2);
    builder->taskSyncInfos_[0].queueType = SkQueueType::AIV;
    builder->taskSyncInfos_[1].queueType = SkQueueType::AIC;

    builder->taskSyncInfos_[0].vecSendInfo[1] = SyncDirection::VEC_TO_CUB;
    builder->taskSyncInfos_[0].vecRecvInfo[1] = SyncDirection::CUB_TO_VEC;
    builder->taskSyncInfos_[1].cubSendInfo[0] = SyncDirection::CUB_TO_VEC;
    builder->taskSyncInfos_[1].cubRecvInfo[0] = SyncDirection::VEC_TO_CUB;

    builder->PrintSyncInfo("ut-print-sync");
}

TEST_F(SkTaskBuilderTest, ExtractIntraStreamSync_StreamFusionOffMultiStreamBranch)
{
    opts->AddOption(std::make_unique<NumberOptOption>("stream_fusion", aclskOtionType::STREAM_FUSION, 0, 0, 1));

    auto* k0 = CreateKernelNodeEx(30001, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
    auto* k1 = CreateKernelNodeEx(30002, 1, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
    (void)k0;
    (void)k1;

    std::vector<SuperKernelBaseNode*> tasks = {graph->GetNodeById(30001), graph->GetNodeById(30002)};
    ASSERT_TRUE(builder->InitTaskSyncInfos(tasks));
    builder->ExtractIntraStreamSync(tasks);
}

TEST_F(SkTaskBuilderTest, InsertSyncEvent_GenSyncDirectionDenseBranches)
{
    builder->taskSyncInfos_.clear();
    builder->taskSyncInfos_.resize(2);

    // pre=MIX, curr=MIX -> MIX_TO_MIX
    builder->taskSyncInfos_[0] = TaskSyncInfo{};
    builder->taskSyncInfos_[1] = TaskSyncInfo{};
    builder->taskSyncInfos_[0].queueType = SkQueueType::MIX_1_1;
    builder->taskSyncInfos_[1].queueType = SkQueueType::MIX_1_2;
    builder->InsertSyncEvent(0, 1);
    EXPECT_TRUE(builder->taskSyncInfos_[0].cubSendInfo.find(1) != builder->taskSyncInfos_[0].cubSendInfo.end());
    EXPECT_TRUE(builder->taskSyncInfos_[0].vecSendInfo.find(1) != builder->taskSyncInfos_[0].vecSendInfo.end());

    // pre=MIX, curr=AIC -> VEC_TO_CUB
    builder->taskSyncInfos_[0] = TaskSyncInfo{};
    builder->taskSyncInfos_[1] = TaskSyncInfo{};
    builder->taskSyncInfos_[0].queueType = SkQueueType::MIX_1_2;
    builder->taskSyncInfos_[1].queueType = SkQueueType::AIC;
    builder->InsertSyncEvent(0, 1);
    EXPECT_TRUE(builder->taskSyncInfos_[0].vecSendInfo.find(1) != builder->taskSyncInfos_[0].vecSendInfo.end());

    // pre=AIC, curr=MIX -> CUB_TO_VEC
    builder->taskSyncInfos_[0] = TaskSyncInfo{};
    builder->taskSyncInfos_[1] = TaskSyncInfo{};
    builder->taskSyncInfos_[0].queueType = SkQueueType::AIC;
    builder->taskSyncInfos_[1].queueType = SkQueueType::MIX_1_1;
    builder->InsertSyncEvent(0, 1);
    EXPECT_TRUE(builder->taskSyncInfos_[0].cubSendInfo.find(1) != builder->taskSyncInfos_[0].cubSendInfo.end());

    // pre=AIV, curr=MIX -> VEC_TO_CUB
    builder->taskSyncInfos_[0] = TaskSyncInfo{};
    builder->taskSyncInfos_[1] = TaskSyncInfo{};
    builder->taskSyncInfos_[0].queueType = SkQueueType::AIV;
    builder->taskSyncInfos_[1].queueType = SkQueueType::MIX_1_1;
    builder->InsertSyncEvent(0, 1);
    EXPECT_TRUE(builder->taskSyncInfos_[0].vecSendInfo.find(1) != builder->taskSyncInfos_[0].vecSendInfo.end());

    // pre=UNKNOWN(default) -> fallback VEC_TO_CUB
    builder->taskSyncInfos_[0] = TaskSyncInfo{};
    builder->taskSyncInfos_[1] = TaskSyncInfo{};
    builder->taskSyncInfos_[0].queueType = SkQueueType::UNKNOWN;
    builder->taskSyncInfos_[1].queueType = SkQueueType::AIV;
    builder->InsertSyncEvent(0, 1);
    EXPECT_TRUE(builder->taskSyncInfos_[0].vecSendInfo.find(1) != builder->taskSyncInfos_[0].vecSendInfo.end());
}

TEST_F(SkTaskBuilderTest, GenEntryArgs_MemcpyAicFail_ReturnEmpty)
{
    SkTask aic;
    SkTask aiv;
    ASSERT_TRUE(aic.taskQue.Init(4));
    ASSERT_TRUE(aiv.taskQue.Init(4));

    SkDfxInfo dfx{};
    SkUtSetSecurecMemcpyFailOnCall(1);
    DeviceArgsPtr devArgs = builder->GenEntryArgs(aic, aiv, &dfx, 1);
    EXPECT_EQ(devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, GenEntryArgs_MemcpyAivFail_ReturnEmpty)
{
    SkTask aic;
    SkTask aiv;
    ASSERT_TRUE(aic.taskQue.Init(4));
    ASSERT_TRUE(aiv.taskQue.Init(4));

    SkDfxInfo dfx{};
    SkUtSetSecurecMemcpyFailOnCall(2);
    DeviceArgsPtr devArgs = builder->GenEntryArgs(aic, aiv, &dfx, 1);
    EXPECT_EQ(devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, GenEntryArgs_MemsetCounterFail_ReturnEmpty)
{
    SkTask aic;
    SkTask aiv;
    ASSERT_TRUE(aic.taskQue.Init(4));
    ASSERT_TRUE(aiv.taskQue.Init(4));

    SkDfxInfo dfx{};
    SkUtSetSecurecMemsetFailOnCall(1);
    DeviceArgsPtr devArgs = builder->GenEntryArgs(aic, aiv, &dfx, 1);
    EXPECT_EQ(devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, GenEntryArgs_MemsetWorkspaceFail_ReturnEmpty)
{
    SkTask aic;
    SkTask aiv;
    ASSERT_TRUE(aic.taskQue.Init(4));
    ASSERT_TRUE(aiv.taskQue.Init(4));

    SkDfxInfo dfx{};
    SkUtSetSecurecMemsetFailOnCall(2);
    DeviceArgsPtr devArgs = builder->GenEntryArgs(aic, aiv, &dfx, 1);
    EXPECT_EQ(devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, GenEntryArgs_MemcpyDfxFail_ReturnEmpty)
{
    SkTask aic;
    SkTask aiv;
    ASSERT_TRUE(aic.taskQue.Init(4));
    ASSERT_TRUE(aiv.taskQue.Init(4));

    SkDfxInfo dfx{};
    SkUtSetSecurecMemcpyFailOnCall(3);
    DeviceArgsPtr devArgs = builder->GenEntryArgs(aic, aiv, &dfx, 1);
    EXPECT_EQ(devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, Build_DfxMemsetFail_ReturnEmpty)
{
    std::vector<SuperKernelBaseNode*> tasks;
    tasks.push_back(CreateKernelNodeEx(40001, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY));

    SkUtSetSecurecMemsetFailOnCall(1);
    SkLaunchInfo launchInfo = builder->Build("Unknown", tasks, {});
    EXPECT_EQ(launchInfo.entryInfo.skEntryFunc, nullptr);
    EXPECT_EQ(launchInfo.devArgs.Get(), nullptr);
}
