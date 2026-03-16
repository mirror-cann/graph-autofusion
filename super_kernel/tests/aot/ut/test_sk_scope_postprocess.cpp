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
#include <vector>

#define private public
#define protected public
#include "sk_graph.h"
#include "sk_scope_postprocess.h"
#include "sk_node.h"

class SuperKernelScopePostprocessTest : public testing::Test {
protected:
    void SetUp() override
    {
        graph = std::make_unique<SuperKernelGraph>();
    }

    std::unique_ptr<SuperKernelGraph> graph;
};

TEST_F(SuperKernelScopePostprocessTest, PostProcess_NotifyWaitPairCancelled_ReturnEmpty)
{
    auto notifyNode = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_EVENT_RECORD, 0, 0, INVALID_TASK_ID);
    notifyNode->SetNodeType(SkNodeType::NODE_NOTIFY);
    notifyNode->SetNodeId(1);
    notifyNode->SetNextNodeId(2);
    notifyNode->nodeInfos.syncInfos.eventId = 100;

    auto waitNode = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_EVENT_WAIT, 1, 0, 1);
    waitNode->SetNodeType(SkNodeType::NODE_WAIT);
    waitNode->SetNodeId(2);
    waitNode->SetPreNodeId(1);
    waitNode->SetNextNodeId(INVALID_TASK_ID);
    waitNode->nodeInfos.syncInfos.eventId = 100;

    SuperKernelScopeInfo scopeInfo;
    scopeInfo.nodes.push_back(notifyNode.get());
    scopeInfo.nodes.push_back(waitNode.get());

    ScopeStreamInfo streamInfo;
    streamInfo.streamIdx = 0;
    streamInfo.headNodeIdx = 1;
    streamInfo.tailNodeIdx = 2;
    streamInfo.nodeSize = 2;
    scopeInfo.scopeStreamInfos.push_back(streamInfo);

    graph->graphMap[1] = std::move(notifyNode);
    graph->graphMap[2] = std::move(waitNode);

    SuperKernelScopePostProcessor postProcessor(*graph);
    SuperKernelProcessedScopeInfo processed = postProcessor.PostProcess(scopeInfo);

    EXPECT_TRUE(processed.nodes.empty());
    EXPECT_TRUE(processed.updateStreamInfos.empty());
    EXPECT_EQ(processed.skMainNodeId, INVALID_TASK_ID);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_NotifyOneToManyWaits_AllCancelled)
{
    auto notifyNode = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_EVENT_RECORD, 0, 0, INVALID_TASK_ID);
    notifyNode->SetNodeId(3);
    notifyNode->SetNextNodeId(4);
    notifyNode->nodeInfos.syncInfos.eventId = 200;

    auto waitNode1 = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_EVENT_WAIT, 1, 0, 3);
    waitNode1->SetNodeId(4);
    waitNode1->SetPreNodeId(3);
    waitNode1->SetNextNodeId(5);
    waitNode1->nodeInfos.syncInfos.eventId = 200;

    auto waitNode2 = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_EVENT_WAIT, 2, 1, INVALID_TASK_ID);
    waitNode2->SetNodeId(5);
    waitNode2->SetPreNodeId(INVALID_TASK_ID);
    waitNode2->SetNextNodeId(INVALID_TASK_ID);
    waitNode2->nodeInfos.syncInfos.eventId = 200;

    SuperKernelScopeInfo scopeInfo;
    scopeInfo.nodes.push_back(notifyNode.get());
    scopeInfo.nodes.push_back(waitNode1.get());
    scopeInfo.nodes.push_back(waitNode2.get());

    ScopeStreamInfo stream0;
    stream0.streamIdx = 0;
    stream0.headNodeIdx = 3;
    stream0.tailNodeIdx = 4;
    stream0.nodeSize = 2;

    ScopeStreamInfo stream1;
    stream1.streamIdx = 1;
    stream1.headNodeIdx = 5;
    stream1.tailNodeIdx = 5;
    stream1.nodeSize = 1;

    scopeInfo.scopeStreamInfos = {stream0, stream1};

    graph->graphMap[3] = std::move(notifyNode);
    graph->graphMap[4] = std::move(waitNode1);
    graph->graphMap[5] = std::move(waitNode2);

    SuperKernelScopePostProcessor postProcessor(*graph);
    SuperKernelProcessedScopeInfo processed = postProcessor.PostProcess(scopeInfo);

    EXPECT_TRUE(processed.nodes.empty());
    EXPECT_TRUE(processed.updateStreamInfos.empty());
    EXPECT_EQ(processed.skMainNodeId, INVALID_TASK_ID);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_NotifyWithoutWait_Retained)
{
    auto notifyNode = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_EVENT_RECORD, 0, 0, INVALID_TASK_ID);
    notifyNode->SetNodeType(SkNodeType::NODE_NOTIFY);
    notifyNode->SetNodeId(6);
    notifyNode->SetPreNodeId(INVALID_TASK_ID);
    notifyNode->SetNextNodeId(INVALID_TASK_ID);
    notifyNode->nodeInfos.syncInfos.eventId = 300;

    SuperKernelScopeInfo scopeInfo;
    scopeInfo.nodes.push_back(notifyNode.get());

    ScopeStreamInfo stream0;
    stream0.streamIdx = 0;
    stream0.headNodeIdx = 6;
    stream0.tailNodeIdx = 6;
    stream0.nodeSize = 1;
    scopeInfo.scopeStreamInfos.push_back(stream0);

    graph->graphMap[6] = std::move(notifyNode);

    SuperKernelScopePostProcessor postProcessor(*graph);
    SuperKernelProcessedScopeInfo processed = postProcessor.PostProcess(scopeInfo);

    EXPECT_TRUE(processed.nodes.empty());
    EXPECT_TRUE(processed.updateStreamInfos.empty());
    EXPECT_EQ(processed.skMainNodeId, INVALID_TASK_ID);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_SingleKernel_SelectMainNode)
{
    auto kernelNode = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 3, INVALID_TASK_ID);
    kernelNode->SetNodeId(11);
    kernelNode->SetPreNodeId(INVALID_TASK_ID);
    kernelNode->SetNextNodeId(INVALID_TASK_ID);

    SuperKernelScopeInfo scopeInfo;
    scopeInfo.nodes.push_back(kernelNode.get());

    ScopeStreamInfo streamInfo;
    streamInfo.streamIdx = 3;
    streamInfo.headNodeIdx = 11;
    streamInfo.tailNodeIdx = 11;
    streamInfo.nodeSize = 1;
    scopeInfo.scopeStreamInfos.push_back(streamInfo);

    graph->graphMap[11] = std::move(kernelNode);

    SuperKernelScopePostProcessor postProcessor(*graph);
    SuperKernelProcessedScopeInfo processed = postProcessor.PostProcess(scopeInfo);

    ASSERT_EQ(processed.nodes.size(), 1U);
    EXPECT_EQ(processed.nodes[0]->GetNodeId(), 11U);
    EXPECT_EQ(processed.skMainNodeId, 11U);

    ASSERT_EQ(processed.updateStreamInfos.size(), 1U);
    EXPECT_EQ(processed.updateStreamInfos[0].streamIdx, 3U);
    EXPECT_EQ(processed.updateStreamInfos[0].headNodeIdx, 11U);
    EXPECT_EQ(processed.updateStreamInfos[0].tailNodeIdx, 11U);
    EXPECT_EQ(processed.updateStreamInfos[0].nodeSize, 1U);
    EXPECT_TRUE(processed.updateStreamInfos[0].customParams.empty());
    EXPECT_TRUE(processed.eventNodes.empty());
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_StreamHeadMissing_ReturnEmpty)
{
    auto kernelNode = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_TASK_ID);
    kernelNode->SetNodeId(31);
    kernelNode->SetPreNodeId(INVALID_TASK_ID);
    kernelNode->SetNextNodeId(INVALID_TASK_ID);

    SuperKernelScopeInfo scopeInfo;
    scopeInfo.nodes.push_back(kernelNode.get());

    ScopeStreamInfo streamInfo;
    streamInfo.streamIdx = 0;
    streamInfo.headNodeIdx = 999;
    streamInfo.tailNodeIdx = 31;
    streamInfo.nodeSize = 1;
    scopeInfo.scopeStreamInfos.push_back(streamInfo);

    graph->graphMap[31] = std::move(kernelNode);

    SuperKernelScopePostProcessor postProcessor(*graph);
    SuperKernelProcessedScopeInfo processed = postProcessor.PostProcess(scopeInfo);

    EXPECT_TRUE(processed.nodes.empty());
    EXPECT_TRUE(processed.updateStreamInfos.empty());
    EXPECT_EQ(processed.skMainNodeId, INVALID_TASK_ID);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_FrontWaitFeaturePath_ReturnEmpty)
{
    auto stream0Node0 = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_TASK_ID);
    stream0Node0->SetNodeId(10);
    stream0Node0->SetPreNodeId(INVALID_TASK_ID);
    stream0Node0->SetNextNodeId(11);

    auto stream0Node1 = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 1, 0, 10);
    stream0Node1->SetNodeId(11);
    stream0Node1->SetPreNodeId(10);
    stream0Node1->SetNextNodeId(INVALID_TASK_ID);

    auto stream1Node0 = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 1, 19);
    stream1Node0->SetNodeId(20);
    stream1Node0->SetPreNodeId(19);
    stream1Node0->SetNextNodeId(INVALID_TASK_ID);

    SuperKernelScopeInfo scopeInfo;
    scopeInfo.nodes = {stream0Node0.get(), stream0Node1.get(), stream1Node0.get()};

    ScopeStreamInfo stream0;
    stream0.streamIdx = 0;
    stream0.headNodeIdx = 10;
    stream0.tailNodeIdx = 11;
    stream0.nodeSize = 2;

    ScopeStreamInfo stream1;
    stream1.streamIdx = 1;
    stream1.headNodeIdx = 20;
    stream1.tailNodeIdx = 20;
    stream1.nodeSize = 1;

    scopeInfo.scopeStreamInfos = {stream0, stream1};

    graph->graphMap[10] = std::move(stream0Node0);
    graph->graphMap[11] = std::move(stream0Node1);
    graph->graphMap[20] = std::move(stream1Node0);

    SuperKernelScopePostProcessor postProcessor(*graph);
    SuperKernelProcessedScopeInfo processed = postProcessor.PostProcess(scopeInfo);

    EXPECT_TRUE(processed.nodes.empty());
    EXPECT_TRUE(processed.updateStreamInfos.empty());
    EXPECT_EQ(processed.skMainNodeId, INVALID_TASK_ID);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_BackBlockFeaturePath_ReturnEmpty)
{
    auto stream0Node0 = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_TASK_ID);
    stream0Node0->SetNodeId(40);
    stream0Node0->SetPreNodeId(INVALID_TASK_ID);
    stream0Node0->SetNextNodeId(INVALID_TASK_ID);

    auto stream1Node0 = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 1, INVALID_TASK_ID);
    stream1Node0->SetNodeId(50);
    stream1Node0->SetPreNodeId(INVALID_TASK_ID);
    stream1Node0->SetNextNodeId(51);

    auto stream1Node1 = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 1, 1, 50);
    stream1Node1->SetNodeId(51);
    stream1Node1->SetPreNodeId(50);
    stream1Node1->SetNextNodeId(INVALID_TASK_ID);

    SuperKernelScopeInfo scopeInfo;
    scopeInfo.nodes = {stream0Node0.get(), stream1Node0.get(), stream1Node1.get()};

    ScopeStreamInfo stream0;
    stream0.streamIdx = 0;
    stream0.headNodeIdx = 40;
    stream0.tailNodeIdx = 40;
    stream0.nodeSize = 1;

    ScopeStreamInfo stream1;
    stream1.streamIdx = 1;
    stream1.headNodeIdx = 50;
    stream1.tailNodeIdx = 50;
    stream1.nodeSize = 1;

    scopeInfo.scopeStreamInfos = {stream0, stream1};

    graph->graphMap[40] = std::move(stream0Node0);
    graph->graphMap[50] = std::move(stream1Node0);
    graph->graphMap[51] = std::move(stream1Node1);

    SuperKernelScopePostProcessor postProcessor(*graph);
    SuperKernelProcessedScopeInfo processed = postProcessor.PostProcess(scopeInfo);

    EXPECT_TRUE(processed.nodes.empty());
    EXPECT_TRUE(processed.updateStreamInfos.empty());
    EXPECT_EQ(processed.skMainNodeId, INVALID_TASK_ID);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_NoKernelCandidate_ReturnEmpty)
{
    auto notifyNode = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_EVENT_RECORD, 0, 0, INVALID_TASK_ID);
    notifyNode->SetNodeType(SkNodeType::NODE_NOTIFY);
    notifyNode->SetNodeId(60);
    notifyNode->SetPreNodeId(INVALID_TASK_ID);
    notifyNode->SetNextNodeId(INVALID_TASK_ID);
    notifyNode->nodeInfos.syncInfos.eventId = 101;

    SuperKernelScopeInfo scopeInfo;
    scopeInfo.nodes.push_back(notifyNode.get());

    ScopeStreamInfo streamInfo;
    streamInfo.streamIdx = 0;
    streamInfo.headNodeIdx = 60;
    streamInfo.tailNodeIdx = 60;
    streamInfo.nodeSize = 1;
    scopeInfo.scopeStreamInfos.push_back(streamInfo);

    graph->graphMap[60] = std::move(notifyNode);

    SuperKernelScopePostProcessor postProcessor(*graph);
    SuperKernelProcessedScopeInfo processed = postProcessor.PostProcess(scopeInfo);

    EXPECT_TRUE(processed.nodes.empty());
    EXPECT_TRUE(processed.updateStreamInfos.empty());
    EXPECT_EQ(processed.skMainNodeId, INVALID_TASK_ID);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_MultiStreamKernelOnly_Success)
{
    auto stream0Node0 = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_TASK_ID);
    stream0Node0->SetNodeId(70);
    stream0Node0->SetPreNodeId(INVALID_TASK_ID);
    stream0Node0->SetNextNodeId(INVALID_TASK_ID);

    auto stream1Node0 = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 1, INVALID_TASK_ID);
    stream1Node0->SetNodeId(80);
    stream1Node0->SetPreNodeId(INVALID_TASK_ID);
    stream1Node0->SetNextNodeId(INVALID_TASK_ID);

    SuperKernelScopeInfo scopeInfo;
    scopeInfo.nodes = {stream0Node0.get(), stream1Node0.get()};

    ScopeStreamInfo stream0;
    stream0.streamIdx = 0;
    stream0.headNodeIdx = 70;
    stream0.tailNodeIdx = 70;
    stream0.nodeSize = 1;

    ScopeStreamInfo stream1;
    stream1.streamIdx = 1;
    stream1.headNodeIdx = 80;
    stream1.tailNodeIdx = 80;
    stream1.nodeSize = 1;

    scopeInfo.scopeStreamInfos = {stream0, stream1};

    graph->graphMap[70] = std::move(stream0Node0);
    graph->graphMap[80] = std::move(stream1Node0);

    SuperKernelScopePostProcessor postProcessor(*graph);
    SuperKernelProcessedScopeInfo processed = postProcessor.PostProcess(scopeInfo);

    ASSERT_EQ(processed.nodes.size(), 2U);
    ASSERT_EQ(processed.updateStreamInfos.size(), 2U);
    EXPECT_EQ(processed.skMainNodeId, 70U);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_MainSelectReserveBoundaryAndMissingNext_ReturnEmpty)
{
    auto stream0Node0 = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, 1);
    stream0Node0->SetNodeId(90);
    stream0Node0->SetPreNodeId(1);
    stream0Node0->SetNextNodeId(INVALID_TASK_ID);

    auto stream1Node0 = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 1, 2);
    stream1Node0->SetNodeId(100);
    stream1Node0->SetPreNodeId(2);
    stream1Node0->SetNextNodeId(101);

    SuperKernelScopeInfo scopeInfo;
    scopeInfo.nodes = {stream0Node0.get(), stream1Node0.get()};

    ScopeStreamInfo stream0;
    stream0.streamIdx = 0;
    stream0.headNodeIdx = 90;
    stream0.tailNodeIdx = 90;
    stream0.nodeSize = 1;

    ScopeStreamInfo stream1;
    stream1.streamIdx = 1;
    stream1.headNodeIdx = 100;
    stream1.tailNodeIdx = 999; // Tail missing in graph; reserve-step traversal uses missing next path.
    stream1.nodeSize = 1;

    scopeInfo.scopeStreamInfos = {stream0, stream1};

    graph->graphMap[90] = std::move(stream0Node0);
    graph->graphMap[100] = std::move(stream1Node0);

    SuperKernelScopePostProcessor postProcessor(*graph);
    SuperKernelProcessedScopeInfo processed = postProcessor.PostProcess(scopeInfo);

    EXPECT_TRUE(processed.nodes.empty());
    EXPECT_TRUE(processed.updateStreamInfos.empty());
    EXPECT_EQ(processed.skMainNodeId, INVALID_TASK_ID);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_FrontWaitMoveWorkNodePath_ReturnEmpty)
{
    auto s0n0 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, 1);
    s0n0->SetNodeId(200);
    s0n0->SetPreNodeId(1);
    s0n0->SetNextNodeId(201);

    auto s0n1 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 1, 0, 200);
    s0n1->SetNodeId(201);
    s0n1->SetPreNodeId(200);
    s0n1->SetNextNodeId(INVALID_TASK_ID);

    auto s1n0 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 1, 2);
    s1n0->SetNodeId(300);
    s1n0->SetPreNodeId(2);
    s1n0->SetNextNodeId(301);

    auto s1n1 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 1, 1, 300);
    s1n1->SetNodeId(301);
    s1n1->SetPreNodeId(300);
    s1n1->SetNextNodeId(INVALID_TASK_ID);

    SuperKernelScopeInfo scopeInfo;
    scopeInfo.nodes = {s0n0.get(), s0n1.get(), s1n0.get(), s1n1.get()};

    ScopeStreamInfo stream0;
    stream0.streamIdx = 0;
    stream0.headNodeIdx = 200;
    stream0.tailNodeIdx = 201;
    stream0.nodeSize = 2;

    ScopeStreamInfo stream1;
    stream1.streamIdx = 1;
    stream1.headNodeIdx = 300;
    stream1.tailNodeIdx = 301;
    stream1.nodeSize = 2;

    scopeInfo.scopeStreamInfos = {stream0, stream1};

    graph->graphMap[200] = std::move(s0n0);
    graph->graphMap[201] = std::move(s0n1);
    graph->graphMap[300] = std::move(s1n0);
    graph->graphMap[301] = std::move(s1n1);

    SuperKernelScopePostProcessor postProcessor(*graph);
    SuperKernelProcessedScopeInfo processed = postProcessor.PostProcess(scopeInfo);

    EXPECT_TRUE(processed.nodes.empty());
    EXPECT_TRUE(processed.updateStreamInfos.empty());
    EXPECT_EQ(processed.skMainNodeId, INVALID_TASK_ID);
}
