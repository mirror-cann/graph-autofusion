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
#include <gmock/gmock.h>
#include <memory>
#include <vector>

#define private public
#define protected public
#include "sk_graph.h"
#include "sk_node.h"
#include "sk_scope_split.h"

// Helper function to create a kernel node
std::unique_ptr<SuperKernelKernelNode> CreateKernelNode(uint64_t nodeId, uint32_t streamIdx,
                                                       uint64_t preNodeId, uint64_t nextNodeId,
                                                       uint32_t numBlocks = 1,
                                                       SkKernelType kernelType = SkKernelType::AIC_ONLY) {
    auto node = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, nodeId, streamIdx, preNodeId);
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);

    // Set kernel parameters
    auto& nodeInfos = const_cast<NodeInfos&>(node->GetNodeInfos());
    nodeInfos.kernelInfos.numBlocks = numBlocks;
    nodeInfos.kernelInfos.kernelType = kernelType;
    // Set vecNum and cubeNum based on kernelType and numBlocks
    if (kernelType == SkKernelType::AIC_ONLY || kernelType == SkKernelType::MIX_AIC_1_0) {
        nodeInfos.kernelInfos.cubeNum = numBlocks;
        nodeInfos.kernelInfos.vecNum = 0;
    } else if (kernelType == SkKernelType::AIV_ONLY || kernelType == SkKernelType::MIX_AIV_1_0) {
        nodeInfos.kernelInfos.cubeNum = 0;
        nodeInfos.kernelInfos.vecNum = numBlocks;
    } else if (kernelType == SkKernelType::MIX_AIC_1_1) {
        nodeInfos.kernelInfos.cubeNum = numBlocks;
        nodeInfos.kernelInfos.vecNum = numBlocks;
    } else if (kernelType == SkKernelType::MIX_AIC_1_2) {
        nodeInfos.kernelInfos.cubeNum = numBlocks;
        nodeInfos.kernelInfos.vecNum = numBlocks << 1;
    }

    // Mark as fusible
    node->isFusible = true;

    return node;
}

// Helper function to create a wait node
std::unique_ptr<SuperKernelMemoryNode> CreateWaitNode(uint64_t nodeId, uint32_t streamIdx,
                                                      uint64_t preNodeId, uint64_t nextNodeId,
                                                      uint64_t notifyId) {
    auto node = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_VALUE_WAIT, nodeId, streamIdx, preNodeId);
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);

    // Set wait parameters
    auto& nodeInfos = const_cast<NodeInfos&>(node->GetNodeInfos());
    nodeInfos.syncInfos.correspondingNotifyNodeId = notifyId;

    node->isFusible = true;
    // Manually set nodeType since InitNode() is not called
    node->nodeType = SkNodeType::NODE_WAIT;

    return node;
}

// Helper function to create a notify node
std::unique_ptr<SuperKernelMemoryNode> CreateNotifyNode(uint64_t nodeId, uint32_t streamIdx,
                                                         uint64_t preNodeId, uint64_t nextNodeId,
                                                         uint64_t eventId,
                                                         const std::vector<uint64_t>& waitNodeIds = std::vector<uint64_t>()) {
    auto node = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_VALUE_WRITE, nodeId, streamIdx, preNodeId);
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);

    // Set notify parameters
    auto& nodeInfos = const_cast<NodeInfos&>(node->GetNodeInfos());
    nodeInfos.syncInfos.eventId = eventId;
    nodeInfos.syncInfos.correspondingWaitNodeIds = waitNodeIds;

    node->isFusible = true;
    // Manually set nodeType since InitNode() is not called
    node->nodeType = SkNodeType::NODE_NOTIFY;

    return node;
}

// Helper function to create a reset node
std::unique_ptr<SuperKernelMemoryNode> CreateResetNode(uint64_t nodeId, uint32_t streamIdx,
                                                        uint64_t preNodeId, uint64_t nextNodeId,
                                                        uint64_t eventId) {
    auto node = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_VALUE_WRITE, nodeId, streamIdx, preNodeId);
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);

    // Set reset parameters
    auto& nodeInfos = const_cast<NodeInfos&>(node->GetNodeInfos());
    nodeInfos.syncInfos.eventId = eventId;

    node->isFusible = true;
    // Manually set nodeType since InitNode() is not called
    node->nodeType = SkNodeType::NODE_RESET;

    return node;
}

/**
 * @brief Test fixture class for notify/wait interface tests
 */
class NotifyWaitInterfaceTest : public testing::Test {
protected:
    void SetUp() override {
        graph = std::make_unique<SuperKernelGraph>();
        splitter = std::make_unique<SuperKernelScopeSplitter>(*graph);
    }

    void TearDown() override {
        splitter.reset();
        graph.reset();
    }

    std::unique_ptr<SuperKernelGraph> graph;
    std::unique_ptr<SuperKernelScopeSplitter> splitter;
};

// ==================== Test Case 1: Notify-Wait Interface - One-to-Many Relationship ====================

TEST_F(NotifyWaitInterfaceTest, TestCase1_NotifyWaitOneToMany) {
    // Stream 0: [Notify(id=1, eventId=100)] -> [K2(id=2)]
    // Stream 1: [K3(id=3)] -> [Wait(id=4, notifyId=1)] -> [K5(id=5)]
    // Stream 2: [K6(id=6)] -> [Wait(id=7, notifyId=1)] -> [K8(id=8)]

    // Create notify node with two wait nodes (one-to-many)
    std::vector<uint64_t> waitNodeIds = {4, 7};
    graph->graphMap[1] = CreateNotifyNode(1, 0, INVALID_TASK_ID, 2, 100, waitNodeIds);
    graph->graphMap[2] = CreateKernelNode(2, 0, 1, INVALID_TASK_ID);

    graph->graphMap[3] = CreateKernelNode(3, 1, INVALID_TASK_ID, 4);
    graph->graphMap[4] = CreateWaitNode(4, 1, 3, 5, 1);
    graph->graphMap[5] = CreateKernelNode(5, 1, 4, INVALID_TASK_ID);

    graph->graphMap[6] = CreateKernelNode(6, 2, INVALID_TASK_ID, 7);
    graph->graphMap[7] = CreateWaitNode(7, 2, 6, 8, 1);
    graph->graphMap[8] = CreateKernelNode(8, 2, 7, INVALID_TASK_ID);

    // Set head nodes
    graph->headNodes = {1, 3, 6};

    // Verify notify node returns correct wait node IDs (one-to-many)
    auto* notifyNode = graph->graphMap[1].get();
    ASSERT_NE(notifyNode, nullptr);
    EXPECT_EQ(notifyNode->GetNodeType(), SkNodeType::NODE_NOTIFY);

    std::vector<uint64_t> retrievedWaitNodeIds = notifyNode->GetCorrespondingWaitNodeIds();
    EXPECT_EQ(retrievedWaitNodeIds.size(), 2);
    EXPECT_TRUE(std::find(retrievedWaitNodeIds.begin(), retrievedWaitNodeIds.end(), 4) != retrievedWaitNodeIds.end());
    EXPECT_TRUE(std::find(retrievedWaitNodeIds.begin(), retrievedWaitNodeIds.end(), 7) != retrievedWaitNodeIds.end());

    // Verify wait nodes return correct notify node ID (many-to-one)
    auto* waitNode1 = graph->graphMap[4].get();
    ASSERT_NE(waitNode1, nullptr);
    EXPECT_EQ(waitNode1->GetNodeType(), SkNodeType::NODE_WAIT);
    EXPECT_EQ(waitNode1->GetCorrespondingNotifyNodeId(), 1);

    auto* waitNode2 = graph->graphMap[7].get();
    ASSERT_NE(waitNode2, nullptr);
    EXPECT_EQ(waitNode2->GetNodeType(), SkNodeType::NODE_WAIT);
    EXPECT_EQ(waitNode2->GetCorrespondingNotifyNodeId(), 1);
}

// ==================== Test Case 2: Notify-Wait Interface - Empty Wait Node List ====================

TEST_F(NotifyWaitInterfaceTest, TestCase2_NotifyWithNoWaitNodes) {
    // Stream 0: [Notify(id=1, eventId=100)] -> [K2(id=2)]

    // Create notify node with no wait nodes
    graph->graphMap[1] = CreateNotifyNode(1, 0, INVALID_TASK_ID, 2, 100);
    graph->graphMap[2] = CreateKernelNode(2, 0, 1, INVALID_TASK_ID);

    // Set head nodes
    graph->headNodes = {1};

    // Verify notify node returns empty wait node list
    auto* notifyNode = graph->graphMap[1].get();
    ASSERT_NE(notifyNode, nullptr);
    EXPECT_EQ(notifyNode->GetNodeType(), SkNodeType::NODE_NOTIFY);

    std::vector<uint64_t> retrievedWaitNodeIds = notifyNode->GetCorrespondingWaitNodeIds();
    EXPECT_TRUE(retrievedWaitNodeIds.empty());
}

// ==================== Test Case 3: Memory Notify-Wait Interface ====================

TEST_F(NotifyWaitInterfaceTest, TestCase3_MemoryNotifyWaitInterface) {
    // Stream 0: [MemoryNotify(id=1, eventId=200)] -> [K2(id=2)]
    // Stream 1: [K3(id=3)] -> [MemoryWait(id=4, notifyId=1)] -> [K5(id=5)]

    // Create memory notify node with wait node
    std::vector<uint64_t> waitNodeIds = {4};
    auto memoryNotifyNode = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_VALUE_WRITE, 1, 0, INVALID_TASK_ID);
    memoryNotifyNode->SetNodeId(1);
    memoryNotifyNode->SetNextNodeId(2);
    auto& notifyInfos = const_cast<NodeInfos&>(memoryNotifyNode->GetNodeInfos());
    notifyInfos.syncInfos.eventId = 200;
    notifyInfos.syncInfos.correspondingWaitNodeIds = waitNodeIds;
    memoryNotifyNode->isFusible = true;
    // Manually set nodeType since InitNode() is not called
    memoryNotifyNode->nodeType = SkNodeType::NODE_NOTIFY;
    graph->graphMap[1] = std::move(memoryNotifyNode);
    graph->graphMap[2] = CreateKernelNode(2, 0, 1, INVALID_TASK_ID);

    graph->graphMap[3] = CreateKernelNode(3, 1, INVALID_TASK_ID, 4);

    auto memoryWaitNode = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_VALUE_WAIT, 4, 1, 3);
    memoryWaitNode->SetNodeId(4);
    memoryWaitNode->SetNextNodeId(5);
    auto& waitInfos = const_cast<NodeInfos&>(memoryWaitNode->GetNodeInfos());
    waitInfos.syncInfos.eventId = 200;
    waitInfos.syncInfos.correspondingNotifyNodeId = 1;
    memoryWaitNode->isFusible = true;
    // Manually set nodeType since InitNode() is not called
    memoryWaitNode->nodeType = SkNodeType::NODE_WAIT;
    graph->graphMap[4] = std::move(memoryWaitNode);
    graph->graphMap[5] = CreateKernelNode(5, 1, 4, INVALID_TASK_ID);

    // Set head nodes
    graph->headNodes = {1, 3};

    // Verify memory notify node returns correct wait node IDs
    auto* notifyNode = graph->graphMap[1].get();
    ASSERT_NE(notifyNode, nullptr);
    EXPECT_EQ(notifyNode->GetNodeType(), SkNodeType::NODE_NOTIFY);

    std::vector<uint64_t> retrievedWaitNodeIds = notifyNode->GetCorrespondingWaitNodeIds();
    EXPECT_EQ(retrievedWaitNodeIds.size(), 1);
    EXPECT_EQ(retrievedWaitNodeIds[0], 4);

    // Verify memory wait node returns correct notify node ID
    auto* waitNode = graph->graphMap[4].get();
    ASSERT_NE(waitNode, nullptr);
    EXPECT_EQ(waitNode->GetNodeType(), SkNodeType::NODE_WAIT);
    EXPECT_EQ(waitNode->GetCorrespondingNotifyNodeId(), 1);
}

// ==================== Test Case 4: Base Class Default Return Values ====================

TEST_F(NotifyWaitInterfaceTest, TestCase4_BaseClassDefaultReturns) {
    // Create a kernel node (should return default values for notify/wait methods)
    graph->graphMap[1] = CreateKernelNode(1, 0, INVALID_TASK_ID, INVALID_TASK_ID);
    graph->headNodes = {1};

    auto* kernelNode = graph->graphMap[1].get();
    ASSERT_NE(kernelNode, nullptr);

    // Base class should return empty vector for GetCorrespondingWaitNodeIds
    std::vector<uint64_t> waitIds = kernelNode->GetCorrespondingWaitNodeIds();
    EXPECT_TRUE(waitIds.empty());

    // Base class should return INVALID_TASK_ID for GetCorrespondingNotifyNodeId
    uint64_t notifyId = kernelNode->GetCorrespondingNotifyNodeId();
    EXPECT_EQ(notifyId, INVALID_TASK_ID);
}
