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
        nullptr, SkNodeType::NODE_KERNEL, nodeId, streamIdx, preNodeId);
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);

    // Set kernel parameters
    auto& nodeInfos = const_cast<NodeInfos&>(node->GetNodeInfos());
    nodeInfos.kernelInfos.numBlocks = numBlocks;
    nodeInfos.kernelInfos.kernelType = kernelType;

    // Mark as fusible
    node->isFusible = true;

    return node;
}

// Helper function to create a wait node
std::unique_ptr<SuperKernelEventWaitNode> CreateWaitNode(uint64_t nodeId, uint32_t streamIdx,
                                                      uint64_t preNodeId, uint64_t nextNodeId,
                                                      uint64_t notifyId) {
    auto node = std::make_unique<SuperKernelEventWaitNode>(
        nullptr, SkNodeType::NODE_WAIT, nodeId, streamIdx, preNodeId);
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);

    // Set wait parameters
    auto& nodeInfos = const_cast<NodeInfos&>(node->GetNodeInfos());
    nodeInfos.syncInfos.notifyNodeId = notifyId;

    node->isFusible = true;

    return node;
}

// Helper function to create a notify node
std::unique_ptr<SuperKernelEventNotifyNode> CreateNotifyNode(uint64_t nodeId, uint32_t streamIdx,
                                                         uint64_t preNodeId, uint64_t nextNodeId,
                                                         uint64_t eventId, uint64_t waitNodeId) {
    auto node = std::make_unique<SuperKernelEventNotifyNode>(
        nullptr, SkNodeType::NODE_NOTIFY, nodeId, streamIdx, preNodeId);
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);

    // Set notify parameters
    auto& nodeInfos = const_cast<NodeInfos&>(node->GetNodeInfos());
    nodeInfos.syncInfos.eventId = eventId;
    nodeInfos.syncInfos.waitNodeId = waitNodeId;

    node->isFusible = true;

    return node;
}

// Helper function to create a reset node
std::unique_ptr<SuperKernelMemoryResetNode> CreateResetNode(uint64_t nodeId, uint32_t streamIdx,
                                                        uint64_t preNodeId, uint64_t nextNodeId,
                                                        uint64_t eventId) {
    auto node = std::make_unique<SuperKernelMemoryResetNode>(
        nullptr, SkNodeType::NODE_RESET, nodeId, streamIdx, preNodeId);
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);

    // Set reset parameters
    auto& nodeInfos = const_cast<NodeInfos&>(node->GetNodeInfos());
    nodeInfos.syncInfos.eventId = eventId;

    node->isFusible = true;

    return node;
}

// Helper function to create an unfusible node
std::unique_ptr<SuperKernelKernelNode> CreateUnfusibleNode(uint64_t nodeId, uint32_t streamIdx,
                                                          uint64_t preNodeId, uint64_t nextNodeId) {
    auto node = std::make_unique<SuperKernelKernelNode>(
        nullptr, SkNodeType::NODE_KERNEL, nodeId, streamIdx, preNodeId);
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);
    node->isFusible = false; // Mark as unfusible

    return node;
}

/**
 * @brief Test fixture class for multi-stream graph splitting tests
 */
class MultiStreamSplitterTest : public testing::Test {
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

// ==================== Test Case 1: Basic Multi-Stream Fusion ====================

TEST_F(MultiStreamSplitterTest, TestCase1_BasicMultiStreamFusion) {
    // Stream 0: [K1(id=1)] -> [K2(id=2)] -> [K3(id=3)]
    // Stream 1: [K4(id=4)] -> [K5(id=5)] -> [K6(id=6)]

    // Add nodes to graph
    graph->graphMap[1] = CreateKernelNode(1, 0, INVALID_TASK_ID, 2);
    graph->graphMap[2] = CreateKernelNode(2, 0, 1, 3);
    graph->graphMap[3] = CreateKernelNode(3, 0, 2, INVALID_TASK_ID);

    graph->graphMap[4] = CreateKernelNode(4, 1, INVALID_TASK_ID, 5);
    graph->graphMap[5] = CreateKernelNode(5, 1, 4, 6);
    graph->graphMap[6] = CreateKernelNode(6, 1, 5, INVALID_TASK_ID);

    // Set head nodes
    graph->headNodes = {1, 4};

    // Split graph
    bool result = splitter->SplitGraph();
    EXPECT_TRUE(result);

    // Verify results
    const auto& scopeInfos = splitter->GetScopeInfos();
    EXPECT_GE(scopeInfos.size(), 1);

    // All nodes should be in scopes
    std::set<uint64_t> allNodesInScopes;
    for (const auto& scopeInfo : scopeInfos) {
        for (const auto* node : scopeInfo.nodes) {
            allNodesInScopes.insert(node->GetNodeId());
        }
    }

    EXPECT_EQ(allNodesInScopes.size(), 6);
    EXPECT_TRUE(allNodesInScopes.count(1));
    EXPECT_TRUE(allNodesInScopes.count(2));
    EXPECT_TRUE(allNodesInScopes.count(3));
    EXPECT_TRUE(allNodesInScopes.count(4));
    EXPECT_TRUE(allNodesInScopes.count(5));
    EXPECT_TRUE(allNodesInScopes.count(6));
}

// ==================== Test Case 2: Single Cross-Stream Wait-Notify ====================

TEST_F(MultiStreamSplitterTest, TestCase2_SingleCrossStreamWaitNotify) {
    // Stream 0: [K1(id=1)] -> [Wait1(id=3)] -> [K2(id=5)]
    // Stream 1: [K3(id=2)] -> [Notify1(id=4)] -> [K4(id=6)]
    // Event1: Wait1(id=3) 等待 Notify1(id=4)

    // Add nodes
    graph->graphMap[1] = CreateKernelNode(1, 0, INVALID_TASK_ID, 3);
    graph->graphMap[3] = CreateWaitNode(3, 0, 1, 5, 4); // Wait for Notify1
    graph->graphMap[5] = CreateKernelNode(5, 0, 3, INVALID_TASK_ID);

    graph->graphMap[2] = CreateKernelNode(2, 1, INVALID_TASK_ID, 4);
    graph->graphMap[4] = CreateNotifyNode(4, 1, 2, 6, 100, 3); // EventId=100, waitNodeId=3
    graph->graphMap[6] = CreateKernelNode(6, 1, 4, INVALID_TASK_ID);

    // Set event association
    graph->eventToNodes[100].notifyNodeId = 4;
    graph->eventToNodes[100].waitNodeIdList.insert(3);

    // Set head nodes
    graph->headNodes = {1, 2};

    // Split graph
    bool result = splitter->SplitGraph();
    EXPECT_TRUE(result);

    // Verify results
    const auto& scopeInfos = splitter->GetScopeInfos();

    // All nodes should be in one scope
    EXPECT_GE(scopeInfos.size(), 1);

    std::set<uint64_t> allNodesInScopes;
    for (const auto& scopeInfo : scopeInfos) {
        for (const auto* node : scopeInfo.nodes) {
            allNodesInScopes.insert(node->GetNodeId());
        }
    }

    EXPECT_EQ(allNodesInScopes.size(), 6);
    EXPECT_TRUE(allNodesInScopes.count(1));
    EXPECT_TRUE(allNodesInScopes.count(2));
    EXPECT_TRUE(allNodesInScopes.count(3));
    EXPECT_TRUE(allNodesInScopes.count(4));
    EXPECT_TRUE(allNodesInScopes.count(5));
    EXPECT_TRUE(allNodesInScopes.count(6));

    // Verify stream order
    // Stream 0: 1 -> 3 -> 5
    // Stream 1: 2 -> 4 -> 6
    for (const auto& scopeInfo : scopeInfos) {
        if (scopeInfo.nodes.size() >= 6) {
            bool foundStream0[3] = {false, false, false};
            bool foundStream1[3] = {false, false, false};
            for (size_t i = 0; i < scopeInfo.nodes.size(); ++i) {
                uint64_t nodeId = scopeInfo.nodes[i]->GetNodeId();
                uint32_t streamIdx = scopeInfo.nodes[i]->GetStreamIdxInGraph();
                if (streamIdx == 0) {
                    if (nodeId == 1) foundStream0[0] = true;
                    if (nodeId == 3) foundStream0[1] = true;
                    if (nodeId == 5) foundStream0[2] = true;
                } else if (streamIdx == 1) {
                    if (nodeId == 2) foundStream1[0] = true;
                    if (nodeId == 4) foundStream1[1] = true;
                    if (nodeId == 6) foundStream1[2] = true;
                }
            }
            EXPECT_TRUE(foundStream0[0] && foundStream0[1] && foundStream0[2]);
            EXPECT_TRUE(foundStream1[0] && foundStream1[1] && foundStream1[2]);
        }
    }
}

// ==================== Test Case 3: Multiple Wait-Notify ====================

TEST_F(MultiStreamSplitterTest, TestCase3_MultipleWaitNotify) {
    // Stream 0: [K1(id=1)] -> [Wait1(id=4)] -> [K2(id=7)]
    // Stream 1: [K3(id=2)] -> [Notify1(id=5)] -> [Wait2(id=8)] -> [K4(id=10)]
    // Stream 2: [K5(id=3)] -> [Notify2(id=6)] -> [K6(id=9)]
    // Event1: Wait1(id=4) 等待 Notify1(id=5)
    // Event2: Wait2(id=8) 等待 Notify2(id=6)

    // Add nodes
    graph->graphMap[1] = CreateKernelNode(1, 0, INVALID_TASK_ID, 4);
    graph->graphMap[4] = CreateWaitNode(4, 0, 1, 7, 5); // Wait for Notify1
    graph->graphMap[7] = CreateKernelNode(7, 0, 4, INVALID_TASK_ID);

    graph->graphMap[2] = CreateKernelNode(2, 1, INVALID_TASK_ID, 5);
    graph->graphMap[5] = CreateNotifyNode(5, 1, 2, 8, 100, 4); // EventId=100
    graph->graphMap[8] = CreateWaitNode(8, 1, 5, 10, 6); // Wait for Notify2
    graph->graphMap[10] = CreateKernelNode(10, 1, 8, INVALID_TASK_ID);

    graph->graphMap[3] = CreateKernelNode(3, 2, INVALID_TASK_ID, 6);
    graph->graphMap[6] = CreateNotifyNode(6, 2, 3, 9, 200, 8); // EventId=200
    graph->graphMap[9] = CreateKernelNode(9, 2, 6, INVALID_TASK_ID);

    // Set event associations
    graph->eventToNodes[100].notifyNodeId = 5;
    graph->eventToNodes[100].waitNodeIdList.insert(4);

    graph->eventToNodes[200].notifyNodeId = 6;
    graph->eventToNodes[200].waitNodeIdList.insert(8);

    // Set head nodes
    graph->headNodes = {1, 2, 3};

    // Split graph
    bool result = splitter->SplitGraph();
    EXPECT_TRUE(result);

    // Verify results
    const auto& scopeInfos = splitter->GetScopeInfos();
    EXPECT_GE(scopeInfos.size(), 1);

    std::set<uint64_t> allNodesInScopes;
    for (const auto& scopeInfo : scopeInfos) {
        for (const auto* node : scopeInfo.nodes) {
            allNodesInScopes.insert(node->GetNodeId());
        }
    }

    EXPECT_EQ(allNodesInScopes.size(), 9);
}

// ==================== Test Case 4: Unfusible Node (Single Stream) ====================

TEST_F(MultiStreamSplitterTest, TestCase4_UnfusibleNodeSingleStream) {
    // Stream 0: [K1(可融合)] -> [UF1(不可融合)] -> [K2(可融合)]

    // Add nodes
    graph->graphMap[1] = CreateKernelNode(1, 0, INVALID_TASK_ID, 2);
    graph->graphMap[2] = CreateUnfusibleNode(2, 0, 1, 3); // Unfusible
    graph->graphMap[3] = CreateKernelNode(3, 0, 2, INVALID_TASK_ID);

    // Set head nodes
    graph->headNodes = {1};

    // Split graph
    bool result = splitter->SplitGraph();
    EXPECT_TRUE(result);

    // Verify results
    const auto& scopeInfos = splitter->GetScopeInfos();

    // Should have 2 scopes: {K1} and {K2}
    // UF1 should not be in any scope
    std::set<uint64_t> allNodesInScopes;
    for (const auto& scopeInfo : scopeInfos) {
        for (const auto* node : scopeInfo.nodes) {
            allNodesInScopes.insert(node->GetNodeId());
        }
    }

    EXPECT_EQ(allNodesInScopes.size(), 2);
    EXPECT_TRUE(allNodesInScopes.count(1));
    EXPECT_TRUE(allNodesInScopes.count(3));
    EXPECT_FALSE(allNodesInScopes.count(2)); // UF1 should not be in scope
}

// ==================== Test Case 5: Unfusible Node (Multi-Stream) ====================

TEST_F(MultiStreamSplitterTest, TestCase5_UnfusibleNodeMultiStream) {
    // Stream 0: [K1(可融合)] -> [UF1(不可融合)] -> [K2(可融合)]
    // Stream 1: [K3(可融合)]

    // Add nodes
    graph->graphMap[1] = CreateKernelNode(1, 0, INVALID_TASK_ID, 2);
    graph->graphMap[2] = CreateUnfusibleNode(2, 0, 1, 3); // Unfusible
    graph->graphMap[3] = CreateKernelNode(3, 0, 2, INVALID_TASK_ID);

    graph->graphMap[4] = CreateKernelNode(4, 1, INVALID_TASK_ID, INVALID_TASK_ID);

    // Set head nodes
    graph->headNodes = {1, 4};

    // Split graph
    bool result = splitter->SplitGraph();
    EXPECT_TRUE(result);

    // Verify results
    const auto& scopeInfos = splitter->GetScopeInfos();

    std::set<uint64_t> allNodesInScopes;
    for (const auto& scopeInfo : scopeInfos) {
        for (const auto* node : scopeInfo.nodes) {
            allNodesInScopes.insert(node->GetNodeId());
        }
    }

    // Should have K1, K3, K2 in scopes, UF1 not in any scope
    EXPECT_EQ(allNodesInScopes.size(), 3);
    EXPECT_TRUE(allNodesInScopes.count(1));
    EXPECT_TRUE(allNodesInScopes.count(3));
    EXPECT_TRUE(allNodesInScopes.count(4));
    EXPECT_FALSE(allNodesInScopes.count(2));
}

// ==================== Test Case 6: Reset Node ====================

TEST_F(MultiStreamSplitterTest, TestCase6_ResetNode) {
    // Stream 0: [K1(id=1)] -> [Wait1(id=3)] -> [K2(id=5)] -> [Reset1(id=7)] -> [K3(id=9)]
    // Stream 1: [K4(id=2)] -> [Notify1(id=4)] -> [K5(id=6)] -> [Notify2(id=8)] -> [K6(id=10)]
    // Event1: Wait1(id=3) 等待 Notify1(id=4)
    // Event2: Reset1(id=7) 重置 Event2

    // Add nodes
    graph->graphMap[1] = CreateKernelNode(1, 0, INVALID_TASK_ID, 3);
    graph->graphMap[3] = CreateWaitNode(3, 0, 1, 5, 4); // Wait for Notify1
    graph->graphMap[5] = CreateKernelNode(5, 0, 3, 7);
    graph->graphMap[7] = CreateResetNode(7, 0, 5, 9, 200); // Reset Event2
    graph->graphMap[9] = CreateKernelNode(9, 0, 7, INVALID_TASK_ID);

    graph->graphMap[2] = CreateKernelNode(2, 1, INVALID_TASK_ID, 4);
    graph->graphMap[4] = CreateNotifyNode(4, 1, 2, 6, 100, 3); // EventId=100
    graph->graphMap[6] = CreateKernelNode(6, 1, 4, 8);
    graph->graphMap[8] = CreateNotifyNode(8, 1, 6, 10, 200, 0); // EventId=200
    graph->graphMap[10] = CreateKernelNode(10, 1, 8, INVALID_TASK_ID);

    // Set event associations
    graph->eventToNodes[100].notifyNodeId = 4;
    graph->eventToNodes[100].waitNodeIdList.insert(3);

    graph->eventToNodes[200].notifyNodeId = 8;
    graph->eventToNodes[200].resetNodeId = 7;

    // Set head nodes
    graph->headNodes = {1, 2};

    // Split graph
    bool result = splitter->SplitGraph();
    EXPECT_TRUE(result);

    // Verify results
    const auto& scopeInfos = splitter->GetScopeInfos();
    EXPECT_GE(scopeInfos.size(), 1);

    std::set<uint64_t> allNodesInScopes;
    for (const auto& scopeInfo : scopeInfos) {
        for (const auto* node : scopeInfo.nodes) {
            allNodesInScopes.insert(node->GetNodeId());
        }
    }

    // All nodes should be in scopes
    EXPECT_EQ(allNodesInScopes.size(), 9);
}

// ==================== Test Case 7: Multiple Waits for Same Notify ====================

TEST_F(MultiStreamSplitterTest, TestCase7_MultipleWaitsSameNotify) {
    // Stream 0: [K1(id=1)] -> [Wait1(id=4)] -> [K2(id=7)]
    // Stream 1: [K3(id=2)] -> [Wait2(id=5)] -> [K4(id=8)]
    // Stream 2: [K5(id=3)] -> [Notify1(id=6)] -> [K6(id=9)]
    // Event1: Wait1(id=4) 和 Wait2(id=5) 都等待 Notify1(id=6)

    // Add nodes
    graph->graphMap[1] = CreateKernelNode(1, 0, INVALID_TASK_ID, 4);
    graph->graphMap[4] = CreateWaitNode(4, 0, 1, 7, 6); // Wait for Notify1
    graph->graphMap[7] = CreateKernelNode(7, 0, 4, INVALID_TASK_ID);

    graph->graphMap[2] = CreateKernelNode(2, 1, INVALID_TASK_ID, 5);
    graph->graphMap[5] = CreateWaitNode(5, 1, 2, 8, 6); // Wait for Notify1
    graph->graphMap[8] = CreateKernelNode(8, 1, 5, INVALID_TASK_ID);

    graph->graphMap[3] = CreateKernelNode(3, 2, INVALID_TASK_ID, 6);
    graph->graphMap[6] = CreateNotifyNode(6, 2, 3, 9, 100, 0); // EventId=100
    graph->graphMap[9] = CreateKernelNode(9, 2, 6, INVALID_TASK_ID);

    // Set event associations
    graph->eventToNodes[100].notifyNodeId = 6;
    graph->eventToNodes[100].waitNodeIdList.insert(4);
    graph->eventToNodes[100].waitNodeIdList.insert(5);

    // Set head nodes
    graph->headNodes = {1, 2, 3};

    // Split graph
    bool result = splitter->SplitGraph();
    EXPECT_TRUE(result);

    // Verify results
    const auto& scopeInfos = splitter->GetScopeInfos();
    EXPECT_GE(scopeInfos.size(), 1);

    std::set<uint64_t> allNodesInScopes;
    for (const auto& scopeInfo : scopeInfos) {
        for (const auto* node : scopeInfo.nodes) {
            allNodesInScopes.insert(node->GetNodeId());
        }
    }

    EXPECT_EQ(allNodesInScopes.size(), 9);
}

// ==================== Test Case 8: Complex Scenario ====================

TEST_F(MultiStreamSplitterTest, TestCase8_ComplexScenario) {
    // Stream 0: [K1(id=1)] -> [K2(id=3)] -> [Wait1(id=5)] -> [K3(id=7)]
    //          -> [UF1(id=9,不可融合)] -> [K4(id=11)] -> [K5(id=13)]
    // Stream 1: [K6(id=2)] -> [Notify1(id=4)] -> [K7(id=6)] -> [Wait2(id=8)] -> [K8(id=10)]
    // Stream 2: [K9(id=12)] -> [Notify2(id=14)] -> [K10(id=16)]
    // Event1: Wait1(id=5) 等待 Notify1(id=4)
    // Event2: Wait2(id=8) 等待 Notify2(id=14)

    // Add nodes for Stream 0
    graph->graphMap[1] = CreateKernelNode(1, 0, INVALID_TASK_ID, 3);
    graph->graphMap[3] = CreateKernelNode(3, 0, 1, 5);
    graph->graphMap[5] = CreateWaitNode(5, 0, 3, 7, 4); // Wait for Notify1
    graph->graphMap[7] = CreateKernelNode(7, 0, 5, 9);
    graph->graphMap[9] = CreateUnfusibleNode(9, 0, 7, 11); // Unfusible
    graph->graphMap[11] = CreateKernelNode(11, 0, 9, 13);
    graph->graphMap[13] = CreateKernelNode(13, 0, 11, INVALID_TASK_ID);

    // Add nodes for Stream 1
    graph->graphMap[2] = CreateKernelNode(2, 1, INVALID_TASK_ID, 4);
    graph->graphMap[4] = CreateNotifyNode(4, 1, 2, 6, 100, 5); // EventId=100
    graph->graphMap[6] = CreateKernelNode(6, 1, 4, 8);
    graph->graphMap[8] = CreateWaitNode(8, 1, 6, 10, 14); // Wait for Notify2
    graph->graphMap[10] = CreateKernelNode(10, 1, 8, INVALID_TASK_ID);

    // Add nodes for Stream 2
    graph->graphMap[12] = CreateKernelNode(12, 2, INVALID_TASK_ID, 14);
    graph->graphMap[14] = CreateNotifyNode(14, 2, 12, 16, 200, 8); // EventId=200
    graph->graphMap[16] = CreateKernelNode(16, 2, 14, INVALID_TASK_ID);

    // Set event associations
    graph->eventToNodes[100].notifyNodeId = 4;
    graph->eventToNodes[100].waitNodeIdList.insert(5);

    graph->eventToNodes[200].notifyNodeId = 14;
    graph->eventToNodes[200].waitNodeIdList.insert(8);

    // Set head nodes
    graph->headNodes = {1, 2, 12};

    // Split graph
    bool result = splitter->SplitGraph();
    EXPECT_TRUE(result);

    // Verify results
    const auto& scopeInfos = splitter->GetScopeInfos();

    // Count total nodes in all scopes
    std::set<uint64_t> allNodesInScopes;
    for (const auto& scopeInfo : scopeInfos) {
        for (const auto* node : scopeInfo.nodes) {
            allNodesInScopes.insert(node->GetNodeId());
        }
    }

    // Expected nodes: 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 16
    // Not expected: 9 (UF1), 11 (K4 depends on UF1 order)
    // Actually, K4 should be in a scope after UF1 is skipped
    EXPECT_TRUE(allNodesInScopes.count(1));
    EXPECT_TRUE(allNodesInScopes.count(2));
    EXPECT_TRUE(allNodesInScopes.count(3));
    EXPECT_TRUE(allNodesInScopes.count(4));
    EXPECT_TRUE(allNodesInScopes.count(5));
    EXPECT_TRUE(allNodesInScopes.count(6));
    EXPECT_TRUE(allNodesInScopes.count(7));
    EXPECT_TRUE(allNodesInScopes.count(10));
    EXPECT_TRUE(allNodesInScopes.count(12));
    EXPECT_TRUE(allNodesInScopes.count(13));
    EXPECT_TRUE(allNodesInScopes.count(14));
    EXPECT_TRUE(allNodesInScopes.count(16));
    EXPECT_FALSE(allNodesInScopes.count(9)); // UF1 should not be in scope
}

// ==================== Test Case 9: Empty Skip Handling ====================

TEST_F(MultiStreamSplitterTest, TestCase9_EmptySkipHandling) {
    // Stream 0: [K1(可融合)] -> [UF1(不可融合)] -> [UF2(不可融合)] -> [K2(可融合)]

    // Add nodes
    graph->graphMap[1] = CreateKernelNode(1, 0, INVALID_TASK_ID, 2);
    graph->graphMap[2] = CreateUnfusibleNode(2, 0, 1, 3); // UF1
    graph->graphMap[3] = CreateUnfusibleNode(3, 0, 2, 4); // UF2
    graph->graphMap[4] = CreateKernelNode(4, 0, 3, INVALID_TASK_ID);

    // Set head nodes
    graph->headNodes = {1};

    // Split graph
    bool result = splitter->SplitGraph();
    EXPECT_TRUE(result);

    // Verify results
    const auto& scopeInfos = splitter->GetScopeInfos();

    std::set<uint64_t> allNodesInScopes;
    for (const auto& scopeInfo : scopeInfos) {
        for (const auto* node : scopeInfo.nodes) {
            allNodesInScopes.insert(node->GetNodeId());
        }
    }

    // Should have K1 and K2 in scopes
    EXPECT_EQ(allNodesInScopes.size(), 2);
    EXPECT_TRUE(allNodesInScopes.count(1));
    EXPECT_TRUE(allNodesInScopes.count(4));
    EXPECT_FALSE(allNodesInScopes.count(2)); // UF1
    EXPECT_FALSE(allNodesInScopes.count(3)); // UF2
}

// ==================== Test Case 10: Single Stream (Fallback to Single Stream Logic) ====================

TEST_F(MultiStreamSplitterTest, TestCase10_SingleStreamFallback) {
    // Stream 0: [K1(id=1)] -> [K2(id=2)] -> [K3(id=3)]

    // Add nodes
    graph->graphMap[1] = CreateKernelNode(1, 0, INVALID_TASK_ID, 2);
    graph->graphMap[2] = CreateKernelNode(2, 0, 1, 3);
    graph->graphMap[3] = CreateKernelNode(3, 0, 2, INVALID_TASK_ID);

    // Set head nodes
    graph->headNodes = {1};

    // Split graph
    bool result = splitter->SplitGraph();
    EXPECT_TRUE(result);

    // Verify results
    const auto& scopeInfos = splitter->GetScopeInfos();
    EXPECT_GE(scopeInfos.size(), 1);

    std::set<uint64_t> allNodesInScopes;
    for (const auto& scopeInfo : scopeInfos) {
        for (const auto* node : scopeInfo.nodes) {
            allNodesInScopes.insert(node->GetNodeId());
        }
    }

    EXPECT_EQ(allNodesInScopes.size(), 3);
}
