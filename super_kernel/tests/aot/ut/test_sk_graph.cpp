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
#include <gmock/gmock.h>
#include <memory>
#include <vector>

#define private public
#define protected public
#include "sk_graph.h"
#include "sk_node.h"
#include "sk_lock_detector.h"

/**
 * @brief Test fixture class for SuperKernelGraph unit tests
 */
class SuperKernelGraphTest : public testing::Test {
protected:
    void SetUp() override {
        graph = std::make_unique<SuperKernelGraph>();
    }

    void TearDown() override {
        graph.reset();
    }

    std::unique_ptr<SuperKernelGraph> graph;
};

// ==================== GetSortedNodeIds Empty Graph Tests ====================

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_EmptyGraph)
{
    auto sortedIds = graph->GetSortedNodeIds();
    EXPECT_TRUE(sortedIds.empty());
    EXPECT_EQ(sortedIds.size(), 0);
}

// ==================== GetSortedNodeIds Single Node Tests ====================

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_SingleNode)
{
    auto node = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(5);
    graph->graphMap[5] = std::move(node);

    auto sortedIds = graph->GetSortedNodeIds();
    EXPECT_EQ(sortedIds.size(), 1);
    EXPECT_EQ(sortedIds[0], 5);
}

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_SingleNodeWithMinId)
{
    auto node = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(0);
    graph->graphMap[0] = std::move(node);

    auto sortedIds = graph->GetSortedNodeIds();
    EXPECT_EQ(sortedIds.size(), 1);
    EXPECT_EQ(sortedIds[0], 0);
}

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_SingleNodeWithMaxId)
{
    auto node = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(100);
    graph->graphMap[100] = std::move(node);

    auto sortedIds = graph->GetSortedNodeIds();
    EXPECT_EQ(sortedIds.size(), 1);
    EXPECT_EQ(sortedIds[0], 100);
}

// ==================== GetSortedNodeIds Multiple Nodes Tests ====================

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_MultipleNodes_Sequential)
{
    graph->graphMap[1] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[2] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[3] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);

    auto sortedIds = graph->GetSortedNodeIds();
    EXPECT_EQ(sortedIds.size(), 3);
    EXPECT_EQ(sortedIds[0], 1);
    EXPECT_EQ(sortedIds[1], 2);
    EXPECT_EQ(sortedIds[2], 3);
}

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_MultipleNodes_ReverseOrder)
{
    graph->graphMap[3] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[2] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[1] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);

    auto sortedIds = graph->GetSortedNodeIds();
    EXPECT_EQ(sortedIds.size(), 3);
    EXPECT_EQ(sortedIds[0], 1);
    EXPECT_EQ(sortedIds[1], 2);
    EXPECT_EQ(sortedIds[2], 3);
}

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_MultipleNodes_RandomOrder)
{
    graph->graphMap[5] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[2] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[8] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[1] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[10] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);

    auto sortedIds = graph->GetSortedNodeIds();
    EXPECT_EQ(sortedIds.size(), 5);
    EXPECT_EQ(sortedIds[0], 1);
    EXPECT_EQ(sortedIds[1], 2);
    EXPECT_EQ(sortedIds[2], 5);
    EXPECT_EQ(sortedIds[3], 8);
    EXPECT_EQ(sortedIds[4], 10);
}

// ==================== GetSortedNodeIds Boundary Tests ====================

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_Boundary_ZeroId)
{
    graph->graphMap[0] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[1] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[2] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);

    auto sortedIds = graph->GetSortedNodeIds();
    EXPECT_EQ(sortedIds.size(), 3);
    EXPECT_EQ(sortedIds[0], 0);
    EXPECT_EQ(sortedIds[1], 1);
    EXPECT_EQ(sortedIds[2], 2);
}

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_Boundary_LargeIdGap)
{
    graph->graphMap[1] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[1000] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[500] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);

    auto sortedIds = graph->GetSortedNodeIds();
    EXPECT_EQ(sortedIds.size(), 3);
    EXPECT_EQ(sortedIds[0], 1);
    EXPECT_EQ(sortedIds[1], 500);
    EXPECT_EQ(sortedIds[2], 1000);
}

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_Boundary_NonContiguousIds)
{
    graph->graphMap[10] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[20] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[30] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);

    auto sortedIds = graph->GetSortedNodeIds();
    EXPECT_EQ(sortedIds.size(), 3);
    EXPECT_EQ(sortedIds[0], 10);
    EXPECT_EQ(sortedIds[1], 20);
    EXPECT_EQ(sortedIds[2], 30);
}

// ==================== GetSortedNodeIds Multiple Calls Tests ====================

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_MultipleCalls_SameGraph)
{
    graph->graphMap[3] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[1] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[2] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);

    auto sortedIds1 = graph->GetSortedNodeIds();
    auto sortedIds2 = graph->GetSortedNodeIds();

    EXPECT_EQ(sortedIds1.size(), sortedIds2.size());
    for (size_t i = 0; i < sortedIds1.size(); ++i) {
        EXPECT_EQ(sortedIds1[i], sortedIds2[i]);
    }
}

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_MultipleCalls_AfterModification)
{
    graph->graphMap[2] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[1] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);

    auto sortedIds1 = graph->GetSortedNodeIds();
    EXPECT_EQ(sortedIds1.size(), 2);
    EXPECT_EQ(sortedIds1[0], 1);
    EXPECT_EQ(sortedIds1[1], 2);

    graph->graphMap[3] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);

    auto sortedIds2 = graph->GetSortedNodeIds();
    EXPECT_EQ(sortedIds2.size(), 3);
    EXPECT_EQ(sortedIds2[0], 1);
    EXPECT_EQ(sortedIds2[1], 2);
    EXPECT_EQ(sortedIds2[2], 3);
}

// ==================== GetSortedNodeIds Mixed Node Types Tests ====================

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_MixedNodeTypes)
{
    graph->graphMap[1] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[3] = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_VALUE_WRITE, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[2] = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_VALUE_WAIT, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);

    auto sortedIds = graph->GetSortedNodeIds();
    EXPECT_EQ(sortedIds.size(), 3);
    EXPECT_EQ(sortedIds[0], 1);
    EXPECT_EQ(sortedIds[1], 2);
    EXPECT_EQ(sortedIds[2], 3);
}

// ==================== GetSortedNodeIds Large Scale Tests ====================

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_LargeScale)
{
    const uint32_t numNodes = 100;
    for (uint32_t i = 0; i < numNodes; ++i) {
        uint64_t nodeId = numNodes - i;  // Insert in reverse order
        graph->graphMap[nodeId] = std::make_unique<SuperKernelKernelNode>(
            nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    }

    auto sortedIds = graph->GetSortedNodeIds();
    EXPECT_EQ(sortedIds.size(), numNodes);

    for (uint32_t i = 0; i < numNodes; ++i) {
        EXPECT_EQ(sortedIds[i], i + 1);
    }
}

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_LargeScale_RandomInsertion)
{
    const uint32_t numNodes = 50;
    std::vector<uint64_t> nodeIds = {5, 23, 1, 45, 12, 8, 33, 17, 50, 3,
                                     27, 9, 41, 2, 37, 19, 48, 6, 30, 14,
                                     25, 11, 39, 4, 35, 20, 44, 7, 29, 13,
                                     49, 10, 42, 18, 36, 15, 46, 22, 38, 26,
                                     47, 21, 40, 24, 34, 16, 43, 31, 32, 28};

    for (uint64_t nodeId : nodeIds) {
        graph->graphMap[nodeId] = std::make_unique<SuperKernelKernelNode>(
            nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    }

    auto sortedIds = graph->GetSortedNodeIds();
    EXPECT_EQ(sortedIds.size(), numNodes);

    std::vector<uint64_t> expectedIds = nodeIds;
    std::sort(expectedIds.begin(), expectedIds.end());

    for (uint32_t i = 0; i < numNodes; ++i) {
        EXPECT_EQ(sortedIds[i], expectedIds[i]);
    }
}

// ==================== GetSortedNodeIds Const Correctness Tests ====================

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_ConstMethod)
{
    graph->graphMap[2] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[1] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);

    const SuperKernelGraph* constGraph = graph.get();
    auto sortedIds = constGraph->GetSortedNodeIds();

    EXPECT_EQ(sortedIds.size(), 2);
    EXPECT_EQ(sortedIds[0], 1);
    EXPECT_EQ(sortedIds[1], 2);
}

// ==================== GetSortedNodeIds Duplicate Values Tests ====================

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_NoDuplicateIds)
{
    graph->graphMap[1] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[2] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[3] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);

    auto sortedIds = graph->GetSortedNodeIds();
    EXPECT_EQ(sortedIds.size(), 3);

    std::sort(sortedIds.begin(), sortedIds.end());
    auto last = std::unique(sortedIds.begin(), sortedIds.end());
    EXPECT_EQ(last, sortedIds.end());
}

// ==================== GetSortedNodeIds Integrity Tests ====================

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_Integrity_AllIdsPresent)
{
    std::vector<uint64_t> expectedIds = {5, 10, 15, 20, 25};

    for (uint64_t nodeId : expectedIds) {
        graph->graphMap[nodeId] = std::make_unique<SuperKernelKernelNode>(
            nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    }

    auto sortedIds = graph->GetSortedNodeIds();
    EXPECT_EQ(sortedIds.size(), expectedIds.size());

    for (uint64_t expectedId : expectedIds) {
        auto it = std::find(sortedIds.begin(), sortedIds.end(), expectedId);
        EXPECT_NE(it, sortedIds.end());
    }
}

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_Integrity_NoExtraIds)
{
    graph->graphMap[3] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[1] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    graph->graphMap[2] = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);

    auto sortedIds = graph->GetSortedNodeIds();
    EXPECT_EQ(sortedIds.size(), 3);
    EXPECT_EQ(sortedIds[0], 1);
    EXPECT_EQ(sortedIds[1], 2);
    EXPECT_EQ(sortedIds[2], 3);
}

// ==================== BitsetToString Tests ====================

TEST_F(SuperKernelGraphTest, BitsetToString_EmptyScopeNameToIdx)
{
    graph->scopeNameToIdx.clear();
    std::bitset<MAX_SCOPE_NUM> flags;
    flags.set(0);
    std::string result = graph->BitsetToString(flags);
    EXPECT_EQ(result, "0");
}

TEST_F(SuperKernelGraphTest, BitsetToString_WithScopeNames)
{
    graph->scopeNameToIdx["scope_a"] = 0;
    graph->scopeNameToIdx["scope_b"] = 1;
    std::bitset<MAX_SCOPE_NUM> flags;
    flags.set(0);
    flags.set(1);
    std::string result = graph->BitsetToString(flags);
    EXPECT_EQ(result.size(), 2);
}

TEST_F(SuperKernelGraphTest, BitsetToString_SingleFlag)
{
    graph->scopeNameToIdx["scope_a"] = 0;
    std::bitset<MAX_SCOPE_NUM> flags;
    flags.set(0);
    std::string result = graph->BitsetToString(flags);
    EXPECT_EQ(result, "1");
}

// ==================== GetNodeById Tests ====================

TEST_F(SuperKernelGraphTest, GetNodeById_ExistingNode)
{
    auto node = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(5);
    graph->graphMap[5] = std::move(node);

    auto* retrievedNode = graph->GetNodeById(5);
    EXPECT_NE(retrievedNode, nullptr);
    EXPECT_EQ(retrievedNode->GetNodeId(), 5);
}

TEST_F(SuperKernelGraphTest, GetNodeById_NonExistingNode)
{
    auto* retrievedNode = graph->GetNodeById(999);
    EXPECT_EQ(retrievedNode, nullptr);
}

// ==================== AddNode Tests ====================

TEST_F(SuperKernelGraphTest, AddNode_KernelNode)
{
    auto node = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(10);

    bool result = graph->AddNode(std::move(node));
    EXPECT_TRUE(result);
    EXPECT_EQ(graph->graphMap.size(), 1);
    EXPECT_NE(graph->GetNodeById(10), nullptr);
}

TEST_F(SuperKernelGraphTest, AddNode_DuplicateNodeId)
{
    auto node1 = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    node1->SetNodeId(10);
    graph->graphMap[10] = std::move(node1);

    auto node2 = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    node2->SetNodeId(10);

    bool result = graph->AddNode(std::move(node2));
    EXPECT_FALSE(result);
    EXPECT_EQ(graph->graphMap.size(), 1);
}

// ==================== ExpandUpdateNodes Tests ====================

TEST_F(SuperKernelGraphTest, ExpandUpdateNodes_EmptyList)
{
    std::vector<SuperKernelBaseNode*> customNodes;
    bool result = graph->ExpandUpdateNodes(customNodes);
    EXPECT_TRUE(result);
    EXPECT_EQ(graph->needUpdateNodes.size(), 0);
}

TEST_F(SuperKernelGraphTest, ExpandUpdateNodes_AddNodes)
{
    auto node1 = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    node1->SetNodeId(10);
    SuperKernelBaseNode* node1Ptr = node1.get();
    graph->graphMap[10] = std::move(node1);

    auto node2 = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    node2->SetNodeId(20);
    SuperKernelBaseNode* node2Ptr = node2.get();
    graph->graphMap[20] = std::move(node2);

    std::vector<SuperKernelBaseNode*> customNodes = {node1Ptr, node2Ptr};
    bool result = graph->ExpandUpdateNodes(customNodes);
    EXPECT_TRUE(result);
    EXPECT_EQ(graph->needUpdateNodes.size(), 2);
}

TEST_F(SuperKernelGraphTest, ExpandUpdateNodes_NullNode)
{
    std::vector<SuperKernelBaseNode*> customNodes = {nullptr};
    bool result = graph->ExpandUpdateNodes(customNodes);
    EXPECT_TRUE(result);
    EXPECT_EQ(graph->needUpdateNodes.size(), 0);
}

TEST_F(SuperKernelGraphTest, ExpandUpdateNodes_DuplicateAdd)
{
    auto node = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(10);
    SuperKernelBaseNode* nodePtr = node.get();
    graph->graphMap[10] = std::move(node);

    std::vector<SuperKernelBaseNode*> customNodes1 = {nodePtr};
    graph->ExpandUpdateNodes(customNodes1);
    EXPECT_EQ(graph->needUpdateNodes.size(), 1);

    std::vector<SuperKernelBaseNode*> customNodes2 = {nodePtr};
    graph->ExpandUpdateNodes(customNodes2);
    EXPECT_EQ(graph->needUpdateNodes.size(), 1);
}

// ==================== GetStreamByIndex Tests ====================

TEST_F(SuperKernelGraphTest, GetStreamByIndex_ValidIndex)
{
    graph->streams.resize(3);
    graph->streams[0] = reinterpret_cast<aclrtStream>(0x100);
    graph->streams[1] = reinterpret_cast<aclrtStream>(0x200);
    graph->streams[2] = reinterpret_cast<aclrtStream>(0x300);

    auto* stream = graph->GetStreamByIndex(1);
    EXPECT_EQ(stream, reinterpret_cast<aclrtStream>(0x200));
}

TEST_F(SuperKernelGraphTest, GetStreamByIndex_OutOfBounds)
{
    graph->streams.resize(2);
    auto* stream = graph->GetStreamByIndex(5);
    EXPECT_EQ(stream, nullptr);
}

// ==================== GetScopeNameByIdx Tests ====================

TEST_F(SuperKernelGraphTest, GetScopeNameByIdx_ExistingIdx)
{
    graph->scopeIdxToName[0] = "scope_a";
    graph->scopeIdxToName[1] = "scope_b";

    std::string scopeName;
    bool result = graph->GetScopeNameByIdx(0, scopeName);
    EXPECT_TRUE(result);
    EXPECT_EQ(scopeName, "scope_a");
}

TEST_F(SuperKernelGraphTest, GetScopeNameByIdx_NonExistingIdx)
{
    graph->scopeIdxToName[0] = "scope_a";

    std::string scopeName;
    bool result = graph->GetScopeNameByIdx(5, scopeName);
    EXPECT_FALSE(result);
}

// ==================== GetOriginalScopeInfos Tests ====================

TEST_F(SuperKernelGraphTest, GetOriginalScopeInfos_Empty)
{
    const auto& scopeInfos = graph->GetOriginalScopeInfos();
    EXPECT_EQ(scopeInfos.size(), 0);
}

TEST_F(SuperKernelGraphTest, GetOriginalScopeInfos_WithData)
{
    OriginalScopeInfo info;
    info.scopeId = 0;
    info.nodeIds = {1, 2, 3};
    graph->originalScopeInfos_.push_back(info);

    const auto& scopeInfos = graph->GetOriginalScopeInfos();
    EXPECT_EQ(scopeInfos.size(), 1);
    EXPECT_EQ(scopeInfos[0].nodeIds.size(), 3);
}

// ==================== GetStreams Tests ====================

TEST_F(SuperKernelGraphTest, GetStreams_Empty)
{
    const auto& streams = graph->GetStreams();
    EXPECT_EQ(streams.size(), 0);
}

TEST_F(SuperKernelGraphTest, GetStreams_WithStreams)
{
    graph->streams.resize(3);
    graph->streams[0] = reinterpret_cast<aclrtStream>(0x100);
    graph->streams[1] = reinterpret_cast<aclrtStream>(0x200);
    graph->streams[2] = reinterpret_cast<aclrtStream>(0x300);

    const auto& streams = graph->GetStreams();
    EXPECT_EQ(streams.size(), 3);
}

// ==================== GetHeadNodes Tests ====================

TEST_F(SuperKernelGraphTest, GetHeadNodes_Empty)
{
    const auto& headNodes = graph->GetHeadNodes();
    EXPECT_EQ(headNodes.size(), 0);
}

TEST_F(SuperKernelGraphTest, GetHeadNodes_WithNodes)
{
    graph->headNodes = {10, 20, 30};
    const auto& headNodes = graph->GetHeadNodes();
    EXPECT_EQ(headNodes.size(), 3);
    EXPECT_EQ(headNodes[0], 10);
    EXPECT_EQ(headNodes[1], 20);
    EXPECT_EQ(headNodes[2], 30);
}

// ==================== GetNodeSizeInStream Tests ====================

TEST_F(SuperKernelGraphTest, GetNodeSizeInStream_Empty)
{
    const auto& nodeSize = graph->GetNodeSizeInStream();
    EXPECT_EQ(nodeSize.size(), 0);
}

TEST_F(SuperKernelGraphTest, GetNodeSizeInStream_WithData)
{
    graph->nodeSizeInStream = {5, 10, 3};
    const auto& nodeSize = graph->GetNodeSizeInStream();
    EXPECT_EQ(nodeSize.size(), 3);
    EXPECT_EQ(nodeSize[0], 5);
    EXPECT_EQ(nodeSize[1], 10);
    EXPECT_EQ(nodeSize[2], 3);
}

// ==================== EventAssociation Tests ====================

TEST_F(SuperKernelGraphTest, AddEventAssociateNotify_Success)
{
    auto node = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_EVENT_RECORD, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(10);
    node->SetNodeType(SkNodeType::NODE_NOTIFY);
    SuperKernelBaseNode* nodePtr = node.get();
    graph->graphMap[10] = std::move(node);

    bool result = graph->AddEventAssociateNotify(100, nodePtr);
    EXPECT_TRUE(result);
    EXPECT_EQ(graph->eventToNodes[100].notifyNodeId, 10);
}

TEST_F(SuperKernelGraphTest, AddEventAssociateNotify_Duplicate)
{
    auto node1 = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_EVENT_RECORD, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    node1->SetNodeId(10);
    node1->SetNodeType(SkNodeType::NODE_NOTIFY);
    graph->graphMap[10] = std::move(node1);
    graph->eventToNodes[100].notifyNodeId = 10;

    auto node2 = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_EVENT_RECORD, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    node2->SetNodeId(20);
    node2->SetNodeType(SkNodeType::NODE_NOTIFY);
    SuperKernelBaseNode* node2Ptr = node2.get();
    graph->graphMap[20] = std::move(node2);

    bool result = graph->AddEventAssociateNotify(100, node2Ptr);
    EXPECT_FALSE(result);
}

TEST_F(SuperKernelGraphTest, AddEventAssociateWait_Success)
{
    auto node = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_EVENT_WAIT, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(10);
    node->SetNodeType(SkNodeType::NODE_WAIT);
    SuperKernelBaseNode* nodePtr = node.get();
    graph->graphMap[10] = std::move(node);

    bool result = graph->AddEventAssociateWait(100, nodePtr);
    EXPECT_TRUE(result);
    EXPECT_TRUE(graph->eventToNodes[100].waitNodeIdList.count(10) > 0);
}

TEST_F(SuperKernelGraphTest, AddEventAssociateReset_Success)
{
    auto node = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_EVENT_RESET, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(10);
    node->SetNodeType(SkNodeType::NODE_RESET);
    SuperKernelBaseNode* nodePtr = node.get();
    graph->graphMap[10] = std::move(node);

    bool result = graph->AddEventAssociateReset(100, nodePtr);
    EXPECT_TRUE(result);
    EXPECT_TRUE(graph->eventToNodes[100].resetNodeIdList.count(10) > 0);
}

// ==================== MemoryAssociation Tests ====================

TEST_F(SuperKernelGraphTest, AddMemoryAssociateWrite_Success)
{
    auto node = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_VALUE_WRITE, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(10);
    node->SetNodeType(SkNodeType::NODE_MEMORY_WRITE);
    SuperKernelBaseNode* nodePtr = node.get();
    graph->graphMap[10] = std::move(node);

    bool result = graph->AddMemoryAssociateWrite(100, nodePtr);
    EXPECT_TRUE(result);
    EXPECT_TRUE(graph->memoryToNodes[100].writeNodeIdList.count(10) > 0);
}

TEST_F(SuperKernelGraphTest, AddMemoryAssociateWait_Success)
{
    auto node = std::make_unique<SuperKernelMemoryNode>(
        nullptr, ACL_MODEL_RI_TASK_VALUE_WAIT, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(10);
    node->SetNodeType(SkNodeType::NODE_MEMORY_WAIT);
    SuperKernelBaseNode* nodePtr = node.get();
    graph->graphMap[10] = std::move(node);

    bool result = graph->AddMemoryAssociateWait(100, nodePtr);
    EXPECT_TRUE(result);
    EXPECT_TRUE(graph->memoryToNodes[100].waitNodeIdList.count(10) > 0);
}

// ==================== SuperKernelNodeFactory Tests ====================

TEST_F(SuperKernelGraphTest, CreateNode_KernelTask)
{
    auto task = std::make_unique<aclmdlRITask>(nullptr);
    auto node = SuperKernelNodeFactory::CreateNode(std::move(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
    EXPECT_NE(node, nullptr);
}

TEST_F(SuperKernelGraphTest, CreateNode_EventRecordTask)
{
    auto task = std::make_unique<aclmdlRITask>(nullptr);
    auto node = SuperKernelNodeFactory::CreateNode(std::move(task), ACL_MODEL_RI_TASK_EVENT_RECORD, 0, 0, 0, INVALID_TASK_ID);
    EXPECT_NE(node, nullptr);
}

TEST_F(SuperKernelGraphTest, CreateNode_EventWaitTask)
{
    auto task = std::make_unique<aclmdlRITask>(nullptr);
    auto node = SuperKernelNodeFactory::CreateNode(std::move(task), ACL_MODEL_RI_TASK_EVENT_WAIT, 0, 0, 0, INVALID_TASK_ID);
    EXPECT_NE(node, nullptr);
}

TEST_F(SuperKernelGraphTest, CreateNode_ValueWriteTask)
{
    auto task = std::make_unique<aclmdlRITask>(nullptr);
    auto node = SuperKernelNodeFactory::CreateNode(std::move(task), ACL_MODEL_RI_TASK_VALUE_WRITE, 0, 0, 0, INVALID_TASK_ID);
    EXPECT_NE(node, nullptr);
}

TEST_F(SuperKernelGraphTest, CreateNode_ValueWaitTask)
{
    auto task = std::make_unique<aclmdlRITask>(nullptr);
    auto node = SuperKernelNodeFactory::CreateNode(std::move(task), ACL_MODEL_RI_TASK_VALUE_WAIT, 0, 0, 0, INVALID_TASK_ID);
    EXPECT_NE(node, nullptr);
}

TEST_F(SuperKernelGraphTest, CreateNode_DefaultTask)
{
    auto task = std::make_unique<aclmdlRITask>(nullptr);
    auto node = SuperKernelNodeFactory::CreateNode(std::move(task), ACL_MODEL_RI_TASK_DEFAULT, 0, 0, 0, INVALID_TASK_ID);
    EXPECT_NE(node, nullptr);
    EXPECT_EQ(node->GetNodeType(), SkNodeType::NODE_DEFAULT);
}

// ==================== CollectFusionFailStats Tests ====================

TEST_F(SuperKernelGraphTest, CollectFusionFailStats_EmptyGraph)
{
    auto stats = graph->CollectFusionFailStats();
    EXPECT_EQ(stats.fusibleCount, 0);
    EXPECT_EQ(stats.unfusibleCount, 0);
}

TEST_F(SuperKernelGraphTest, CollectFusionFailStats_WithNodes)
{
    auto node1 = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    node1->SetNodeId(10);
    node1->SetIsFusible(true);
    graph->graphMap[10] = std::move(node1);

    auto node2 = std::make_unique<SuperKernelKernelNode>(
        nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    node2->SetNodeId(20);
    node2->SetIsFusible(false);
    node2->SetFusionFailReason(FusionFailReason::UNSUPPORT_EVENT_TYPE);
    graph->graphMap[20] = std::move(node2);

    auto stats = graph->CollectFusionFailStats();
    EXPECT_EQ(stats.fusibleCount, 1);
    EXPECT_EQ(stats.unfusibleCount, 1);
}
