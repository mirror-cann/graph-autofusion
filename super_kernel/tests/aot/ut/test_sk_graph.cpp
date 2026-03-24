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
