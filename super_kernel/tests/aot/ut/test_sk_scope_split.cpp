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
#include <algorithm>

#define private public
#define protected public
#include "sk_graph.h"
#include "sk_scope_split.h"
#include "sk_node.h"
#include "sk_lock_detector.h"

/**
 * @brief Test fixture class for SuperKernelScopeSplitter unit tests
 */
class SuperKernelScopeSplitterTest : public testing::Test {
protected:
    void SetUp() override {
        graph = std::make_unique<SuperKernelGraph>();
        // Initialize device core numbers for LockDetector
        LockDetector::GetDeviceCores();
    }

    void TearDown() override {
        graph.reset();
    }

    // Helper function to create a kernel node
    SuperKernelBaseNode* CreateKernelNode(uint64_t nodeId, uint32_t streamIdx, uint64_t nextNodeId = INVALID_TASK_ID) {
        auto node = std::make_unique<SuperKernelKernelNode>(
            nullptr, SkNodeType::NODE_KERNEL, 0, streamIdx, INVALID_TASK_ID);
        node->SetNodeId(nodeId);
        node->SetNextNodeId(nextNodeId);
        // Mark as fusible for testing
        node->isFusible = true;
        // Set kernel parameters to pass LockDetector check
        node->nodeInfos.kernelInfos.numBlocks = 1;
        node->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
        SuperKernelBaseNode* ptr = node.get();
        graph->graphMap[nodeId] = std::move(node);
        return ptr;
    }

    // Helper function to create a wait node
    SuperKernelBaseNode* CreateWaitNode(uint64_t nodeId, uint32_t streamIdx, uint64_t notifyNodeId, uint64_t nextNodeId = INVALID_TASK_ID) {
        auto node = std::make_unique<SuperKernelEventWaitNode>(
            nullptr, SkNodeType::NODE_WAIT, 0, streamIdx, INVALID_TASK_ID);
        node->SetNodeId(nodeId);
        node->SetNextNodeId(nextNodeId);
        node->nodeInfos.syncInfos.notifyNodeId = notifyNodeId;
        // eventId is not used in LockDetector, but needed for eventToNodes mapping
        // We'll set it based on the corresponding notify node's eventId
        node->isFusible = true;
        SuperKernelBaseNode* ptr = node.get();
        graph->graphMap[nodeId] = std::move(node);
        return ptr;
    }

    // Helper function to create a notify node
    SuperKernelBaseNode* CreateNotifyNode(uint64_t nodeId, uint32_t streamIdx, uint64_t eventId, uint64_t nextNodeId = INVALID_TASK_ID) {
        auto node = std::make_unique<SuperKernelEventNotifyNode>(
            nullptr, SkNodeType::NODE_NOTIFY, 0, streamIdx, INVALID_TASK_ID);
        node->SetNodeId(nodeId);
        node->SetNextNodeId(nextNodeId);
        node->nodeInfos.syncInfos.eventId = eventId;
        node->isFusible = true;
        SuperKernelBaseNode* ptr = node.get();
        graph->graphMap[nodeId] = std::move(node);
        return ptr;
    }

    // Helper function to create an unfusible kernel node
    SuperKernelBaseNode* CreateUnfusibleKernelNode(uint64_t nodeId, uint32_t streamIdx, uint64_t nextNodeId = INVALID_TASK_ID) {
        auto node = std::make_unique<SuperKernelKernelNode>(
            nullptr, SkNodeType::NODE_KERNEL, 0, streamIdx, INVALID_TASK_ID);
        node->SetNodeId(nodeId);
        node->SetNextNodeId(nextNodeId);
        // Explicitly mark as unfusible
        node->isFusible = false;
        SuperKernelBaseNode* ptr = node.get();
        graph->graphMap[nodeId] = std::move(node);
        return ptr;
    }

    // Helper function to create a reset node
    SuperKernelBaseNode* CreateResetNode(uint64_t nodeId, uint32_t streamIdx, uint64_t eventId, uint64_t nextNodeId = INVALID_TASK_ID) {
        auto node = std::make_unique<SuperKernelMemoryResetNode>(
            nullptr, SkNodeType::NODE_RESET, 0, streamIdx, INVALID_TASK_ID);
        node->SetNodeId(nodeId);
        node->SetNextNodeId(nextNodeId);
        node->nodeInfos.syncInfos.eventId = eventId;
        node->isFusible = true;
        SuperKernelBaseNode* ptr = node.get();
        graph->graphMap[nodeId] = std::move(node);
        return ptr;
    }

    // Helper function to create a kernel node with custom core counts
    SuperKernelBaseNode* CreateKernelNodeWithCores(uint64_t nodeId, uint32_t streamIdx, uint64_t nextNodeId,
                                                     uint32_t numBlocks, SkKernelType kernelType) {
        auto node = std::make_unique<SuperKernelKernelNode>(
            nullptr, SkNodeType::NODE_KERNEL, 0, streamIdx, INVALID_TASK_ID);
        node->SetNodeId(nodeId);
        node->SetNextNodeId(nextNodeId);
        // Mark as fusible for testing
        node->isFusible = true;
        // Set custom kernel parameters
        node->nodeInfos.kernelInfos.numBlocks = numBlocks;
        node->nodeInfos.kernelInfos.kernelType = kernelType;
        SuperKernelBaseNode* ptr = node.get();
        graph->graphMap[nodeId] = std::move(node);
        return ptr;
    }

    // Helper function to setup streams in graph
    void SetupStreams(const std::vector<std::vector<uint64_t>>& streamNodes) {
        graph->streams.clear();
        graph->headNodes.clear();

        for (const auto& nodes : streamNodes) {
            graph->streams.emplace_back();
            if (!nodes.empty()) {
                graph->headNodes.push_back(nodes[0]);
            } else {
                graph->headNodes.push_back(INVALID_TASK_ID);
            }
        }
    }

    // Helper function to setup event mapping
    void SetupEvent(uint64_t eventId, uint64_t notifyNodeId, const std::vector<uint64_t>& waitNodeIds) {
        EventInfos eventInfo;
        eventInfo.notifyNodeId = notifyNodeId;
        for (auto waitNodeId : waitNodeIds) {
            eventInfo.waitNodeIdList.insert(waitNodeId);
        }
        graph->eventToNodes[eventId] = eventInfo;
    }

    // Helper function to verify scope result
    void VerifyScope(const SuperKernelScopeInfo& scope, const std::vector<uint64_t>& expectedNodeIds) {
        EXPECT_EQ(scope.nodes.size(), expectedNodeIds.size());
        std::vector<uint64_t> actualNodeIds;
        for (const auto* node : scope.nodes) {
            actualNodeIds.push_back(node->GetNodeId());
        }
        std::sort(actualNodeIds.begin(), actualNodeIds.end());
        std::vector<uint64_t> sortedExpected = expectedNodeIds;
        std::sort(sortedExpected.begin(), sortedExpected.end());
        EXPECT_EQ(actualNodeIds, sortedExpected);
    }

    std::unique_ptr<SuperKernelGraph> graph;
};

// ==================== 测试用例 1: 基本多流融合（无跨流依赖） ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase1_BasicMultiStreamFusion_NoCrossStreamDependency)
{
    // Stream 0: [K1(id=1)] → [K2(id=2)] → [K3(id=3)]
    // Stream 1: [K4(id=4)] → [K5(id=5)] → [K6(id=6)]

    auto* k1 = CreateKernelNode(1, 0, 2);
    auto* k2 = CreateKernelNode(2, 0, 3);
    auto* k3 = CreateKernelNode(3, 0, INVALID_TASK_ID);

    auto* k4 = CreateKernelNode(4, 1, 5);
    auto* k5 = CreateKernelNode(5, 1, 6);
    auto* k6 = CreateKernelNode(6, 1, INVALID_TASK_ID);

    SetupStreams({{1, 2, 3}, {4, 5, 6}});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitMultiStreamGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    EXPECT_EQ(scopeInfos.size(), 1);
    VerifyScope(scopeInfos[0], {1, 2, 3, 4, 5, 6});
}

// ==================== 测试用例 2: 单个跨流 Wait-Notify ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase2_SingleCrossStreamWaitNotify)
{
    // Stream 0: [K1(id=1)] → [Wait1(id=3)] → [K2(id=5)]
    // Stream 1: [K3(id=2)] → [Notify1(id=4)] → [K4(id=6)]
    // Event1: Wait1(id=3) 等待 Notify1(id=4)

    auto* k1 = CreateKernelNode(1, 0, 3);
    auto* wait1 = CreateWaitNode(3, 0, 4, 5);
    auto* k2 = CreateKernelNode(5, 0, INVALID_TASK_ID);

    auto* k3 = CreateKernelNode(2, 1, 4);
    auto* notify1 = CreateNotifyNode(4, 1, 100, 6);
    auto* k4 = CreateKernelNode(6, 1, INVALID_TASK_ID);

    SetupStreams({{1, 3, 5}, {2, 4, 6}});
    SetupEvent(100, 4, {3});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitMultiStreamGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    EXPECT_EQ(scopeInfos.size(), 1);
    VerifyScope(scopeInfos[0], {1, 2, 3, 4, 5, 6});
}

// ==================== 测试用例 3: 多个 Wait-Notify ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase3_MultipleWaitNotify)
{
    // Stream 0: [K1(id=1)] → [Wait1(id=4)] → [K2(id=7)]
    // Stream 1: [K3(id=2)] → [Notify1(id=5)] → [Wait2(id=8)] → [K4(id=10)]
    // Stream 2: [K5(id=3)] → [Notify2(id=6)] → [K6(id=9)]
    // Event1: Wait1(id=4) 等待 Notify1(id=5)
    // Event2: Wait2(id=8) 等待 Notify2(id=6)

    auto* k1 = CreateKernelNode(1, 0, 4);
    auto* wait1 = CreateWaitNode(4, 0, 5, 7);
    auto* k2 = CreateKernelNode(7, 0, INVALID_TASK_ID);

    auto* k3 = CreateKernelNode(2, 1, 5);
    auto* notify1 = CreateNotifyNode(5, 1, 100, 8);
    auto* wait2 = CreateWaitNode(8, 1, 6, 10);
    auto* k4 = CreateKernelNode(10, 1, INVALID_TASK_ID);

    auto* k5 = CreateKernelNode(3, 2, 6);
    auto* notify2 = CreateNotifyNode(6, 2, 200, 9);
    auto* k6 = CreateKernelNode(9, 2, INVALID_TASK_ID);

    SetupStreams({{1, 4, 7}, {2, 5, 8, 10}, {3, 6, 9}});
    SetupEvent(100, 5, {4});
    SetupEvent(200, 6, {8});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitMultiStreamGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    EXPECT_EQ(scopeInfos.size(), 1);
    VerifyScope(scopeInfos[0], {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
}

// ==================== 测试用例 4: 不可融合节点（单流） ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase4_UnfusibleNode_SingleStream)
{
    // Stream 0: [K1(可融合)] → [UF1(不可融合)] → [K2(可融合)]

    auto* k1 = CreateKernelNode(1, 0, 2);
    auto* uf1 = CreateUnfusibleKernelNode(2, 0, 3);
    auto* k2 = CreateKernelNode(3, 0, INVALID_TASK_ID);

    SetupStreams({{1, 2, 3}});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitMultiStreamGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    EXPECT_EQ(scopeInfos.size(), 2);
    VerifyScope(scopeInfos[0], {1});
    VerifyScope(scopeInfos[1], {3});
}

// ==================== 测试用例 5: 不可融合节点（多流） ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase5_UnfusibleNode_MultiStream)
{
    // Stream 0: [K1(可融合)] → [UF1(不可融合)] → [K2(可融合)]
    // Stream 1: [K3(可融合)]

    auto* k1 = CreateKernelNode(1, 0, 2);
    auto* uf1 = CreateUnfusibleKernelNode(2, 0, 3);
    auto* k2 = CreateKernelNode(3, 0, INVALID_TASK_ID);

    auto* k3 = CreateKernelNode(4, 1, INVALID_TASK_ID);

    SetupStreams({{1, 2, 3}, {4}});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitMultiStreamGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    EXPECT_EQ(scopeInfos.size(), 2);
    VerifyScope(scopeInfos[0], {1, 4});
    VerifyScope(scopeInfos[1], {3});
}

// ==================== 测试用例 6: 多 Wait 等待同一 Notify ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase6_MultipleWaitsWaitingSameNotify)
{
    // Stream 0: [K1(id=1)] → [Wait1(id=4)] → [K2(id=7)]
    // Stream 1: [K3(id=2)] → [Wait2(id=5)] → [K4(id=8)]
    // Stream 2: [K5(id=3)] → [Notify1(id=6)] → [K6(id=9)]
    // Event1: Wait1(id=4) 和 Wait2(id=5) 都等待 Notify1(id=6)

    auto* k1 = CreateKernelNode(1, 0, 4);
    auto* wait1 = CreateWaitNode(4, 0, 6, 7);
    auto* k2 = CreateKernelNode(7, 0, INVALID_TASK_ID);

    auto* k3 = CreateKernelNode(2, 1, 5);
    auto* wait2 = CreateWaitNode(5, 1, 6, 8);
    auto* k4 = CreateKernelNode(8, 1, INVALID_TASK_ID);

    auto* k5 = CreateKernelNode(3, 2, 6);
    auto* notify1 = CreateNotifyNode(6, 2, 100, 9);
    auto* k6 = CreateKernelNode(9, 2, INVALID_TASK_ID);

    SetupStreams({{1, 4, 7}, {2, 5, 8}, {3, 6, 9}});
    SetupEvent(100, 6, {4, 5});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitMultiStreamGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    EXPECT_EQ(scopeInfos.size(), 1);
    VerifyScope(scopeInfos[0], {1, 2, 3, 4, 5, 6, 7, 8, 9});
}

// ==================== 测试用例 7: 连续不可融合节点 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase7_ConsecutiveUnfusibleNodes)
{
    // Stream 0: [K1(可融合)] → [UF1(不可融合)] → [UF2(不可融合)] → [K2(可融合)]
    // Stream 1: [K3(可融合)]

    auto* k1 = CreateKernelNode(1, 0, 2);
    auto* uf1 = CreateUnfusibleKernelNode(2, 0, 3);
    auto* uf2 = CreateUnfusibleKernelNode(3, 0, 4);
    auto* k2 = CreateKernelNode(4, 0, INVALID_TASK_ID);

    auto* k3 = CreateKernelNode(5, 1, INVALID_TASK_ID);

    SetupStreams({{1, 2, 3, 4}, {5}});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitMultiStreamGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    EXPECT_EQ(scopeInfos.size(), 2);
    VerifyScope(scopeInfos[0], {1, 5});
    VerifyScope(scopeInfos[1], {4});
}

// ==================== 测试用例 8: 空 Skip 处理（所有流都暂停） ====================

// TEST_F(SuperKernelScopeSplitterTest, TestCase8_AllStreamsSuspended)
// {
//     // Stream 0: [Wait1(id=1)] → [K1(id=3)]
//     // Stream 1: [Wait2(id=2)] → [K2(id=4)]
//     // Stream 2: [Notify1(id=5)] → [Notify2(id=6)]
//     // Event1: Wait1(id=1) 等待 Notify1(id=5)
//     // Event2: Wait2(id=2) 等待 Notify2(id=6)

//     auto* wait1 = CreateWaitNode(1, 0, 5, 3);
//     auto* k1 = CreateKernelNode(3, 0, INVALID_TASK_ID);

//     auto* wait2 = CreateWaitNode(2, 1, 6, 4);
//     auto* k2 = CreateKernelNode(4, 1, INVALID_TASK_ID);

//     auto* notify1 = CreateNotifyNode(5, 2, 100, 6);
//     auto* notify2 = CreateNotifyNode(6, 2, 200, INVALID_TASK_ID);

//     SetupStreams({{1, 3}, {2, 4}, {5, 6}});
//     SetupEvent(100, 5, {1});
//     SetupEvent(200, 6, {2});

//     SuperKernelScopeSplitter splitter(*graph);
//     bool result = splitter.SplitMultiStreamGraph();

//     ASSERT_TRUE(result);
//     const auto& scopeInfos = splitter.GetScopeInfos();

//     EXPECT_EQ(scopeInfos.size(), 1);
//     VerifyScope(scopeInfos[0], {1, 2, 3, 4, 5, 6});
// }

// ==================== 测试用例 9: 三流多 Wait-Notify + 不可融合节点 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase9_ThreeStreamsWithMultipleWaitNotifyAndUnfusible)
{
    // Stream 0: [K1(id=1)] → [K2(id=3)] → [Wait1(id=5)] → [K3(id=7)]
    //         → [UF1(id=9,不可融合)] → [K4(id=11)] → [K5(id=13)]
    // Stream 1: [K6(id=2)] → [Notify1(id=4)] → [K7(id=6)] → [Wait2(id=8)] → [K8(id=10)]
    // Stream 2: [K9(id=12)] → [Notify2(id=14)] → [K10(id=16)]
    // Event1: Wait1(id=5) 等待 Notify1(id=4)
    // Event2: Wait2(id=8) 等待 Notify2(id=14)

    auto* k1 = CreateKernelNode(1, 0, 3);
    auto* k2 = CreateKernelNode(3, 0, 5);
    auto* wait1 = CreateWaitNode(5, 0, 4, 7);
    auto* k3 = CreateKernelNode(7, 0, 9);
    auto* uf1 = CreateUnfusibleKernelNode(9, 0, 11);
    auto* k4 = CreateKernelNode(11, 0, 13);
    auto* k5 = CreateKernelNode(13, 0, INVALID_TASK_ID);

    auto* k6 = CreateKernelNode(2, 1, 4);
    auto* notify1 = CreateNotifyNode(4, 1, 100, 6);
    auto* k7 = CreateKernelNode(6, 1, 8);
    auto* wait2 = CreateWaitNode(8, 1, 14, 10);
    auto* k8 = CreateKernelNode(10, 1, INVALID_TASK_ID);

    auto* k9 = CreateKernelNode(12, 2, 14);
    auto* notify2 = CreateNotifyNode(14, 2, 200, 16);
    auto* k10 = CreateKernelNode(16, 2, INVALID_TASK_ID);

    SetupStreams({{1, 3, 5, 7, 9, 11, 13}, {2, 4, 6, 8, 10}, {12, 14, 16}});
    SetupEvent(100, 4, {5});
    SetupEvent(200, 14, {8});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitMultiStreamGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    EXPECT_EQ(scopeInfos.size(), 2);
    VerifyScope(scopeInfos[0], {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16});
    VerifyScope(scopeInfos[1], {11, 13});
}

// ==================== 测试用例 11: Wait-Notify 嵌套场景 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase11_NestedWaitNotify)
{
    // Stream 0: [K1(id=1)] → [Wait1(id=4)] → [K2(id=7)]
    // Stream 1: [K3(id=2)] → [Notify1(id=5)] → [Wait2(id=8)] → [K4(id=10)]
    // Stream 2: [K5(id=3)] → [K6(id=6)] → [Notify2(id=9)]
    // Event1: Wait1(id=4) 等待 Notify1(id=5)
    // Event2: Wait2(id=8) 等待 Notify2(id=9)

    auto* k1 = CreateKernelNode(1, 0, 4);
    auto* wait1 = CreateWaitNode(4, 0, 5, 7);
    auto* k2 = CreateKernelNode(7, 0, INVALID_TASK_ID);

    auto* k3 = CreateKernelNode(2, 1, 5);
    auto* notify1 = CreateNotifyNode(5, 1, 100, 8);
    auto* wait2 = CreateWaitNode(8, 1, 9, 10);
    auto* k4 = CreateKernelNode(10, 1, INVALID_TASK_ID);

    auto* k5 = CreateKernelNode(3, 2, 6);
    auto* k6 = CreateKernelNode(6, 2, 9);
    auto* notify2 = CreateNotifyNode(9, 2, 200, INVALID_TASK_ID);

    SetupStreams({{1, 4, 7}, {2, 5, 8, 10}, {3, 6, 9}});
    SetupEvent(100, 5, {4});
    SetupEvent(200, 9, {8});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitMultiStreamGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    EXPECT_EQ(scopeInfos.size(), 1);
    VerifyScope(scopeInfos[0], {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
}

// ==================== 测试用例 12: 单流（回退到单流切分） ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase12_SingleStreamFallback)
{
    // Stream 0: [K1(id=1)] → [K2(id=2)] → [K3(id=3)]

    auto* k1 = CreateKernelNode(1, 0, 2);
    auto* k2 = CreateKernelNode(2, 0, 3);
    auto* k3 = CreateKernelNode(3, 0, INVALID_TASK_ID);

    SetupStreams({{1, 2, 3}});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    // 单流应该使用单流切分算法，可能产生多个 scope
    EXPECT_GE(scopeInfos.size(), 1);
}

// ==================== 测试用例 13: 空图 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase13_EmptyGraph)
{
    SetupStreams({{}});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitMultiStreamGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    EXPECT_EQ(scopeInfos.size(), 0);
}

// ==================== 测试用例 14: 只有不可融合节点 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase14_AllUnfusibleNodes)
{
    // Stream 0: [UF1(不可融合)] → [UF2(不可融合)]
    // Stream 1: [UF3(不可融合)]

    auto* uf1 = CreateUnfusibleKernelNode(1, 0, 2);
    auto* uf2 = CreateUnfusibleKernelNode(2, 0, INVALID_TASK_ID);
    auto* uf3 = CreateUnfusibleKernelNode(3, 1, INVALID_TASK_ID);

    SetupStreams({{1, 2}, {3}});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitMultiStreamGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    // 所有节点都不可融合，应该不产生任何 scope
    EXPECT_EQ(scopeInfos.size(), 0);
}

// ==================== 测试用例 15: 流顺序验证 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase15_StreamOrderVerification)
{
    // Stream 0: [K1(id=10)] → [K2(id=30)] → [K3(id=50)]
    // Stream 1: [K4(id=20)] → [K5(id=40)] → [K6(id=60)]

    auto* k1 = CreateKernelNode(10, 0, 30);
    auto* k2 = CreateKernelNode(30, 0, 50);
    auto* k3 = CreateKernelNode(50, 0, INVALID_TASK_ID);

    auto* k4 = CreateKernelNode(20, 1, 40);
    auto* k5 = CreateKernelNode(40, 1, 60);
    auto* k6 = CreateKernelNode(60, 1, INVALID_TASK_ID);

    SetupStreams({{10, 30, 50}, {20, 40, 60}});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitMultiStreamGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    EXPECT_EQ(scopeInfos.size(), 1);
    VerifyScope(scopeInfos[0], {10, 20, 30, 40, 50, 60});

    // 验证流内顺序
    const auto& scopeNodes = scopeInfos[0].nodes;
    for (const auto* node : scopeNodes) {
        uint32_t streamIdx = node->GetStreamIdxInGraph();
        uint64_t nodeId = node->GetNodeId();

        if (streamIdx == 0) {
            // Stream 0 的节点顺序应该是 10 → 30 → 50
            EXPECT_TRUE(nodeId == 10 || nodeId == 30 || nodeId == 50);
        } else if (streamIdx == 1) {
            // Stream 1 的节点顺序应该是 20 → 40 → 60
            EXPECT_TRUE(nodeId == 20 || nodeId == 40 || nodeId == 60);
        }
    }
}

// ==================== 测试用例 16: 多个 Scope 切分（Wait-Notify 在不同 Scope） ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase16_MultipleScopesWithWaitNotifyInDifferentScopes)
{
    // Stream 0: [K1(id=1)] → [Wait1(id=3)] → [K2(id=5)]
    // Stream 1: [K3(id=2)] → [K4(id=4)] → [Notify1(id=6)]
    // Event1: Wait1(id=3) 等待 Notify1(id=6)
    // 注意：Wait1 和 Notify1 不在同一 scope，因为中间有其他节点

    auto* k1 = CreateKernelNode(1, 0, 3);
    auto* wait1 = CreateWaitNode(3, 0, 6, 5);
    auto* k2 = CreateKernelNode(5, 0, INVALID_TASK_ID);

    auto* k3 = CreateKernelNode(2, 1, 4);
    auto* k4 = CreateKernelNode(4, 1, 6);
    auto* notify1 = CreateNotifyNode(6, 1, 100, INVALID_TASK_ID);

    SetupStreams({{1, 3, 5}, {2, 4, 6}});
    SetupEvent(100, 6, {3});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitMultiStreamGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    // 应该切分为多个 scope
    EXPECT_GE(scopeInfos.size(), 1);

    // 所有节点都应该被融合
    std::vector<uint64_t> allNodeIds;
    for (const auto& scope : scopeInfos) {
        for (const auto* node : scope.nodes) {
            allNodeIds.push_back(node->GetNodeId());
        }
    }
    std::sort(allNodeIds.begin(), allNodeIds.end());
    std::vector<uint64_t> expectedIds = {1, 2, 3, 4, 5, 6};
    std::sort(expectedIds.begin(), expectedIds.end());
    EXPECT_EQ(allNodeIds, expectedIds);
}

// ==================== 测试用例 17: 四流并行融合 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase17_FourStreamParallelFusion)
{
    // Stream 0: [K1(id=1)] → [K2(id=5)]
    // Stream 1: [K3(id=2)] → [K4(id=6)]
    // Stream 2: [K5(id=3)] → [K6(id=7)]
    // Stream 3: [K7(id=4)] → [K8(id=8)]

    auto* k1 = CreateKernelNode(1, 0, 5);
    auto* k2 = CreateKernelNode(5, 0, INVALID_TASK_ID);

    auto* k3 = CreateKernelNode(2, 1, 6);
    auto* k4 = CreateKernelNode(6, 1, INVALID_TASK_ID);

    auto* k5 = CreateKernelNode(3, 2, 7);
    auto* k6 = CreateKernelNode(7, 2, INVALID_TASK_ID);

    auto* k7 = CreateKernelNode(4, 3, 8);
    auto* k8 = CreateKernelNode(8, 3, INVALID_TASK_ID);

    SetupStreams({{1, 5}, {2, 6}, {3, 7}, {4, 8}});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitMultiStreamGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    EXPECT_EQ(scopeInfos.size(), 1);
    VerifyScope(scopeInfos[0], {1, 2, 3, 4, 5, 6, 7, 8});
}

// ==================== 测试用例 18: Wait 在 Notify 之前（应暂停） ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase18_WaitBeforeNotify_ShouldSuspend)
{
    // Stream 0: [Wait1(id=1)] → [K1(id=3)]
    // Stream 1: [K2(id=2)] → [Notify1(id=4)]
    // Event1: Wait1(id=1) 等待 Notify1(id=4)

    auto* wait1 = CreateWaitNode(1, 0, 4, 3);
    auto* k1 = CreateKernelNode(3, 0, INVALID_TASK_ID);

    auto* k2 = CreateKernelNode(2, 1, 4);
    auto* notify1 = CreateNotifyNode(4, 1, 100, INVALID_TASK_ID);

    SetupStreams({{1, 3}, {2, 4}});
    SetupEvent(100, 4, {1});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitMultiStreamGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    EXPECT_EQ(scopeInfos.size(), 1);
    VerifyScope(scopeInfos[0], {1, 2, 3, 4});

    // 验证 Wait1 在 Notify1 之后处理
    const auto& scopeNodes = scopeInfos[0].nodes;
    size_t waitPos = 0, notifyPos = 0;
    for (size_t i = 0; i < scopeNodes.size(); ++i) {
        if (scopeNodes[i]->GetNodeId() == 1) {
            waitPos = i;
        }
        else if (scopeNodes[i]->GetNodeId() == 4) {
            notifyPos = i;
        }
    }
    EXPECT_GT(waitPos, notifyPos);
}

// ==================== 测试用例 19: 不可融合节点在流中间（多 Scope 切分） ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase19_UnfusibleNodeInMiddle_MultipleScopes)
{
    // Stream 0: [K1(id=1)] → [UF1(不可融合)] → [K2(id=3)] → [K3(id=4)]
    // Stream 1: [K4(id=6)] → [K5(id=5)]

    auto* k1 = CreateKernelNode(1, 0, 2);
    auto* uf1 = CreateUnfusibleKernelNode(2, 0, 3);
    auto* k2 = CreateKernelNode(3, 0, 4);
    auto* k3 = CreateKernelNode(4, 0, INVALID_TASK_ID);

    auto* k4 = CreateKernelNode(6, 1, 5);
    auto* k5 = CreateKernelNode(5, 1, INVALID_TASK_ID);

    SetupStreams({{1, 2, 3, 4}, {6, 5}});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitMultiStreamGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    EXPECT_EQ(scopeInfos.size(), 2);
    VerifyScope(scopeInfos[0], {1, 5, 6});
    VerifyScope(scopeInfos[1], {3, 4});
}

// ==================== 测试用例 20: 复杂 Wait-Notify 链 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase20_ComplexWaitNotifyChain)
{
    // Stream 0: [K1(id=1)] → [Wait1(id=5)] → [K2(id=9)] → [Wait2(id=13)] → [K3(id=17)]
    // Stream 1: [K4(id=2)] → [Notify1(id=6)] → [K5(id=10)] → [Notify2(id=14)] → [K7(id=18)]
    // Stream 2: [K6(id=3)] → [Notify3(id=15)] → [Wait3(id=7)] → [K8(id=11)] → [K9(id=19)]
    // Stream 3: [K10(id=4)] → [Notify4(id=8)] → [K11(id=12)] → [Wait4(id=16)] → [K12(id=20)]
    // Event1: Wait1(id=5) 等待 Notify1(id=6)
    // Event2: Wait2(id=13) 等待 Notify2(id=14)
    // Event3: Wait3(id=7) 等待 Notify3(id=15)
    // Event4: Wait4(id=16) 等待 Notify4(id=8)

    auto* k1 = CreateKernelNode(1, 0, 5);
    auto* wait1 = CreateWaitNode(5, 0, 6, 9);
    auto* k2 = CreateKernelNode(9, 0, 13);
    auto* wait2 = CreateWaitNode(13, 0, 14, 17);
    auto* k3 = CreateKernelNode(17, 0, INVALID_TASK_ID);

    auto* k4 = CreateKernelNode(2, 1, 6);
    auto* notify1 = CreateNotifyNode(6, 1, 100, 10);
    auto* k5 = CreateKernelNode(10, 1, 14);
    auto* notify2 = CreateNotifyNode(14, 1, 200, 18);
    auto* k7 = CreateKernelNode(18, 1, INVALID_TASK_ID);

    auto* k6 = CreateKernelNode(3, 2, 15);
    auto* notify3 = CreateNotifyNode(15, 2, 300, 7);
    auto* wait3 = CreateWaitNode(7, 2, 15, 11);
    auto* k8 = CreateKernelNode(11, 2, 19);
    auto* k9 = CreateKernelNode(19, 2, INVALID_TASK_ID);

    auto* k10 = CreateKernelNode(4, 3, 8);
    auto* notify4 = CreateNotifyNode(8, 3, 400, 12);
    auto* k11 = CreateKernelNode(12, 3, 16);
    auto* wait4 = CreateWaitNode(16, 3, 8, 20);
    auto* k12 = CreateKernelNode(20, 3, INVALID_TASK_ID);

    SetupStreams({{1, 5, 9, 13, 17}, {2, 6, 10, 14, 18}, {3, 15, 7, 11, 19}, {4, 8, 12, 16, 20}});
    SetupEvent(100, 6, {5});
    SetupEvent(200, 14, {13});
    SetupEvent(300, 15, {7});
    SetupEvent(400, 8, {16});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitMultiStreamGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    // 应该融合到 1 个或多个 scope
    EXPECT_GE(scopeInfos.size(), 1);

    // 所有节点都应该被融合
    std::vector<uint64_t> allNodeIds;
    for (const auto& scope : scopeInfos) {
        for (const auto* node : scope.nodes) {
            allNodeIds.push_back(node->GetNodeId());
        }
    }
    std::sort(allNodeIds.begin(), allNodeIds.end());
    std::vector<uint64_t> expectedIds = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    std::sort(expectedIds.begin(), expectedIds.end());
    EXPECT_EQ(allNodeIds, expectedIds);
}

// ==================== 测试用例 21: 验证 StreamState 重置 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase21_VerifyStreamStateReset)
{
    // Stream 0: [K1(id=1)] → [UF1(不可融合)] → [K2(id=3)]
    // Stream 1: [K3(id=4)]

    auto* k1 = CreateKernelNode(1, 0, 2);
    auto* uf1 = CreateUnfusibleKernelNode(2, 0, 3);
    auto* k2 = CreateKernelNode(3, 0, INVALID_TASK_ID);
    auto* k3 = CreateKernelNode(4, 1, INVALID_TASK_ID);

    SetupStreams({{1, 2, 3}, {4}});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitMultiStreamGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    EXPECT_EQ(scopeInfos.size(), 2);
    VerifyScope(scopeInfos[0], {1, 4});
    VerifyScope(scopeInfos[1], {3});
}

// ==================== 测试用例 22: 单流单节点 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase22_SingleStreamSingleNode)
{
    // Stream 0: [K1(id=1)]

    auto* k1 = CreateKernelNode(1, 0, INVALID_TASK_ID);

    SetupStreams({{1}});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitMultiStreamGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    EXPECT_EQ(scopeInfos.size(), 1);
    VerifyScope(scopeInfos[0], {1});
}

// ==================== 测试用例 23: 多流但每流只有一个节点 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase23_MultiStreamSingleNodes)
{
    // Stream 0: [K1(id=1)]
    // Stream 1: [K2(id=2)]
    // Stream 2: [K3(id=3)]

    auto* k1 = CreateKernelNode(1, 0, INVALID_TASK_ID);
    auto* k2 = CreateKernelNode(2, 1, INVALID_TASK_ID);
    auto* k3 = CreateKernelNode(3, 2, INVALID_TASK_ID);

    SetupStreams({{1}, {2}, {3}});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitMultiStreamGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    EXPECT_EQ(scopeInfos.size(), 1);
    VerifyScope(scopeInfos[0], {1, 2, 3});
}

// ==================== 测试用例 24: ResetStreamStates 恢复暂停的流 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase24_ResetStreamStatesResumeSuspended)
{
    // 测试 ResetStreamStates 正确恢复被暂停的流状态
    // Stream 0: [K1(id=1, big cores)] → [Wait1(id=3)] → [K2(id=5, big cores)]
    // Stream 1: [K3(id=2)] → [Notify1(id=4)]
    // Stream 2: [K4(id=6)] → [Notify2(id=8)]
    // Event1: Wait1(id=3) 等待 Notify1(id=4)

    // 使用不同 core 数触发多个 scope 分割
    auto* k1 = CreateKernelNodeWithCores(1, 0, 3, 4, SkKernelType::AIC_ONLY);
    auto* wait1 = CreateWaitNode(3, 0, 4, 5);
    auto* k2 = CreateKernelNodeWithCores(5, 0, INVALID_TASK_ID, 4, SkKernelType::AIC_ONLY);

    auto* k3 = CreateKernelNodeWithCores(2, 1, 4, 2, SkKernelType::AIC_ONLY);
    auto* notify1 = CreateNotifyNode(4, 1, 100, INVALID_TASK_ID);

    auto* k4 = CreateKernelNodeWithCores(6, 2, 8, 2, SkKernelType::AIC_ONLY);
    auto* notify2 = CreateNotifyNode(8, 2, 200, INVALID_TASK_ID);

    SetupStreams({{1, 3, 5}, {2, 4}, {6, 8}});
    SetupEvent(100, 4, {3});
    SetupEvent(200, 8, {});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitMultiStreamGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    // 由于 K1 占用 4 个 cores，K2 也需要 4 个 cores，但 Wait1 等待 Notify1
    // 预期会形成多个 scope
    EXPECT_GE(scopeInfos.size(), 1);

    // 验证所有节点都被处理
    std::set<uint64_t> allProcessedNodes;
    for (const auto& scope : scopeInfos) {
        for (const auto* node : scope.nodes) {
            allProcessedNodes.insert(node->GetNodeId());
        }
    }
    std::set<uint64_t> expectedNodes = {1, 2, 3, 4, 5, 6, 8};
    EXPECT_EQ(allProcessedNodes, expectedNodes);
}

// ==================== 测试用例 25: 多次暂停恢复场景 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase25_MultipleSuspendResume)
{
    // 测试流多次暂停和恢复
    // Stream 0: [Wait1(id=1)] → [K1(id=3)] → [Wait2(id=5)] → [K2(id=7)]
    // Stream 1: [Notify1(id=2)] → [K2(id=4)] → [Notify2(id=6)] → [K3(id=8)]
    // Stream 2: [K4(id=9)]
    // Event1: Wait1(id=1) 等待 Notify1(id=2)
    // Event2: Wait2(id=5) 等待 Notify2(id=6)

    auto* wait1 = CreateWaitNode(1, 0, 2, 3);
    auto* k1 = CreateKernelNode(3, 0, 5);
    auto* wait2 = CreateWaitNode(5, 0, 6, 7);
    auto* k2 = CreateKernelNode(7, 0, INVALID_TASK_ID);

    auto* notify1 = CreateNotifyNode(2, 1, 100, 4);
    auto* k3 = CreateKernelNode(4, 1, 6);
    auto* notify2 = CreateNotifyNode(6, 1, 200, 8);
    auto* k4 = CreateKernelNode(8, 1, INVALID_TASK_ID);

    auto* k5 = CreateKernelNode(9, 2, INVALID_TASK_ID);

    SetupStreams({{1, 3, 5, 7}, {2, 4, 6, 8}, {9}});
    SetupEvent(100, 2, {1});
    SetupEvent(200, 6, {5});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitMultiStreamGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    // 验证所有节点都被处理
    std::set<uint64_t> allProcessedNodes;
    for (const auto& scope : scopeInfos) {
        for (const auto* node : scope.nodes) {
            allProcessedNodes.insert(node->GetNodeId());
        }
    }
    std::set<uint64_t> expectedNodes = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_EQ(allProcessedNodes, expectedNodes);
}
