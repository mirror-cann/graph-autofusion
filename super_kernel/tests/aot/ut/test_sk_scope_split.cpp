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
#include "mockcpp/mockcpp.hpp"
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
        // Clear any lingering mock state from previous tests
        GlobalMockObject::verify();
        graph = std::make_unique<SuperKernelGraph>();
        LockDetector::GetDeviceCores();
    }

    void TearDown() override {
        graph.reset();
        // Clear mock state
        GlobalMockObject::verify();
    }

    // Helper function to create a kernel node
    SuperKernelBaseNode* CreateKernelNode(uint64_t nodeId, uint32_t streamIdx, uint64_t nextNodeId = INVALID_TASK_ID) {
        auto node = std::make_unique<SuperKernelKernelNode>(
            nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, streamIdx, INVALID_TASK_ID);
        node->nodeType = SkNodeType::NODE_KERNEL;
        node->SetNodeId(nodeId);
        node->SetNextNodeId(nextNodeId);
        // Mark as fusible for testing
        node->isFusible = true;
        // Set kernel parameters to pass LockDetector check
        node->nodeInfos.kernelInfos.numBlocks = 1;
        node->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
        node->nodeInfos.kernelInfos.cubeNum = 1;
        node->nodeInfos.kernelInfos.vecNum = 0;
        SuperKernelBaseNode* ptr = node.get();
        graph->graphMap[nodeId] = std::move(node);
        return ptr;
    }

    // Helper function to create a wait node
    SuperKernelBaseNode* CreateWaitNode(uint64_t nodeId, uint32_t streamIdx, uint64_t notifyNodeId, uint64_t nextNodeId = INVALID_TASK_ID) {
        auto node = std::make_unique<SuperKernelMemoryNode>(
            nullptr, ACL_MODEL_RI_TASK_VALUE_WAIT, 0, streamIdx, INVALID_TASK_ID);
        node->SetNodeId(nodeId);
        node->SetNextNodeId(nextNodeId);
        node->nodeInfos.syncInfos.correspondingNotifyNodeId = notifyNodeId;
        // eventId is not used in LockDetector, but needed for eventToNodes mapping
        // We'll set it based on the corresponding notify node's eventId
        node->isFusible = true;
        // Manually set nodeType since InitNode() is not called
        node->nodeType = SkNodeType::NODE_WAIT;
        SuperKernelBaseNode* ptr = node.get();
        graph->graphMap[nodeId] = std::move(node);
        return ptr;
    }

    // Helper function to create a notify node
    SuperKernelBaseNode* CreateNotifyNode(uint64_t nodeId, uint32_t streamIdx, uint64_t eventId, uint64_t nextNodeId = INVALID_TASK_ID) {
        auto node = std::make_unique<SuperKernelMemoryNode>(
            nullptr, ACL_MODEL_RI_TASK_VALUE_WRITE, 0, streamIdx, INVALID_TASK_ID);
        node->SetNodeId(nodeId);
        node->SetNextNodeId(nextNodeId);
        node->nodeInfos.syncInfos.eventId = eventId;
        node->isFusible = true;
        // Manually set nodeType since InitNode() is not called
        node->nodeType = SkNodeType::NODE_NOTIFY;
        SuperKernelBaseNode* ptr = node.get();
        graph->graphMap[nodeId] = std::move(node);
        return ptr;
    }

    // Helper function to create an unfusible kernel node
    SuperKernelBaseNode* CreateUnfusibleKernelNode(uint64_t nodeId, uint32_t streamIdx, uint64_t nextNodeId = INVALID_TASK_ID) {
        auto node = std::make_unique<SuperKernelKernelNode>(
            nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, streamIdx, INVALID_TASK_ID);
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
        auto node = std::make_unique<SuperKernelMemoryNode>(
            nullptr, ACL_MODEL_RI_TASK_VALUE_WRITE, 0, streamIdx, INVALID_TASK_ID);
        node->SetNodeId(nodeId);
        node->SetNextNodeId(nextNodeId);
        node->nodeInfos.syncInfos.eventId = eventId;
        node->isFusible = true;
        // Manually set nodeType since InitNode() is not called
        node->nodeType = SkNodeType::NODE_RESET;
        SuperKernelBaseNode* ptr = node.get();
        graph->graphMap[nodeId] = std::move(node);
        return ptr;
    }

    // Helper function to create a kernel node with custom core counts
    SuperKernelBaseNode* CreateKernelNodeWithCores(uint64_t nodeId, uint32_t streamIdx, uint64_t nextNodeId,
                                                     uint32_t numBlocks, SkKernelType kernelType) {
        auto node = std::make_unique<SuperKernelKernelNode>(
            nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, streamIdx, INVALID_TASK_ID);
        node->SetNodeId(nodeId);
        node->SetNextNodeId(nextNodeId);
        // Mark as fusible for testing
        node->isFusible = true;
        // Set custom kernel parameters
        node->nodeInfos.kernelInfos.numBlocks = numBlocks;
        node->nodeInfos.kernelInfos.kernelType = kernelType;
        // Calculate vecNum and cubeNum based on kernelType and numBlocks
        if (kernelType == SkKernelType::AIC_ONLY || kernelType == SkKernelType::MIX_AIC_1_0) {
            node->nodeInfos.kernelInfos.cubeNum = numBlocks;
            node->nodeInfos.kernelInfos.vecNum = 0;
        } else if (kernelType == SkKernelType::AIV_ONLY || kernelType == SkKernelType::MIX_AIV_1_0) {
            node->nodeInfos.kernelInfos.cubeNum = 0;
            node->nodeInfos.kernelInfos.vecNum = numBlocks;
        } else if (kernelType == SkKernelType::MIX_AIC_1_1) {
            node->nodeInfos.kernelInfos.cubeNum = numBlocks;
            node->nodeInfos.kernelInfos.vecNum = numBlocks;
        } else if (kernelType == SkKernelType::MIX_AIC_1_2) {
            node->nodeInfos.kernelInfos.cubeNum = numBlocks;
            node->nodeInfos.kernelInfos.vecNum = numBlocks << 1;
        }
        SuperKernelBaseNode* ptr = node.get();
        graph->graphMap[nodeId] = std::move(node);
        return ptr;
    }

    // Helper function to create a scope begin node (fusible)
    SuperKernelBaseNode* CreateScopeBeginNode(uint64_t nodeId, uint32_t streamIdx, const std::string& scopeName, uint64_t nextNodeId = INVALID_TASK_ID) {
        auto node = std::make_unique<SuperKernelKernelNode>(
            nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, streamIdx, INVALID_TASK_ID);
        node->SetNodeId(nodeId);
        node->SetNextNodeId(nextNodeId);
        node->isFusible = true;
        node->SetIsScopeNode(true);
        node->nodeInfos.kernelInfos.numBlocks = 1;
        node->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
        node->nodeInfos.kernelInfos.cubeNum = 1;
        node->nodeInfos.kernelInfos.vecNum = 0;
        // Set scope name through reflection (need to access private member)
        SuperKernelKernelNode* kernelNode = static_cast<SuperKernelKernelNode*>(node.get());
        kernelNode->scopeName = scopeName;
        kernelNode->isScopeBegin = true;
        SuperKernelBaseNode* ptr = node.get();
        graph->graphMap[nodeId] = std::move(node);
        return ptr;
    }

    // Helper function to create a scope end node (fusible)
    SuperKernelBaseNode* CreateScopeEndNode(uint64_t nodeId, uint32_t streamIdx, const std::string& scopeName, uint64_t nextNodeId = INVALID_TASK_ID) {
        auto node = std::make_unique<SuperKernelKernelNode>(
            nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, streamIdx, INVALID_TASK_ID);
        node->SetNodeId(nodeId);
        node->SetNextNodeId(nextNodeId);
        node->isFusible = true;
        node->SetIsScopeNode(true);
        node->nodeInfos.kernelInfos.numBlocks = 1;
        node->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
        node->nodeInfos.kernelInfos.cubeNum = 1;
        node->nodeInfos.kernelInfos.vecNum = 0;
        // Set scope name through reflection
        SuperKernelKernelNode* kernelNode = static_cast<SuperKernelKernelNode*>(node.get());
        kernelNode->scopeName = scopeName;
        kernelNode->isScopeBegin = false;
        kernelNode->isScopeEnd = true;
        SuperKernelBaseNode* ptr = node.get();
        graph->graphMap[nodeId] = std::move(node);
        return ptr;
    }

    // Helper function to create an unfusible scope begin node
    SuperKernelBaseNode* CreateUnfusibleScopeBeginNode(uint64_t nodeId, uint32_t streamIdx, uint64_t nextNodeId = INVALID_TASK_ID) {
        auto node = std::make_unique<SuperKernelKernelNode>(
            nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, streamIdx, INVALID_TASK_ID);
        node->SetNodeId(nodeId);
        node->SetNextNodeId(nextNodeId);
        node->isFusible = false;
        node->SetIsScopeNode(true);
        node->nodeInfos.kernelInfos.numBlocks = 1;
        node->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
        node->nodeInfos.kernelInfos.cubeNum = 1;
        node->nodeInfos.kernelInfos.vecNum = 0;
        SuperKernelKernelNode* kernelNode = static_cast<SuperKernelKernelNode*>(node.get());
        kernelNode->scopeName = "default_sk_scope_name";
        kernelNode->isScopeBegin = true;
        SuperKernelBaseNode* ptr = node.get();
        graph->graphMap[nodeId] = std::move(node);
        return ptr;
    }

    // Helper function to create an unfusible scope end node
    SuperKernelBaseNode* CreateUnfusibleScopeEndNode(uint64_t nodeId, uint32_t streamIdx, uint64_t nextNodeId = INVALID_TASK_ID) {
        auto node = std::make_unique<SuperKernelKernelNode>(
            nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, streamIdx, INVALID_TASK_ID);
        node->SetNodeId(nodeId);
        node->SetNextNodeId(nextNodeId);
        node->isFusible = false;
        node->SetIsScopeNode(true);
        node->nodeInfos.kernelInfos.numBlocks = 1;
        node->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
        node->nodeInfos.kernelInfos.cubeNum = 1;
        node->nodeInfos.kernelInfos.vecNum = 0;
        SuperKernelKernelNode* kernelNode = static_cast<SuperKernelKernelNode*>(node.get());
        kernelNode->scopeName = "default_sk_scope_name";
        kernelNode->isScopeBegin = false;
        kernelNode->isScopeEnd = true;
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
    bool result = splitter.SplitGraph();

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
    bool result = splitter.SplitGraph();

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
    bool result = splitter.SplitGraph();

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
    bool result = splitter.SplitGraph();

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
    bool result = splitter.SplitGraph();

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
    bool result = splitter.SplitGraph();

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
    bool result = splitter.SplitGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    EXPECT_EQ(scopeInfos.size(), 2);
    VerifyScope(scopeInfos[0], {1, 5});
    VerifyScope(scopeInfos[1], {4});
}

// ==================== 测试用例 8: 空 Skip 处理（所有流都暂停） ====================

// ==================== 测试用例 8: 所有流都挂起 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase8_AllStreamsSuspended)
{
    // Stream 0: [Wait1(id=1)] → [K1(id=3)]
    // Stream 1: [Wait2(id=2)] → [K2(id=4)]
    // Stream 2: [Notify1(id=5)] → [Notify2(id=6)]
    // Event1: Wait1(id=1) 等待 Notify1(id=5)
    // Event2: Wait2(id=2) 等待 Notify2(id=6)

    auto* wait1 = CreateWaitNode(1, 0, 5, 3);
    auto* k1 = CreateKernelNode(3, 0, INVALID_TASK_ID);

    auto* wait2 = CreateWaitNode(2, 1, 6, 4);
    auto* k2 = CreateKernelNode(4, 1, INVALID_TASK_ID);

    auto* notify1 = CreateNotifyNode(5, 2, 100, 6);
    auto* notify2 = CreateNotifyNode(6, 2, 200, INVALID_TASK_ID);

    SetupStreams({{1, 3}, {2, 4}, {5, 6}});
    SetupEvent(100, 5, {1});
    SetupEvent(200, 6, {2});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    EXPECT_EQ(scopeInfos.size(), 1);
    VerifyScope(scopeInfos[0], {1, 2, 3, 4, 5, 6});
}

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
    bool result = splitter.SplitGraph();

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
    bool result = splitter.SplitGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    EXPECT_EQ(scopeInfos.size(), 1);
    VerifyScope(scopeInfos[0], {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
}

// ==================== 测试用例 12: 单流（统一 SplitGraph 路径） ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase12_SingleStreamUnifiedPath)
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

    // 单流现在也走统一的 SplitGraph 路径
    EXPECT_EQ(scopeInfos.size(), 1);
    VerifyScope(scopeInfos[0], {1, 2, 3});
}

// ==================== 测试用例 13: 空图 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase13_EmptyGraph)
{
    SetupStreams({{}});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitGraph();

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
    bool result = splitter.SplitGraph();

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
    bool result = splitter.SplitGraph();

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
    bool result = splitter.SplitGraph();

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
    bool result = splitter.SplitGraph();

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
    bool result = splitter.SplitGraph();

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
    bool result = splitter.SplitGraph();

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
    bool result = splitter.SplitGraph();

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
    bool result = splitter.SplitGraph();

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
    bool result = splitter.SplitGraph();

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
    bool result = splitter.SplitGraph();

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
    bool result = splitter.SplitGraph();

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
    bool result = splitter.SplitGraph();

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

// ==================== 测试用例 26: 无scope节点，整张图参与切分 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase26_NoScopeNodes_FullGraphFusion)
{
    // 测试场景：用户未添加任何scope节点，代表整张图参与切分
    // Stream 0: [K1(id=1)] → [K2(id=2)] → [K3(id=3)]
    // Stream 1: [K4(id=4)] → [K5(id=5)] → [K6(id=6)]
    // 预期：所有节点融合到一个scope

    auto* k1 = CreateKernelNode(1, 0, 2);
    auto* k2 = CreateKernelNode(2, 0, 3);
    auto* k3 = CreateKernelNode(3, 0, INVALID_TASK_ID);

    auto* k4 = CreateKernelNode(4, 1, 5);
    auto* k5 = CreateKernelNode(5, 1, 6);
    auto* k6 = CreateKernelNode(6, 1, INVALID_TASK_ID);

    SetupStreams({{1, 2, 3}, {4, 5, 6}});

    // 初始化graph，确保没有scope标记
    graph->scopeNameToIdx.clear();

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    // 预期：所有节点融合到一个scope
    EXPECT_EQ(scopeInfos.size(), 1);
    VerifyScope(scopeInfos[0], {1, 2, 3, 4, 5, 6});
}

// ==================== 测试用例 27: 单个可融合scope ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase27_SingleFusibleScope)
{
    // 测试场景：单个可融合scope，scope内的节点融合
    // Stream 0: [K1(id=1)] → [ScopeBegin_A(id=2)] → [K2(id=3)] → [K3(id=4)] → [ScopeEnd_A(id=5)] → [K4(id=6)]
    // 预期：
    //   - 节点2, 3, 4, 5的scopeBitFlags第0位为1（属于scope_A）
    //   - 节点1, 6的scopeBitFlags全为0（不属于任何scope）

    auto* k1 = CreateKernelNode(1, 0, 2);
    auto* scopeBegin = CreateScopeBeginNode(2, 0, "scope_A", 3);
    auto* k2 = CreateKernelNode(3, 0, 4);
    auto* k3 = CreateKernelNode(4, 0, 5);
    auto* scopeEnd = CreateScopeEndNode(5, 0, "scope_A", 6);
    auto* k4 = CreateKernelNode(6, 0, INVALID_TASK_ID);

    SetupStreams({{1, 2, 3, 4, 5, 6}});

    // 初始化scope名称
    graph->scopeNameToIdx["scope_A"] = 0;

    // 执行scope标记更新
    graph->UpdateNodeScopeBitFlags();

    // 验证scopeBitFlags
    // K1: 不属于任何scope
    EXPECT_EQ(k1->GetScopeBitFlags().count(), 0);

    // ScopeBegin: 属于scope_A
    EXPECT_TRUE(scopeBegin->GetScopeBitFlags().test(0));
    EXPECT_EQ(scopeBegin->GetScopeBitFlags().count(), 1);

    // K2, K3: 属于scope_A
    EXPECT_TRUE(k2->GetScopeBitFlags().test(0));
    EXPECT_EQ(k2->GetScopeBitFlags().count(), 1);
    EXPECT_TRUE(k3->GetScopeBitFlags().test(0));
    EXPECT_EQ(k3->GetScopeBitFlags().count(), 1);

    // ScopeEnd: 属于scope_A
    EXPECT_TRUE(scopeEnd->GetScopeBitFlags().test(0));
    EXPECT_EQ(scopeEnd->GetScopeBitFlags().count(), 1);

    // K4: 不属于任何scope
    EXPECT_EQ(k4->GetScopeBitFlags().count(), 0);
}

// ==================== 测试用例 28: 多个可融合scope ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase28_MultipleFusibleScopes)
{
    // 测试场景：多个独立的可融合scope
    // Stream 0: [ScopeBegin_A(id=1)] → [K1(id=2)] → [ScopeEnd_A(id=3)] → [K2(id=4)] → [ScopeBegin_B(id=5)] → [K3(id=6)] → [ScopeEnd_B(id=7)]
    // 预期：
    //   - 节点1, 2, 3的scopeBitFlags第0位为1（属于scope_A）
    //   - 节点5, 6, 7的scopeBitFlags第1位为1（属于scope_B）
    //   - 节点4的scopeBitFlags全为0

    auto* scopeBeginA = CreateScopeBeginNode(1, 0, "scope_A", 2);
    auto* k1 = CreateKernelNode(2, 0, 3);
    auto* scopeEndA = CreateScopeEndNode(3, 0, "scope_A", 4);
    auto* k2 = CreateKernelNode(4, 0, 5);
    auto* scopeBeginB = CreateScopeBeginNode(5, 0, "scope_B", 6);
    auto* k3 = CreateKernelNode(6, 0, 7);
    auto* scopeEndB = CreateScopeEndNode(7, 0, "scope_B", INVALID_TASK_ID);

    SetupStreams({{1, 2, 3, 4, 5, 6, 7}});

    // 初始化scope名称
    graph->scopeNameToIdx["scope_A"] = 0;
    graph->scopeNameToIdx["scope_B"] = 1;

    // 执行scope标记更新
    graph->UpdateNodeScopeBitFlags();

    // 验证scope_A内的节点（第0位为1）
    EXPECT_TRUE(scopeBeginA->GetScopeBitFlags().test(0));
    EXPECT_TRUE(k1->GetScopeBitFlags().test(0));
    EXPECT_TRUE(scopeEndA->GetScopeBitFlags().test(0));

    // K2: 在scope_A和scope_B之间
    EXPECT_EQ(k2->GetScopeBitFlags().count(), 0);

    // 验证scope_B内的节点（第1位为1）
    EXPECT_TRUE(scopeBeginB->GetScopeBitFlags().test(1));
    EXPECT_TRUE(k3->GetScopeBitFlags().test(1));
    EXPECT_TRUE(scopeEndB->GetScopeBitFlags().test(1));

    // 验证scope_A和scope_B的标记不会互相干扰
    EXPECT_FALSE(scopeBeginA->GetScopeBitFlags().test(1));
    EXPECT_FALSE(k1->GetScopeBitFlags().test(1));
    EXPECT_FALSE(scopeEndA->GetScopeBitFlags().test(1));

    EXPECT_FALSE(scopeBeginB->GetScopeBitFlags().test(0));
    EXPECT_FALSE(k3->GetScopeBitFlags().test(0));
    EXPECT_FALSE(scopeEndB->GetScopeBitFlags().test(0));
}

// ==================== 测试用例 29: 不可融合scope ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase29_UnfusibleScope)
{
    // 测试场景：仅有unFuseEnable为false的标记，整张图切分，但unFuseEnable为false中间的算子不可融
    // Stream 0: [K1(id=1)] → [UnfusibleBegin(id=2)] → [K2(id=3)] → [K3(id=4)] → [UnfusibleEnd(id=5)] → [K4(id=6)]
    // 预期：生成2个scope，第一个scope只有K1(id=1)，第二个scope只有K4(id=6)，其他节点(2,3,4,5)都不应该被包含在任何scope中

    auto* k1 = CreateKernelNode(1, 0, 2);
    auto* unfusibleBegin = CreateUnfusibleScopeBeginNode(2, 0, 3);
    auto* k2 = CreateKernelNode(3, 0, 4);
    auto* k3 = CreateKernelNode(4, 0, 5);
    auto* unfusibleEnd = CreateUnfusibleScopeEndNode(5, 0, 6);
    auto* k4 = CreateKernelNode(6, 0, INVALID_TASK_ID);

    SetupStreams({{1, 2, 3, 4, 5, 6}});

    // 执行scope标记更新，将K2和K3标记为不可融合
    graph->UpdateNodeScopeBitFlags();

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    // 验证生成了2个scope
    EXPECT_EQ(scopeInfos.size(), 2);

    // 验证scope 0只有K1(id=1)
    EXPECT_EQ(scopeInfos[0].nodes.size(), 1);
    EXPECT_EQ(scopeInfos[0].nodes[0]->GetNodeId(), 1);

    // 验证scope 1只有K4(id=6)
    EXPECT_EQ(scopeInfos[1].nodes.size(), 1);
    EXPECT_EQ(scopeInfos[1].nodes[0]->GetNodeId(), 6);

    // 验证不可融合的节点没有被包含在任何scope中
    std::set<uint64_t> allProcessedNodes;
    for (const auto& scope : scopeInfos) {
        for (const auto* node : scope.nodes) {
            allProcessedNodes.insert(node->GetNodeId());
        }
    }
    std::set<uint64_t> expectedNodes = {1, 6}; // 只有K1和K4被处理
    EXPECT_EQ(allProcessedNodes, expectedNodes);

    // 验证K2和K3被标记为不可融合
    EXPECT_FALSE(k2->IsFusible());
    EXPECT_FALSE(k3->IsFusible());
    EXPECT_TRUE(k1->IsFusible());
    EXPECT_TRUE(k4->IsFusible());
}

// ==================== 测试用例 30: 可融合scope + 不可融合scope混合 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase30_MixedFusibleAndUnfusibleScopes)
{
    // 测试场景：可融合和不可融合scope混合
    // Stream 0: [ScopeBegin_A(id=1)] → [K1(id=2)] → [ScopeEnd_A(id=3)] → [UnfusibleBegin(id=4)] → [K2(id=5)] → [UnfusibleEnd(id=6)] → [K3(id=7)]
    // 预期：生成2个scope，第一个scope包含[ScopeBegin_A(1), K1(2), ScopeEnd_A(3)]，第二个scope包含[K3(7)]，不可融合的节点(4,5,6)不应该被包含在任何scope中

    auto* scopeBeginA = CreateScopeBeginNode(1, 0, "scope_A", 2);
    auto* k1 = CreateKernelNode(2, 0, 3);
    auto* scopeEndA = CreateScopeEndNode(3, 0, "scope_A", 4);
    auto* unfusibleBegin = CreateUnfusibleScopeBeginNode(4, 0, 5);
    auto* k2 = CreateKernelNode(5, 0, 6);
    auto* unfusibleEnd = CreateUnfusibleScopeEndNode(6, 0, 7);
    auto* k3 = CreateKernelNode(7, 0, INVALID_TASK_ID);

    SetupStreams({{1, 2, 3, 4, 5, 6, 7}});

    // 初始化scope名称
    graph->scopeNameToIdx["scope_A"] = 0;

    // 执行scope标记更新，将K2标记为不可融合
    graph->UpdateNodeScopeBitFlags();

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    // 验证生成了2个scope
    EXPECT_EQ(scopeInfos.size(), 2);

    // 验证scope 0只包含K1(2)，ScopeBegin_A(1)和ScopeEnd_A(3)是scope节点，不放入nodes中
    EXPECT_EQ(scopeInfos[0].nodes.size(), 1);
    EXPECT_EQ(scopeInfos[0].nodes[0]->GetNodeId(), 2);

    // 验证scope 1只包含K3(id=7)
    EXPECT_EQ(scopeInfos[1].nodes.size(), 1);
    EXPECT_EQ(scopeInfos[1].nodes[0]->GetNodeId(), 7);

    // 验证不可融合的节点没有被包含在任何scope中
    std::set<uint64_t> allProcessedNodes;
    for (const auto& scope : scopeInfos) {
        for (const auto* node : scope.nodes) {
            allProcessedNodes.insert(node->GetNodeId());
        }
    }
    std::set<uint64_t> expectedNodes = {2, 7}; // 只有K1和K3是普通kernel节点
    EXPECT_EQ(allProcessedNodes, expectedNodes);

    // 验证融合状态
    EXPECT_TRUE(k1->IsFusible());
    EXPECT_FALSE(k2->IsFusible());
    EXPECT_TRUE(k3->IsFusible());
}

// ==================== 测试用例 31: 多流中相同scope名称 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase31_SameScopeNameAcrossStreams)
{
    // 测试场景：多个流中使用相同的scope名称
    // Stream 0: [ScopeBegin_A(id=1)] → [K1(id=2)] → [ScopeEnd_A(id=3)]
    // Stream 1: [K2(id=4)] → [ScopeBegin_A(id=5)] → [K3(id=6)] → [ScopeEnd_A(id=7)]
    // 预期：相同scope名称的节点使用相同的bit标记

    auto* scopeBeginA0 = CreateScopeBeginNode(1, 0, "scope_A", 2);
    auto* k1 = CreateKernelNode(2, 0, 3);
    auto* scopeEndA0 = CreateScopeEndNode(3, 0, "scope_A", INVALID_TASK_ID);

    auto* k2 = CreateKernelNode(4, 1, 5);
    auto* scopeBeginA1 = CreateScopeBeginNode(5, 1, "scope_A", 6);
    auto* k3 = CreateKernelNode(6, 1, 7);
    auto* scopeEndA1 = CreateScopeEndNode(7, 1, "scope_A", INVALID_TASK_ID);

    SetupStreams({{1, 2, 3}, {4, 5, 6, 7}});

    // 初始化scope名称（相同名称使用相同索引）
    graph->scopeNameToIdx["scope_A"] = 0;

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    // 验证所有普通节点都被处理（scope节点不放入nodes中）
    std::set<uint64_t> allProcessedNodes;
    for (const auto& scope : scopeInfos) {
        for (const auto* node : scope.nodes) {
            allProcessedNodes.insert(node->GetNodeId());
        }
    }
    std::set<uint64_t> expectedNodes = {2, 4, 6}; // 只有普通kernel节点
    EXPECT_EQ(allProcessedNodes, expectedNodes);
}

// ==================== 测试用例 32: 嵌套scope ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase32_NestedScopes)
{
    // 测试场景：嵌套的scope
    // Stream 0: [ScopeBegin_A(id=1)] → [ScopeBegin_B(id=2)] → [K1(id=3)] → [ScopeEnd_B(id=4)] → [K2(id=5)] → [ScopeEnd_A(id=6)]
    // 预期：K1同时属于scope A和scope B，K2只属于scope A

    auto* scopeBeginA = CreateScopeBeginNode(1, 0, "scope_A", 2);
    auto* scopeBeginB = CreateScopeBeginNode(2, 0, "scope_B", 3);
    auto* k1 = CreateKernelNode(3, 0, 4);
    auto* scopeEndB = CreateScopeEndNode(4, 0, "scope_B", 5);
    auto* k2 = CreateKernelNode(5, 0, 6);
    auto* scopeEndA = CreateScopeEndNode(6, 0, "scope_A", INVALID_TASK_ID);

    SetupStreams({{1, 2, 3, 4, 5, 6}});

    // 初始化scope名称
    graph->scopeNameToIdx["scope_A"] = 0;
    graph->scopeNameToIdx["scope_B"] = 1;

    // 执行scope标记更新
    graph->UpdateNodeScopeBitFlags();

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    // 验证所有普通节点都被处理（scope节点不放入nodes中）
    std::set<uint64_t> allProcessedNodes;
    for (const auto& scope : scopeInfos) {
        for (const auto* node : scope.nodes) {
            allProcessedNodes.insert(node->GetNodeId());
        }
    }
    std::set<uint64_t> expectedNodes = {3, 5}; // 只有普通kernel节点K1和K2
    EXPECT_EQ(allProcessedNodes, expectedNodes);

    // 验证K1的scopeBitFlags同时包含scope_A和scope_B
    auto k1BitFlags = k1->GetScopeBitFlags();
    EXPECT_TRUE(k1BitFlags.test(0)); // scope_A
    EXPECT_TRUE(k1BitFlags.test(1)); // scope_B

    // 验证K2的scopeBitFlags只包含scope_A
    auto k2BitFlags = k2->GetScopeBitFlags();
    EXPECT_TRUE(k2BitFlags.test(0)); // scope_A
    EXPECT_FALSE(k2BitFlags.test(1)); // scope_B
}

// ==================== 测试用例 33: Scope与跨流依赖 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase33_ScopeWithCrossStreamDependency)
{
    // 测试场景：scope与跨流Wait-Notify依赖
    // Stream 0: [ScopeBegin_A(id=1)] → [K1(id=2)] → [Wait1(id=3)] → [K2(id=4)] → [ScopeEnd_A(id=5)]
    // Stream 1: [K3(id=6)] → [Notify1(id=7)] → [K4(id=8)]
    // Event1: Wait1(id=3) 等待 Notify1(id=7)
    // 预期：scope内的节点（K1, K2）融合，Wait1/Notify1作为同步点

    auto* scopeBeginA = CreateScopeBeginNode(1, 0, "scope_A", 2);
    auto* k1 = CreateKernelNode(2, 0, 3);
    auto* wait1 = CreateWaitNode(3, 0, 7, 4);
    auto* k2 = CreateKernelNode(4, 0, 5);
    auto* scopeEndA = CreateScopeEndNode(5, 0, "scope_A", INVALID_TASK_ID);

    auto* k3 = CreateKernelNode(6, 1, 7);
    auto* notify1 = CreateNotifyNode(7, 1, 100, 8);
    auto* k4 = CreateKernelNode(8, 1, INVALID_TASK_ID);

    SetupStreams({{1, 2, 3, 4, 5}, {6, 7, 8}});
    SetupEvent(100, 7, {3});

    // 初始化scope名称
    graph->scopeNameToIdx["scope_A"] = 0;

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    // 验证所有非scope节点都被处理（scope节点不放入nodes中，但wait/notify会被添加）
    std::set<uint64_t> allProcessedNodes;
    for (const auto& scope : scopeInfos) {
        for (const auto* node : scope.nodes) {
            allProcessedNodes.insert(node->GetNodeId());
        }
    }
    std::set<uint64_t> expectedNodes = {2, 3, 4, 6, 7, 8}; // K1, Wait1, K2, K3, Notify1, K4
    EXPECT_EQ(allProcessedNodes, expectedNodes);
}

// ==================== 测试用例 34: 超过最大scope数量限制 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase34_ExceedMaxScopeNumLimit)
{
    // 测试场景：超过MAX_SCOPE_NUM限制
    // 预期：超过限制的scope不生效

    // 创建65个不同的scope（超过MAX_SCOPE_NUM=64）
    for (uint32_t i = 0; i < MAX_SCOPE_NUM + 1; ++i) {
        graph->scopeNameToIdx["scope_" + std::to_string(i)] = i;
    }

    auto* k1 = CreateKernelNode(1, 0, 2);
    auto* scopeBegin = CreateScopeBeginNode(2, 0, "scope_64", 3);
    auto* k2 = CreateKernelNode(3, 0, INVALID_TASK_ID);

    SetupStreams({{1, 2, 3}});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    // 验证所有节点都被处理（scope节点不放入nodes中）
    std::set<uint64_t> allProcessedNodes;
    for (const auto& scope : scopeInfos) {
        for (const auto* node : scope.nodes) {
            allProcessedNodes.insert(node->GetNodeId());
        }
    }
    std::set<uint64_t> expectedNodes = {1, 3}; // scopeBegin(2)是scope节点，不放入nodes中
    EXPECT_EQ(allProcessedNodes, expectedNodes);
}

// ==================== 测试用例 35: Scope与不可融合节点混合 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase35_ScopeWithUnfusibleNodes)
{
    // 测试场景：scope内包含不可融合节点
    // Stream 0: [ScopeBegin_A(id=1)] → [K1(id=2)] → [Unfusible_K(id=3)] → [K2(id=4)] → [ScopeEnd_A(id=5)]
    // 预期：生成2个scope，第一个scope包含[ScopeBegin_A(1), K1(2)]，第二个scope包含[K2(4), ScopeEnd_A(5)]，Unfusible_K(3)不应该被包含在任何scope中

    auto* scopeBeginA = CreateScopeBeginNode(1, 0, "scope_A", 2);
    auto* k1 = CreateKernelNode(2, 0, 3);
    auto* unfusibleK = CreateUnfusibleKernelNode(3, 0, 4);
    auto* k2 = CreateKernelNode(4, 0, 5);
    auto* scopeEndA = CreateScopeEndNode(5, 0, "scope_A", INVALID_TASK_ID);

    SetupStreams({{1, 2, 3, 4, 5}});

    // 初始化scope名称
    graph->scopeNameToIdx["scope_A"] = 0;

    // 执行scope标记更新
    graph->UpdateNodeScopeBitFlags();

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    // 验证生成了2个scope
    EXPECT_EQ(scopeInfos.size(), 2);

    // 验证scope 0只包含K1(2)，ScopeBegin_A(1)是scope节点，不放入nodes中
    EXPECT_EQ(scopeInfos[0].nodes.size(), 1);
    EXPECT_EQ(scopeInfos[0].nodes[0]->GetNodeId(), 2);

    // 验证scope 1只包含K2(4)，ScopeEnd_A(5)是scope节点，不放入nodes中
    EXPECT_EQ(scopeInfos[1].nodes.size(), 1);
    EXPECT_EQ(scopeInfos[1].nodes[0]->GetNodeId(), 4);

    // 验证不可融合的节点没有被包含在任何scope中
    std::set<uint64_t> allProcessedNodes;
    for (const auto& scope : scopeInfos) {
        for (const auto* node : scope.nodes) {
            allProcessedNodes.insert(node->GetNodeId());
        }
    }
    std::set<uint64_t> expectedNodes = {2, 4}; // 只有K1和K2是普通kernel节点
    EXPECT_EQ(allProcessedNodes, expectedNodes);

    // 验证融合状态
    EXPECT_TRUE(k1->IsFusible());
    EXPECT_FALSE(unfusibleK->IsFusible());
    EXPECT_TRUE(k2->IsFusible());
}

// ==================== 测试用例 36: Scope begin/end不成对（只有begin） ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase36_UnpairedScopeBegin)
{
    // 测试场景：scope begin/end不成对，只有begin没有end
    // Stream 0: [K1(id=1)] → [ScopeBegin_A(id=2)] → [K2(id=3)] → [K3(id=4)]
    // 预期：UpdateNodeScopeBitFlags会检测到未关闭的scope并报错

    auto* k1 = CreateKernelNode(1, 0, 2);
    auto* scopeBegin = CreateScopeBeginNode(2, 0, "scope_A", 3);
    auto* k2 = CreateKernelNode(3, 0, 4);
    auto* k3 = CreateKernelNode(4, 0, INVALID_TASK_ID);

    SetupStreams({{1, 2, 3, 4}});

    // 初始化scope名称
    graph->scopeNameToIdx["scope_A"] = 0;

    // 执行scope标记更新（应该检测到未关闭的scope）
    // 注意：这里不检查返回值，因为函数内部会记录错误日志
    graph->UpdateNodeScopeBitFlags();

    // 验证scope标记（即使未关闭，begin之后的节点也应该被标记）
    EXPECT_TRUE(k2->GetScopeBitFlags().test(0));
    EXPECT_TRUE(k3->GetScopeBitFlags().test(0));
}

// ==================== 测试用例 37: Scope begin/end不成对（只有end） ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase37_UnpairedScopeEnd)
{
    // 测试场景：scope begin/end不成对，只有end没有begin
    // Stream 0: [K1(id=1)] → [ScopeEnd_A(id=2)] → [K2(id=3)] → [K3(id=4)]
    // 预期：UpdateNodeScopeBitFlags会检测到没有匹配begin的scope end并报错

    auto* k1 = CreateKernelNode(1, 0, 2);
    auto* scopeEnd = CreateScopeEndNode(2, 0, "scope_A", 3);
    auto* k2 = CreateKernelNode(3, 0, 4);
    auto* k3 = CreateKernelNode(4, 0, INVALID_TASK_ID);

    SetupStreams({{1, 2, 3, 4}});

    // 初始化scope名称
    graph->scopeNameToIdx["scope_A"] = 0;

    // 执行scope标记更新（应该检测到没有匹配begin的scope end）
    graph->UpdateNodeScopeBitFlags();

    // 验证scope标记（scopeEnd本身应该有当前的scopeBitFlags）
    // K1, K2, K3不应该有scope_A的标记（因为begin不存在）
    EXPECT_FALSE(k1->GetScopeBitFlags().test(0));
}

// ==================== 测试用例 38: 重复的scope begin ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase38_DuplicateScopeBegin)
{
    // 测试场景：重复的scope begin
    // Stream 0: [ScopeBegin_A(id=1)] → [K1(id=2)] → [ScopeBegin_A(id=3)] → [K2(id=4)] → [ScopeEnd_A(id=5)]
    // 预期：UpdateNodeScopeBitFlags会检测到重复的scope begin并报错

    auto* scopeBegin1 = CreateScopeBeginNode(1, 0, "scope_A", 2);
    auto* k1 = CreateKernelNode(2, 0, 3);
    auto* scopeBegin2 = CreateScopeBeginNode(3, 0, "scope_A", 4);
    auto* k2 = CreateKernelNode(4, 0, 5);
    auto* scopeEnd = CreateScopeEndNode(5, 0, "scope_A", INVALID_TASK_ID);

    SetupStreams({{1, 2, 3, 4, 5}});

    // 初始化scope名称
    graph->scopeNameToIdx["scope_A"] = 0;

    // 执行scope标记更新（应该检测到重复的scope begin）
    graph->UpdateNodeScopeBitFlags();

    // K1应该在第一个scope内
    EXPECT_TRUE(k1->GetScopeBitFlags().test(0));
}

// ==================== 测试用例 39: 多个scope的scopeBitFlags验证 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase39_MultipleScopeBitFlagsVerification)
{
    // 测试场景：多个独立的scope，验证scopeBitFlags
    // Stream 0: [K1(id=1)] → [ScopeBegin_A(id=2)] → [K2(id=3)] → [ScopeEnd_A(id=4)] →
    //          [K3(id=5)] → [ScopeBegin_B(id=6)] → [K4(id=7)] → [ScopeEnd_B(id=8)] → [K5(id=9)]
    // 预期：
    //   - 节点2, 3, 4的scopeBitFlags第0位为1（属于scope_A）
    //   - 节点6, 7, 8的scopeBitFlags第1位为1（属于scope_B）
    //   - 节点1, 5, 9的scopeBitFlags全为0

    auto* k1 = CreateKernelNode(1, 0, 2);
    auto* scopeBeginA = CreateScopeBeginNode(2, 0, "scope_A", 3);
    auto* k2 = CreateKernelNode(3, 0, 4);
    auto* scopeEndA = CreateScopeEndNode(4, 0, "scope_A", 5);
    auto* k3 = CreateKernelNode(5, 0, 6);
    auto* scopeBeginB = CreateScopeBeginNode(6, 0, "scope_B", 7);
    auto* k4 = CreateKernelNode(7, 0, 8);
    auto* scopeEndB = CreateScopeEndNode(8, 0, "scope_B", 9);
    auto* k5 = CreateKernelNode(9, 0, INVALID_TASK_ID);

    SetupStreams({{1, 2, 3, 4, 5, 6, 7, 8, 9}});

    // 初始化scope名称
    graph->scopeNameToIdx["scope_A"] = 0;
    graph->scopeNameToIdx["scope_B"] = 1;

    // 执行scope标记更新
    graph->UpdateNodeScopeBitFlags();

    // 验证scopeBitFlags
    // K1: 不属于任何scope
    EXPECT_EQ(k1->GetScopeBitFlags().count(), 0);

    // scope_A内的节点（第0位为1）
    EXPECT_TRUE(scopeBeginA->GetScopeBitFlags().test(0));
    EXPECT_TRUE(k2->GetScopeBitFlags().test(0));
    EXPECT_TRUE(scopeEndA->GetScopeBitFlags().test(0));

    // K3: 在scope_A和scope_B之间
    EXPECT_EQ(k3->GetScopeBitFlags().count(), 0);

    // scope_B内的节点（第1位为1）
    EXPECT_TRUE(scopeBeginB->GetScopeBitFlags().test(1));
    EXPECT_TRUE(k4->GetScopeBitFlags().test(1));
    EXPECT_TRUE(scopeEndB->GetScopeBitFlags().test(1));

    // K5: 不属于任何scope
    EXPECT_EQ(k5->GetScopeBitFlags().count(), 0);

    // 验证scope_A和scope_B的标记不会互相干扰
    EXPECT_FALSE(scopeBeginA->GetScopeBitFlags().test(1));
    EXPECT_FALSE(k2->GetScopeBitFlags().test(1));
    EXPECT_FALSE(scopeEndA->GetScopeBitFlags().test(1));

    EXPECT_FALSE(scopeBeginB->GetScopeBitFlags().test(0));
    EXPECT_FALSE(k4->GetScopeBitFlags().test(0));
    EXPECT_FALSE(scopeEndB->GetScopeBitFlags().test(0));
}

// ==================== 测试用例 40: 纯unfusible scope（关键测试） ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase40_PureUnfusibleScope)
{
    // 测试场景：只有unfusible scope，没有fusible scope
    // Stream 0: [K1(id=1)] → [ScopeBegin_Unfusible(id=2)] → [K2(id=3)] → [K3(id=4)] → [ScopeEnd_Unfusible(id=5)] → [K4(id=6)]
    // 预期：
    //   - 节点1: 可融合（在scope外）
    //   - 节点2, 3, 4: 不可融合（在unfusible scope内）
    //   - 节点5: 不可融合（scope end节点）
    //   - 节点6: 可融合（在scope外）

    auto* k1 = CreateKernelNode(1, 0, 2);
    auto* scopeBegin = CreateUnfusibleScopeBeginNode(2, 0, 3);
    auto* k2 = CreateKernelNode(3, 0, 4);
    auto* k3 = CreateKernelNode(4, 0, 5);
    auto* scopeEnd = CreateUnfusibleScopeEndNode(5, 0, 6);
    auto* k4 = CreateKernelNode(6, 0, INVALID_TASK_ID);

    SetupStreams({{1, 2, 3, 4, 5, 6}});

    // 注意：unfusible scope没有scopeName，所以scopeNameToIdx为空
    // 这是关键测试点：UpdateNodeScopeBitFlags不应该直接return
    EXPECT_TRUE(graph->scopeNameToIdx.empty());

    // 执行scope标记更新
    graph->UpdateNodeScopeBitFlags();

    // 验证节点1：在scope外，应该可融合
    EXPECT_TRUE(k1->IsFusible()) << "K1 should be fusible (outside scope)";
    EXPECT_EQ(k1->GetScopeBitFlags().count(), 0) << "K1 should have no scope flags";

    // 验证scope begin：不可融合
    EXPECT_FALSE(scopeBegin->IsFusible()) << "ScopeBegin should be unfusible";
    EXPECT_EQ(scopeBegin->GetScopeBitFlags().count(), 0) << "ScopeBegin should have no scope flags (unfusible)";

    // 验证节点2和3：在unfusible scope内，应该不可融合
    EXPECT_FALSE(k2->IsFusible()) << "K2 should be unfusible (inside unfusible scope)";
    EXPECT_EQ(k2->GetScopeBitFlags().count(), 0) << "K2 should have no scope flags (unfusible scope)";

    EXPECT_FALSE(k3->IsFusible()) << "K3 should be unfusible (inside unfusible scope)";
    EXPECT_EQ(k3->GetScopeBitFlags().count(), 0) << "K3 should have no scope flags (unfusible scope)";

    // 验证scope end：不可融合
    EXPECT_FALSE(scopeEnd->IsFusible()) << "ScopeEnd should be unfusible";
    EXPECT_EQ(scopeEnd->GetScopeBitFlags().count(), 0) << "ScopeEnd should have no scope flags (unfusible)";

    // 验证节点4：在scope外，应该可融合
    EXPECT_TRUE(k4->IsFusible()) << "K4 should be fusible (outside scope)";
    EXPECT_EQ(k4->GetScopeBitFlags().count(), 0) << "K4 should have no scope flags";
}

// ==================== 测试用例 41: 多个unfusible scope ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase41_MultipleUnfusibleScopes)
{
    // 测试场景：多个独立的unfusible scope
    // Stream 0: [K1(id=1)] → [UnfusibleBegin1(id=2)] → [K2(id=3)] → [UnfusibleEnd1(id=4)] →
    //          [K3(id=5)] → [UnfusibleBegin2(id=6)] → [K4(id=7)] → [UnfusibleEnd2(id=8)] → [K5(id=9)]
    // 预期：
    //   - K1, K3, K5: 可融合
    //   - K2, K4: 不可融合

    auto* k1 = CreateKernelNode(1, 0, 2);
    auto* scopeBegin1 = CreateUnfusibleScopeBeginNode(2, 0, 3);
    auto* k2 = CreateKernelNode(3, 0, 4);
    auto* scopeEnd1 = CreateUnfusibleScopeEndNode(4, 0, 5);
    auto* k3 = CreateKernelNode(5, 0, 6);
    auto* scopeBegin2 = CreateUnfusibleScopeBeginNode(6, 0, 7);
    auto* k4 = CreateKernelNode(7, 0, 8);
    auto* scopeEnd2 = CreateUnfusibleScopeEndNode(8, 0, 9);
    auto* k5 = CreateKernelNode(9, 0, INVALID_TASK_ID);

    SetupStreams({{1, 2, 3, 4, 5, 6, 7, 8, 9}});

    // 执行scope标记更新
    graph->UpdateNodeScopeBitFlags();

    // 验证可融合的节点
    EXPECT_TRUE(k1->IsFusible());
    EXPECT_TRUE(k3->IsFusible());
    EXPECT_TRUE(k5->IsFusible());

    // 验证不可融合的节点
    EXPECT_FALSE(k2->IsFusible());
    EXPECT_FALSE(k4->IsFusible());
}

// ==================== 测试用例 42: fusible和unfusible scope混合 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase42_MixedFusibleAndUnfusibleScopes)
{
    // 测试场景：fusible scope和unfusible scope混合
    // Stream 0: [FusibleBegin_A(id=1)] → [K1(id=2)] → [UnfusibleBegin(id=3)] → [K2(id=4)] →
    //          [UnfusibleEnd(id=5)] → [K3(id=6)] → [FusibleEnd_A(id=7)]
    // 预期：
    //   - K1: 可融合（只在fusible scope中，scopeBitFlags第0位为1）
    //   - K2: 不可融合（在unfusible scope中，虽然在fusible scope中但被unfusible覆盖）
    //   - K3: 可融合（回到fusible scope中，scopeBitFlags第0位为1）

    auto* fusibleBeginA = CreateScopeBeginNode(1, 0, "scope_A", 2);
    auto* k1 = CreateKernelNode(2, 0, 3);
    auto* unfusibleBegin = CreateUnfusibleScopeBeginNode(3, 0, 4);
    auto* k2 = CreateKernelNode(4, 0, 5);
    auto* unfusibleEnd = CreateUnfusibleScopeEndNode(5, 0, 6);
    auto* k3 = CreateKernelNode(6, 0, 7);
    auto* fusibleEndA = CreateScopeEndNode(7, 0, "scope_A", INVALID_TASK_ID);

    SetupStreams({{1, 2, 3, 4, 5, 6, 7}});

    // 初始化scope名称
    graph->scopeNameToIdx["scope_A"] = 0;

    // 执行scope标记更新
    graph->UpdateNodeScopeBitFlags();

    // 验证K1：在fusible scope_A中，可融合，scopeBitFlags第0位为1
    EXPECT_TRUE(k1->IsFusible()) << "K1 should be fusible";
    EXPECT_TRUE(k1->GetScopeBitFlags().test(0)) << "K1 should have scope_A flag";
    EXPECT_EQ(k1->GetScopeBitFlags().count(), 1) << "K1 should have exactly 1 scope flag";

    // 验证K2：在fusible scope_A和unfusible scope中，不可融合（unfusible覆盖fusible）
    EXPECT_FALSE(k2->IsFusible()) << "K2 should be unfusible (unfusible scope overrides fusible)";
    // 注意：K2的scopeBitFlags应该包含scope_A（因为它还在fusible scope中）
    EXPECT_TRUE(k2->GetScopeBitFlags().test(0)) << "K2 should have scope_A flag";
    EXPECT_EQ(k2->GetScopeBitFlags().count(), 1) << "K2 should have exactly 1 scope flag";

    // 验证K3：在fusible scope_A中，可融合，scopeBitFlags第0位为1
    EXPECT_TRUE(k3->IsFusible()) << "K3 should be fusible";
    EXPECT_TRUE(k3->GetScopeBitFlags().test(0)) << "K3 should have scope_A flag";
    EXPECT_EQ(k3->GetScopeBitFlags().count(), 1) << "K3 should have exactly 1 scope flag";
}

// ==================== 测试用例 43: 嵌套的fusible和unfusible scope ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase43_NestedFusibleAndUnfusibleScopes)
{
    // 测试场景：嵌套的fusible和unfusible scope
    // Stream 0: [FusibleBegin_A(id=1)] → [UnfusibleBegin(id=2)] → [FusibleBegin_B(id=3)] →
    //          [K1(id=4)] → [FusibleEnd_B(id=5)] → [K2(id=6)] → [UnfusibleEnd(id=7)] →
    //          [K3(id=8)] → [FusibleEnd_A(id=9)]
    // 预期：
    //   - K1: 不可融合（在外层unfusible scope中，虽然在内层fusible scope_B中）
    //   - K2: 不可融合（在unfusible scope中）
    //   - K3: 可融合（在fusible scope_A中）

    auto* fusibleBeginA = CreateScopeBeginNode(1, 0, "scope_A", 2);
    auto* unfusibleBegin = CreateUnfusibleScopeBeginNode(2, 0, 3);
    auto* fusibleBeginB = CreateScopeBeginNode(3, 0, "scope_B", 4);
    auto* k1 = CreateKernelNode(4, 0, 5);
    auto* fusibleEndB = CreateScopeEndNode(5, 0, "scope_B", 6);
    auto* k2 = CreateKernelNode(6, 0, 7);
    auto* unfusibleEnd = CreateUnfusibleScopeEndNode(7, 0, 8);
    auto* k3 = CreateKernelNode(8, 0, 9);
    auto* fusibleEndA = CreateScopeEndNode(9, 0, "scope_A", INVALID_TASK_ID);

    SetupStreams({{1, 2, 3, 4, 5, 6, 7, 8, 9}});

    // 初始化scope名称
    graph->scopeNameToIdx["scope_A"] = 0;
    graph->scopeNameToIdx["scope_B"] = 1;

    // 执行scope标记更新
    graph->UpdateNodeScopeBitFlags();

    // 验证K1：在fusible scope_A, unfusible scope, fusible scope_B中，不可融合
    EXPECT_FALSE(k1->IsFusible()) << "K1 should be unfusible (unfusible outer scope overrides)";
    EXPECT_TRUE(k1->GetScopeBitFlags().test(0)) << "K1 should have scope_A flag";
    EXPECT_TRUE(k1->GetScopeBitFlags().test(1)) << "K1 should have scope_B flag";
    EXPECT_EQ(k1->GetScopeBitFlags().count(), 2) << "K1 should have 2 scope flags";

    // 验证K2：在fusible scope_A和unfusible scope中，不可融合
    EXPECT_FALSE(k2->IsFusible()) << "K2 should be unfusible";
    EXPECT_TRUE(k2->GetScopeBitFlags().test(0)) << "K2 should have scope_A flag";
    EXPECT_EQ(k2->GetScopeBitFlags().count(), 1) << "K2 should have 1 scope flag";

    // 验证K3：只在fusible scope_A中，可融合
    EXPECT_TRUE(k3->IsFusible()) << "K3 should be fusible";
    EXPECT_TRUE(k3->GetScopeBitFlags().test(0)) << "K3 should have scope_A flag";
    EXPECT_EQ(k3->GetScopeBitFlags().count(), 1) << "K3 should have 1 scope flag";
}

// ==================== 测试用例 44: unfusible scope跨越整个图 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase44_UnfusibleScopeSpanningWholeGraph)
{
    // 测试场景：unfusible scope跨越整个图
    // Stream 0: [UnfusibleBegin(id=1)] → [K1(id=2)] → [K2(id=3)] → [K3(id=4)] → [UnfusibleEnd(id=5)]
    // 预期：所有节点都不可融合

    auto* unfusibleBegin = CreateUnfusibleScopeBeginNode(1, 0, 2);
    auto* k1 = CreateKernelNode(2, 0, 3);
    auto* k2 = CreateKernelNode(3, 0, 4);
    auto* k3 = CreateKernelNode(4, 0, 5);
    auto* unfusibleEnd = CreateUnfusibleScopeEndNode(5, 0, INVALID_TASK_ID);

    SetupStreams({{1, 2, 3, 4, 5}});

    // 执行scope标记更新
    graph->UpdateNodeScopeBitFlags();

    // 验证所有节点都不可融合
    EXPECT_FALSE(unfusibleBegin->IsFusible());
    EXPECT_FALSE(k1->IsFusible());
    EXPECT_FALSE(k2->IsFusible());
    EXPECT_FALSE(k3->IsFusible());
    EXPECT_FALSE(unfusibleEnd->IsFusible());
}

// ==================== 测试用例 45: 重复的unfusible scope begin/end ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase45_DuplicateUnfusibleScopeBeginEnd)
{
    // 测试场景：重复的unfusible scope begin/end
    // Stream 0: [UnfusibleBegin1(id=1)] → [K1(id=2)] → [UnfusibleBegin2(id=3)] →
    //          [K2(id=4)] → [UnfusibleEnd1(id=5)] → [K3(id=6)] → [UnfusibleEnd2(id=7)]
    // 预期：
    //   - K1: 不可融合（在第一个unfusible scope中）
    //   - K2: 不可融合（在两个unfusible scope中）
    //   - K3: 不可融合（仍在第一个unfusible scope中）

    auto* unfusibleBegin1 = CreateUnfusibleScopeBeginNode(1, 0, 2);
    auto* k1 = CreateKernelNode(2, 0, 3);
    auto* unfusibleBegin2 = CreateUnfusibleScopeBeginNode(3, 0, 4);
    auto* k2 = CreateKernelNode(4, 0, 5);
    auto* unfusibleEnd1 = CreateUnfusibleScopeEndNode(5, 0, 6);
    auto* k3 = CreateKernelNode(6, 0, 7);
    auto* unfusibleEnd2 = CreateUnfusibleScopeEndNode(7, 0, INVALID_TASK_ID);

    SetupStreams({{1, 2, 3, 4, 5, 6, 7}});

    // 执行scope标记更新
    graph->UpdateNodeScopeBitFlags();

    // 验证所有节点都不可融合
    EXPECT_FALSE(k1->IsFusible());
    EXPECT_FALSE(k2->IsFusible());
    EXPECT_FALSE(k3->IsFusible());
}

// ==================== 测试用例 46: unfusible scope在fusible scope内部 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase46_UnfusibleScopeInsideFusibleScope)
{
    // 测试场景：unfusible scope完全在fusible scope内部
    // Stream 0: [FusibleBegin_A(id=1)] → [K1(id=2)] → [UnfusibleBegin(id=3)] →
    //          [K2(id=4)] → [UnfusibleEnd(id=5)] → [K3(id=6)] → [FusibleEnd_A(id=7)]
    // 预期：
    //   - K1, K3: 可融合（只在fusible scope中）
    //   - K2: 不可融合（在unfusible scope中）

    auto* fusibleBeginA = CreateScopeBeginNode(1, 0, "scope_A", 2);
    auto* k1 = CreateKernelNode(2, 0, 3);
    auto* unfusibleBegin = CreateUnfusibleScopeBeginNode(3, 0, 4);
    auto* k2 = CreateKernelNode(4, 0, 5);
    auto* unfusibleEnd = CreateUnfusibleScopeEndNode(5, 0, 6);
    auto* k3 = CreateKernelNode(6, 0, 7);
    auto* fusibleEndA = CreateScopeEndNode(7, 0, "scope_A", INVALID_TASK_ID);

    SetupStreams({{1, 2, 3, 4, 5, 6, 7}});

    // 初始化scope名称
    graph->scopeNameToIdx["scope_A"] = 0;

    // 执行scope标记更新
    graph->UpdateNodeScopeBitFlags();

    // 验证K1：可融合，在fusible scope_A中
    EXPECT_TRUE(k1->IsFusible());
    EXPECT_TRUE(k1->GetScopeBitFlags().test(0));

    // 验证K2：不可融合，在unfusible scope中
    EXPECT_FALSE(k2->IsFusible());
    EXPECT_TRUE(k2->GetScopeBitFlags().test(0)); // 仍在fusible scope_A中

    // 验证K3：可融合，在fusible scope_A中
    EXPECT_TRUE(k3->IsFusible());
    EXPECT_TRUE(k3->GetScopeBitFlags().test(0));
}

// ==================== 测试用例 47: unfusible scope与fusible scope并列 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase47_UnfusibleAndFusibleScopesSideBySide)
{
    // 测试场景：unfusible scope和fusible scope并列
    // Stream 0: [K1(id=1)] → [UnfusibleBegin(id=2)] → [K2(id=3)] → [UnfusibleEnd(id=4)] →
    //          [K3(id=5)] → [FusibleBegin_A(id=6)] → [K4(id=7)] → [FusibleEnd_A(id=8)] → [K5(id=9)]
    // 预期：
    //   - K1, K3, K5: 可融合
    //   - K2: 不可融合（在unfusible scope中）
    //   - K4: 可融合（在fusible scope中）

    auto* k1 = CreateKernelNode(1, 0, 2);
    auto* unfusibleBegin = CreateUnfusibleScopeBeginNode(2, 0, 3);
    auto* k2 = CreateKernelNode(3, 0, 4);
    auto* unfusibleEnd = CreateUnfusibleScopeEndNode(4, 0, 5);
    auto* k3 = CreateKernelNode(5, 0, 6);
    auto* fusibleBeginA = CreateScopeBeginNode(6, 0, "scope_A", 7);
    auto* k4 = CreateKernelNode(7, 0, 8);
    auto* fusibleEndA = CreateScopeEndNode(8, 0, "scope_A", 9);
    auto* k5 = CreateKernelNode(9, 0, INVALID_TASK_ID);

    SetupStreams({{1, 2, 3, 4, 5, 6, 7, 8, 9}});

    // 初始化scope名称
    graph->scopeNameToIdx["scope_A"] = 0;

    // 执行scope标记更新
    graph->UpdateNodeScopeBitFlags();

    // 验证可融合的节点
    EXPECT_TRUE(k1->IsFusible());
    EXPECT_TRUE(k3->IsFusible());
    EXPECT_TRUE(k4->IsFusible());
    EXPECT_TRUE(k5->IsFusible());

    // 验证不可融合的节点
    EXPECT_FALSE(k2->IsFusible());

    // 验证scopeBitFlags
    EXPECT_EQ(k1->GetScopeBitFlags().count(), 0);
    EXPECT_EQ(k3->GetScopeBitFlags().count(), 0);
    EXPECT_TRUE(k4->GetScopeBitFlags().test(0));
    EXPECT_EQ(k5->GetScopeBitFlags().count(), 0);
}

// ==================== 测试用例 48: 未关闭的unfusible scope ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase48_UnclosedUnfusibleScope)
{
    // 测试场景：未关闭的unfusible scope
    // Stream 0: [K1(id=1)] → [UnfusibleBegin(id=2)] → [K2(id=3)] → [K3(id=4)]
    // 预期：K2和K3不可融合（在unfusible scope中），K1可融合

    auto* k1 = CreateKernelNode(1, 0, 2);
    auto* unfusibleBegin = CreateUnfusibleScopeBeginNode(2, 0, 3);
    auto* k2 = CreateKernelNode(3, 0, 4);
    auto* k3 = CreateKernelNode(4, 0, INVALID_TASK_ID);

    SetupStreams({{1, 2, 3, 4}});

    // 执行scope标记更新（应该记录警告但继续处理）
    graph->UpdateNodeScopeBitFlags();

    // 验证节点1：可融合
    EXPECT_TRUE(k1->IsFusible());

    // 验证节点2和3：不可融合（在未关闭的unfusible scope中）
    EXPECT_FALSE(k2->IsFusible());
    EXPECT_FALSE(k3->IsFusible());
}

// ==================== 测试用例 49: 只有unfusible scope，无普通节点 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase49_OnlyUnfusibleScopeNoRegularNodes)
{
    // 测试场景：只有unfusible scope begin和end，没有普通节点
    // Stream 0: [UnfusibleBegin(id=1)] → [UnfusibleEnd(id=2)]

    auto* unfusibleBegin = CreateUnfusibleScopeBeginNode(1, 0, 2);
    auto* unfusibleEnd = CreateUnfusibleScopeEndNode(2, 0, INVALID_TASK_ID);

    SetupStreams({{1, 2}});

    // 执行scope标记更新
    graph->UpdateNodeScopeBitFlags();

    // 验证两个节点都不可融合
    EXPECT_FALSE(unfusibleBegin->IsFusible());
    EXPECT_FALSE(unfusibleEnd->IsFusible());
}

// ==================== 测试用例 50: 空图（无任何节点） ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase50_EmptyGraph)
{
    // 测试场景：空图，没有任何节点
    SetupStreams({{}});

    // 执行scope标记更新（应该正常完成，不崩溃）
    EXPECT_NO_THROW(graph->UpdateNodeScopeBitFlags());
}

// ==================== ScopeBitFlags 相关测试用例 ====================

// ==================== 测试用例 51: 不同 ScopeBitFlags 的节点应分配到不同 Scope ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase51_DifferentScopeBitFlagsSplitIntoDifferentScopes)
{
    // 场景：两个流中的节点具有不同的 ScopeBitFlags，应该分配到不同的 Scope
    // Stream 0: K1(scope=0) → K2(scope=1)
    // Stream 1: K3(scope=0)
    // 期望：K1 和 K3 在 Scope 0，K2 在 Scope 1

    auto* k1 = CreateKernelNode(1, 0, 2);
    std::bitset<MAX_SCOPE_NUM> flags0;
    flags0.set(0);
    k1->SetScopeBitFlags(flags0);

    auto* k2 = CreateKernelNode(2, 0, INVALID_TASK_ID);
    std::bitset<MAX_SCOPE_NUM> flags1;
    flags1.set(1);
    k2->SetScopeBitFlags(flags1);

    auto* k3 = CreateKernelNode(3, 1, INVALID_TASK_ID);
    k3->SetScopeBitFlags(flags0);

    SetupStreams({{1, 2}, {3}});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    // 应该产生 2 个 scope
    EXPECT_EQ(scopeInfos.size(), 2);

    // Scope 0: K1, K3 (相同 ScopeBitFlags)
    // Scope 1: K2 (不同 ScopeBitFlags)
    EXPECT_EQ(scopeInfos[0].nodes.size(), 2);
    EXPECT_EQ(scopeInfos[1].nodes.size(), 1);

    // 验证 ScopeBitFlags
    EXPECT_EQ(scopeInfos[0].scopeBitFlags, flags0);
    EXPECT_EQ(scopeInfos[1].scopeBitFlags, flags1);
}

// ==================== 测试用例 52: 无 ScopeBitFlags 的节点只能和无 ScopeBitFlags 的节点融合 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase52_NoScopeBitFlagsOnlyFuseWithNoFlags)
{
    // 场景：没有设置 ScopeBitFlags 的节点只能和其他没有设置的节点融合
    // Stream 0: K1(scope=0) → K2(no flags) → K3(scope=1)
    // 期望：K1 在 Scope 0，K2 在独立 Scope，K3 在 Scope 1

    auto* k1 = CreateKernelNode(1, 0, 2);
    std::bitset<MAX_SCOPE_NUM> flags0;
    flags0.set(0);
    k1->SetScopeBitFlags(flags0);

    auto* k2 = CreateKernelNode(2, 0, 3);
    // K2 不设置 ScopeBitFlags

    auto* k3 = CreateKernelNode(3, 0, INVALID_TASK_ID);
    std::bitset<MAX_SCOPE_NUM> flags1;
    flags1.set(1);
    k3->SetScopeBitFlags(flags1);

    SetupStreams({{1, 2, 3}});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    // 应该产生 3 个 scope
    EXPECT_EQ(scopeInfos.size(), 3);

    // Scope 0: K1 (scope=0)
    // Scope 1: K2 (no flags)
    // Scope 2: K3 (scope=1)
    EXPECT_EQ(scopeInfos[0].nodes.size(), 1);
    EXPECT_EQ(scopeInfos[1].nodes.size(), 1);
    EXPECT_EQ(scopeInfos[2].nodes.size(), 1);
    EXPECT_EQ(scopeInfos[0].nodes[0]->GetNodeId(), 1);
    EXPECT_EQ(scopeInfos[1].nodes[0]->GetNodeId(), 2);
    EXPECT_EQ(scopeInfos[2].nodes[0]->GetNodeId(), 3);
}

// ==================== 测试用例 53: 多个 ScopeBitFlags 位同时设置 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase53_MultipleScopeBitFlagsBits)
{
    // 场景：节点的 ScopeBitFlags 有多个位设置，只有完全匹配才能融合
    // Stream 0: K1(flags=011) → K2(flags=011) → K3(flags=001)
    // Stream 1: K4(flags=011) → K5(flags=111)
    // 期望：K1, K2, K4 在 Scope 0 (flags=011), K3 在 Scope 1, K5 在 Scope 2

    auto* k1 = CreateKernelNode(1, 0, 2);
    std::bitset<MAX_SCOPE_NUM> flags011;
    flags011.set(0);
    flags011.set(1);
    k1->SetScopeBitFlags(flags011);

    auto* k2 = CreateKernelNode(2, 0, 3);
    k2->SetScopeBitFlags(flags011);

    auto* k3 = CreateKernelNode(3, 0, INVALID_TASK_ID);
    std::bitset<MAX_SCOPE_NUM> flags001;
    flags001.set(0);
    k3->SetScopeBitFlags(flags001);

    auto* k4 = CreateKernelNode(4, 1, 5);
    k4->SetScopeBitFlags(flags011);

    auto* k5 = CreateKernelNode(5, 1, INVALID_TASK_ID);
    std::bitset<MAX_SCOPE_NUM> flags111;
    flags111.set(0);
    flags111.set(1);
    flags111.set(2);
    k5->SetScopeBitFlags(flags111);

    SetupStreams({{1, 2, 3}, {4, 5}});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    // 应该产生 3 个 scope
    EXPECT_EQ(scopeInfos.size(), 3);

    // Scope 0: K1, K2, K4 (flags=011)
    EXPECT_EQ(scopeInfos[0].nodes.size(), 3);
    EXPECT_EQ(scopeInfos[0].scopeBitFlags, flags011);

    // Scope 1: K3 (flags=001) 或 K5 (flags=111)，取决于处理顺序
    // Scope 2: 另一个
}

// ==================== 测试用例 54: 复杂场景 - 30+ 节点多流多 ScopeBitFlags ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase54_ComplexMultiStreamMultiScopeBitFlags)
{
    // 复杂场景：4 个流，32 个节点，4 种 ScopeBitFlags，包含事件同步
    // 
    // Stream 0 (scope0): K1 → K2 → K3 → K4 → K5 → K6 → K7 → K8
    // Stream 1 (scope1): K9 → K10 → K11 → K12 → K13 → K14 → K15 → K16
    // Stream 2 (scope0): K17 → K18 → Notify1 → K20 → K21 → K22 → K23 → K24
    // Stream 3 (scope1): K25 → K26 → Wait1 → K28 → K29 → K30 → K31 → K32
    //
    // Event: Notify1(K19) → Wait1(K27)
    //
    // 期望结果：
    // - Scope 0: K1-K8, K17-K24 (scope0, 包含 Notify1)
    // - Scope 1: K9-K16, K25-K32 (scope1, 包含 Wait1)

    std::bitset<MAX_SCOPE_NUM> scope0;
    scope0.set(0);
    std::bitset<MAX_SCOPE_NUM> scope1;
    scope1.set(1);

    // Stream 0: K1-K8 (scope0)
    for (int i = 1; i <= 8; ++i) {
        auto* node = CreateKernelNode(i, 0, i < 8 ? i + 1 : INVALID_TASK_ID);
        node->SetScopeBitFlags(scope0);
    }

    // Stream 1: K9-K16 (scope1)
    for (int i = 9; i <= 16; ++i) {
        auto* node = CreateKernelNode(i, 1, i < 16 ? i + 1 : INVALID_TASK_ID);
        node->SetScopeBitFlags(scope1);
    }

    // Stream 2: K17-K24 (scope0), K19 是 Notify
    for (int i = 17; i <= 24; ++i) {
        if (i == 19) {
            auto* node = CreateNotifyNode(19, 2, 100, 20);  // eventId=100
            node->SetScopeBitFlags(scope0);
        } else {
            auto* node = CreateKernelNode(i, 2, i < 24 ? i + 1 : INVALID_TASK_ID);
            node->SetScopeBitFlags(scope0);
        }
    }

    // Stream 3: K25-K32 (scope1), K27 是 Wait
    for (int i = 25; i <= 32; ++i) {
        if (i == 27) {
            auto* node = CreateWaitNode(27, 3, 19, 28);  // 等待 Notify1(K19)
            node->SetScopeBitFlags(scope1);
        } else {
            auto* node = CreateKernelNode(i, 3, i < 32 ? i + 1 : INVALID_TASK_ID);
            node->SetScopeBitFlags(scope1);
        }
    }

    SetupStreams({
        {1, 2, 3, 4, 5, 6, 7, 8},           // Stream 0
        {9, 10, 11, 12, 13, 14, 15, 16},   // Stream 1
        {17, 18, 19, 20, 21, 22, 23, 24},  // Stream 2
        {25, 26, 27, 28, 29, 30, 31, 32}   // Stream 3
    });

    SetupEvent(100, 19, {27});  // Event 100: Notify1(19) → Wait1(27)

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    // 应该产生 2 个 scope
    EXPECT_EQ(scopeInfos.size(), 2);

    // 验证每个 scope 的 ScopeBitFlags
    int scope0Count = 0, scope1Count = 0;
    for (const auto& scope : scopeInfos) {
        if (scope.scopeBitFlags == scope0) scope0Count++;
        else if (scope.scopeBitFlags == scope1) scope1Count++;
    }
    EXPECT_EQ(scope0Count, 1);
    EXPECT_EQ(scope1Count, 1);

    // 验证总节点数 = 32
    size_t totalNodes = 0;
    for (const auto& scope : scopeInfos) {
        totalNodes += scope.nodes.size();
    }
    EXPECT_EQ(totalNodes, 32);

    // 验证每个 scope 的节点数
    // Scope 0: 16 nodes (Stream 0 + Stream 2)
    // Scope 1: 16 nodes (Stream 1 + Stream 3)
    EXPECT_EQ(scopeInfos[0].nodes.size(), 16);
    EXPECT_EQ(scopeInfos[1].nodes.size(), 16);
}

// ==================== 测试用例 55: ScopeBitFlags 与事件同步交互 ====================

TEST_F(SuperKernelScopeSplitterTest, TestCase55_ScopeBitFlagsWithEventSynchronization)
{
    // 场景：Wait 节点跨越不同 ScopeBitFlags 的边界
    // Stream 0: K1(scope=0) → Wait1(scope=0) → K3(scope=1)
    // Stream 1: K4(scope=0) → Notify1(scope=0) → K6(scope=1)
    // Event: Notify1 → Wait1
    //
    // 处理流程：
    // 1. K1 和 K4 先被发现 (scope=0)
    // 2. Wait1 因 notify 未访问而被 suspend
    // 3. Notify1 被处理，resume Wait1
    // 4. Wait1 检查 ScopeBitFlags，匹配 scope0，可以加入
    // 5. K3 和 K6 检查 ScopeBitFlags，不匹配当前 scope，terminate
    // 6. 新 scope 开始处理 K3 和 K6

    std::bitset<MAX_SCOPE_NUM> scope0;
    scope0.set(0);
    std::bitset<MAX_SCOPE_NUM> scope1;
    scope1.set(1);

    // Stream 0: K1 → Wait1 → K3
    auto* k1 = CreateKernelNode(1, 0, 2);
    k1->SetScopeBitFlags(scope0);
    auto* wait1 = CreateWaitNode(2, 0, 5, 3);  // 等待 Notify1(id=5)
    wait1->SetScopeBitFlags(scope0);
    auto* k3 = CreateKernelNode(3, 0, INVALID_TASK_ID);
    k3->SetScopeBitFlags(scope1);

    // Stream 1: K4 → Notify1 → K6
    auto* k4 = CreateKernelNode(4, 1, 5);
    k4->SetScopeBitFlags(scope0);
    auto* notify1 = CreateNotifyNode(5, 1, 100, 6);  // eventId=100
    notify1->SetScopeBitFlags(scope0);
    auto* k6 = CreateKernelNode(6, 1, INVALID_TASK_ID);
    k6->SetScopeBitFlags(scope1);

    SetupStreams({{1, 2, 3}, {4, 5, 6}});
    SetupEvent(100, 5, {2});

    SuperKernelScopeSplitter splitter(*graph);
    bool result = splitter.SplitGraph();

    ASSERT_TRUE(result);
    const auto& scopeInfos = splitter.GetScopeInfos();

    // 应该产生 2 个 scope
    EXPECT_EQ(scopeInfos.size(), 2);

    // Scope 0: K1, K4, Notify1(5), Wait1(2) (scope=0)
    // 节点按 ID 排序: K1(1), Wait1(2), K3(3), K4(4), Notify1(5), K6(6)
    // 但 K3 和 K6 因为 scopeBitFlags 不匹配，在第二个 scope
    EXPECT_EQ(scopeInfos[0].scopeBitFlags, scope0);
    EXPECT_EQ(scopeInfos[0].nodes.size(), 4);

    // Scope 1: K3(3), K6(6) (scope=1)
    EXPECT_EQ(scopeInfos[1].scopeBitFlags, scope1);
    EXPECT_EQ(scopeInfos[1].nodes.size(), 2);
}
