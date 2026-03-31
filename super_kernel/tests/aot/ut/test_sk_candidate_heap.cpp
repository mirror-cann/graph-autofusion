/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#define private public
#define protected public
#include "sk_candidate_heap.h"
#include "sk_graph.h"
#include "sk_node.h"
#include "sk_types.h"

class SkCandidateHeapTest : public testing::Test {
protected:
    void SetUp() override
    {
        graph = std::make_unique<SuperKernelGraph>();
    }

    void TearDown() override
    {
        graph.reset();
    }

    // 创建 Kernel 节点并添加到 graph 中
    uint64_t CreateKernelNode(uint64_t nodeId, uint32_t streamIdx = 0,
                              SkKernelType kernelType = SkKernelType::AIC_ONLY)
    {
        auto node = std::make_unique<SuperKernelKernelNode>(
            nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, streamIdx, INVALID_STREAM_ID, INVALID_TASK_ID);
        node->SetNodeType(SkNodeType::NODE_KERNEL);
        node->SetNodeId(nodeId);
        node->nodeInfos.kernelInfos.kernelType = kernelType;
        node->nodeInfos.kernelInfos.numBlocks = 1;
        node->nodeInfos.kernelInfos.funcName = "test_kernel";
        graph->graphMap[nodeId] = std::move(node);
        return nodeId;
    }

    // 创建 Notify 节点并添加到 graph 中
    uint64_t CreateNotifyNode(uint64_t nodeId, uint32_t streamIdx = 0)
    {
        auto node = std::make_unique<SuperKernelMemoryNode>(
            nullptr, ACL_MODEL_RI_TASK_EVENT_RECORD, 0, streamIdx, INVALID_STREAM_ID, INVALID_TASK_ID);
        node->SetNodeType(SkNodeType::NODE_NOTIFY);
        node->SetNodeId(nodeId);
        node->nodeInfos.syncInfos.eventId = nodeId * 10;
        graph->graphMap[nodeId] = std::move(node);
        return nodeId;
    }

    // 创建 Wait 节点并添加到 graph 中
    uint64_t CreateWaitNode(uint64_t nodeId, uint32_t streamIdx = 0)
    {
        auto node = std::make_unique<SuperKernelMemoryNode>(
            nullptr, ACL_MODEL_RI_TASK_EVENT_WAIT, 0, streamIdx, INVALID_STREAM_ID, INVALID_TASK_ID);
        node->SetNodeType(SkNodeType::NODE_WAIT);
        node->SetNodeId(nodeId);
        node->nodeInfos.syncInfos.eventId = nodeId * 10;
        graph->graphMap[nodeId] = std::move(node);
        return nodeId;
    }

    // 创建 Reset 节点并添加到 graph 中
    uint64_t CreateResetNode(uint64_t nodeId, uint32_t streamIdx = 0)
    {
        auto node = std::make_unique<SuperKernelMemoryNode>(
            nullptr, ACL_MODEL_RI_TASK_EVENT_RESET, 0, streamIdx, INVALID_STREAM_ID, INVALID_TASK_ID);
        node->SetNodeType(SkNodeType::NODE_RESET);
        node->SetNodeId(nodeId);
        node->nodeInfos.syncInfos.eventId = nodeId * 10;
        graph->graphMap[nodeId] = std::move(node);
        return nodeId;
    }

    std::unique_ptr<SuperKernelGraph> graph;
};

// ==================== 基本操作测试 ====================

// 测试空堆 (heapMode = CUSTOMIZE_QUEUE)
TEST_F(SkCandidateHeapTest, Empty_InitiallyTrue)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    EXPECT_TRUE(heap.empty());
    EXPECT_EQ(heap.size(), 0U);
    EXPECT_FALSE(heap.HasKernelNodes());
}

// 测试 Push 单个 Kernel 节点
TEST_F(SkCandidateHeapTest, Push_SingleKernelNode)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    uint64_t nodeId = CreateKernelNode(1);
    heap.push(nodeId);

    EXPECT_FALSE(heap.empty());
    EXPECT_EQ(heap.size(), 1U);
    EXPECT_TRUE(heap.HasKernelNodes());
}

// 测试 Push 无效节点 ID
TEST_F(SkCandidateHeapTest, Push_InvalidNodeId_Ignored)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    heap.push(999);  // 不存在的 nodeId
    EXPECT_TRUE(heap.empty());
}

// 测试 Pop 空堆
TEST_F(SkCandidateHeapTest, Pop_EmptyHeap_ReturnZero)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    EXPECT_EQ(heap.pop(), 0U);
}

// 测试 Push 和 Pop 单个 Kernel 节点
TEST_F(SkCandidateHeapTest, PushPop_SingleKernelNode)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    uint64_t nodeId = CreateKernelNode(1);
    heap.push(nodeId);

    uint64_t poppedId = heap.pop();
    EXPECT_EQ(poppedId, nodeId);
    EXPECT_TRUE(heap.empty());
}

// ==================== heapMode = PRIORITY_QUEUE 测试 (简单模式) ====================

// 测试 heapMode = PRIORITY_QUEUE 的基本操作
TEST_F(SkCandidateHeapTest, HeapMode0_BasicOperations)
{
    SkCandidateHeap heap(*graph, SkHeapType::PRIORITY_QUEUE);
    
    heap.push(5);
    heap.push(2);
    heap.push(8);
    
    EXPECT_FALSE(heap.empty());
    EXPECT_EQ(heap.size(), 3U);
    
    // heapMode = 0 使用 priority_queue，按 nodeId 升序弹出
    EXPECT_EQ(heap.pop(), 2U);
    EXPECT_EQ(heap.pop(), 5U);
    EXPECT_EQ(heap.pop(), 8U);
    EXPECT_TRUE(heap.empty());
}

// 测试 heapMode = PRIORITY_QUEUE 的 clear
TEST_F(SkCandidateHeapTest, HeapMode0_Clear)
{
    SkCandidateHeap heap(*graph, SkHeapType::PRIORITY_QUEUE);
    heap.push(1);
    heap.push(2);
    heap.push(3);
    
    heap.clear();
    
    EXPECT_TRUE(heap.empty());
    EXPECT_EQ(heap.size(), 0U);
}

// 测试 heapMode = PRIORITY_QUEUE 的 reset
TEST_F(SkCandidateHeapTest, HeapMode0_Reset)
{
    SkCandidateHeap heap(*graph, SkHeapType::PRIORITY_QUEUE);
    heap.push(1);
    heap.push(2);
    heap.push(3);
    
    heap.reset();
    
    EXPECT_TRUE(heap.empty());
    EXPECT_EQ(heap.size(), 0U);
}

// ==================== 非 Kernel 节点测试 ====================

// 测试 Push Notify 节点
TEST_F(SkCandidateHeapTest, Push_NotifyNode)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    uint64_t nodeId = CreateNotifyNode(1);
    heap.push(nodeId);

    EXPECT_FALSE(heap.empty());
    EXPECT_EQ(heap.size(), 1U);
    EXPECT_FALSE(heap.HasKernelNodes());
}

// 测试 Push Wait 节点
TEST_F(SkCandidateHeapTest, Push_WaitNode)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    uint64_t nodeId = CreateWaitNode(1);
    heap.push(nodeId);

    EXPECT_FALSE(heap.empty());
    EXPECT_EQ(heap.size(), 1U);
    EXPECT_FALSE(heap.HasKernelNodes());
}

// 测试 Push Reset 节点
TEST_F(SkCandidateHeapTest, Push_ResetNode)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    uint64_t nodeId = CreateResetNode(1);
    heap.push(nodeId);

    EXPECT_FALSE(heap.empty());
    EXPECT_EQ(heap.size(), 1U);
    EXPECT_FALSE(heap.HasKernelNodes());
}

// 测试非 Kernel 节点按 nodeId 排序
TEST_F(SkCandidateHeapTest, Pop_NonKernelNodes_SortedByNodeId)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    uint64_t notify3 = CreateNotifyNode(3);
    uint64_t wait1 = CreateWaitNode(1);
    uint64_t reset2 = CreateResetNode(2);

    heap.push(notify3);
    heap.push(wait1);
    heap.push(reset2);

    // 应该按 nodeId 升序弹出
    EXPECT_EQ(heap.pop(), 1U);
    EXPECT_EQ(heap.pop(), 2U);
    EXPECT_EQ(heap.pop(), 3U);
    EXPECT_TRUE(heap.empty());
}

// ==================== Kernel 节点优先级测试 ====================

// 测试 Kernel 节点优先于非 Kernel 节点
TEST_F(SkCandidateHeapTest, Pop_KernelPrioritizedOverNonKernel)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    uint64_t notify1 = CreateNotifyNode(1);
    uint64_t kernel2 = CreateKernelNode(2);
    uint64_t wait3 = CreateWaitNode(3);

    heap.push(notify1);
    heap.push(kernel2);
    heap.push(wait3);

    // Kernel 节点应该优先弹出，即使它的 nodeId 不是最小的
    uint64_t poppedId = heap.pop();
    EXPECT_EQ(poppedId, kernel2);
}

// 测试混合节点：先弹出所有 Kernel，再弹出非 Kernel
TEST_F(SkCandidateHeapTest, Pop_MixedNodes_KernelFirst)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    uint64_t notify1 = CreateNotifyNode(1);
    uint64_t kernel2 = CreateKernelNode(2);
    uint64_t kernel3 = CreateKernelNode(3);
    uint64_t wait4 = CreateWaitNode(4);

    heap.push(notify1);
    heap.push(kernel2);
    heap.push(kernel3);
    heap.push(wait4);

    // 先弹出 kernel 节点（按 nodeId 排序）
    EXPECT_EQ(heap.pop(), 2U);
    EXPECT_EQ(heap.pop(), 3U);
    // 然后弹出非 kernel 节点（按 nodeId 排序）
    EXPECT_EQ(heap.pop(), 1U);
    EXPECT_EQ(heap.pop(), 4U);
    EXPECT_TRUE(heap.empty());
}

// ==================== Kernel Type 选择规则测试 ====================

// 测试第一次选择：没有 MIX 时选择 nodeId 最小
TEST_F(SkCandidateHeapTest, Pop_FirstSelection_NoMix_SmallestNodeId)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    uint64_t kernel3 = CreateKernelNode(3);
    uint64_t kernel1 = CreateKernelNode(1);
    uint64_t kernel2 = CreateKernelNode(2);

    heap.push(kernel3);
    heap.push(kernel1);
    heap.push(kernel2);

    // 第一次选择：没有 MIX 节点，选择 nodeId 最小的
    EXPECT_EQ(heap.pop(), 1U);
}

// 测试第一次选择：优先选择 MIX 节点
TEST_F(SkCandidateHeapTest, Pop_FirstSelection_PreferMix)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    uint64_t kernel1 = CreateKernelNode(1, 0, SkKernelType::AIC_ONLY);   // CUBE, nodeId 最小
    uint64_t kernel2 = CreateKernelNode(2, 0, SkKernelType::MIX_AIC_1_1); // MIX
    uint64_t kernel3 = CreateKernelNode(3, 0, SkKernelType::AIV_ONLY);   // VEC

    heap.push(kernel1);
    heap.push(kernel2);
    heap.push(kernel3);

    // 第一次选择：优先选择 MIX 节点，即使 nodeId 不是最小
    EXPECT_EQ(heap.pop(), 2U);
}

// 测试第一次选择：多个 MIX 时选择 id 最小的 MIX
TEST_F(SkCandidateHeapTest, Pop_FirstSelection_MultipleMix_SmallestId)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    uint64_t kernel1 = CreateKernelNode(1, 0, SkKernelType::AIC_ONLY);   // CUBE, nodeId 最小
    uint64_t kernel2 = CreateKernelNode(2, 1, SkKernelType::MIX_AIC_1_1); // MIX, stream 1
    uint64_t kernel3 = CreateKernelNode(3, 0, SkKernelType::MIX_AIC_1_2); // MIX, stream 0
    uint64_t kernel4 = CreateKernelNode(4, 0, SkKernelType::AIV_ONLY);   // VEC

    heap.push(kernel1);
    heap.push(kernel2);
    heap.push(kernel3);
    heap.push(kernel4);

    // 第一次选择：选择 id 最小的 MIX 节点 (kernel2)
    EXPECT_EQ(heap.pop(), 2U);
}

// 测试 MIX 类型后优先选择 MIX
TEST_F(SkCandidateHeapTest, Pop_AfterMix_PrioritizeMix)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    // 先 push 一个 MIX 类型节点
    uint64_t mixNode = CreateKernelNode(1, 0, SkKernelType::MIX_AIC_1_1);
    uint64_t vecNode = CreateKernelNode(2, 0, SkKernelType::AIV_ONLY);
    uint64_t anotherMix = CreateKernelNode(3, 1, SkKernelType::MIX_AIC_1_2);  // 不同流

    heap.push(mixNode);
    heap.push(vecNode);
    heap.push(anotherMix);

    // 第一次选择：优先选择 MIX 节点 (nodeId 最小的 MIX)
    EXPECT_EQ(heap.pop(), 1U);

    // 第二次选择：前序是 MIX，应该优先选不同流的 MIX 节点
    EXPECT_EQ(heap.pop(), 3U);
}

// 测试 VEC 类型后优先选择 CUBE
TEST_F(SkCandidateHeapTest, Pop_AfterVec_PrioritizeCube)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    uint64_t vecNode = CreateKernelNode(1, 0, SkKernelType::AIV_ONLY);
    uint64_t anotherVec = CreateKernelNode(2, 0, SkKernelType::AIV_ONLY);
    uint64_t cubeNode = CreateKernelNode(3, 1, SkKernelType::AIC_ONLY);  // 不同流

    heap.push(vecNode);
    heap.push(anotherVec);
    heap.push(cubeNode);

    // 第一次选择：没有 MIX 节点，选择 nodeId 最小的 VEC 节点
    EXPECT_EQ(heap.pop(), 1U);

    // 第二次选择：前序是 VEC，应该优先选不同流的 CUBE 节点
    EXPECT_EQ(heap.pop(), 3U);
}

// 测试 CUBE 类型后优先选择 VEC
TEST_F(SkCandidateHeapTest, Pop_AfterCube_PrioritizeVec)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    uint64_t cubeNode = CreateKernelNode(1, 0, SkKernelType::AIC_ONLY);
    uint64_t anotherCube = CreateKernelNode(2, 0, SkKernelType::AIC_ONLY);
    uint64_t vecNode = CreateKernelNode(3, 1, SkKernelType::AIV_ONLY);  // 不同流

    heap.push(cubeNode);
    heap.push(anotherCube);
    heap.push(vecNode);

    // 第一次选择：没有 MIX 节点，选择 nodeId 最小的 CUBE 节点
    EXPECT_EQ(heap.pop(), 1U);

    // 第二次选择：前序是 CUBE，应该优先选不同流的 VEC 节点
    EXPECT_EQ(heap.pop(), 3U);
}

// 测试优先选择不同流的节点
TEST_F(SkCandidateHeapTest, Pop_PrioritizeDifferentStream)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    uint64_t node1 = CreateKernelNode(1, 0);  // stream 0
    uint64_t node2 = CreateKernelNode(2, 0);  // stream 0
    uint64_t node3 = CreateKernelNode(3, 1);  // stream 1

    heap.push(node1);
    heap.push(node2);
    heap.push(node3);

    // 第一次选择：没有 MIX 节点，选择 nodeId 最小的 node1 (stream 0)
    EXPECT_EQ(heap.pop(), 1U);

    // 第二次选择：应该优先选不同流的 node3
    EXPECT_EQ(heap.pop(), 3U);
}

// ==================== Clear 和 Reset 测试 ====================

// 测试 Clear
TEST_F(SkCandidateHeapTest, Clear_EmptiesHeap)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    CreateKernelNode(1);
    CreateNotifyNode(2);
    heap.push(1);
    heap.push(2);

    heap.clear();

    EXPECT_TRUE(heap.empty());
    EXPECT_EQ(heap.size(), 0U);
    EXPECT_FALSE(heap.HasKernelNodes());
}

// 测试 Reset - 初始化所有变量和状态
TEST_F(SkCandidateHeapTest, Reset_InitializesAllVariablesAndState)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    
    // Push 一些节点并 Pop 以改变状态
    uint64_t kernel1 = CreateKernelNode(1, 0, SkKernelType::AIV_ONLY);
    uint64_t notify2 = CreateNotifyNode(2);
    uint64_t kernel3 = CreateKernelNode(3, 1, SkKernelType::AIC_ONLY);
    
    heap.push(kernel1);
    heap.push(notify2);
    heap.push(kernel3);
    
    // Pop 两个 kernel 节点以更新状态
    heap.pop();  // kernel1 (VEC, stream 0)
    heap.pop();  // kernel3 (CUBE, stream 1)
    
    // 验证状态已被更新
    EXPECT_EQ(heap.GetPrevKernelTypeClass(), SkCandidateHeap::KernelTypeClass::CUBE);
    EXPECT_EQ(heap.GetPrevStreamIdx(), 1U);
    EXPECT_FALSE(heap.empty());  // notify2 还在
    
    // 调用 Reset
    heap.reset();
    
    // 验证所有状态已被重置
    EXPECT_TRUE(heap.empty());
    EXPECT_EQ(heap.size(), 0U);
    EXPECT_FALSE(heap.HasKernelNodes());
    EXPECT_EQ(heap.GetPrevKernelTypeClass(), SkCandidateHeap::KernelTypeClass::OTHER);
    EXPECT_EQ(heap.GetPrevStreamIdx(), 0U);
}

// 测试 Reset 后可以正常使用
TEST_F(SkCandidateHeapTest, Reset_CanReuseAfterReset)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    
    // 第一次使用
    CreateKernelNode(1, 0, SkKernelType::AIC_ONLY);
    CreateNotifyNode(2);
    heap.push(1);
    heap.push(2);
    heap.pop();
    
    // Reset
    heap.reset();
    
    // 第二次使用
    uint64_t kernel3 = CreateKernelNode(3, 1, SkKernelType::AIV_ONLY);
    uint64_t wait4 = CreateWaitNode(4);
    
    heap.push(kernel3);
    heap.push(wait4);
    
    // 验证可以正常工作
    EXPECT_FALSE(heap.empty());
    EXPECT_EQ(heap.size(), 2U);
    EXPECT_TRUE(heap.HasKernelNodes());
    
    // 第一次选择：没有 MIX 节点，选择 nodeId 最小的 kernel 节点
    uint64_t poppedId = heap.pop();
    EXPECT_EQ(poppedId, kernel3);
    EXPECT_EQ(heap.GetPrevKernelTypeClass(), SkCandidateHeap::KernelTypeClass::VEC);
}

// 测试 ResetSelectionState
TEST_F(SkCandidateHeapTest, ResetSelectionState_ResetsState)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    CreateKernelNode(1, 0, SkKernelType::AIV_ONLY);
    heap.push(1);
    heap.pop();  // 这会更新 prevKernelTypeClass 和 prevStreamIdx

    // 验证状态已被更新
    EXPECT_EQ(heap.GetPrevKernelTypeClass(), SkCandidateHeap::KernelTypeClass::VEC);
    EXPECT_EQ(heap.GetPrevStreamIdx(), 0U);

    heap.ResetSelectionState();

    EXPECT_EQ(heap.GetPrevKernelTypeClass(), SkCandidateHeap::KernelTypeClass::OTHER);
    // isFirstSelection 应该被重置
}

// 测试 SetPrevKernelTypeClass
TEST_F(SkCandidateHeapTest, SetPrevKernelTypeClass)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    heap.SetPrevKernelTypeClass(SkCandidateHeap::KernelTypeClass::MIX);
    EXPECT_EQ(heap.GetPrevKernelTypeClass(), SkCandidateHeap::KernelTypeClass::MIX);
}

// 测试 SetPrevStreamIdx
TEST_F(SkCandidateHeapTest, SetPrevStreamIdx)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    heap.SetPrevStreamIdx(5);
    EXPECT_EQ(heap.GetPrevStreamIdx(), 5U);
}

// ==================== 辅助函数测试 ====================

// 测试 IsMixKernelType
TEST_F(SkCandidateHeapTest, IsMixKernelType_ValidTypes_ReturnTrue)
{
    EXPECT_TRUE(SkCandidateHeap::IsMixKernelType(SkKernelType::MIX_AIC_1_1));
    EXPECT_TRUE(SkCandidateHeap::IsMixKernelType(SkKernelType::MIX_AIC_1_2));
}

TEST_F(SkCandidateHeapTest, IsMixKernelType_InvalidTypes_ReturnFalse)
{
    EXPECT_FALSE(SkCandidateHeap::IsMixKernelType(SkKernelType::AIC_ONLY));
    EXPECT_FALSE(SkCandidateHeap::IsMixKernelType(SkKernelType::AIV_ONLY));
    EXPECT_FALSE(SkCandidateHeap::IsMixKernelType(SkKernelType::MIX_AIV_1_0));
    EXPECT_FALSE(SkCandidateHeap::IsMixKernelType(SkKernelType::MIX_AIC_1_0));
    EXPECT_FALSE(SkCandidateHeap::IsMixKernelType(SkKernelType::DEFAULT));
}

// 测试 IsVecKernelType
TEST_F(SkCandidateHeapTest, IsVecKernelType_ValidTypes_ReturnTrue)
{
    EXPECT_TRUE(SkCandidateHeap::IsVecKernelType(SkKernelType::AIV_ONLY));
    EXPECT_TRUE(SkCandidateHeap::IsVecKernelType(SkKernelType::MIX_AIV_1_0));
}

TEST_F(SkCandidateHeapTest, IsVecKernelType_InvalidTypes_ReturnFalse)
{
    EXPECT_FALSE(SkCandidateHeap::IsVecKernelType(SkKernelType::AIC_ONLY));
    EXPECT_FALSE(SkCandidateHeap::IsVecKernelType(SkKernelType::MIX_AIC_1_0));
    EXPECT_FALSE(SkCandidateHeap::IsVecKernelType(SkKernelType::MIX_AIC_1_1));
    EXPECT_FALSE(SkCandidateHeap::IsVecKernelType(SkKernelType::MIX_AIC_1_2));
    EXPECT_FALSE(SkCandidateHeap::IsVecKernelType(SkKernelType::DEFAULT));
}

// 测试 IsCubeKernelType
TEST_F(SkCandidateHeapTest, IsCubeKernelType_ValidTypes_ReturnTrue)
{
    EXPECT_TRUE(SkCandidateHeap::IsCubeKernelType(SkKernelType::AIC_ONLY));
    EXPECT_TRUE(SkCandidateHeap::IsCubeKernelType(SkKernelType::MIX_AIC_1_0));
}

TEST_F(SkCandidateHeapTest, IsCubeKernelType_InvalidTypes_ReturnFalse)
{
    EXPECT_FALSE(SkCandidateHeap::IsCubeKernelType(SkKernelType::AIV_ONLY));
    EXPECT_FALSE(SkCandidateHeap::IsCubeKernelType(SkKernelType::MIX_AIV_1_0));
    EXPECT_FALSE(SkCandidateHeap::IsCubeKernelType(SkKernelType::MIX_AIC_1_1));
    EXPECT_FALSE(SkCandidateHeap::IsCubeKernelType(SkKernelType::MIX_AIC_1_2));
    EXPECT_FALSE(SkCandidateHeap::IsCubeKernelType(SkKernelType::DEFAULT));
}

// 测试 GetKernelTypeClass
TEST_F(SkCandidateHeapTest, GetKernelTypeClass_MixTypes_ReturnMix)
{
    EXPECT_EQ(static_cast<int>(SkCandidateHeap::GetKernelTypeClass(SkKernelType::MIX_AIC_1_1)),
              static_cast<int>(SkCandidateHeap::KernelTypeClass::MIX));
    EXPECT_EQ(static_cast<int>(SkCandidateHeap::GetKernelTypeClass(SkKernelType::MIX_AIC_1_2)),
              static_cast<int>(SkCandidateHeap::KernelTypeClass::MIX));
}

TEST_F(SkCandidateHeapTest, GetKernelTypeClass_VecTypes_ReturnVec)
{
    EXPECT_EQ(static_cast<int>(SkCandidateHeap::GetKernelTypeClass(SkKernelType::AIV_ONLY)),
              static_cast<int>(SkCandidateHeap::KernelTypeClass::VEC));
    EXPECT_EQ(static_cast<int>(SkCandidateHeap::GetKernelTypeClass(SkKernelType::MIX_AIV_1_0)),
              static_cast<int>(SkCandidateHeap::KernelTypeClass::VEC));
}

TEST_F(SkCandidateHeapTest, GetKernelTypeClass_CubeTypes_ReturnCube)
{
    EXPECT_EQ(static_cast<int>(SkCandidateHeap::GetKernelTypeClass(SkKernelType::AIC_ONLY)),
              static_cast<int>(SkCandidateHeap::KernelTypeClass::CUBE));
    EXPECT_EQ(static_cast<int>(SkCandidateHeap::GetKernelTypeClass(SkKernelType::MIX_AIC_1_0)),
              static_cast<int>(SkCandidateHeap::KernelTypeClass::CUBE));
}

TEST_F(SkCandidateHeapTest, GetKernelTypeClass_DefaultType_ReturnOther)
{
    EXPECT_EQ(static_cast<int>(SkCandidateHeap::GetKernelTypeClass(SkKernelType::DEFAULT)),
              static_cast<int>(SkCandidateHeap::KernelTypeClass::OTHER));
}

// ==================== 复杂场景测试 ====================

// 测试复杂混合场景
TEST_F(SkCandidateHeapTest, ComplexScenario_MixedNodeTypes)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);

    // 创建多种类型的节点
    uint64_t kernel1 = CreateKernelNode(1, 0, SkKernelType::AIV_ONLY);   // VEC, stream 0
    uint64_t kernel2 = CreateKernelNode(2, 1, SkKernelType::AIC_ONLY);   // CUBE, stream 1
    uint64_t kernel3 = CreateKernelNode(3, 0, SkKernelType::AIV_ONLY);   // VEC, stream 0
    uint64_t notify4 = CreateNotifyNode(4, 0);
    uint64_t wait5 = CreateWaitNode(5, 1);
    uint64_t kernel6 = CreateKernelNode(6, 1, SkKernelType::AIC_ONLY);   // CUBE, stream 1

    heap.push(kernel1);
    heap.push(kernel2);
    heap.push(kernel3);
    heap.push(notify4);
    heap.push(wait5);
    heap.push(kernel6);

    // 弹出顺序：
    // 1. kernel1 (VEC, stream 0) - 第一次选择，没有 MIX，nodeId 最小
    // 2. kernel2 (CUBE, stream 1) - 前序是 VEC，优先选不同流的 CUBE
    // 3. kernel3 (VEC, stream 0) - 前序是 CUBE，优先选不同流的 VEC
    // 4. kernel6 (CUBE, stream 1) - 前序是 VEC，优先选不同流的 CUBE
    // 5. notify4 - 非 kernel 节点，按 nodeId 排序
    // 6. wait5 - 非 kernel 节点，按 nodeId 排序

    EXPECT_EQ(heap.pop(), 1U);
    EXPECT_EQ(heap.pop(), 2U);
    EXPECT_EQ(heap.pop(), 3U);
    EXPECT_EQ(heap.pop(), 6U);
    EXPECT_EQ(heap.pop(), 4U);
    EXPECT_EQ(heap.pop(), 5U);
    EXPECT_TRUE(heap.empty());
}

// 测试所有节点都在同一流的情况
TEST_F(SkCandidateHeapTest, AllSameStream_SelectByRuleThenNodeId)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);

    // 所有节点都在 stream 0
    uint64_t kernel1 = CreateKernelNode(1, 0, SkKernelType::AIV_ONLY);   // VEC
    uint64_t kernel2 = CreateKernelNode(2, 0, SkKernelType::AIC_ONLY);   // CUBE
    uint64_t kernel3 = CreateKernelNode(3, 0, SkKernelType::AIV_ONLY);   // VEC

    heap.push(kernel1);
    heap.push(kernel2);
    heap.push(kernel3);

    // 第一次选择：没有 MIX 节点，选择 nodeId 最小的 kernel1
    EXPECT_EQ(heap.pop(), 1U);

    // 前序是 VEC，优先选 CUBE，但都在同一流
    // 应该选择 kernel2 (CUBE)
    EXPECT_EQ(heap.pop(), 2U);

    // 前序是 CUBE，优先选 VEC
    EXPECT_EQ(heap.pop(), 3U);
}

// ==================== 构造函数测试 ====================

// 测试构造函数初始化状态
TEST_F(SkCandidateHeapTest, Constructor_InitializesState)
{
    SkCandidateHeap heap(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    
    EXPECT_EQ(heap.GetPrevKernelTypeClass(), SkCandidateHeap::KernelTypeClass::OTHER);
    EXPECT_EQ(heap.GetPrevStreamIdx(), 0U);
}

// 测试移动构造
TEST_F(SkCandidateHeapTest, MoveConstructor_Works)
{
    SkCandidateHeap heap1(*graph, SkHeapType::CUSTOMIZE_QUEUE);
    uint64_t kernel1 = CreateKernelNode(1);
    heap1.push(kernel1);
    heap1.pop();

    SkCandidateHeap heap2(std::move(heap1));
    EXPECT_TRUE(heap2.empty());
    EXPECT_EQ(heap2.GetPrevKernelTypeClass(), SkCandidateHeap::KernelTypeClass::CUBE);
}
