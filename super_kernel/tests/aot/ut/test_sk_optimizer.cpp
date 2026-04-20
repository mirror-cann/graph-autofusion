/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
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
#include "sk_optimizer.h"
#include "sk_options_manager.h"
#include "sk_node.h"
#include "sk_candidate_heap.h"

class SkOptimizerTaskReorderTest : public testing::Test {
protected:
    void SetUp() override
    {
        graph = std::make_unique<SuperKernelGraph>();
        opts = std::make_unique<SuperKernelOptionsManager>();
        optimizer = std::make_unique<SuperKernelOptimizer>(*opts);
    }

    SuperKernelBaseNode* CreateKernelNode(uint64_t nodeId, uint32_t streamIdx)
    {
        auto node = std::make_unique<SuperKernelKernelNode>(
            nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, streamIdx, INVALID_STREAM_ID, INVALID_TASK_ID);
        node->SetNodeType(SkNodeType::NODE_KERNEL);
        node->SetNodeId(nodeId);
        node->SetPreNodeId(INVALID_TASK_ID);
        node->SetNextNodeId(INVALID_TASK_ID);
        node->nodeInfos.kernelInfos.funcName = "k";
        auto* ptr = node.get();
        graph->graphMap[nodeId] = std::move(node);
        return ptr;
    }

    SuperKernelBaseNode* CreateWaitNode(uint64_t nodeId, uint32_t streamIdx)
    {
        auto node = std::make_unique<SuperKernelMemoryNode>(
            nullptr, ACL_MODEL_RI_TASK_EVENT_WAIT, 0, streamIdx, INVALID_STREAM_ID, INVALID_TASK_ID);
        node->SetNodeType(SkNodeType::NODE_WAIT);
        node->SetNodeId(nodeId);
        node->SetPreNodeId(INVALID_TASK_ID);
        node->SetNextNodeId(INVALID_TASK_ID);
        auto* ptr = node.get();
        graph->graphMap[nodeId] = std::move(node);
        return ptr;
    }

    std::vector<uint64_t> ToNodeIds(const std::vector<SuperKernelBaseNode*>& nodes)
    {
        std::vector<uint64_t> nodeIds;
        nodeIds.reserve(nodes.size());
        for (const auto* node : nodes) {
            nodeIds.push_back(node == nullptr ? INVALID_TASK_ID : node->GetNodeId());
        }
        return nodeIds;
    }

    void EnableCustomizeQueueReorder()
    {
        auto option = std::make_unique<NumberOptOption>(
            "auto_op_parallel", aclskOptionType::AUTO_OP_PARALLEL,
            static_cast<uint32_t>(SkHeapType::CUSTOMIZE_QUEUE), 0, 0xFFFFFFFF);
        opts->AddOption(std::move(option));
    }

    std::unique_ptr<SuperKernelGraph> graph;
    std::unique_ptr<SuperKernelOptionsManager> opts;
    std::unique_ptr<SuperKernelOptimizer> optimizer;
};

TEST_F(SkOptimizerTaskReorderTest, ReorderWaitNodesForTaskBuild_DefaultDisabled_KeepOriginalOrder)
{
    std::vector<SuperKernelBaseNode*> taskNodes = {
        CreateKernelNode(1, 0),
        CreateWaitNode(2, 1),
        CreateKernelNode(3, 1),
    };

    std::vector<SuperKernelBaseNode*> reorderedTaskNodes = optimizer->ReorderWaitNodesForTaskBuild(taskNodes);

    EXPECT_EQ(ToNodeIds(reorderedTaskNodes), (std::vector<uint64_t>{1, 2, 3}));
}

TEST_F(SkOptimizerTaskReorderTest, ReorderWaitNodesForTaskBuild_CustomizeQueue_WaitMovesBeforeLaterSameStreamKernel)
{
    EnableCustomizeQueueReorder();
    std::vector<SuperKernelBaseNode*> taskNodes = {
        CreateKernelNode(1, 0),
        CreateWaitNode(2, 1),
        CreateKernelNode(3, 1),
    };

    std::vector<SuperKernelBaseNode*> reorderedTaskNodes = optimizer->ReorderWaitNodesForTaskBuild(taskNodes);

    EXPECT_EQ(ToNodeIds(reorderedTaskNodes), (std::vector<uint64_t>{1, 2, 3}));
}

TEST_F(SkOptimizerTaskReorderTest, ReorderWaitNodesForTaskBuild_CustomizeQueue_WaitWithoutLaterSameStreamKernelStaysInPlace)
{
    EnableCustomizeQueueReorder();
    std::vector<SuperKernelBaseNode*> taskNodes = {
        CreateKernelNode(1, 0),
        CreateWaitNode(2, 1),
        CreateKernelNode(3, 0),
    };

    std::vector<SuperKernelBaseNode*> reorderedTaskNodes = optimizer->ReorderWaitNodesForTaskBuild(taskNodes);

    EXPECT_EQ(ToNodeIds(reorderedTaskNodes), (std::vector<uint64_t>{1, 2, 3}));
}

TEST_F(SkOptimizerTaskReorderTest, ReorderWaitNodesForTaskBuild_CustomizeQueue_ConsecutiveWaitsSameStreamKeepRelativeOrder)
{
    EnableCustomizeQueueReorder();
    std::vector<SuperKernelBaseNode*> taskNodes = {
        CreateWaitNode(10, 1),
        CreateKernelNode(20, 0),
        CreateWaitNode(11, 1),
        CreateKernelNode(12, 1),
    };

    std::vector<SuperKernelBaseNode*> reorderedTaskNodes = optimizer->ReorderWaitNodesForTaskBuild(taskNodes);

    EXPECT_EQ(ToNodeIds(reorderedTaskNodes), (std::vector<uint64_t>{20, 10, 11, 12}));
}

TEST_F(SkOptimizerTaskReorderTest, ReorderWaitNodesForTaskBuild_CustomizeQueue_ConsecutiveWaitsDifferentStreamsAttachToOwnKernel)
{
    EnableCustomizeQueueReorder();
    std::vector<SuperKernelBaseNode*> taskNodes = {
        CreateWaitNode(10, 1),
        CreateWaitNode(11, 2),
        CreateKernelNode(20, 0),
        CreateKernelNode(12, 1),
        CreateKernelNode(13, 2),
    };

    std::vector<SuperKernelBaseNode*> reorderedTaskNodes = optimizer->ReorderWaitNodesForTaskBuild(taskNodes);

    EXPECT_EQ(ToNodeIds(reorderedTaskNodes), (std::vector<uint64_t>{20, 10, 12, 11, 13}));
}

TEST_F(SkOptimizerTaskReorderTest, ReorderWaitNodesForTaskBuild_CustomizeQueue_OutOfOrderTargetsDoNotFlushPrefix)
{
    EnableCustomizeQueueReorder();
    std::vector<SuperKernelBaseNode*> taskNodes = {
        CreateWaitNode(10, 1),
        CreateWaitNode(11, 2),
        CreateKernelNode(13, 2),
        CreateKernelNode(12, 1),
    };

    std::vector<SuperKernelBaseNode*> reorderedTaskNodes = optimizer->ReorderWaitNodesForTaskBuild(taskNodes);

    EXPECT_EQ(ToNodeIds(reorderedTaskNodes), (std::vector<uint64_t>{11, 13, 10, 12}));
}

TEST_F(SkOptimizerTaskReorderTest, ReorderWaitNodesForTaskBuild_CustomizeQueue_AllNodesSameStream_KeepOriginalOrder)
{
    EnableCustomizeQueueReorder();
    std::vector<SuperKernelBaseNode*> taskNodes = {
        CreateKernelNode(1, 3),
        CreateWaitNode(2, 3),
        CreateKernelNode(3, 3),
        CreateWaitNode(4, 3),
        CreateKernelNode(5, 3),
    };

    std::vector<SuperKernelBaseNode*> reorderedTaskNodes = optimizer->ReorderWaitNodesForTaskBuild(taskNodes);

    EXPECT_EQ(ToNodeIds(reorderedTaskNodes), (std::vector<uint64_t>{1, 2, 3, 4, 5}));
}

TEST_F(SkOptimizerTaskReorderTest, ReorderWaitNodesForTaskBuild_CustomizeQueue_MultipleWaitsTargetSameKernelKeepRelativeOrder)
{
    EnableCustomizeQueueReorder();
    std::vector<SuperKernelBaseNode*> taskNodes = {
        CreateKernelNode(1, 0),
        CreateWaitNode(2, 2),
        CreateKernelNode(3, 1),
        CreateWaitNode(4, 2),
        CreateKernelNode(5, 2),
    };

    std::vector<SuperKernelBaseNode*> reorderedTaskNodes = optimizer->ReorderWaitNodesForTaskBuild(taskNodes);

    EXPECT_EQ(ToNodeIds(reorderedTaskNodes), (std::vector<uint64_t>{1, 3, 2, 4, 5}));
}

TEST_F(SkOptimizerTaskReorderTest, ReorderWaitNodesForTaskBuild_CustomizeQueue_NonWaitRelativeOrderPreserved)
{
    EnableCustomizeQueueReorder();
    std::vector<SuperKernelBaseNode*> taskNodes = {
        CreateKernelNode(1, 0),
        CreateWaitNode(2, 1),
        CreateKernelNode(3, 0),
        CreateKernelNode(4, 1),
        CreateKernelNode(5, 0),
    };

    std::vector<SuperKernelBaseNode*> reorderedTaskNodes = optimizer->ReorderWaitNodesForTaskBuild(taskNodes);

    EXPECT_EQ(ToNodeIds(reorderedTaskNodes), (std::vector<uint64_t>{1, 3, 2, 4, 5}));
}
