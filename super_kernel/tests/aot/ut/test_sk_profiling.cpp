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
#include <memory>
#include <cstring>

#define private public
#define protected public
#include "sk_graph.h"
#include "sk_node.h"
#include "sk_optimizer.h"
#include "sk_options_manager.h"
#include "sk_event_recorder.h"

static std::unique_ptr<SuperKernelKernelNode> CreateKernelNodeWithCacheInfo(
    uint64_t nodeId, uint32_t tensorNum = 0, uint64_t attrId = 0, uint32_t opFlag = 0) {
    // Create node using factory method with correct constructor parameters
    auto node = static_cast<SuperKernelKernelNode*>(
        SuperKernelNodeFactory::CreateNode(
            std::make_unique<aclmdlRITask>(nullptr),
            ACL_MODEL_RI_TASK_KERNEL,
            nodeId, 0, INVALID_STREAM_ID, INVALID_TASK_ID).release());
    
    std::unique_ptr<SuperKernelKernelNode> kernelNode(node);
    kernelNode->SetNodeId(nodeId);
    
    // Set nodeType directly since InitNode() requires real task object
    kernelNode->SetNodeType(SkNodeType::NODE_KERNEL);

    auto& nodeInfos = const_cast<NodeInfos&>(kernelNode->GetNodeInfos());
    nodeInfos.kernelInfos.numBlocks = 8;
    nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;

    if (tensorNum > 0) {
        size_t cacheInfoSize = sizeof(CacheopInfoBasic) + tensorNum * sizeof(MsrofTensorData);
        void* cacheInfoMem = malloc(cacheInfoSize);
        if (cacheInfoMem != nullptr) {
            memset_s(cacheInfoMem, cacheInfoSize, 0, cacheInfoSize);
            CacheopInfoBasic* cacheInfo = static_cast<CacheopInfoBasic*>(cacheInfoMem);
            cacheInfo->tensorNum = tensorNum;
            cacheInfo->attrId = attrId;
            cacheInfo->opFlag = opFlag;
            nodeInfos.kernelInfos.opInfoPtr = cacheInfo;
            nodeInfos.kernelInfos.opInfoSize = cacheInfoSize;
        }
    }
    kernelNode->isFusible = true;
    return kernelNode;
}

class SkProfilingTest : public testing::Test {
protected:
    void SetUp() override {
        opts = std::make_unique<SuperKernelOptionsManager>();
    }
    void TearDown() override {
        for (auto* node : scopeInfo.extInfo.filteredNodes) {
            if (node != nullptr) {
                auto& nodeInfos = const_cast<NodeInfos&>(node->GetNodeInfos());
                if (nodeInfos.kernelInfos.opInfoPtr != nullptr) {
                    free(const_cast<void*>(nodeInfos.kernelInfos.opInfoPtr));
                    nodeInfos.kernelInfos.opInfoPtr = nullptr;
                }
                delete node;
            }
        }
        scopeInfo.extInfo.filteredNodes.clear();
        // launchInfo.cacheInfo is managed by graph.shapeInfoPtrList, cleared automatically
        graph.ClearShapeInfoPtrList();
    }
    std::unique_ptr<SuperKernelOptionsManager> opts;
    SuperKernelScopeInfo scopeInfo;
    SkLaunchInfo launchInfo;
    SuperKernelGraph graph;
};

// ============================================================================
// numBlocks 编码测试
// ============================================================================

TEST_F(SkProfilingTest, NumBlocks_Encoding_Mix11) {
    scopeInfo.extInfo.filteredNodes.push_back(CreateKernelNodeWithCacheInfo(1, 1).release());
    launchInfo.entryInfo.numBlocks = 8;
    launchInfo.entryInfo.entryType = SkKernelType::MIX_AIC_1_1;

    SkProfiling(scopeInfo, launchInfo, graph);

    CacheopInfoBasic* result = static_cast<CacheopInfoBasic*>(launchInfo.cacheInfo);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->numBlocks, (1U << 16) | 8);
}

TEST_F(SkProfilingTest, NumBlocks_Encoding_Mix12) {
    scopeInfo.extInfo.filteredNodes.push_back(CreateKernelNodeWithCacheInfo(1, 1).release());
    launchInfo.entryInfo.numBlocks = 4;
    launchInfo.entryInfo.entryType = SkKernelType::MIX_AIC_1_2;

    SkProfiling(scopeInfo, launchInfo, graph);

    CacheopInfoBasic* result = static_cast<CacheopInfoBasic*>(launchInfo.cacheInfo);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->numBlocks, (2U << 16) | 4);
}

TEST_F(SkProfilingTest, NumBlocks_NoEncoding_AicOnly) {
    scopeInfo.extInfo.filteredNodes.push_back(CreateKernelNodeWithCacheInfo(1, 1).release());
    launchInfo.entryInfo.numBlocks = 64;
    launchInfo.entryInfo.entryType = SkKernelType::AIC_ONLY;

    SkProfiling(scopeInfo, launchInfo, graph);

    CacheopInfoBasic* result = static_cast<CacheopInfoBasic*>(launchInfo.cacheInfo);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->numBlocks, 64);
}

TEST_F(SkProfilingTest, NumBlocks_NoEncoding_AivOnly) {
    scopeInfo.extInfo.filteredNodes.push_back(CreateKernelNodeWithCacheInfo(1, 1).release());
    launchInfo.entryInfo.numBlocks = 128;
    launchInfo.entryInfo.entryType = SkKernelType::AIV_ONLY;

    SkProfiling(scopeInfo, launchInfo, graph);

    CacheopInfoBasic* result = static_cast<CacheopInfoBasic*>(launchInfo.cacheInfo);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->numBlocks, 128);
}

TEST_F(SkProfilingTest, NumBlocks_Boundary_Zero) {
    scopeInfo.extInfo.filteredNodes.push_back(CreateKernelNodeWithCacheInfo(1, 1).release());
    launchInfo.entryInfo.numBlocks = 0;
    launchInfo.entryInfo.entryType = SkKernelType::MIX_AIC_1_1;

    SkProfiling(scopeInfo, launchInfo, graph);

    CacheopInfoBasic* result = static_cast<CacheopInfoBasic*>(launchInfo.cacheInfo);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->numBlocks, static_cast<uint32_t>(1 << 16));  // (1 << 16) | 0
}

TEST_F(SkProfilingTest, NumBlocks_Boundary_MaxUint16) {
    scopeInfo.extInfo.filteredNodes.push_back(CreateKernelNodeWithCacheInfo(1, 1).release());
    launchInfo.entryInfo.numBlocks = 65535;
    launchInfo.entryInfo.entryType = SkKernelType::MIX_AIC_1_1;

    SkProfiling(scopeInfo, launchInfo, graph);

    CacheopInfoBasic* result = static_cast<CacheopInfoBasic*>(launchInfo.cacheInfo);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->numBlocks, (1U << 16) | 65535);
}

// ============================================================================
// CacheopInfoBasic 字段测试
// ============================================================================

// taskType 字段测试
TEST_F(SkProfilingTest, TaskType_AlwaysMixAic) {
    scopeInfo.extInfo.filteredNodes.push_back(CreateKernelNodeWithCacheInfo(1, 0).release());
    launchInfo.entryInfo.numBlocks = 8;
    launchInfo.entryInfo.entryType = SkKernelType::AIC_ONLY;

    SkProfiling(scopeInfo, launchInfo, graph);

    CacheopInfoBasic* result = static_cast<CacheopInfoBasic*>(launchInfo.cacheInfo);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->taskType, MSPROF_GE_TASK_TYPE_MIX_AIC);
}

// nodeId 字段测试 (基于 entryType)
TEST_F(SkProfilingTest, NodeId_Mix11) {
    scopeInfo.extInfo.filteredNodes.push_back(CreateKernelNodeWithCacheInfo(1, 0).release());
    launchInfo.entryInfo.numBlocks = 8;
    launchInfo.entryInfo.entryType = SkKernelType::MIX_AIC_1_1;
    launchInfo.skFuncName = "Unknown";

    SkProfiling(scopeInfo, launchInfo, graph);

    CacheopInfoBasic* result = static_cast<CacheopInfoBasic*>(launchInfo.cacheInfo);
    ASSERT_NE(result, nullptr);
    uint64_t expectedNodeId = MsprofStr2Id("Unknown", strlen("Unknown"));
    EXPECT_EQ(result->nodeId, expectedNodeId);
}

// opType 字段测试 (固定为 "SuperKernel")
TEST_F(SkProfilingTest, OpType_AlwaysSuperKernel) {
    scopeInfo.extInfo.filteredNodes.push_back(CreateKernelNodeWithCacheInfo(1, 0).release());
    launchInfo.entryInfo.numBlocks = 8;
    launchInfo.entryInfo.entryType = SkKernelType::AIC_ONLY;

    SkProfiling(scopeInfo, launchInfo, graph);

    CacheopInfoBasic* result = static_cast<CacheopInfoBasic*>(launchInfo.cacheInfo);
    ASSERT_NE(result, nullptr);
    uint64_t expectedOpType = MsprofStr2Id("SuperKernel", strlen("SuperKernel"));
    EXPECT_EQ(result->opType, expectedOpType);
}

// attrId 字段测试
TEST_F(SkProfilingTest, AttrId_SingleNode) {
    scopeInfo.extInfo.filteredNodes.push_back(CreateKernelNodeWithCacheInfo(1, 1, 12345, 0).release());
    launchInfo.entryInfo.numBlocks = 8;
    launchInfo.entryInfo.entryType = SkKernelType::AIC_ONLY;

    SkProfiling(scopeInfo, launchInfo, graph);

    CacheopInfoBasic* result = static_cast<CacheopInfoBasic*>(launchInfo.cacheInfo);
    ASSERT_NE(result, nullptr);
    uint64_t expectedAttrId = MsprofStr2Id("unknown", strlen("unknown"));
    EXPECT_EQ(result->attrId, expectedAttrId);
}

TEST_F(SkProfilingTest, AttrId_MultipleNodes) {
    scopeInfo.extInfo.filteredNodes.push_back(CreateKernelNodeWithCacheInfo(1, 1, 100, 0).release());
    scopeInfo.extInfo.filteredNodes.push_back(CreateKernelNodeWithCacheInfo(2, 1, 200, 0).release());
    launchInfo.entryInfo.numBlocks = 8;
    launchInfo.entryInfo.entryType = SkKernelType::AIC_ONLY;

    SkProfiling(scopeInfo, launchInfo, graph);

    CacheopInfoBasic* result = static_cast<CacheopInfoBasic*>(launchInfo.cacheInfo);
    ASSERT_NE(result, nullptr);
    // attrId is combined as string "100|200" then hashed
    uint64_t expectedAttrId = MsprofStr2Id("unknown|unknown", strlen("unknown|unknown"));
    EXPECT_EQ(result->attrId, expectedAttrId);
}

// opFlag 字段测试
TEST_F(SkProfilingTest, OpFlag_SingleNode) {
    scopeInfo.extInfo.filteredNodes.push_back(CreateKernelNodeWithCacheInfo(1, 1, 0, 5).release());
    launchInfo.entryInfo.numBlocks = 8;
    launchInfo.entryInfo.entryType = SkKernelType::AIC_ONLY;

    SkProfiling(scopeInfo, launchInfo, graph);

    CacheopInfoBasic* result = static_cast<CacheopInfoBasic*>(launchInfo.cacheInfo);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->opFlag, static_cast<uint32_t>(1));
}

TEST_F(SkProfilingTest, OpFlag_Aggregation) {
    scopeInfo.extInfo.filteredNodes.push_back(CreateKernelNodeWithCacheInfo(1, 1, 0, 1).release());
    scopeInfo.extInfo.filteredNodes.push_back(CreateKernelNodeWithCacheInfo(2, 1, 0, 2).release());
    scopeInfo.extInfo.filteredNodes.push_back(CreateKernelNodeWithCacheInfo(3, 1, 0, 4).release());
    launchInfo.entryInfo.numBlocks = 8;
    launchInfo.entryInfo.entryType = SkKernelType::AIC_ONLY;

    SkProfiling(scopeInfo, launchInfo, graph);

    CacheopInfoBasic* result = static_cast<CacheopInfoBasic*>(launchInfo.cacheInfo);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->opFlag, static_cast<uint32_t>(1));  // First node's opFlag
}

// tensorNum 字段测试
TEST_F(SkProfilingTest, TensorNum_EmptyNodes) {
    launchInfo.entryInfo.numBlocks = 8;
    launchInfo.entryInfo.entryType = SkKernelType::AIC_ONLY;

    SkProfiling(scopeInfo, launchInfo, graph);

    CacheopInfoBasic* result = static_cast<CacheopInfoBasic*>(launchInfo.cacheInfo);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tensorNum, 0);
}

TEST_F(SkProfilingTest, TensorNum_SingleNode) {
    scopeInfo.extInfo.filteredNodes.push_back(CreateKernelNodeWithCacheInfo(1, 3).release());
    launchInfo.entryInfo.numBlocks = 8;
    launchInfo.entryInfo.entryType = SkKernelType::AIC_ONLY;

    SkProfiling(scopeInfo, launchInfo, graph);

    CacheopInfoBasic* result = static_cast<CacheopInfoBasic*>(launchInfo.cacheInfo);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tensorNum, 3);
}

TEST_F(SkProfilingTest, TensorNum_MultipleNodes) {
    scopeInfo.extInfo.filteredNodes.push_back(CreateKernelNodeWithCacheInfo(1, 2).release());
    scopeInfo.extInfo.filteredNodes.push_back(CreateKernelNodeWithCacheInfo(2, 3).release());
    launchInfo.entryInfo.numBlocks = 8;
    launchInfo.entryInfo.entryType = SkKernelType::AIC_ONLY;

    SkProfiling(scopeInfo, launchInfo, graph);

    CacheopInfoBasic* result = static_cast<CacheopInfoBasic*>(launchInfo.cacheInfo);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tensorNum, 5);  // 2 + 3
}

// reserve 字段测试
TEST_F(SkProfilingTest, Reserve_DefaultZero) {
    scopeInfo.extInfo.filteredNodes.push_back(CreateKernelNodeWithCacheInfo(1, 0).release());
    launchInfo.entryInfo.numBlocks = 8;
    launchInfo.entryInfo.entryType = SkKernelType::AIC_ONLY;

    SkProfiling(scopeInfo, launchInfo, graph);

    CacheopInfoBasic* result = static_cast<CacheopInfoBasic*>(launchInfo.cacheInfo);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->reserve2, static_cast<uint64_t>(0));
}
