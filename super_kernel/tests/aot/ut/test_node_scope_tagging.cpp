/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file test_node_scope_tagging.cpp
 * \brief Unit tests for verifying scope tagging functionality on nodes
 */

#include <gtest/gtest.h>
#include <memory>
#include <bitset>

#include "sk_node.h"
#include "sk_types.h"

class TestNodeScopeTagging : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }

    // Helper function to create a kernel node
    std::unique_ptr<SuperKernelKernelNode> CreateKernelNode(uint64_t nodeId = 1, uint32_t streamIdx = 0) {
        auto task = std::make_unique<aclrtTask>();
        auto node = std::make_unique<SuperKernelKernelNode>(
            std::move(task),
            SkNodeType::NODE_KERNEL,
            nodeId,
            streamIdx,
            INVALID_TASK_ID
        );
        node->SetNodeId(nodeId);
        return node;
    }
};

// Test 1: Verify initial state of scope flags
TEST_F(TestNodeScopeTagging, InitialScopeFlagsAreZero) {
    auto node = CreateKernelNode(1, 0);
    
    EXPECT_EQ(node->GetScopeBitFlags().count(), 0);
    EXPECT_FALSE(node->GetScopeBitFlags().any());
    EXPECT_TRUE(node->GetScopeBitFlags().none());
    for (size_t i = 0; i < MAX_SCOPE_NUM; ++i) {
        EXPECT_FALSE(node->GetScopeBitFlags().test(i));
    }
}

// Test 2: Verify setting single scope bit flag
TEST_F(TestNodeScopeTagging, SetSingleScopeBitFlag) {
    auto node = CreateKernelNode(1, 0);
    
    std::bitset<MAX_SCOPE_NUM> flags;
    flags.set(0);
    node->SetScopeBitFlags(flags);
    
    EXPECT_TRUE(node->GetScopeBitFlags().test(0));
    EXPECT_EQ(node->GetScopeBitFlags().count(), 1);
    EXPECT_FALSE(node->GetScopeBitFlags().test(1));
}

// Test 3: Verify setting multiple scope bit flags
TEST_F(TestNodeScopeTagging, SetMultipleScopeBitFlags) {
    auto node = CreateKernelNode(1, 0);
    
    std::bitset<MAX_SCOPE_NUM> flags;
    flags.set(0);
    flags.set(2);
    flags.set(5);
    node->SetScopeBitFlags(flags);
    
    EXPECT_TRUE(node->GetScopeBitFlags().test(0));
    EXPECT_FALSE(node->GetScopeBitFlags().test(1));
    EXPECT_TRUE(node->GetScopeBitFlags().test(2));
    EXPECT_FALSE(node->GetScopeBitFlags().test(3));
    EXPECT_FALSE(node->GetScopeBitFlags().test(4));
    EXPECT_TRUE(node->GetScopeBitFlags().test(5));
    EXPECT_EQ(node->GetScopeBitFlags().count(), 3);
}

// Test 4: Verify clearing scope bit flags
TEST_F(TestNodeScopeTagging, ClearScopeBitFlags) {
    auto node = CreateKernelNode(1, 0);
    
    std::bitset<MAX_SCOPE_NUM> flags;
    flags.set(0);
    flags.set(1);
    flags.set(2);
    node->SetScopeBitFlags(flags);
    
    EXPECT_EQ(node->GetScopeBitFlags().count(), 3);
    
    node->ClearScopeBitFlags();
    
    EXPECT_EQ(node->GetScopeBitFlags().count(), 0);
    EXPECT_TRUE(node->GetScopeBitFlags().none());
}

// Test 5: Verify setting isScopeNode flag
TEST_F(TestNodeScopeTagging, SetIsScopeNodeFlag) {
    auto node = CreateKernelNode(1, 0);
    
    EXPECT_FALSE(node->IsScopeNode());
    
    node->SetIsScopeNode(true);
    
    EXPECT_TRUE(node->IsScopeNode());
    
    node->SetIsScopeNode(false);
    
    EXPECT_FALSE(node->IsScopeNode());
}

// Test 6: Verify hasScopeFlags is derived from scopeBitFlags
TEST_F(TestNodeScopeTagging, HasScopeFlagsDerivedFromScopeBitFlags) {
    auto node = CreateKernelNode(1, 0);

    EXPECT_FALSE(node->HasScopeFlags());

    // Setting scopeBitFlags should automatically set hasScopeFlags
    std::bitset<MAX_SCOPE_NUM> flags;
    flags.set(0);
    node->SetScopeBitFlags(flags);

    EXPECT_TRUE(node->HasScopeFlags());

    // Clearing scopeBitFlags should automatically clear hasScopeFlags
    node->ClearScopeBitFlags();

    EXPECT_FALSE(node->HasScopeFlags());
}

// Test 7: Verify scope bit flags are independent between nodes
TEST_F(TestNodeScopeTagging, ScopeFlagsAreIndependentBetweenNodes) {
    auto node1 = CreateKernelNode(1, 0);
    auto node2 = CreateKernelNode(2, 0);
    
    std::bitset<MAX_SCOPE_NUM> flags1;
    flags1.set(0);
    flags1.set(1);
    node1->SetScopeBitFlags(flags1);
    
    std::bitset<MAX_SCOPE_NUM> flags2;
    flags2.set(2);
    flags2.set(3);
    node2->SetScopeBitFlags(flags2);
    
    EXPECT_EQ(node1->GetScopeBitFlags().count(), 2);
    EXPECT_TRUE(node1->GetScopeBitFlags().test(0));
    EXPECT_TRUE(node1->GetScopeBitFlags().test(1));
    EXPECT_FALSE(node1->GetScopeBitFlags().test(2));
    EXPECT_FALSE(node1->GetScopeBitFlags().test(3));
    
    EXPECT_EQ(node2->GetScopeBitFlags().count(), 2);
    EXPECT_FALSE(node2->GetScopeBitFlags().test(0));
    EXPECT_FALSE(node2->GetScopeBitFlags().test(1));
    EXPECT_TRUE(node2->GetScopeBitFlags().test(2));
    EXPECT_TRUE(node2->GetScopeBitFlags().test(3));
}

// Test 8: Verify updating scope bit flags
TEST_F(TestNodeScopeTagging, UpdateScopeBitFlags) {
    auto node = CreateKernelNode(1, 0);
    
    // Set initial flags
    std::bitset<MAX_SCOPE_NUM> flags1;
    flags1.set(0);
    flags1.set(1);
    node->SetScopeBitFlags(flags1);
    EXPECT_EQ(node->GetScopeBitFlags().count(), 2);
    
    // Update to new flags
    std::bitset<MAX_SCOPE_NUM> flags2;
    flags2.set(1);
    flags2.set(2);
    flags2.set(3);
    node->SetScopeBitFlags(flags2);
    
    EXPECT_FALSE(node->GetScopeBitFlags().test(0));
    EXPECT_TRUE(node->GetScopeBitFlags().test(1));
    EXPECT_TRUE(node->GetScopeBitFlags().test(2));
    EXPECT_TRUE(node->GetScopeBitFlags().test(3));
    EXPECT_EQ(node->GetScopeBitFlags().count(), 3);
}

// Test 9: Verify scope bit flags boundary values
TEST_F(TestNodeScopeTagging, ScopeBitFlagsBoundaryValues) {
    auto node = CreateKernelNode(1, 0);
    
    std::bitset<MAX_SCOPE_NUM> flags;
    
    // Set all flags
    flags.set();
    node->SetScopeBitFlags(flags);
    EXPECT_EQ(node->GetScopeBitFlags().count(), MAX_SCOPE_NUM);
    EXPECT_TRUE(node->GetScopeBitFlags().all());
    
    // Clear all flags
    node->ClearScopeBitFlags();
    EXPECT_EQ(node->GetScopeBitFlags().count(), 0);
    EXPECT_TRUE(node->GetScopeBitFlags().none());
}

// Test 10: Verify specific bit position setting
TEST_F(TestNodeScopeTagging, SetSpecificBitPosition) {
    auto node = CreateKernelNode(1, 0);

    for (size_t i = 0; i < std::min(static_cast<size_t>(MAX_SCOPE_NUM), size_t(10)); ++i) {
        std::bitset<MAX_SCOPE_NUM> flags;
        flags.set(i);
        node->SetScopeBitFlags(flags);

        for (size_t j = 0; j < std::min(static_cast<size_t>(MAX_SCOPE_NUM), size_t(10)); ++j) {
            if (j == i) {
                EXPECT_TRUE(node->GetScopeBitFlags().test(j));
            } else {
                EXPECT_FALSE(node->GetScopeBitFlags().test(j));
            }
        }
    }
}

// Test 11: Verify GetScopeName returns empty string for non-scope nodes
TEST_F(TestNodeScopeTagging, GetScopeNameReturnsEmptyForNonScopeNode) {
    auto node = CreateKernelNode(1, 0);
    
    EXPECT_EQ(node->GetScopeName(), "");
}

// Test 12: Verify IsScopeBegin returns false for non-scope nodes
TEST_F(TestNodeScopeTagging, IsScopeBeginReturnsFalseForNonScopeNode) {
    auto node = CreateKernelNode(1, 0);
    
    EXPECT_FALSE(node->IsScopeBegin());
}

// Test 13: Verify multiple nodes with different scope flags
TEST_F(TestNodeScopeTagging, MultipleNodesWithDifferentScopeFlags) {
    const int numNodes = 5;
    std::vector<std::unique_ptr<SuperKernelKernelNode>> nodes;
    
    for (int i = 0; i < numNodes; ++i) {
        nodes.push_back(CreateKernelNode(i + 1, 0));
    }
    
    // Set different flags for each node
    for (int i = 0; i < numNodes; ++i) {
        std::bitset<MAX_SCOPE_NUM> flags;
        flags.set(i % MAX_SCOPE_NUM);
        nodes[i]->SetScopeBitFlags(flags);
    }
    
    // Verify each node has correct flags
    for (int i = 0; i < numNodes; ++i) {
        EXPECT_TRUE(nodes[i]->GetScopeBitFlags().test(i % MAX_SCOPE_NUM));
        EXPECT_EQ(nodes[i]->GetScopeBitFlags().count(), 1);
    }
}

// Test 14: Verify isScopeNode and hasScopeFlags are independent
TEST_F(TestNodeScopeTagging, IsScopeNodeAndHasScopeFlagsAreIndependent) {
    auto node = CreateKernelNode(1, 0);

    // Initially both false
    EXPECT_FALSE(node->IsScopeNode());
    EXPECT_FALSE(node->HasScopeFlags());

    // Set isScopeNode only
    node->SetIsScopeNode(true);
    EXPECT_TRUE(node->IsScopeNode());
    EXPECT_FALSE(node->HasScopeFlags());

    // Set hasScopeFlags only (via scopeBitFlags)
    node->SetIsScopeNode(false);
    std::bitset<MAX_SCOPE_NUM> flags;
    flags.set(0);
    node->SetScopeBitFlags(flags);
    EXPECT_FALSE(node->IsScopeNode());
    EXPECT_TRUE(node->HasScopeFlags());

    // Set both
    node->SetIsScopeNode(true);
    EXPECT_TRUE(node->IsScopeNode());
    EXPECT_TRUE(node->HasScopeFlags());
}

// Test 15: Verify scope bit flags persist after multiple operations
TEST_F(TestNodeScopeTagging, ScopeFlagsPersistAfterMultipleOperations) {
    auto node = CreateKernelNode(1, 0);

    std::bitset<MAX_SCOPE_NUM> flags;
    flags.set(0);
    flags.set(3);
    node->SetScopeBitFlags(flags);

    // Toggle isScopeNode multiple times
    node->SetIsScopeNode(true);
    node->SetIsScopeNode(false);
    node->SetIsScopeNode(true);

    // Modify scopeBitFlags multiple times
    flags.set(5);
    node->SetScopeBitFlags(flags);
    flags.reset(0);
    node->SetScopeBitFlags(flags);

    // Verify scope bit flags are updated correctly
    EXPECT_FALSE(node->GetScopeBitFlags().test(0));
    EXPECT_FALSE(node->GetScopeBitFlags().test(1));
    EXPECT_FALSE(node->GetScopeBitFlags().test(2));
    EXPECT_TRUE(node->GetScopeBitFlags().test(3));
    EXPECT_TRUE(node->GetScopeBitFlags().test(5));
    EXPECT_EQ(node->GetScopeBitFlags().count(), 2);
}

// Test 16: Verify setting empty bit flags
TEST_F(TestNodeScopeTagging, SetEmptyBitFlags) {
    auto node = CreateKernelNode(1, 0);
    
    // Set some flags first
    std::bitset<MAX_SCOPE_NUM> flags1;
    flags1.set(0);
    flags1.set(1);
    node->SetScopeBitFlags(flags1);
    EXPECT_EQ(node->GetScopeBitFlags().count(), 2);
    
    // Set empty flags
    std::bitset<MAX_SCOPE_NUM> flags2;
    node->SetScopeBitFlags(flags2);
    
    EXPECT_EQ(node->GetScopeBitFlags().count(), 0);
    EXPECT_TRUE(node->GetScopeBitFlags().none());
}

// Test 17: Verify clear is equivalent to setting empty flags
TEST_F(TestNodeScopeTagging, ClearEqualsSettingEmptyFlags) {
    auto node1 = CreateKernelNode(1, 0);
    auto node2 = CreateKernelNode(2, 0);
    
    std::bitset<MAX_SCOPE_NUM> flags;
    flags.set(0);
    flags.set(1);
    flags.set(2);
    
    node1->SetScopeBitFlags(flags);
    node2->SetScopeBitFlags(flags);
    
    node1->ClearScopeBitFlags();
    
    std::bitset<MAX_SCOPE_NUM> emptyFlags;
    node2->SetScopeBitFlags(emptyFlags);
    
    EXPECT_EQ(node1->GetScopeBitFlags().count(), 0);
    EXPECT_EQ(node2->GetScopeBitFlags().count(), 0);
    EXPECT_EQ(node1->GetScopeBitFlags(), node2->GetScopeBitFlags());
}

// Test 18: Verify bit flags can be copied
TEST_F(TestNodeScopeTagging, BitFlagsCanBeCopied) {
    auto node1 = CreateKernelNode(1, 0);
    auto node2 = CreateKernelNode(2, 0);
    
    std::bitset<MAX_SCOPE_NUM> flags;
    flags.set(1);
    flags.set(3);
    flags.set(7);
    
    node1->SetScopeBitFlags(flags);
    node2->SetScopeBitFlags(node1->GetScopeBitFlags());
    
    EXPECT_EQ(node1->GetScopeBitFlags(), node2->GetScopeBitFlags());
    EXPECT_TRUE(node2->GetScopeBitFlags().test(1));
    EXPECT_TRUE(node2->GetScopeBitFlags().test(3));
    EXPECT_TRUE(node2->GetScopeBitFlags().test(7));
}

// Test 19: Verify node with multiple scope bits from same scope
TEST_F(TestNodeScopeTagging, NodeWithMultipleScopeBits) {
    auto node = CreateKernelNode(1, 0);
    
    std::bitset<MAX_SCOPE_NUM> flags;
    // Simulate node belongs to multiple scopes
    flags.set(0);  // scope A
    flags.set(5);  // scope F
    flags.set(10); // scope K
    
    node->SetScopeBitFlags(flags);
    
    EXPECT_EQ(node->GetScopeBitFlags().count(), 3);
    EXPECT_TRUE(node->GetScopeBitFlags().test(0));
    EXPECT_TRUE(node->GetScopeBitFlags().test(5));
    EXPECT_TRUE(node->GetScopeBitFlags().test(10));
}

// Test 20: Verify GetScopeBitFlags returns const reference
TEST_F(TestNodeScopeTagging, GetScopeBitFlagsReturnsConstReference) {
    auto node = CreateKernelNode(1, 0);
    
    std::bitset<MAX_SCOPE_NUM> flags;
    flags.set(0);
    node->SetScopeBitFlags(flags);
    
    const auto& flagsRef = node->GetScopeBitFlags();
    EXPECT_TRUE(flagsRef.test(0));
    EXPECT_EQ(flagsRef.count(), 1);
}
