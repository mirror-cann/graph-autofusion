/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "asc_graph_builder.h"
#include "graph_pass/masked_fill_input_reorder_pass.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "platform_context.h"
#include "runtime_stub.h"

using namespace ge;
using namespace af::testing;
using namespace af::ascir_op;

namespace af {
class TestMaskedFillInputReorderPass : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

namespace {
// Helper to get input node name at given index
std::string GetInputNodeName(AscGraph &graph, const std::string &node_name, size_t input_index) {
  for (const auto &node : AscGraphUtils::GetComputeGraph(graph)->GetAllNodes()) {
    if (node->GetName() == node_name) {
      auto asc_node = std::dynamic_pointer_cast<AscNode>(node);
      if (asc_node != nullptr && input_index < asc_node->inputs().size()) {
        return asc_node->inputs()[input_index]->anchor.GetOwnerNode()->GetName();
      }
    }
  }
  return "";
}

// Build a simple MaskedFill graph: Data(x) + Data(mask) + Data(value) -> MaskedFill -> Store -> Output
AscGraph BuildMaskedFillGraph() {
  return AscGraphBuilder("masked_fill_test")
      .Loops({Sym(128), Sym(64)})
      .Data("x", 0, DT_FLOAT)
      .Data("mask", 1, DT_UINT8)
      .Data("value", 2, DT_FLOAT)
      .Load("load_x", "x")
      .Load("load_mask", "mask")
      .Load("load_value", "value")
      .Op<MaskedFill>("masked_fill", {"load_x", "load_mask", "load_value"})
      .Store("store", "masked_fill")
      .Output("output", "store", 0)
      .Build();
}
}  // namespace

/**
 * @tc.name: MaskedFillInputReorderPass_ShouldReorderInputs
 * @tc.number: MaskedFillInputReorderPass_Test_001
 * @tc.desc: Verify that MaskedFill inputs are reordered from (x, mask, value) to (mask, value, x)
 */
TEST_F(TestMaskedFillInputReorderPass, MaskedFillInputReorderPass_ShouldReorderInputs) {
  auto graph = BuildMaskedFillGraph();

  // Before pass: inputs should be (load_x, load_mask, load_value)
  EXPECT_EQ(GetInputNodeName(graph, "masked_fill", 0), "load_x");
  EXPECT_EQ(GetInputNodeName(graph, "masked_fill", 1), "load_mask");
  EXPECT_EQ(GetInputNodeName(graph, "masked_fill", 2), "load_value");

  // Run the pass
  optimize::MaskedFillInputReorderPass pass;
  ASSERT_EQ(pass.RunPass(graph), ge::SUCCESS);

  // After pass: inputs should be reordered to (load_mask, load_value, load_x)
  // Swap(0,1): (x, mask, value) -> (mask, x, value)
  // Swap(1,2): (mask, x, value) -> (mask, value, x)
  EXPECT_EQ(GetInputNodeName(graph, "masked_fill", 0), "load_mask") << "Input 0 should be mask after reorder";
  EXPECT_EQ(GetInputNodeName(graph, "masked_fill", 1), "load_value") << "Input 1 should be value after reorder";
  EXPECT_EQ(GetInputNodeName(graph, "masked_fill", 2), "load_x") << "Input 2 should be x after reorder";
}

/**
 * @tc.name: MaskedFillInputReorderPass_ShouldNotAffectOtherNodes
 * @tc.number: MaskedFillInputReorderPass_Test_002
 * @tc.desc: Verify that non-MaskedFill nodes are not affected by the pass
 */
TEST_F(TestMaskedFillInputReorderPass, MaskedFillInputReorderPass_ShouldNotAffectOtherNodes) {
  auto graph = BuildMaskedFillGraph();

  // Run the pass
  optimize::MaskedFillInputReorderPass pass;
  ASSERT_EQ(pass.RunPass(graph), ge::SUCCESS);

  // Verify Store node still has correct input
  EXPECT_EQ(GetInputNodeName(graph, "store", 0), "masked_fill");

  // Verify Load nodes still have correct inputs
  EXPECT_EQ(GetInputNodeName(graph, "load_x", 0), "x");
  EXPECT_EQ(GetInputNodeName(graph, "load_mask", 0), "mask");
  EXPECT_EQ(GetInputNodeName(graph, "load_value", 0), "value");
}

/**
 * @tc.name: MaskedFillInputReorderPass_ShouldHandleMultipleMaskedFillNodes
 * @tc.number: MaskedFillInputReorderPass_Test_003
 * @tc.desc: Verify that the pass correctly reorders inputs for multiple MaskedFill nodes
 */
TEST_F(TestMaskedFillInputReorderPass, MaskedFillInputReorderPass_ShouldHandleMultipleMaskedFillNodes) {
  auto graph = AscGraphBuilder("multi_masked_fill_test")
                   .Loops({Sym(128), Sym(64)})
                   .Data("x1", 0, DT_FLOAT)
                   .Data("mask1", 1, DT_UINT8)
                   .Data("value1", 2, DT_FLOAT)
                   .Data("x2", 3, DT_FLOAT)
                   .Data("mask2", 4, DT_UINT8)
                   .Data("value2", 5, DT_FLOAT)
                   .Load("load_x1", "x1")
                   .Load("load_mask1", "mask1")
                   .Load("load_value1", "value1")
                   .Load("load_x2", "x2")
                   .Load("load_mask2", "mask2")
                   .Load("load_value2", "value2")
                   .Op<MaskedFill>("masked_fill1", {"load_x1", "load_mask1", "load_value1"})
                   .Op<MaskedFill>("masked_fill2", {"load_x2", "load_mask2", "load_value2"})
                   .Add("add", "masked_fill1", "masked_fill2")
                   .Store("store", "add")
                   .Output("output", "store", 0)
                   .Build();

  // Run the pass
  optimize::MaskedFillInputReorderPass pass;
  ASSERT_EQ(pass.RunPass(graph), ge::SUCCESS);

  // Verify both MaskedFill nodes are reordered
  EXPECT_EQ(GetInputNodeName(graph, "masked_fill1", 0), "load_mask1");
  EXPECT_EQ(GetInputNodeName(graph, "masked_fill1", 1), "load_value1");
  EXPECT_EQ(GetInputNodeName(graph, "masked_fill1", 2), "load_x1");

  EXPECT_EQ(GetInputNodeName(graph, "masked_fill2", 0), "load_mask2");
  EXPECT_EQ(GetInputNodeName(graph, "masked_fill2", 1), "load_value2");
  EXPECT_EQ(GetInputNodeName(graph, "masked_fill2", 2), "load_x2");
}

/**
 * @tc.name: MaskedFillInputReorderPass_ShouldUpdateDtypeSchema
 * @tc.number: MaskedFillInputReorderPass_Test_004
 * @tc.desc: Verify that the pass correctly updates dtype schema after reordering inputs
 */
TEST_F(TestMaskedFillInputReorderPass, MaskedFillInputReorderPass_ShouldUpdateDtypeSchema) {
  auto graph = BuildMaskedFillGraph();

  // Get the MaskedFill node
  auto compute_graph = AscGraphUtils::GetComputeGraph(graph);
  af::AscNodePtr masked_fill_node = nullptr;
  for (const auto &node : compute_graph->GetAllNodes()) {
    if (node->GetName() == "masked_fill") {
      masked_fill_node = std::dynamic_pointer_cast<AscNode>(node);
      break;
    }
  }
  ASSERT_NE(masked_fill_node, nullptr);

  // Run the pass
  optimize::MaskedFillInputReorderPass pass;
  ASSERT_EQ(pass.RunPass(graph), ge::SUCCESS);

  // After pass: dtype schema should match Select(mask, value, x).
  auto op_desc = masked_fill_node->GetOpDescBarePtr();
  EXPECT_EQ(masked_fill_node->GetType(), "Select");
  EXPECT_EQ(op_desc->GetType(), "Select");
  EXPECT_EQ(op_desc->GetInputDesc(0).GetDataType(), DT_UINT8) << "Input 0 dtype should be UINT8 (mask) after reorder";
  EXPECT_EQ(op_desc->GetInputDesc(1).GetDataType(), DT_FLOAT) << "Input 1 dtype should be FLOAT (value) after reorder";
  EXPECT_EQ(op_desc->GetInputDesc(2).GetDataType(), DT_FLOAT) << "Input 2 dtype should be FLOAT (x) after reorder";
  ASSERT_EQ(op_desc->GetIrInputs().size(), 3U);
  EXPECT_EQ(op_desc->GetIrInputs()[0].first, "x1");
  EXPECT_EQ(op_desc->GetIrInputs()[1].first, "x2");
  EXPECT_EQ(op_desc->GetIrInputs()[2].first, "x3");

  // Also verify node input attr dtype is updated
  EXPECT_EQ(static_cast<ge::DataType>(masked_fill_node->inputs[0].attr.dtype), DT_UINT8);
  EXPECT_EQ(static_cast<ge::DataType>(masked_fill_node->inputs[1].attr.dtype), DT_FLOAT);
  EXPECT_EQ(static_cast<ge::DataType>(masked_fill_node->inputs[2].attr.dtype), DT_FLOAT);
}
}  // namespace af
