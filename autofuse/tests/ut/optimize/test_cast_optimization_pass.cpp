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
#include "task_generator/cast_optimization_pass.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "platform_context.h"
#include "runtime_stub.h"

using namespace ge;
using namespace af::testing;
using namespace af::ascir_op;

namespace af {
class TestCastOptimizationPass : public ::testing::Test {
  protected:
    void SetUp() override {
    }
    void TearDown() override {
    }
};

namespace {
size_t CountNodesByType(AscGraph &graph, const std::string &type) {
  size_t count = 0U;
  for (const auto &node : AscGraphUtils::GetComputeGraph(graph)->GetAllNodes()) {
    if (node->GetType() == type) {
      ++count;
    }
  }
  return count;
}

DataType GetNodeOutputDtype(AscGraph &graph, const std::string &name) {
  for (const auto &node : AscGraphUtils::GetComputeGraph(graph)->GetAllNodes()) {
    if (node->GetName() == name) {
      auto asc_node = std::dynamic_pointer_cast<AscNode>(node);
      if (asc_node != nullptr && !asc_node->outputs().empty()) {
        return (*asc_node->outputs().begin())->attr.dtype;
      }
    }
  }
  return DT_UNDEFINED;
}

void AssertNodesExist(AscGraph &graph, const std::vector<std::string> &names, bool should_exist) {
  for (const auto &name : names) {
    bool found = false;
    for (const auto &node : AscGraphUtils::GetComputeGraph(graph)->GetAllNodes()) {
      if (node->GetName() == name) {
        found = true;
        break;
      }
    }
    if (should_exist) {
      EXPECT_TRUE(found) << "node " << name << " should exist but not found";
    } else {
      EXPECT_FALSE(found) << "node " << name << " should not exist but found";
    }
  }
}

AscGraph BuildTwoInputCastConcatGraph(const std::string &name,
                                      DataType data_dtype,
                                      DataType cast_input_dtype,
                                      DataType cast_out_dtype) {
  return AscGraphBuilder(name)
      .Loops({Sym(128), Sym(64)})
      .Data("data0", 0, data_dtype)
      .Data("data1", 1, data_dtype)
      .Load("load0", "data0")
      .Load("load1", "data1")
      .Cast("cast_input0", "load0", cast_input_dtype)
      .Cast("cast_input1", "load1", cast_input_dtype)
      .Concat("concat0", {"cast_input0", "cast_input1"}, 0)
      .Cast("cast_out0", "concat0", cast_out_dtype)
      .Store("store0", "cast_out0")
      .Output("output0", "store0", 0)
      .Build();
}

AscGraph BuildSymbolicAxisConcatCastGraph(const std::string &name, int64_t s1_val, int64_t s2_val) {
  auto s0 = af::Symbol("s0");
  auto s1 = af::Symbol(s1_val);
  auto s2 = af::Symbol(s2_val);
  auto s3 = s1 + s2;
  return AscGraphBuilder(name)
      .Loops({s0, s3})
      .Data("data0", 0, DT_FLOAT)
      .Data("data1", 1, DT_FLOAT)
      .Load("load0", "data0", {s0, s1}, {s1, af::sym::kSymbolOne})
      .Load("load1", "data1", {s0, s2}, {s2, af::sym::kSymbolOne})
      .Concat("concat0", {"load0", "load1"}, 1)
      .Cast("cast_out0", "concat0", DT_FLOAT16)
      .Store("store0", "cast_out0")
      .Output("output0", "store0", 0)
      .Build();
}

AscGraph BuildDowncastDiscontinuityTestGraph(const std::string &name, bool discontinuous) {
  auto s0 = af::Symbol("s0");
  auto s1 = af::Symbol(4);
  auto s2 = af::Symbol(8);
  auto s3 = s1 + s2;
  auto k4 = af::Symbol(4);
  auto big_stride = s0 * s1 * k4;
  auto inner_stride = s1 * k4;
  return AscGraphBuilder(name)
      .Loops({s0, s3})
      .Data("data0", 0, DT_FLOAT)
      .Data("data1", 1, DT_FLOAT)
      .Load("load0",
            "data0",
            {s0, s1},
            discontinuous
              ? std::vector<Expression>{big_stride, inner_stride}
              : std::vector<Expression>{s1, af::sym::kSymbolOne})
      .Load("load1", "data1", {s0, s2}, {s2, af::sym::kSymbolOne})
      .Add("add0", "load0", "load1")
      .Concat("concat0", {"add0"}, 1)
      .Cast("cast_out0", "concat0", DT_FLOAT16)
      .Store("store0", "cast_out0")
      .Output("output0", "store0", 0)
      .Build();
}

} // namespace

TEST_F(TestCastOptimizationPass, NoConcatInGraph_NoChange) {
  auto graph = AscGraphBuilder("test_no_concat")
      .Loops({Sym(128)})
      .Data("data0", 0, DT_FLOAT)
      .Load("load0", "data0")
      .Abs("abs0", "load0")
      .Store("store0", "abs0")
      .Output("output0", "store0")
      .Build();

  ASSERT_EQ(af::optimize::CastOptimizationPass::Run(graph), SUCCESS);
  EXPECT_EQ(CountNodesByType(graph, ascir_op::Cast::Type), 0U);
}

TEST_F(TestCastOptimizationPass, ConcatWithoutCastOutput_NoOptimize) {
  auto graph = AscGraphBuilder("test_concat_no_cast")
      .Loops({Sym(128)})
      .Data("data0", 0, DT_FLOAT)
      .Data("data1", 1, DT_FLOAT)
      .Load("load0", "data0")
      .Load("load1", "data1")
      .Concat("concat0", {"load0", "load1"}, 0)
      .Store("store0", "concat0")
      .Output("output0", "store0")
      .Build();

  ASSERT_EQ(af::optimize::CastOptimizationPass::Run(graph), SUCCESS);
  EXPECT_EQ(CountNodesByType(graph, ascir_op::Cast::Type), 0U);
  EXPECT_EQ(GetNodeOutputDtype(graph, "concat0"), DT_FLOAT);
}

TEST_F(TestCastOptimizationPass, ConcatWithMultipleOutputs_NoOptimize) {
  auto graph = AscGraphBuilder("test_concat_multi_out")
      .Loops({Sym(128)})
      .Data("data0", 0, DT_FLOAT)
      .Load("load0", "data0")
      .Concat("concat0", {"load0"}, 0)
      .Cast("cast0", "concat0", DT_FLOAT16)
      .Store("store0", "cast0")
      .Abs("abs0", "concat0")
      .Store("store1", "abs0")
      .Output("output0", "store0")
      .Output("output1", "store1", 1)
      .Build();

  ASSERT_EQ(af::optimize::CastOptimizationPass::Run(graph), SUCCESS);
  EXPECT_EQ(CountNodesByType(graph, ascir_op::Cast::Type), 1U);
  AssertNodesExist(graph, {"cast0"}, true);
}

TEST_F(TestCastOptimizationPass, ConcatWithNonCastOutput_NoOptimize) {
  auto graph = AscGraphBuilder("test_concat_non_cast_out")
      .Loops({Sym(128)})
      .Data("data0", 0, DT_FLOAT)
      .Load("load0", "data0")
      .Concat("concat0", {"load0"}, 0)
      .Abs("abs0", "concat0")
      .Store("store0", "abs0")
      .Output("output0", "store0")
      .Build();

  ASSERT_EQ(af::optimize::CastOptimizationPass::Run(graph), SUCCESS);
  EXPECT_EQ(CountNodesByType(graph, ascir_op::Cast::Type), 0U);
}

TEST_F(TestCastOptimizationPass, Downcast_PullUpCastBeforeInputs) {
  auto graph = AscGraphBuilder("test_downcast_pullup")
      .Loops({Sym(128), Sym(64)})
      .Data("data0", 0, DT_FLOAT)
      .Data("data1", 1, DT_FLOAT)
      .Load("load0", "data0")
      .Load("load1", "data1")
      .Concat("concat0", {"load0", "load1"}, 0)
      .Cast("cast_out0", "concat0", DT_FLOAT16)
      .Store("store0", "cast_out0")
      .Output("output0", "store0", 0)
      .Build();

  ASSERT_EQ(CountNodesByType(graph, ascir_op::Cast::Type), 1U);
  ASSERT_EQ(af::optimize::CastOptimizationPass::Run(graph), SUCCESS);
  AssertNodesExist(graph, {"cast_out0"}, false);
  EXPECT_EQ(GetNodeOutputDtype(graph, "concat0"), DT_FLOAT16);
  EXPECT_EQ(CountNodesByType(graph, ascir_op::Cast::Type), 2U);
}

TEST_F(TestCastOptimizationPass, Downcast_ReverseInputCastRemoved) {
  auto graph = AscGraphBuilder("test_downcast_reverse_input")
      .Loops({Sym(128), Sym(64)})
      .Data("data0", 0, DT_FLOAT16)
      .Data("data1", 1, DT_FLOAT)
      .Load("load0", "data0")
      .Load("load1", "data1")
      .Cast("cast_input0", "load0", DT_FLOAT)
      .Concat("concat0", {"cast_input0", "load1"}, 0)
      .Cast("cast_out0", "concat0", DT_FLOAT16)
      .Store("store0", "cast_out0")
      .Output("output0", "store0", 0)
      .Build();

  ASSERT_EQ(CountNodesByType(graph, ascir_op::Cast::Type), 2U);
  ASSERT_EQ(af::optimize::CastOptimizationPass::Run(graph), SUCCESS);
  AssertNodesExist(graph, {"cast_out0", "cast_input0"}, false);
  EXPECT_EQ(GetNodeOutputDtype(graph, "concat0"), DT_FLOAT16);
  EXPECT_EQ(CountNodesByType(graph, ascir_op::Cast::Type), 1U);
}

TEST_F(TestCastOptimizationPass, Downcast_AllReverseInputCastsEliminated) {
  auto graph = BuildTwoInputCastConcatGraph("test_downcast_all_reverse", DT_FLOAT16, DT_FLOAT, DT_FLOAT16);

  ASSERT_EQ(CountNodesByType(graph, ascir_op::Cast::Type), 3U);
  ASSERT_EQ(af::optimize::CastOptimizationPass::Run(graph), SUCCESS);
  AssertNodesExist(graph, {"cast_out0", "cast_input0", "cast_input1"}, false);
  EXPECT_EQ(GetNodeOutputDtype(graph, "concat0"), DT_FLOAT16);
  EXPECT_EQ(CountNodesByType(graph, ascir_op::Cast::Type), 0U);
}

TEST_F(TestCastOptimizationPass, Upcast_NoReverseInputCast_NotOptimized) {
  auto graph = AscGraphBuilder("test_upcast_no_reverse")
      .Loops({Sym(128)})
      .Data("data0", 0, DT_FLOAT16)
      .Data("data1", 1, DT_FLOAT16)
      .Load("load0", "data0")
      .Load("load1", "data1")
      .Concat("concat0", {"load0", "load1"}, 0)
      .Cast("cast_out0", "concat0", DT_FLOAT)
      .Store("store0", "cast_out0")
      .Output("output0", "store0", 0)
      .Build();

  ASSERT_EQ(af::optimize::CastOptimizationPass::Run(graph), SUCCESS);
  EXPECT_EQ(CountNodesByType(graph, ascir_op::Cast::Type), 1U);
  AssertNodesExist(graph, {"cast_out0"}, true);
}

TEST_F(TestCastOptimizationPass, Upcast_TransposeAlg_NoOptimize) {
  auto graph = BuildTwoInputCastConcatGraph("test_upcast_transpose", DT_FLOAT, DT_FLOAT16, DT_FLOAT);

  ASSERT_EQ(af::optimize::CastOptimizationPass::Run(graph, 0), SUCCESS);
  AssertNodesExist(graph, {"cast_out0"}, true);
  EXPECT_EQ(GetNodeOutputDtype(graph, "concat0"), DT_FLOAT16);
}

TEST_F(TestCastOptimizationPass, MixedReverseAndNonReverseInputCast) {
  auto graph = AscGraphBuilder("test_mixed_input_cast")
      .Loops({Sym(128), Sym(64)})
      .Data("data0", 0, DT_FLOAT16)
      .Data("data1", 1, DT_FLOAT16)
      .Data("data2", 2, DT_FLOAT)
      .Load("load0", "data0")
      .Load("load1", "data1")
      .Load("load2", "data2")
      .Cast("cast_input0", "load0", DT_FLOAT)
      .Cast("cast_input1", "load1", DT_FLOAT)
      .Concat("concat0", {"cast_input0", "cast_input1", "load2"}, 0)
      .Cast("cast_out0", "concat0", DT_FLOAT16)
      .Store("store0", "cast_out0")
      .Output("output0", "store0", 0)
      .Build();

  ASSERT_EQ(CountNodesByType(graph, ascir_op::Cast::Type), 3U);
  ASSERT_EQ(af::optimize::CastOptimizationPass::Run(graph), SUCCESS);
  AssertNodesExist(graph, {"cast_out0", "cast_input0", "cast_input1"}, false);
  EXPECT_EQ(GetNodeOutputDtype(graph, "concat0"), DT_FLOAT16);
  EXPECT_EQ(CountNodesByType(graph, ascir_op::Cast::Type), 1U);
}

TEST_F(TestCastOptimizationPass, Downcast_TransposeAlg_IgnoreAlignment_Optimize) {
  auto graph = BuildSymbolicAxisConcatCastGraph("test_transpose_ignore_alignment", 3, 5);

  ASSERT_EQ(af::optimize::CastOptimizationPass::Run(graph, 0), SUCCESS);
  AssertNodesExist(graph, {"cast_out0"}, false);
  EXPECT_EQ(GetNodeOutputDtype(graph, "concat0"), DT_FLOAT16);
}

TEST_F(TestCastOptimizationPass, Downcast_GatherAlg_AlignmentDegradation_NoOptimize) {
  auto graph = BuildSymbolicAxisConcatCastGraph("test_gather_alignment_degradation", 3, 5);

  ASSERT_EQ(af::optimize::CastOptimizationPass::Run(graph, 1), SUCCESS);
  AssertNodesExist(graph, {"cast_out0"}, true);
  EXPECT_EQ(GetNodeOutputDtype(graph, "concat0"), DT_FLOAT);
}

TEST_F(TestCastOptimizationPass, Downcast_GatherAlg_NoAlignmentDegradation_Optimize) {
  auto graph = BuildSymbolicAxisConcatCastGraph("test_no_degradation", 4, 8);

  ASSERT_EQ(af::optimize::CastOptimizationPass::Run(graph, 1), SUCCESS);
  AssertNodesExist(graph, {"cast_out0"}, false);
  EXPECT_EQ(GetNodeOutputDtype(graph, "concat0"), DT_FLOAT16);
  EXPECT_EQ(CountNodesByType(graph, ascir_op::Cast::Type), 2U);
}

TEST_F(TestCastOptimizationPass, Downcast_SharedNonCastInput_OnlyOneCastInserted) {
  auto graph = AscGraphBuilder("test_shared_noncast_downcast")
      .Loops({Sym(128), Sym(64)})
      .Data("data0", 0, DT_FLOAT)
      .Load("load0", "data0")
      .Relu("relu0", "load0")
      .Concat("concat0", {"relu0", "relu0"}, 0)
      .Cast("cast_out0", "concat0", DT_FLOAT16)
      .Store("store0", "cast_out0")
      .Output("output0", "store0", 0)
      .Build();

  ASSERT_EQ(CountNodesByType(graph, ascir_op::Cast::Type), 1U);
  ASSERT_EQ(af::optimize::CastOptimizationPass::Run(graph), SUCCESS);
  AssertNodesExist(graph, {"cast_out0"}, false);
  EXPECT_EQ(GetNodeOutputDtype(graph, "concat0"), DT_FLOAT16);
  EXPECT_EQ(CountNodesByType(graph, ascir_op::Cast::Type), 1U);
}

TEST_F(TestCastOptimizationPass, Downcast_GatherAlg_MultipleDiscontinuitiesViaMultiInputNode_Optimize) {
  auto graph = BuildDowncastDiscontinuityTestGraph("test_multi_input_discontinuity", true);
  ASSERT_EQ(af::optimize::CastOptimizationPass::Run(graph, 1), SUCCESS);
  AssertNodesExist(graph, {"cast_out0"}, false);
  EXPECT_EQ(GetNodeOutputDtype(graph, "concat0"), DT_FLOAT16);
}

TEST_F(TestCastOptimizationPass, Downcast_GatherAlg_NoDiscontinuity_Optimize) {
  auto graph = BuildDowncastDiscontinuityTestGraph("test_no_discontinuity", false);
  ASSERT_EQ(af::optimize::CastOptimizationPass::Run(graph, 1), SUCCESS);
  AssertNodesExist(graph, {"cast_out0"}, false);
  EXPECT_EQ(GetNodeOutputDtype(graph, "concat0"), DT_FLOAT16);
}

TEST_F(TestCastOptimizationPass, Downcast_ReverseCastWithMultipleConsumers_BypassNotRemove) {
  auto graph = AscGraphBuilder("test_downcast_reverse_multi_consumer")
      .Loops({Sym(128), Sym(64)})
      .Data("data0", 0, DT_FLOAT16)
      .Data("data1", 1, DT_FLOAT)
      .Load("load0", "data0")
      .Load("load1", "data1")
      .Cast("cast_input0", "load0", DT_FLOAT)
      .Concat("concat0", {"cast_input0", "load1"}, 0)
      .Cast("cast_out0", "concat0", DT_FLOAT16)
      .Store("store0", "cast_out0")
      .Abs("abs0", "cast_input0")
      .Store("store1", "abs0")
      .Output("output0", "store0", 0)
      .Output("output1", "store1", 1)
      .Build();

  ASSERT_EQ(CountNodesByType(graph, ascir_op::Cast::Type), 2U);
  ASSERT_EQ(af::optimize::CastOptimizationPass::Run(graph), SUCCESS);
  AssertNodesExist(graph, {"cast_out0"}, false);
  AssertNodesExist(graph, {"cast_input0"}, true);
  EXPECT_EQ(GetNodeOutputDtype(graph, "concat0"), DT_FLOAT16);
  EXPECT_EQ(CountNodesByType(graph, ascir_op::Cast::Type), 2U);
}

TEST_F(TestCastOptimizationPass, Downcast_SharedSourceReverseCastsWithMultipleConsumers_BypassBoth) {
  auto graph = AscGraphBuilder("test_downcast_shared_reverse_multi")
      .Loops({Sym(128), Sym(64)})
      .Data("data0", 0, DT_FLOAT16)
      .Data("data1", 1, DT_FLOAT)
      .Load("load0", "data0")
      .Load("load1", "data1")
      .Cast("cast_input0", "load0", DT_FLOAT)
      .Cast("cast_input1", "load0", DT_FLOAT)
      .Concat("concat0", {"cast_input0", "cast_input1", "load1"}, 0)
      .Cast("cast_out0", "concat0", DT_FLOAT16)
      .Store("store0", "cast_out0")
      .Abs("abs0", "cast_input0")
      .Store("store1", "abs0")
      .Output("output0", "store0", 0)
      .Output("output1", "store1", 1)
      .Build();

  ASSERT_EQ(CountNodesByType(graph, ascir_op::Cast::Type), 3U);
  ASSERT_EQ(af::optimize::CastOptimizationPass::Run(graph), SUCCESS);
  AssertNodesExist(graph, {"cast_out0", "cast_input1"}, false);
  AssertNodesExist(graph, {"cast_input0"}, true);
  EXPECT_EQ(GetNodeOutputDtype(graph, "concat0"), DT_FLOAT16);
  EXPECT_EQ(CountNodesByType(graph, ascir_op::Cast::Type), 2U);
}

} // namespace af
