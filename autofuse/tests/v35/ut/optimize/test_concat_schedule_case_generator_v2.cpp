/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asc_graph_builder.h"
#include "gtest/gtest.h"
#include "ascendc_ir.h"
#include "ascir_ops_utils.h"
#include "asc_graph_utils.h"
#include "platform_context.h"
#include "task_generator/concat_schedule_case_generator.h"
#include "task_generator/concat_group_partitioner.h"
#include "platform/platform_factory.h"
#include "runtime_stub.h"
#include "task_generator/concat_score_function_generator.h"

namespace schedule {
using namespace optimize;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;

class ConcatScheduleCaseGeneratorV2Test : public ::testing::Test {
 protected:
  void SetUp() override {
    // dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<af::RuntimeStubV2>();
    ge::RuntimeStub::SetInstance(stub_v2);
  }
  void TearDown() override {
    // dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }

  static void CreateConcatAscGraph(af::AscGraph &graph, const std::string &head_dim,
                                   const std::vector<std::string> &concat_dims,
                                   const std::vector<std::string> &tail_dims) {
    std::vector<af::Expression> concat_axis_sizes;
    af::Expression concat_axis_size = af::Symbol(0);
    for (const auto &dim : concat_dims) {
      if (dim[0] == 's') {
        concat_axis_sizes.emplace_back(graph.CreateSizeVar(dim));
      } else {
        concat_axis_sizes.emplace_back(af::Symbol(std::strtol(dim.c_str(), nullptr, 10)));
      }
      concat_axis_size = concat_axis_size + concat_axis_sizes.back();
    }
    std::vector<af::Expression> tail_dim_sizes;
    for (const auto &dim : tail_dims) {
      if (dim[0] == 's') {
        tail_dim_sizes.emplace_back(graph.CreateSizeVar(dim));
      } else {
        tail_dim_sizes.emplace_back(af::Symbol(std::strtol(dim.c_str(), nullptr, 10)));
      }
    }

    auto head_dim_size = graph.CreateSizeVar(head_dim);
    auto z0 = graph.CreateAxis("z0", head_dim_size);
    auto z1 = graph.CreateAxis("z1", concat_axis_size);
    std::vector<af::Axis> tail_axes;
    tail_axes.reserve(tail_dims.size());
    for (size_t i = 0U; i < tail_dims.size(); ++i) {
      tail_axes.emplace_back(graph.CreateAxis("z" + std::to_string(2 + i), tail_dim_sizes[i]));
    }

    std::vector<std::shared_ptr<Data>> data_ops;
    std::vector<af::AscOpOutput> concat_inputs;
    af::Expression concat_dim_size = af::Symbol(0);
    for (size_t i = 0; i < concat_axis_sizes.size(); ++i) {
      std::string name = "x" + std::to_string(i);
      auto x_op = std::make_shared<Data>(name.c_str(), graph);
      x_op->y.dtype = af::DT_FLOAT16;
      *x_op->y.repeats = {head_dim_size, concat_axis_sizes[i]};
      for (const auto &dim_size : tail_dim_sizes) {
        x_op->y.repeats->push_back(dim_size);
      }
      x_op->ir_attr.SetIndex(static_cast<int64_t>(i));
      data_ops.emplace_back(x_op);
      concat_inputs.push_back(x_op->y);
      concat_dim_size = concat_dim_size + concat_axis_sizes[i];
    }

    af::ascir_op::Concat concat_op("concat");
    concat_op.x = {concat_inputs[0], concat_inputs[1], concat_inputs[2], concat_inputs[3], concat_inputs[4]};
    concat_op.y.repeats->emplace_back(head_dim_size);
    concat_op.y.repeats->emplace_back(concat_dim_size);
    for (const auto &dim_size : tail_dim_sizes) {
      concat_op.y.repeats->push_back(dim_size);
    }
  }

  static std::string ExpressToStr(const std::vector<af::Expression> &exprs) {
    std::stringstream ss;
    for (auto &size_expr : exprs) {
      ss << std::string(size_expr.Str().get()) << ", ";
    }
    return ss.str();
  }

  static std::string RepeatsToStr(const af::AscGraph &graph, const char *node_name) {
    auto node = graph.FindNode(node_name);
    if (node == nullptr) {
      return "";
    }
    return ExpressToStr(node->outputs[0].attr.repeats);
  }

  static std::string StridesToStr(const af::AscGraph &graph, const char *node_name) {
    auto node = graph.FindNode(node_name);
    if (node == nullptr) {
      return "";
    }
    return ExpressToStr(node->outputs[0].attr.strides);
  }
};

TEST_F(ConcatScheduleCaseGeneratorV2Test, ConcatTailDim_SplitConcat) {
  af::AscGraph graph("concat_last_dim_graph");
  std::vector<int> concat_dim_sizes{412,
                                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                                    16, 16, 16,};
  auto s0 = graph.CreateSizeVar("s0");
  auto concat_size = af::Expression(af::Symbol(0));
  std::vector<std::shared_ptr<Data>> data_ops;
  std::vector<af::AscOpOutput> outputs;
  for (size_t i = 0; i < concat_dim_sizes.size(); ++i) {
    af::Expression s_i;
    s_i = graph.CreateSizeVar(concat_dim_sizes[i]);
    concat_size = (concat_size + s_i);
    auto data_op = std::make_shared<Data>(("Data" + std::to_string(i + 1)).c_str(), graph);
    data_op->y.dtype = af::DT_FLOAT;
    *data_op->y.repeats = {s0, s_i};
    *data_op->y.strides = {s_i, af::ops::One};
    data_ops.emplace_back(data_op);
    outputs.emplace_back(data_ops.back()->y);
  }

  af::ascir_op::Concat concat_op("concat");
  concat_op.x = outputs;
  concat_op.y.dtype = af::DT_FLOAT;
  *concat_op.y.repeats = {s0, concat_size};
  *concat_op.y.strides = {concat_size, af::ops::One};

  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);

  optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  std::vector<optimize::ConcatGroupPartitioner::ConcatGroup> groups;
  ASSERT_EQ(partitioner.PartitionGroups(groups), af::SUCCESS);
  size_t index = 0;
  size_t last_end = 0;
  for (const auto &group : groups) {
    std::cout << "index: " << index << ", start: " << group.start << ", end: " << group.end << ", type: " << group.group_type << std::endl;
    std::vector<int> dims(concat_dim_sizes.begin() + static_cast<int64_t>(group.start),
                          concat_dim_sizes.begin() + static_cast<int64_t>(group.end));
    std::cout << "  " << af::ToString(dims) << "count = " << group.end - group.start << ", size = " << group.size << std::endl;
    EXPECT_EQ(group.start, last_end);
    last_end = group.end;
    ++index;
  }
  EXPECT_EQ(groups.size(), 6);
  EXPECT_EQ(groups[0].end - groups[0].start, 17);
}

TEST_F(ConcatScheduleCaseGeneratorV2Test, ConcatTailDim_SplitConcat_LargeRowNum) {
  af::AscGraph graph("concat_last_dim_graph");
  std::vector<int> concat_dim_sizes{64, 6, 28, 42};
  auto s0 = graph.CreateSizeVar(64 * 64);
  auto concat_size = af::Expression(af::Symbol(0));
  std::vector<std::shared_ptr<Data>> data_ops;
  std::vector<af::AscOpOutput> outputs;
  for (size_t i = 0; i < concat_dim_sizes.size(); ++i) {
    af::Expression s_i;
    s_i = graph.CreateSizeVar(concat_dim_sizes[i]);
    concat_size = (concat_size + s_i);
    auto data_op = std::make_shared<Data>(("Data" + std::to_string(i + 1)).c_str(), graph);
    data_op->y.dtype = af::DT_FLOAT;
    *data_op->y.repeats = {s0, s_i};
    *data_op->y.strides = {s_i, af::ops::One};
    data_ops.emplace_back(data_op);
    outputs.emplace_back(data_ops.back()->y);
  }

  af::ascir_op::Concat concat_op("concat");
  concat_op.x = outputs;
  concat_op.y.dtype = af::DT_FLOAT;
  *concat_op.y.repeats = {s0, concat_size};
  *concat_op.y.strides = {concat_size, af::ops::One};

  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);

  optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  std::vector<optimize::ConcatGroupPartitioner::ConcatGroup> groups;
  ASSERT_EQ(partitioner.PartitionGroups(groups), af::SUCCESS);
  size_t index = 0;
  size_t last_end = 0;
  for (const auto &group : groups) {
    std::cout << "index: " << index << ", start: " << group.start << ", end: " << group.end << ", type: " << group.group_type << std::endl;
    std::vector<int> dims(concat_dim_sizes.begin() + static_cast<int64_t>(group.start),
                          concat_dim_sizes.begin() + static_cast<int64_t>(group.end));
    std::cout << "  " << af::ToString(dims) << "count = " << group.end - group.start << ", size = " << group.size << std::endl;
    EXPECT_EQ(group.start, last_end);
    last_end = group.end;
    ++index;
  }
  EXPECT_EQ(groups.size(), 1);
}

TEST_F(ConcatScheduleCaseGeneratorV2Test, ConcatFirstDim_InsertAxis) {
  af::AscGraph graph("concat_last_dim_graph");
  std::vector<int> concat_dim_sizes{1, 2, 1, 1, 1, 1};
  auto s0 = graph.CreateSizeVar(1);
  auto concat_size = af::Expression(af::Symbol(0));
  std::vector<std::shared_ptr<Data>> data_ops;
  std::vector<af::AscOpOutput> outputs;
  for (size_t i = 0; i < concat_dim_sizes.size(); ++i) {
    af::Expression s_i;
    s_i = graph.CreateSizeVar(concat_dim_sizes[i]);
    concat_size = (concat_size + s_i);
    auto data_op = std::make_shared<Data>(("Data" + std::to_string(i + 1)).c_str(), graph);
    data_op->y.dtype = af::DT_FLOAT;
    *data_op->y.repeats = {s0, s_i};
    *data_op->y.strides = {s_i, af::ops::One};
    data_ops.emplace_back(data_op);
    outputs.emplace_back(data_ops.back()->y);
  }

  af::ascir_op::Concat concat_op("concat");
  concat_op.x = outputs;
  concat_op.y.dtype = af::DT_FLOAT;
  *concat_op.y.repeats = {s0, concat_size};
  *concat_op.y.strides = {concat_size, af::ops::One};

  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);

  optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  std::vector<optimize::ConcatGroupPartitioner::ConcatGroup> groups;
  ASSERT_EQ(partitioner.PartitionGroups(groups), af::SUCCESS);
  size_t index = 0;
  size_t last_end = 0;
  for (const auto &group : groups) {
    std::cout << "index: " << index << ", start: " << group.start << ", end: " << group.end << ", type: " << group.group_type << std::endl;
    std::vector<int> dims(concat_dim_sizes.begin() + static_cast<int64_t>(group.start),
                          concat_dim_sizes.begin() + static_cast<int64_t>(group.end));
    std::cout << "  " << af::ToString(dims) << "count = " << group.end - group.start << ", size = " << group.size << std::endl;
    EXPECT_EQ(group.start, last_end);
    last_end = group.end;
    ++index;
  }
}

TEST_F(ConcatScheduleCaseGeneratorV2Test, OptimizeSameShapeConcat) {
  auto dtype = af::DT_INT16;
  auto s0 = af::Symbol("s0");
  auto s1 = af::Symbol(7);
  auto s2 = s1 + s1;
  auto graph = af::testing::AscGraphBuilder("test_graph")
                   .Loops({s0, s2})
                   .Data("data0", 0, dtype)
                   .Load("load0", "data0", {s0, s1}, {s1, af::sym::kSymbolOne})
                   .Data("data1", 1, dtype)
                   .Load("load1", "data1", {s0, s1}, {s1, af::sym::kSymbolOne})
                   .Relu("relu0", "load1")
                   .Concat("concat", {"load0", "relu0"})
                   .Store("store", "concat")
                   .Output("out", "store")
                   .Build();
  auto concat_node = graph.FindNode("concat");
  std::vector<std::string> score_functions;
  std::vector<::ascir::ImplGraph> graphs;
  optimize::ConcatFusionCaseGenerator generator;
  EXPECT_EQ(generator.Generate(graph, graphs, score_functions), SUCCESS);
  EXPECT_EQ(graphs.size(), 2);
  EXPECT_TRUE(graphs[0].FindNode("load0_ub_cpy_input_0") != nullptr);
}

// 测试场景：Concat 输出有 Downcast Cast，Generate 后 Cast 被优化消除
// 图结构：
//   data0[FP32] -> load0 ----\
//                             concat0[FP32] -> cast_out0[FP32→FP16] -> store0 -> output0[FP16]
//   data1[FP32] -> load1 ---/
// 优化后（在 Generate 生成的 ImplGraph 中）：
//   cast_out0 被移除，concat 输出 dtype 变为 FP16，Cast 被拉到输入端
// 预期：生成的图中 cast_out0 不存在，Cast 数量从 1 变为 2（两个新输入 Cast）
TEST_F(ConcatScheduleCaseGeneratorV2Test, DowncastCastOptimizeAfterGenerate) {
  auto s0 = af::Symbol("s0");
  auto s1 = af::Symbol(8);
  auto s2 = s1 + s1;
  auto graph = af::testing::AscGraphBuilder("test_downcast_concat")
                   .Loops({s0, s2})
                   .Data("data0", 0, af::DT_FLOAT)
                   .Load("load0", "data0", {s0, s1}, {s1, af::sym::kSymbolOne})
                   .Data("data1", 1, af::DT_FLOAT)
                   .Load("load1", "data1", {s0, s1}, {s1, af::sym::kSymbolOne})
                   .Concat("concat", {"load0", "load1"})
                   .Cast("cast_out0", "concat", af::DT_FLOAT16)
                   .Store("store0", "cast_out0")
                   .Output("output0", "store0", 0, af::DT_FLOAT16)
                   .Build();

  std::vector<std::string> score_functions;
  std::vector<::ascir::ImplGraph> graphs;
  optimize::ConcatFusionCaseGenerator generator;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), SUCCESS);
  ASSERT_GE(graphs.size(), 1U);

  size_t cast_count = 0U;
  for (const auto &node : graphs[0].GetAllNodes()) {
    if (node->GetType() == Cast::Type) {
      ++cast_count;
    }
  }
  EXPECT_TRUE(graphs[0].FindNode("cast_out0") == nullptr);
  EXPECT_EQ(cast_count, 2U);
}

TEST_F(ConcatScheduleCaseGeneratorV2Test, InputWithTransposeOrReduce_ForceSingleGroup) {
  auto s0 = af::Symbol("s0");
  auto s1 = af::Symbol(4);
  auto s2 = s1 + s1 + s1;
  auto graph = af::testing::AscGraphBuilder("test_input_transpose_reduce")
                   .Loops({s0, s2})
                   .Data("data0", 0, af::DT_FLOAT16)
                   .Load("load0", "data0", {s0, s1}, {s1, af::sym::kSymbolOne})
                   .Data("data1", 1, af::DT_FLOAT16)
                   .Load("load1", "data1", {s0, s1}, {s1, af::sym::kSymbolOne})
                   .Transpose("transpose0", "load1", {1, 0})
                   .Data("data2", 2, af::DT_FLOAT16)
                   .Load("load2", "data2", {s0, s1}, {s1, af::sym::kSymbolOne})
                   .Concat("concat", {"load0", "transpose0", "load2"}, 1)
                   .Store("store", "concat")
                   .Output("out", "store", 0, af::DT_FLOAT16)
                   .Build();

  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);

  optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  std::vector<optimize::ConcatGroupPartitioner::ConcatGroup> groups;
  ASSERT_EQ(partitioner.PartitionGroups(groups), af::SUCCESS);

  size_t input_with_transpose = 1UL;
  bool found_single_group_for_transpose = false;
  for (const auto &group : groups) {
    if (group.start == input_with_transpose && group.end == input_with_transpose + 1) {
      found_single_group_for_transpose = true;
    }
  }
  EXPECT_TRUE(found_single_group_for_transpose);
}

TEST_F(ConcatScheduleCaseGeneratorV2Test, InputWithReduce_ForceSingleGroup) {
  auto s0 = af::Symbol("s0");
  auto s1 = af::Symbol(4);
  auto s2 = s1 + s1 + s1;
  auto graph = af::testing::AscGraphBuilder("test_input_reduce")
      .Loops({s0, s2})
      .Data("data0", 0, af::DT_FLOAT16)
      .Load("load0", "data0", {s0, s1}, {s1, af::sym::kSymbolOne})
      .Data("data1", 1, af::DT_FLOAT16)
      .Load("load1", "data1", {s0, s1}, {s1, af::sym::kSymbolOne})
      .Sum("sum0", "load1", {1})
      .Data("data2", 2, af::DT_FLOAT16)
      .Load("load2", "data2", {s0, s1}, {s1, af::sym::kSymbolOne})
      .Concat("concat", {"load0", "sum0", "load2"}, 1)
      .Store("store", "concat")
      .Output("out", "store", 0, af::DT_FLOAT16)
      .Build();

  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);

  optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  std::vector<optimize::ConcatGroupPartitioner::ConcatGroup> groups;
  ASSERT_EQ(partitioner.PartitionGroups(groups), af::SUCCESS);

  size_t input_with_reduce = 1UL;
  bool found_single_group_for_reduce = false;
  for (const auto &group : groups) {
    if (group.start == input_with_reduce && group.end == input_with_reduce + 1) {
      found_single_group_for_reduce = true;
    }
  }
  EXPECT_TRUE(found_single_group_for_reduce);
}

TEST_F(ConcatScheduleCaseGeneratorV2Test, SixInputsWithSharedReduceAndTranspose) {
  auto s0 = af::Symbol("s0");
  auto s1 = af::Symbol(4);
  auto s2 = s1 + s1 + s1 + s1 + s1 + s1;
  auto graph = af::testing::AscGraphBuilder("test_6input_reduce_transpose")
      .Loops({s0, s2})
      .Data("data0", 0, af::DT_FLOAT16)
      .Load("load0", "data0", {s0, s1}, {s1, af::sym::kSymbolOne})
      .Data("data_r", 1, af::DT_FLOAT16)
      .Load("load_r", "data_r", {s0, s1}, {s1, af::sym::kSymbolOne})
      .Sum("reduce0", "load_r", {1})
      .Data("data2", 2, af::DT_FLOAT16)
      .Load("load2", "data2", {s0, s1}, {s1, af::sym::kSymbolOne})
      .Data("data_t", 3, af::DT_FLOAT16)
      .Load("load_t", "data_t", {s0, s1}, {s1, af::sym::kSymbolOne})
      .Transpose("transpose0", "load_t", {1, 0})
      .Data("data4", 4, af::DT_FLOAT16)
      .Load("load4", "data4", {s0, s1}, {s1, af::sym::kSymbolOne})
      .Concat("concat", {"reduce0", "load0", "transpose0", "load2", "load4", "reduce0"}, 1)
      .Store("store", "concat")
      .Output("out", "store", 0, af::DT_FLOAT16)
      .Build();

  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);

  optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  std::vector<optimize::ConcatGroupPartitioner::ConcatGroup> groups;
  ASSERT_EQ(partitioner.PartitionGroups(groups), af::SUCCESS);

  std::set<size_t> forced_single;
  for (const auto &group : groups) {
    if (group.end - group.start == 1) {
      forced_single.insert(group.start);
    }
  }
  EXPECT_TRUE(forced_single.find(0) != forced_single.end());
  EXPECT_TRUE(forced_single.find(2) != forced_single.end());
  EXPECT_TRUE(forced_single.find(5) != forced_single.end());
  EXPECT_TRUE(forced_single.find(3) == forced_single.end());
  EXPECT_TRUE(forced_single.find(4) == forced_single.end());
  size_t total_inputs = 6UL;
  size_t covered = 0UL;
  for (const auto &group : groups) {
    covered += group.end - group.start;
  }
  EXPECT_EQ(covered, total_inputs);
  EXPECT_FALSE(partitioner.HasRecompute());
}

TEST_F(ConcatScheduleCaseGeneratorV2Test, Generate_SkipsUBConcatForGraphWithTransposeOrReduce) {
  auto s0 = af::Symbol("s0");
  auto s1 = af::Symbol(4);
  auto s2 = s1 + s1;

  auto graph_with_reduce = af::testing::AscGraphBuilder("graph_with_reduce")
      .Loops({s0, s2})
      .Data("data0", 0, af::DT_FLOAT16)
      .Load("load0", "data0", {s0, s1}, {s1, af::sym::kSymbolOne})
      .Data("data1", 1, af::DT_FLOAT16)
      .Load("load1", "data1", {s0, s1}, {s1, af::sym::kSymbolOne})
      .Sum("reduce0", "load1", {1})
      .Concat("concat", {"load0", "reduce0"}, 1)
      .Store("store", "concat")
      .Output("out", "store", 0, af::DT_FLOAT16)
      .Build();

  std::vector<std::string> score_functions;
  std::vector<::ascir::ImplGraph> graphs_with_reduce;
  optimize::ConcatFusionCaseGenerator generator;
  EXPECT_EQ(generator.Generate(graph_with_reduce, graphs_with_reduce, score_functions), SUCCESS);

  auto graph_no_reduce = af::testing::AscGraphBuilder("graph_no_reduce")
      .Loops({s0, s2})
      .Data("data0", 0, af::DT_FLOAT16)
      .Load("load0", "data0", {s0, s1}, {s1, af::sym::kSymbolOne})
      .Data("data1", 1, af::DT_FLOAT16)
      .Load("load1", "data1", {s0, s1}, {s1, af::sym::kSymbolOne})
      .Concat("concat", {"load0", "load1"}, 1)
      .Store("store", "concat")
      .Output("out", "store", 0, af::DT_FLOAT16)
      .Build();

  std::vector<std::string> score_functions2;
  std::vector<::ascir::ImplGraph> graphs_no_reduce;
  optimize::ConcatFusionCaseGenerator generator2;
  EXPECT_EQ(generator2.Generate(graph_no_reduce, graphs_no_reduce, score_functions2), SUCCESS);
  EXPECT_FALSE(graphs_no_reduce.empty());

  EXPECT_TRUE(graphs_no_reduce.size() > graphs_with_reduce.size());
}
}  // namespace schedule