/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "gtest/gtest.h"
#define private public
#include "expr_gen/arg_list_reorder.h"

namespace att {
namespace {
std::map<std::string, size_t> BuildArgIdMap(const std::vector<AttAxisPtr> &arg_list) {
  std::map<std::string, size_t> arg_id_map;
  for (size_t i = 0; i < arg_list.size(); i++) {
    arg_id_map[arg_list[i]->name] = i;
  }
  return arg_id_map;
}

std::shared_ptr<Tensor> MakeTensor(const std::string &name, const std::vector<SubAxis *> &dims,
                                   const std::vector<ge::Expression> &repeat,
                                   const std::vector<ge::Expression> &stride) {
  auto tensor = std::make_shared<Tensor>();
  tensor->name = name;
  tensor->dim_info = dims;
  tensor->repeat = repeat;
  tensor->stride = stride;
  return tensor;
}

NodeInfo MakeNodeInfo(const std::string &name, const std::string &type,
                      const std::vector<std::shared_ptr<Tensor>> &inputs,
                      const std::vector<std::shared_ptr<Tensor>> &outputs) {
  NodeInfo node;
  node.name = name;
  node.node_type = type;
  node.inputs = inputs;
  node.outputs = outputs;
  return node;
}

std::shared_ptr<TuningSpace> BuildMatMulLoadTuningSpace(const NodeInfo &matmul, const NodeInfo &load,
                                                         std::vector<std::unique_ptr<SubAxis>> &axes) {
  auto tuning_space = std::make_shared<TuningSpace>();
  tuning_space->node_infos = {matmul, load};
  for (auto &axis : axes) {
    tuning_space->sub_axes.emplace_back(std::move(axis));
  }
  return tuning_space;
}

ModelInfo BuildModelInfo(const std::vector<std::string> &names) {
  ModelInfo model_info;
  for (const auto &name : names) {
    auto axis = std::make_shared<AttAxis>();
    axis->name = name;
    model_info.arg_list.push_back(axis);
  }
  return model_info;
}

struct LayerNormTestCtx {
  std::unique_ptr<SubAxis> z0 = std::make_unique<SubAxis>();
  std::unique_ptr<SubAxis> z1 = std::make_unique<SubAxis>();
  std::unique_ptr<SubAxis> z2 = std::make_unique<SubAxis>();
  std::unique_ptr<SubAxis> z0z1 = std::make_unique<SubAxis>();
  std::unique_ptr<SubAxis> z0z1t = std::make_unique<SubAxis>();
  std::unique_ptr<SubAxis> z2t = std::make_unique<SubAxis>();
  std::shared_ptr<TuningSpace> tuning_space = std::make_shared<TuningSpace>();
  ModelInfo model_info;

  LayerNormTestCtx() {
    z0->name = "z0";
    z1->name = "z1";
    z2->name = "z2";
    z0z1->name = "z0z1";
    z0z1t->name = "z0z1t";
    z2t->name = "z2t";

    auto att_z0 = std::make_shared<AttAxis>();
    auto att_z1 = std::make_shared<AttAxis>();
    auto att_z2 = std::make_shared<AttAxis>();
    auto att_z0z1 = std::make_shared<AttAxis>();
    auto att_z0z1t = std::make_shared<AttAxis>();
    auto att_z2t = std::make_shared<AttAxis>();

    att_z0->name = "z0";
    att_z1->name = "z1";
    att_z2->name = "z2";
    att_z0z1->name = "z0z1";
    att_z0z1t->name = "z0z1t";
    att_z2t->name = "z2t";

    att_z0z1->from_axis.emplace_back(att_z0.get());
    att_z0z1->from_axis.emplace_back(att_z1.get());
    att_z0z1t->from_axis.emplace_back(att_z0z1.get());
    att_z2t->from_axis.emplace_back(att_z2.get());

    model_info.arg_list = {att_z0, att_z1, att_z2, att_z0z1, att_z0z1t, att_z2t};
  }

  NodeInfo BuildNode(const std::string &node_name, const std::string &input_tensor_name,
                     const std::string &output_tensor_name) {
    NodeInfo node;
    auto tensor_in = std::make_shared<Tensor>();
    auto tensor_out = std::make_shared<Tensor>();

    z0z1t->is_node_innerest_dim = true;
    tensor_in->name = input_tensor_name;
    tensor_in->dim_info = {z0z1t.get(), z2t.get()};
    tensor_in->repeat = {ge::Symbol("z0z1t_size"), ge::Symbol("z2t_size")};
    tensor_in->stride = {ge::Symbol("z2t_size"), ge::Symbol(1, "ONE")};
    tensor_out->name = output_tensor_name;
    tensor_out->dim_info = {z0z1t.get(), z2t.get()};
    tensor_out->repeat = {ge::Symbol(1, "ONE"), ge::Symbol("z2t_size")};
    tensor_out->stride = {ge::Symbol(0, "ZERO"), ge::Symbol(1, "ONE")};

    node.name = node_name;
    node.node_type = "LayerNorm";
    node.inputs = {tensor_in};
    node.outputs = {tensor_out};
    return node;
  }

  void Finalize(const NodeInfo &node) {
    tuning_space->node_infos = {node};
    tuning_space->sub_axes.emplace_back(std::move(z0));
    tuning_space->sub_axes.emplace_back(std::move(z1));
    tuning_space->sub_axes.emplace_back(std::move(z2));
    tuning_space->sub_axes.emplace_back(std::move(z0z1));
    tuning_space->sub_axes.emplace_back(std::move(z0z1t));
    tuning_space->sub_axes.emplace_back(std::move(z2t));
  }
};
}  // namespace

class TestArgListReorder : public ::testing::Test {
 public:
  static void TearDownTestCase() { std::cout << "Test end." << std::endl; }
  static void SetUpTestCase() { std::cout << "Test begin." << std::endl; }
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(TestArgListReorder, case0) {
  auto M = ge::Symbol("M"), K = ge::Symbol("K"), N = ge::Symbol("N");
  auto MM = ge::Symbol("MM"), KK = ge::Symbol("KK"), NN = ge::Symbol("NN");
  auto ONE = ge::Symbol(1, "ONE"), ZERO = ge::Symbol(0, "ZERO");

  auto m = std::make_unique<SubAxis>(), k = std::make_unique<SubAxis>(), n = std::make_unique<SubAxis>();
  m->name = "m", k->name = "k", n->name = "n";
  n->is_node_innerest_dim = true;

  auto matmul = MakeNodeInfo("MatMul", "MatMul",
      {MakeTensor("MatMul_input_0", {m.get(), k.get()}, {M, K}, {MM, KK}),
       MakeTensor("MatMul_input_1", {k.get(), n.get()}, {K, N}, {KK, NN})},
      {MakeTensor("MatMul_output_0", {m.get(), n.get(), k.get()}, {M, N, ONE}, {MM, NN, ZERO})});

  auto a = std::make_unique<SubAxis>(), b = std::make_unique<SubAxis>();
  a->name = "a", b->name = "b";
  auto A = ge::Symbol("A"), B = ge::Symbol("B"), AA = ge::Symbol("AA"), BB = ge::Symbol("BB");

  auto load = MakeNodeInfo("Load", "Load",
      {MakeTensor("Load_input", {a.get(), b.get()}, {A, B}, {AA, BB})},
      {MakeTensor("Load_output", {a.get(), b.get()}, {A, B}, {AA, BB})});

  std::vector<std::unique_ptr<SubAxis>> axes;
  axes.push_back(std::move(m)), axes.push_back(std::move(k)), axes.push_back(std::move(n));
  axes.push_back(std::move(a)), axes.push_back(std::move(b));
  auto tuning_space = BuildMatMulLoadTuningSpace(matmul, load, axes);

  auto model_info = BuildModelInfo({"m", "k", "n", "a", "b"});
  ArgListReorder arg_list_reorder(tuning_space);
  std::vector<AttAxisPtr> tiling_R_arg_list;
  EXPECT_EQ(arg_list_reorder.SortArgList(model_info.arg_list, tiling_R_arg_list), ge::SUCCESS);
  auto arg_id_map = BuildArgIdMap(model_info.arg_list);
  EXPECT_EQ(arg_id_map["k"], 0);
  EXPECT_EQ(arg_id_map["n"], 1);
}

TEST_F(TestArgListReorder, case1) {
  LayerNormTestCtx ctx;
  auto node = ctx.BuildNode("Node1", "node1_input0", "node1_output0");
  ctx.Finalize(node);

  ArgListReorder arg_list_reorder(ctx.tuning_space);
  std::vector<AttAxisPtr> tiling_R_arg_list;
  EXPECT_EQ(arg_list_reorder.SortArgList(ctx.model_info.arg_list, tiling_R_arg_list), ge::SUCCESS);
  auto arg_id_map = BuildArgIdMap(ctx.model_info.arg_list);
  EXPECT_EQ(arg_id_map["z0z1t"] < arg_id_map["z2t"], true);
}

TEST_F(TestArgListReorder, case2) {
  LayerNormTestCtx ctx;
  auto node = ctx.BuildNode("Node1", "node1_output0", "node1_input0");
  // swap inputs/outputs for case2
  auto tmp = node.inputs;
  node.inputs = node.outputs;
  node.outputs = tmp;
  ctx.Finalize(node);

  ArgListReorder arg_list_reorder(ctx.tuning_space);
  std::vector<AttAxisPtr> tiling_R_arg_list;
  EXPECT_EQ(arg_list_reorder.SortArgList(ctx.model_info.arg_list, tiling_R_arg_list), ge::SUCCESS);
  auto arg_id_map = BuildArgIdMap(ctx.model_info.arg_list);
  EXPECT_EQ(arg_id_map["z0z1t"] < arg_id_map["z2t"], true);
}
}  // namespace att
