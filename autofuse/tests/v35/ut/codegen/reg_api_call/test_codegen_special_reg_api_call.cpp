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

#include "graph_utils.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen_kernel.h"
#include "common_utils.h"
#include "reg_api_call/unary_double_output_api_tmp_call.h"
#include "reg_api_call/unary_template_attr_api_tmp_call.h"

using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;

namespace {

void ConfigureLoadNode(const std::shared_ptr<af::AscNode> &load, af::AxisId loop_axis,
                       const std::vector<af::AxisId> &axis, const std::vector<af::Expression> &strides) {
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = loop_axis;
  load->outputs[0].attr.vectorized_axis = axis;
  load->outputs[0].attr.vectorized_strides = strides;
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;
}

void ConfigureOutputTensor(af::AscNode &node, size_t index, ge::DataType dtype, int64_t tensor_id, int64_t que_id,
                           const std::vector<af::AxisId> &axis, const std::vector<af::Expression> &strides) {
  node.outputs[index].attr.vectorized_axis = axis;
  node.outputs[index].attr.vectorized_strides = strides;
  node.outputs[index].attr.dtype = dtype;
  node.outputs[index].attr.mem.position = af::Position::kPositionVecOut;
  node.outputs[index].attr.mem.tensor_id = tensor_id;
  node.outputs[index].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  node.outputs[index].attr.que.id = que_id;
  node.outputs[index].attr.opt.merge_scope = af::kIdNone;
}

void ConfigureElewiseNode(const std::shared_ptr<af::AscNode> &node, af::AxisId loop_axis,
                          const af::Expression &tmp_size) {
  node->attr.api.compute_type = af::ComputeType::kComputeElewise;
  node->attr.api.type = af::ApiType::kAPITypeCompute;
  node->attr.api.unit = af::ComputeUnit::kUnitVector;
  node->attr.sched.loop_axis = loop_axis;
  node->attr.tmp_buffers = {{{tmp_size, -1}, af::MemAttr(), 0}};
}

void AddTilerInfo(codegen::Tiler &tiler, const af::Axis &z0, const af::Axis &z1, const af::Expression &s0,
                  const af::Expression &s1) {
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
}

codegen::ApiTensor MakeApiTensor(int64_t id) {
  codegen::ApiTensor tensor;
  tensor.id = id;
  return tensor;
}

void BuildFrexpGraph(af::AscGraph &graph, const af::Axis &z0, const af::Axis &z1, const af::Expression &s0,
                     const af::Expression &s1) {
  Data x_op("x", graph);
  Load load_op("load");
  Frexp frexp("op_node");
  graph.AddNode(load_op);
  graph.AddNode(frexp);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  frexp.x = load_op.y;
  *frexp.mantissa.axis = {z0.id, z1.id};
  *frexp.mantissa.repeats = {s0, s1};
  *frexp.mantissa.strides = {s1, One};
  *frexp.exponent.axis = {z0.id, z1.id};
  *frexp.exponent.repeats = {s0, s1};
  *frexp.exponent.strides = {s1, One};
}

void BuildShiftedChebyshevGraph(af::AscGraph &graph, const af::Axis &z0, const af::Axis &z1, const af::Expression &s0,
                                const af::Expression &s1) {
  Data x_op("x", graph);
  Load load_op("load");
  ShiftedChebyshevPolynomialT cheb("op_node");
  graph.AddNode(load_op);
  graph.AddNode(cheb);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  cheb.x = load_op.y;
  *cheb.y.axis = {z0.id, z1.id};
  *cheb.y.repeats = {s0, s1};
  *cheb.y.strides = {s1, One};
  cheb.ir_attr.SetN(3);
}

std::string GenerateFrexpResult(const std::shared_ptr<af::AscNode> &load, const std::shared_ptr<af::AscNode> &op_node,
                                codegen::TPipe &tpipe) {
  codegen::UnaryDoubleOutputApiTmpCall call("FrexpExtend");
  EXPECT_EQ(call.Init(op_node), 0);

  auto x1 = MakeApiTensor(load->outputs[0].attr.mem.tensor_id);
  auto y1 = MakeApiTensor(op_node->outputs[0].attr.mem.tensor_id);
  auto y2 = MakeApiTensor(op_node->outputs[1].attr.mem.tensor_id);
  call.inputs.push_back(&x1);
  call.outputs.push_back(y1);
  call.outputs.push_back(y2);

  std::string result;
  call.Generate(tpipe, std::vector<af::AxisId>{}, result);
  return result;
}

std::string GenerateShiftedChebyshevResult(const std::shared_ptr<af::AscNode> &load,
                                           const std::shared_ptr<af::AscNode> &op_node, codegen::TPipe &tpipe) {
  codegen::UnaryTemplateAttrApiTmpCall call("ShiftedChebyshevPolynomialTExtend");
  EXPECT_EQ(call.Init(op_node), 0);

  auto x1 = MakeApiTensor(load->outputs[0].attr.mem.tensor_id);
  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, std::vector<af::AxisId>{}, result);
  return result;
}

}  // namespace

TEST(SpecialApiCall, UnaryDoubleOutputApiTmpCall_Frexp) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  BuildFrexpGraph(graph, z0, z1, s0, s1);

  auto load = graph.FindNode("load");
  ConfigureLoadNode(load, z0.id, {z0.id, z1.id}, {s1, One});

  auto op_node = graph.FindNode("op_node");
  ConfigureElewiseNode(op_node, z0.id, af::Symbol(256));
  ConfigureOutputTensor(*op_node, 0, ge::DT_FLOAT, 1, 2, {z0.id, z1.id}, {s1, One});
  ConfigureOutputTensor(*op_node, 1, ge::DT_INT32, 2, 3, {z0.id, z1.id}, {s1, One});

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(op_node->outputs[0]);
  tpipe.AddTensor(op_node->outputs[1]);

  AddTilerInfo(tiler, z0, z1, s0, s1);

  std::string result = GenerateFrexpResult(load, op_node, tpipe);
  EXPECT_NE(result.find("FrexpExtend("), std::string::npos);
  EXPECT_NE(result.find("tmp_buf_"), std::string::npos);
}

TEST(SpecialApiCall, UnaryTemplateAttrApiTmpCall_ShiftedChebyshev) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  BuildShiftedChebyshevGraph(graph, z0, z1, s0, s1);

  auto load = graph.FindNode("load");
  ConfigureLoadNode(load, z0.id, {z0.id, z1.id}, {s1, One});

  auto op_node = graph.FindNode("op_node");
  ConfigureElewiseNode(op_node, z0.id, af::Symbol(0));
  ConfigureOutputTensor(*op_node, 0, ge::DT_FLOAT, 1, 2, {z0.id, z1.id}, {s1, One});

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(op_node->outputs[0]);

  AddTilerInfo(tiler, z0, z1, s0, s1);

  std::string result = GenerateShiftedChebyshevResult(load, op_node, tpipe);
  EXPECT_NE(result.find("ShiftedChebyshevPolynomialTExtend<float, 3>("), std::string::npos);
  EXPECT_NE(result.find("tmp_buf_"), std::string::npos);
}
