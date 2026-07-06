/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "node_utils_ex.h"
#include "graph_utils.h"
#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen_kernel.h"
#include "codegen_graph_check.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common_utils.h"
#include "utils/api_call_factory.h"
#include "utils/api_call_utils.h"
#include "autofuse_config/auto_fuse_config.h"
#include "graph_pass/masked_fill_input_reorder_pass.h"

using namespace ge;
using namespace af::ops;
using namespace codegen;
using namespace af::ascir_op;

namespace {
std::string ToString(const af::Expression &e) {
  return std::string(e.Serialize().get());
}
}  // namespace

TEST(CodegenKernel, Kernel_DynamicInputDtypeCheck) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  af::ascir_op::Data x1_op("x1", graph);
  x1_op.ir_attr.SetIndex(0);
  af::ascir_op::Data x2_op("x2", graph);
  x2_op.ir_attr.SetIndex(1);

  af::ascir_op::Load load1_op("load1");
  af::ascir_op::Load load2_op("load2");

  af::ascir_op::Concat concat_op("concat");
  af::ascir_op::Store store_op("store");
  af::ascir_op::Output y_op("y");
  y_op.ir_attr.SetIndex(0);

  graph.AddNode(load1_op);
  graph.AddNode(load2_op);
  graph.AddNode(store_op);
  graph.AddNode(y_op);

  x1_op.y.dtype = ge::DT_FLOAT16;
  x2_op.y.dtype = ge::DT_FLOAT16;

  load1_op.x = x1_op.y;
  load1_op.y.dtype = ge::DT_FLOAT16;

  load2_op.x = x2_op.y;
  load2_op.y.dtype = ge::DT_FLOAT16;

  concat_op.x = {load1_op.y, load2_op.y};
  concat_op.y.dtype = ge::DT_FLOAT16;

  store_op.x = concat_op.y;
  store_op.y.dtype = ge::DT_FLOAT16;

  y_op.x = store_op.y;
  y_op.y.dtype = ge::DT_FLOAT16;

  auto x1 = graph.FindNode("x1");
  auto x2 = graph.FindNode("x2");
  auto load1 = graph.FindNode("load1");
  auto load2 = graph.FindNode("load2");
  auto concat = graph.FindNode("concat");
  auto store = graph.FindNode("store");
  auto y = graph.FindNode("y");

  x1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.tensor_id = 0;
  x2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.tensor_id = 1;

  load1->outputs[0].attr.axis = {z0.id};
  load1->outputs[0].attr.vectorized_axis = {z0.id};
  load1->outputs[0].attr.vectorized_strides = {One};
  load1->outputs[0].attr.repeats = {z0.size};
  load1->outputs[0].attr.strides = {One};
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.mem.tensor_id = 2;
  load1->outputs[0].attr.que.id = 0;
  load1->outputs[0].attr.mem.reuse_id = 0;
  load1->outputs[0].attr.que.depth = 2;
  load1->outputs[0].attr.que.buf_num = 2;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  load2->outputs[0].attr.axis = {z0.id};
  load2->outputs[0].attr.vectorized_axis = {z0.id};
  load2->outputs[0].attr.vectorized_strides = {One};
  load2->outputs[0].attr.repeats = {z0.size};
  load2->outputs[0].attr.strides = {One};
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.mem.tensor_id = 3;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.mem.reuse_id = 0;
  load2->outputs[0].attr.que.depth = 2;
  load2->outputs[0].attr.que.buf_num = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  concat->attr.api.unit = af::ComputeUnit::kUnitVector;
  concat->outputs[0].attr.axis = {z0.id};
  concat->outputs[0].attr.vectorized_axis = {z0.id};
  concat->outputs[0].attr.vectorized_strides = {One};
  concat->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  concat->outputs[0].attr.mem.tensor_id = 4;
  concat->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  concat->outputs[0].attr.que.id = 2;
  concat->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.tensor_id = 5;

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.input_nodes.push_back(x1);
  fused_schedule_result.input_nodes.push_back(x2);
  fused_schedule_result.output_nodes.push_back(y);
  codegen::Kernel kernel(graph.GetName());
  auto ret = IsDataTypeSupported(graph);
  EXPECT_EQ(ret, af::SUCCESS);
}

TEST(CodegenKernel, Kernel_MaskedFillReorderedInputDtypeCheck) {
  af::AscGraph graph("test_graph");
  af::ascir_op::Data x_op("x", graph);
  x_op.ir_attr.SetIndex(0);
  af::ascir_op::Data mask_op("mask", graph);
  mask_op.ir_attr.SetIndex(1);
  af::ascir_op::Data value_op("value", graph);
  value_op.ir_attr.SetIndex(2);

  af::ascir_op::MaskedFill masked_fill_op("masked_fill");
  af::ascir_op::Store store_op("store");
  af::ascir_op::Output y_op("y");
  y_op.ir_attr.SetIndex(0);

  graph.AddNode(masked_fill_op);
  graph.AddNode(store_op);
  graph.AddNode(y_op);

  x_op.y.dtype = ge::DT_FLOAT16;
  mask_op.y.dtype = ge::DT_UINT8;
  value_op.y.dtype = ge::DT_FLOAT16;

  masked_fill_op.x = x_op.y;
  masked_fill_op.mask = mask_op.y;
  masked_fill_op.value = value_op.y;
  masked_fill_op.y.dtype = ge::DT_FLOAT16;

  store_op.x = masked_fill_op.y;
  store_op.y.dtype = ge::DT_FLOAT16;
  y_op.x = store_op.y;
  y_op.y.dtype = ge::DT_FLOAT16;

  optimize::MaskedFillInputReorderPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);

  auto ret = IsDataTypeSupported(graph);
  EXPECT_EQ(ret, af::SUCCESS);
}

TEST(CodegenKernel, Kernel_DataTypeRepeatsNodeUnsupportCheck) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  af::ascir_op::Scalar scalar("scalar", graph);
  scalar.ir_attr.SetIndex(1);
  scalar.ir_attr.SetValue("1.0");

  af::ascir_op::Abs abs_op("abs");
  af::ascir_op::Store store_op("store");
  af::ascir_op::Output y_op("y");
  y_op.ir_attr.SetIndex(0);

  graph.AddNode(abs_op);
  graph.AddNode(store_op);
  graph.AddNode(y_op);

  scalar.y.dtype = ge::DT_INT16;
  abs_op.x = scalar.y;
  abs_op.y.dtype = ge::DT_INT16;

  store_op.x = abs_op.y;
  store_op.y.dtype = ge::DT_INT16;

  y_op.x = store_op.y;
  y_op.y.dtype = ge::DT_INT16;

  auto abs = graph.FindNode("abs");
  auto store = graph.FindNode("store");
  auto y = graph.FindNode("y");

  abs->attr.api.unit = af::ComputeUnit::kUnitVector;
  abs->outputs[0].attr.axis = {z0.id};
  abs->outputs[0].attr.vectorized_axis = {z0.id};
  abs->outputs[0].attr.vectorized_strides = {One};
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  abs->outputs[0].attr.mem.tensor_id = 0;
  abs->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  abs->outputs[0].attr.que.id = 0;
  abs->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.tensor_id = 1;

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.output_nodes.push_back(y);
  codegen::Kernel kernel(graph.GetName());
  auto ret = IsDataTypeSupported(graph);
  EXPECT_NE(ret, af::SUCCESS);
  ret = IsRepeatStrideValid(graph);
  EXPECT_NE(ret, af::SUCCESS);
  ret = IsGraphNodeValid(graph);
  EXPECT_NE(ret, af::SUCCESS);
}

TEST(CodegenKernel, Kernel_ShapeConsistencyInValidCheck) {
  af::AscGraph graph("test_graph");
  const af::Expression s0 = graph.CreateSizeVar(3);
  auto z0 = graph.CreateAxis("z0", s0);
  const af::Expression s1 = graph.CreateSizeVar(4);
  auto z1 = graph.CreateAxis("z1", s1);
  const af::Expression s2 = graph.CreateSizeVar(5);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x1_op("x1", graph);
  x1_op.ir_attr.SetIndex(0);
  af::ascir_op::Data x2_op("x2", graph);
  x2_op.ir_attr.SetIndex(1);

  af::ascir_op::Load load1_op("load1");
  af::ascir_op::Load load2_op("load2");
  af::ascir_op::Maximum maximum_op("maximum");
  af::ascir_op::Store store_op("store");
  af::ascir_op::Output y_op("y");
  y_op.ir_attr.SetIndex(0);

  x1_op.y.dtype = ge::DT_FLOAT;
  x2_op.y.dtype = ge::DT_FLOAT;

  load1_op.x = x1_op.y;
  load1_op.y.dtype = ge::DT_FLOAT;

  load2_op.x = x2_op.y;
  load2_op.y.dtype = ge::DT_FLOAT;

  maximum_op.x1 = load1_op.y;
  maximum_op.x2 = load2_op.y;
  maximum_op.y.dtype = ge::DT_FLOAT;

  store_op.x = maximum_op.y;
  store_op.y.dtype = ge::DT_FLOAT;

  y_op.x = store_op.y;
  y_op.y.dtype = ge::DT_FLOAT;

  auto x1 = graph.FindNode("x1");
  auto x2 = graph.FindNode("x2");
  auto load1 = graph.FindNode("load1");
  auto load2 = graph.FindNode("load2");
  auto maximum = graph.FindNode("maximum");
  auto store = graph.FindNode("store");
  auto y = graph.FindNode("y");

  x1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.tensor_id = 0;
  x2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.tensor_id = 1;

  load1->outputs[0].attr.axis = {z0.id, z1.id, z2.id};
  load1->outputs[0].attr.vectorized_axis = {z2.id};
  load1->outputs[0].attr.vectorized_strides = {One};
  load1->outputs[0].attr.repeats = {z1.size, z0.size, z2.size};
  load1->outputs[0].attr.strides = {z0.size * z2.size, z2.size, One};
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.mem.tensor_id = 2;
  load1->outputs[0].attr.que.id = 0;
  load1->outputs[0].attr.mem.reuse_id = 0;
  load1->outputs[0].attr.que.depth = 2;
  load1->outputs[0].attr.que.buf_num = 2;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  load2->outputs[0].attr.axis = {z0.id, z1.id, z2.id};
  load2->outputs[0].attr.vectorized_axis = {z2.id};
  load2->outputs[0].attr.vectorized_strides = {One};
  load2->outputs[0].attr.repeats = {z0.size, z1.size, z2.size};
  load2->outputs[0].attr.strides = {z1.size * z2.size, z2.size, One};
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.mem.tensor_id = 3;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.mem.reuse_id = 0;
  load2->outputs[0].attr.que.depth = 2;
  load2->outputs[0].attr.que.buf_num = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  maximum->attr.api.unit = af::ComputeUnit::kUnitVector;
  maximum->outputs[0].attr.axis = {z0.id, z1.id, z2.id};
  maximum->outputs[0].attr.vectorized_axis = {z0.id};
  maximum->outputs[0].attr.vectorized_strides = {One};
  maximum->outputs[0].attr.repeats = {z0.size, z1.size, z2.size};
  maximum->outputs[0].attr.strides = {z1.size * z2.size, z2.size, One};
  maximum->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  maximum->outputs[0].attr.mem.tensor_id = 4;
  maximum->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  maximum->outputs[0].attr.que.id = 2;
  maximum->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.tensor_id = 5;

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.input_nodes.push_back(x1);
  fused_schedule_result.input_nodes.push_back(x2);
  fused_schedule_result.output_nodes.push_back(y);
  codegen::Kernel kernel(graph.GetName());
  auto ret = CheckGraphValidity(graph);
  EXPECT_NE(ret, af::SUCCESS);
}

TEST(CodegenKernel, Kernel_DynamicShapeConsistencyCheckValid) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  auto s1 = graph.CreateSizeVar("s1");
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar("s2");
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x1_op("x1", graph);
  x1_op.ir_attr.SetIndex(0);
  af::ascir_op::Data x2_op("x2", graph);
  x2_op.ir_attr.SetIndex(1);

  af::ascir_op::Load load1_op("load1");
  af::ascir_op::Load load2_op("load2");
  af::ascir_op::Maximum maximum_op("maximum");
  af::ascir_op::Store store_op("store");
  af::ascir_op::Output y_op("y");
  y_op.ir_attr.SetIndex(0);

  x1_op.y.dtype = ge::DT_FLOAT;
  x2_op.y.dtype = ge::DT_FLOAT;

  load1_op.x = x1_op.y;
  load1_op.y.dtype = ge::DT_FLOAT;

  load2_op.x = x2_op.y;
  load2_op.y.dtype = ge::DT_FLOAT;

  maximum_op.x1 = load1_op.y;
  maximum_op.x2 = load2_op.y;
  maximum_op.y.dtype = ge::DT_FLOAT;

  store_op.x = maximum_op.y;
  store_op.y.dtype = ge::DT_FLOAT;

  y_op.x = store_op.y;
  y_op.y.dtype = ge::DT_FLOAT;

  auto x1 = graph.FindNode("x1");
  auto x2 = graph.FindNode("x2");
  auto load1 = graph.FindNode("load1");
  auto load2 = graph.FindNode("load2");
  auto maximum = graph.FindNode("maximum");
  auto store = graph.FindNode("store");
  auto y = graph.FindNode("y");

  x1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.tensor_id = 0;
  x2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.tensor_id = 1;

  load1->outputs[0].attr.axis = {z0.id, z1.id, z2.id};
  load1->outputs[0].attr.vectorized_axis = {z2.id};
  load1->outputs[0].attr.vectorized_strides = {One};
  load1->outputs[0].attr.repeats = {z1.size, One, One};
  load1->outputs[0].attr.strides = {z2.size, Zero, Zero};
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.mem.tensor_id = 2;
  load1->outputs[0].attr.que.id = 0;
  load1->outputs[0].attr.mem.reuse_id = 0;
  load1->outputs[0].attr.que.depth = 2;
  load1->outputs[0].attr.que.buf_num = 2;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  load2->outputs[0].attr.axis = {z0.id, z1.id, z2.id};
  load2->outputs[0].attr.vectorized_axis = {z2.id};
  load2->outputs[0].attr.vectorized_strides = {One};
  load2->outputs[0].attr.repeats = {One, z1.size, z2.size};
  load2->outputs[0].attr.strides = {Zero, z2.size, One};
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.mem.tensor_id = 3;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.mem.reuse_id = 0;
  load2->outputs[0].attr.que.depth = 2;
  load2->outputs[0].attr.que.buf_num = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  maximum->attr.api.unit = af::ComputeUnit::kUnitVector;
  maximum->outputs[0].attr.axis = {z0.id, z1.id, z2.id};
  maximum->outputs[0].attr.vectorized_axis = {z0.id};
  maximum->outputs[0].attr.vectorized_strides = {One};
  maximum->outputs[0].attr.repeats = {z0.size, z1.size, z2.size};
  maximum->outputs[0].attr.strides = {z1.size * z2.size, z2.size, One};
  maximum->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  maximum->outputs[0].attr.mem.tensor_id = 4;
  maximum->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  maximum->outputs[0].attr.que.id = 2;
  maximum->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.tensor_id = 5;

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.input_nodes.push_back(x1);
  fused_schedule_result.input_nodes.push_back(x2);
  fused_schedule_result.output_nodes.push_back(y);
  codegen::Kernel kernel(graph.GetName());
  auto ret = CheckGraphValidity(graph);
  EXPECT_EQ(ret, af::SUCCESS);
}

TEST(CodegenKernel, Kernel_StaticShapeVecAxisConsistencyInValidCheck) {
  af::AscGraph graph("test_graph");
  const af::Expression s0 = graph.CreateSizeVar(3);
  auto z0 = graph.CreateAxis("z0", s0);
  const af::Expression s1 = graph.CreateSizeVar(4);
  auto z1 = graph.CreateAxis("z1", s1);
  const af::Expression s2 = graph.CreateSizeVar(5);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x1_op("x1", graph);
  x1_op.ir_attr.SetIndex(0);
  af::ascir_op::Data x2_op("x2", graph);
  x2_op.ir_attr.SetIndex(1);

  af::ascir_op::Load load1_op("load1");
  af::ascir_op::Load load2_op("load2");
  af::ascir_op::Pow pow_op("pow");
  af::ascir_op::Store store_op("store");
  af::ascir_op::Output y_op("y");
  y_op.ir_attr.SetIndex(0);

  x1_op.y.dtype = ge::DT_FLOAT;
  x2_op.y.dtype = ge::DT_FLOAT;

  load1_op.x = x1_op.y;
  load1_op.y.dtype = ge::DT_FLOAT;

  load2_op.x = x2_op.y;
  load2_op.y.dtype = ge::DT_FLOAT;

  pow_op.x1 = load1_op.y;
  pow_op.x2 = load2_op.y;
  pow_op.y.dtype = ge::DT_FLOAT;

  store_op.x = pow_op.y;
  store_op.y.dtype = ge::DT_FLOAT;

  y_op.x = store_op.y;
  y_op.y.dtype = ge::DT_FLOAT;

  auto x1 = graph.FindNode("x1");
  auto x2 = graph.FindNode("x2");
  auto load1 = graph.FindNode("load1");
  auto load2 = graph.FindNode("load2");
  auto pow = graph.FindNode("pow");
  auto store = graph.FindNode("store");
  auto y = graph.FindNode("y");

  x1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.tensor_id = 0;
  x2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.tensor_id = 1;

  load1->outputs[0].attr.axis = {z0.id, z1.id, z2.id};
  load1->outputs[0].attr.vectorized_axis = {z2.id};
  load1->outputs[0].attr.vectorized_strides = {One};
  load1->outputs[0].attr.repeats = {z0.size, z1.size, af::Symbol(2)};
  load1->outputs[0].attr.strides = {z1.size * af::Symbol(2), af::Symbol(2), One};
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.mem.tensor_id = 2;
  load1->outputs[0].attr.que.id = 0;
  load1->outputs[0].attr.mem.reuse_id = 0;
  load1->outputs[0].attr.que.depth = 2;
  load1->outputs[0].attr.que.buf_num = 2;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  load2->outputs[0].attr.axis = {z0.id, z1.id, z2.id};
  load2->outputs[0].attr.vectorized_axis = {z2.id};
  load2->outputs[0].attr.vectorized_strides = {One};
  load2->outputs[0].attr.repeats = {z0.size, z1.size, z2.size};
  load2->outputs[0].attr.strides = {z1.size * z2.size, z2.size, One};
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.mem.tensor_id = 3;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.mem.reuse_id = 0;
  load2->outputs[0].attr.que.depth = 2;
  load2->outputs[0].attr.que.buf_num = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  pow->attr.api.unit = af::ComputeUnit::kUnitVector;
  pow->outputs[0].attr.axis = {z0.id, z1.id, z2.id};
  pow->outputs[0].attr.vectorized_axis = {z2.id};
  pow->outputs[0].attr.vectorized_strides = {One};
  pow->outputs[0].attr.repeats = {z0.size, z1.size, z2.size};
  pow->outputs[0].attr.strides = {z1.size * z2.size, z2.size, One};
  pow->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  pow->outputs[0].attr.mem.tensor_id = 4;
  pow->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  pow->outputs[0].attr.que.id = 2;
  pow->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.tensor_id = 5;

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.input_nodes.push_back(x1);
  fused_schedule_result.input_nodes.push_back(x2);
  fused_schedule_result.output_nodes.push_back(y);
  codegen::Kernel kernel(graph.GetName());
  auto ret = CheckGraphValidity(graph);
  EXPECT_NE(ret, af::SUCCESS);
}

TEST(CodegenKernel, Kernel_DynamicShapeVecAxisConsistencyInValidCheck) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  auto s1 = graph.CreateSizeVar("s1");
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar("s2");
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x1_op("x1", graph);
  x1_op.ir_attr.SetIndex(0);
  af::ascir_op::Data x2_op("x2", graph);
  x2_op.ir_attr.SetIndex(1);

  af::ascir_op::Load load1_op("load1");
  af::ascir_op::Load load2_op("load2");
  af::ascir_op::Pow pow_op("pow");
  af::ascir_op::Store store_op("store");
  af::ascir_op::Output y_op("y");
  y_op.ir_attr.SetIndex(0);

  x1_op.y.dtype = ge::DT_FLOAT;
  x2_op.y.dtype = ge::DT_FLOAT;

  load1_op.x = x1_op.y;
  load1_op.y.dtype = ge::DT_FLOAT;

  load2_op.x = x2_op.y;
  load2_op.y.dtype = ge::DT_FLOAT;

  pow_op.x1 = load1_op.y;
  pow_op.x2 = load2_op.y;
  pow_op.y.dtype = ge::DT_FLOAT;

  store_op.x = pow_op.y;
  store_op.y.dtype = ge::DT_FLOAT;

  y_op.x = store_op.y;
  y_op.y.dtype = ge::DT_FLOAT;

  auto x1 = graph.FindNode("x1");
  auto x2 = graph.FindNode("x2");
  auto load1 = graph.FindNode("load1");
  auto load2 = graph.FindNode("load2");
  auto pow = graph.FindNode("pow");
  auto store = graph.FindNode("store");
  auto y = graph.FindNode("y");

  x1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.tensor_id = 0;
  x2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.tensor_id = 1;

  load1->outputs[0].attr.axis = {z0.id, z1.id, z2.id};
  load1->outputs[0].attr.vectorized_axis = {z2.id};
  load1->outputs[0].attr.vectorized_strides = {One};
  load1->outputs[0].attr.repeats = {z0.size, z1.size, af::Symbol(2)};
  load1->outputs[0].attr.strides = {z1.size * af::Symbol(2), af::Symbol(2), One};
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.mem.tensor_id = 2;
  load1->outputs[0].attr.que.id = 0;
  load1->outputs[0].attr.mem.reuse_id = 0;
  load1->outputs[0].attr.que.depth = 2;
  load1->outputs[0].attr.que.buf_num = 2;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  load2->outputs[0].attr.axis = {z0.id, z1.id, z2.id};
  load2->outputs[0].attr.vectorized_axis = {z2.id};
  load2->outputs[0].attr.vectorized_strides = {One};
  load2->outputs[0].attr.repeats = {z0.size, z1.size, z2.size};
  load2->outputs[0].attr.strides = {z1.size * z2.size, z2.size, One};
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.mem.tensor_id = 3;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.mem.reuse_id = 0;
  load2->outputs[0].attr.que.depth = 2;
  load2->outputs[0].attr.que.buf_num = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  pow->attr.api.unit = af::ComputeUnit::kUnitVector;
  pow->outputs[0].attr.axis = {z0.id, z1.id, z2.id};
  pow->outputs[0].attr.vectorized_axis = {z2.id};
  pow->outputs[0].attr.vectorized_strides = {One};
  pow->outputs[0].attr.repeats = {z0.size, z1.size, z2.size};
  pow->outputs[0].attr.strides = {z1.size * z2.size, z2.size, One};
  pow->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  pow->outputs[0].attr.mem.tensor_id = 4;
  pow->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  pow->outputs[0].attr.que.id = 2;
  pow->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.tensor_id = 5;

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.input_nodes.push_back(x1);
  fused_schedule_result.input_nodes.push_back(x2);
  fused_schedule_result.output_nodes.push_back(y);
  codegen::Kernel kernel(graph.GetName());
  auto ret = CheckGraphValidity(graph);
  EXPECT_EQ(ret, af::SUCCESS);
}
