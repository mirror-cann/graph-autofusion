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
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common_utils.h"
#include "utils/api_call_factory.h"
#include "elewise/truediv_api_call.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;

TEST(CodegenKernel, TrueDivExtendWhenInputIsFLOAT) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Data x_op2("x2", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::TrueDiv true_div_op("true_div");
  graph.AddNode(load_op);
  graph.AddNode(true_div_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {s0, s1};
  *load_op2.y.strides = {s1, One};

  true_div_op.x1 = load_op.y;
  true_div_op.x2 = load_op2.y;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;


  auto true_div = graph.FindNode("true_div");
  true_div->attr.api.unit = af::ComputeUnit::kUnitVector;
  true_div->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  true_div->outputs[0].attr.dtype = ge::DT_FLOAT;
  true_div->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  true_div->outputs[0].attr.mem.tensor_id = 2;
  true_div->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  true_div->outputs[0].attr.que.id = 2;
  true_div->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(true_div->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::TrueDivApiCall call("TrueDivExtend");
  EXPECT_EQ(call.Init(true_div), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, std::string{
    "TrueDivExtend<float, float>(local_2[0], local_0[0], local_1[0], local_0_actual_size, tmp_buf_0);\n"
  });
}

TEST(CodegenKernel, TrueDivExtendWhenInputIsINT32) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Data x_op2("x2", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::TrueDiv true_div_op("true_div");
  graph.AddNode(load_op);
  graph.AddNode(true_div_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {s0, s1};
  *load_op2.y.strides = {s1, One};

  true_div_op.x1 = load_op.y;
  true_div_op.x2 = load_op2.y;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.dtype = ge::DT_INT32;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load2->outputs[0].attr.dtype = ge::DT_INT32;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;


  auto true_div = graph.FindNode("true_div");
  true_div->attr.api.unit = af::ComputeUnit::kUnitVector;
  true_div->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  true_div->outputs[0].attr.dtype = ge::DT_FLOAT;
  true_div->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  true_div->outputs[0].attr.mem.tensor_id = 2;
  true_div->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  true_div->outputs[0].attr.que.id = 2;
  true_div->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(true_div->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::TrueDivApiCall call("TrueDivExtend");
  EXPECT_EQ(call.Init(true_div), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, std::string{
    "TrueDivExtend<int32_t, float>(local_2[0], local_0[0], local_1[0], local_0_actual_size, tmp_buf_0);\n"
  });
}

TEST(CodegenKernel, TrueDivExtendsWhenInputIsFLOAT) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::TrueDiv true_div_op("true_div");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1.0");
  graph.AddNode(load_op);
  graph.AddNode(true_div_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  true_div_op.x2 = load_op.y;
  true_div_op.x1 = constant_op.y;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;


  auto constant_node = graph.FindNode("constant");
  constant_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  constant_node->outputs[0].attr.mem.tensor_id = 1;
  constant_node->outputs[0].attr.mem.position = af::Position::kPositionInvalid;


  auto true_div = graph.FindNode("true_div");
  true_div->attr.api.unit = af::ComputeUnit::kUnitVector;
  true_div->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  true_div->outputs[0].attr.dtype = ge::DT_FLOAT;
  true_div->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  true_div->outputs[0].attr.mem.tensor_id = 2;
  true_div->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  true_div->outputs[0].attr.que.id = 2;
  true_div->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(true_div->outputs[0]);
  tpipe.AddTensor("1.0", constant_node->outputs[0], "const_y");

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x2.id = load->outputs[0].attr.mem.tensor_id;
  x1.id = constant_node->outputs[0].attr.mem.tensor_id;

  codegen::TrueDivApiCall call("TrueDivExtend");
  EXPECT_EQ(call.Init(true_div), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, std::string{
    "TrueDivExtends<float, float, false>(local_2[0], local_0[0], (float)1.0, local_0_actual_size, tmp_buf_0);\n"
  });
}

TEST(CodegenKernel, TrueDivExtendsWhenInputIsINT32) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::TrueDiv true_div_op("true_div");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1");
  graph.AddNode(load_op);
  graph.AddNode(true_div_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  true_div_op.x2 = load_op.y;
  true_div_op.x1 = constant_op.y;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.dtype = ge::DT_INT32;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;


  auto constant_node = graph.FindNode("constant");
  constant_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  constant_node->outputs[0].attr.mem.tensor_id = 1;
  constant_node->outputs[0].attr.mem.position = af::Position::kPositionInvalid;
  constant_node->outputs[0].attr.dtype = ge::DT_INT32;


  auto true_div = graph.FindNode("true_div");
  true_div->attr.api.unit = af::ComputeUnit::kUnitVector;
  true_div->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  true_div->outputs[0].attr.dtype = ge::DT_FLOAT;
  true_div->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  true_div->outputs[0].attr.mem.tensor_id = 2;
  true_div->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  true_div->outputs[0].attr.que.id = 2;
  true_div->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(true_div->outputs[0]);
  tpipe.AddTensor("1", constant_node->outputs[0], "const_y");

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x2.id = load->outputs[0].attr.mem.tensor_id;
  x1.id = constant_node->outputs[0].attr.mem.tensor_id;

  codegen::TrueDivApiCall call("TrueDivExtend");
  EXPECT_EQ(call.Init(true_div), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, std::string{
    "TrueDivExtends<int32_t, float, false>(local_2[0], local_0[0], (int32_t)1, local_0_actual_size, tmp_buf_0);\n"
  });
}

TEST(CodegenKernel, TrueDivExtendsWithDoubleInputIsScalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::TrueDiv true_div_op("true_div");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1");
  graph.AddNode(load_op);
  graph.AddNode(true_div_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  true_div_op.x2 = constant_op.y;
  true_div_op.x1 = constant_op.y;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.dtype = ge::DT_INT32;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;


  auto constant_node = graph.FindNode("constant");
  constant_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  constant_node->outputs[0].attr.mem.tensor_id = 1;
  constant_node->outputs[0].attr.mem.position = af::Position::kPositionInvalid;
  constant_node->outputs[0].attr.dtype = ge::DT_INT32;


  auto true_div = graph.FindNode("true_div");
  true_div->attr.api.unit = af::ComputeUnit::kUnitVector;
  true_div->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  true_div->outputs[0].attr.dtype = ge::DT_FLOAT;
  true_div->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  true_div->outputs[0].attr.mem.tensor_id = 2;
  true_div->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  true_div->outputs[0].attr.que.id = 2;
  true_div->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(true_div->outputs[0]);
  tpipe.AddTensor("1", constant_node->outputs[0], "const_y");

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x2.id = constant_node->outputs[0].attr.mem.tensor_id;
  x1.id = constant_node->outputs[0].attr.mem.tensor_id;

  codegen::TrueDivApiCall call("TrueDivExtend");
  EXPECT_EQ(call.Init(true_div), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, std::string{
    "TrueDivExtends<int32_t, float>(local_2[0], (int32_t)1, (int32_t)1);\n"
  });
}
