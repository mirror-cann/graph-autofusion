/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include "gtest/gtest.h"

#include "ascir.h"
#include "ascir_ops.h"
#include "default_reg_func_v2.h"

namespace af {
namespace ascir {
namespace {

std::shared_ptr<AscNode> BuildUnaryFloatNode(AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data("data", graph);
  af::ascir_op::Load load("load");
  af::ascir_op::Tanh op("op");

  data.attr.sched.axis = {z0.id, z1.id};
  data.y.dtype = ge::DT_FLOAT;
  *data.y.axis = {z0.id, z1.id};
  *data.y.repeats = {s0, s1};
  *data.y.strides = {s1, Symbol(1)};

  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, Symbol(1)};
  *load.y.vectorized_axis = {z0.id, z1.id};

  op.x = load.y;
  op.attr.sched.axis = {z0.id, z1.id};
  op.y.dtype = ge::DT_FLOAT;
  *op.y.axis = {z0.id, z1.id};
  *op.y.repeats = {s0, s1};
  *op.y.strides = {s1, Symbol(1)};

  return graph.FindNode("op");
}

void CheckApiLevelTmpBuffer(const std::vector<std::unique_ptr<TmpBufDesc>> &result) {
  ASSERT_EQ(result.size(), 1U);
  EXPECT_EQ(result[0]->size, Symbol(512));
  EXPECT_EQ(result[0]->life_time_axis_id, -1);
}

TEST(CalcModifiedBesselI0TmpSizeV2Test, ShouldReturnApiLevelTmpBuffer)
{
  af::AscGraph graph("test");
  auto node = BuildUnaryFloatNode(graph);
  ASSERT_NE(node, nullptr);
  auto result = CalcModifiedBesselI0TmpSizeV2(*node);
  CheckApiLevelTmpBuffer(result);
}

TEST(CalcModifiedBesselI1TmpSizeV2Test, ShouldReturnApiLevelTmpBuffer)
{
  af::AscGraph graph("test");
  auto node = BuildUnaryFloatNode(graph);
  ASSERT_NE(node, nullptr);
  auto i1_result = CalcModifiedBesselI1TmpSizeV2(*node);
  CheckApiLevelTmpBuffer(i1_result);
}

TEST(CalcModifiedBesselK0TmpSizeV2Test, ShouldReturnApiLevelTmpBuffer)
{
  af::AscGraph graph("test");
  auto node = BuildUnaryFloatNode(graph);
  ASSERT_NE(node, nullptr);
  auto k0_result = CalcModifiedBesselK0TmpSizeV2(*node);
  CheckApiLevelTmpBuffer(k0_result);
}

TEST(CalcModifiedBesselK1TmpSizeV2Test, ShouldReturnApiLevelTmpBuffer)
{
  af::AscGraph graph("test");
  auto node = BuildUnaryFloatNode(graph);
  ASSERT_NE(node, nullptr);
  auto k1_result = CalcModifiedBesselK1TmpSizeV2(*node);
  CheckApiLevelTmpBuffer(k1_result);
}

}  // namespace
}  // namespace ascir
}  // namespace af
