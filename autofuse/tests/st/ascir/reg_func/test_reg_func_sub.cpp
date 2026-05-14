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

#include "graph/operator_reg.h"
#include "graph_utils_ex.h"
#include "node_utils.h"
#include "op_desc_utils.h"

#include "ascir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

#include "../test_util.h"
namespace af{
namespace ascir{
extern std::vector<std::unique_ptr<af::TmpBufDesc>> CalcSubTmpSize(const af::AscNode &node);
using namespace testing;

class CalcSubTmpSizeTest:public::testing::Test{
protected:
  void SetUp() override{}
  void TearDown() override{}
};

TEST_F(CalcSubTmpSizeTest, CalcSubTmpSize_ShouldReturnCorrectSize_WhenNodeIsUbScalar)
{
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z0", s1);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Data x2("x2", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Load load2("load2");
  af::ascir_op::Sub sub("sub");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = ge::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  x2.attr.sched.axis = {z0.id, z1.id};
  x2.y.dtype = ge::DT_FLOAT;
  *x2.y.axis = {z0.id, z1.id};
  *x2.y.repeats = {Symbol(1), Symbol(1)};
  *x2.y.strides = {Symbol(1), Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  load2.x = x2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {Symbol(1), Symbol(1)};
  *load2.y.strides = {Symbol(1), Symbol(1)};
  *load2.y.vectorized_axis = {z0.id, z1.id};

  sub.x1 = load1.y;
  sub.x2 = load2.y;
  sub.attr.sched.axis = {z0.id, z1.id};
  sub.y.dtype = ge::DT_FLOAT;
  *sub.y.axis = {z0.id, z1.id};
  *sub.y.repeats = {s0, s1};
  *sub.y.strides = {s1, Symbol(1)};
  std::shared_ptr<af::AscNode> node = graph.FindNode("sub");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcSubTmpSize(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(8192));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcSubTmpSizeTest, CalcSubTmpSize_ShouldReturnCorrectSize_WhenNodeIsScalar)
{
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z0", s1);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Scalar x2("x2", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Sub sub("sub");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = ge::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  x2.attr.sched.axis = {z0.id, z1.id};
  x2.y.dtype = DT_FLOAT;
  *x2.y.axis = {};
  *x2.y.repeats = {};
  *x2.y.strides = {};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  sub.x1 = load1.y;
  sub.x2 = x2.y;
  sub.attr.sched.axis = {z0.id, z1.id};
  sub.y.dtype = ge::DT_FLOAT;
  *sub.y.axis = {z0.id, z1.id};
  *sub.y.repeats = {s0, s1};
  *sub.y.strides = {s1, Symbol(1)};
  std::shared_ptr<af::AscNode> node = graph.FindNode("sub");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcSubTmpSize(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(8192));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcSubTmpSizeTest, CalcSubTmpSize_ShouldReturnCorrectSize_WhenNodeIsTensor)
{
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto zo = graph.CreateAxis("zo", s1 + s2);
  auto zo_s_0 = graph.CreateAxis("zo_s_0", Axis::Type::kAxisTypeOriginal, s1, {zo.id}, af::kIdNone);
  auto zo_s_1 = graph.CreateAxis("zo_s_1", Axis::Type::kAxisTypeOriginal, s2, {zo.id}, af::kIdNone);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Sub sub("sub");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, zo_s_0.id};
  x1.y.dtype = ge::DT_FLOAT;
  *x1.y.axis = {z0.id, zo_s_0.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, zo_s_0.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, zo_s_0.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, zo_s_0.id};

  sub.x1 = load1.y;
  sub.x2 = load1.y;
  sub.attr.sched.axis = {z0.id, zo_s_0.id};
  sub.y.dtype = ge::DT_FLOAT;
  *sub.y.axis = {z0.id, zo_s_0.id};
  *sub.y.repeats = {s0, s1};
  *sub.y.strides = {s1, Symbol(1)};

  std::shared_ptr<af::AscNode> node = graph.FindNode("sub");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcSubTmpSize(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}
} // namespace asci
} // namespace af