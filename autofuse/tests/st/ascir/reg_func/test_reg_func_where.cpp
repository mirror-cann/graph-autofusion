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

#include "graph/operator_reg_af.h"
#include "graph_utils_ex.h"
#include "node_utils.h"
#include "op_desc_utils.h"

#include "ascir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

#include "../test_util.h"
namespace af {
namespace ascir {
extern std::vector<std::unique_ptr<af::TmpBufDesc>> CalcWhereTmpSize(const af::AscNode &node);
extern std::vector<std::unique_ptr<af::TmpBufDesc>> CalcSelectTmpSize(const af::AscNode &node);
const Expression MAX_TMP_BUFFER_SIZE = af::Symbol(255 * 256 + 32);
using namespace testing;

class CalcWhereTmpSizeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

/**
 * @tc.name: CalcWhereTmpSize_ShouldReturnCorrectSize_WhenNodelsValid
 * @tc.number: CalcWhereTmpSize_Test_001
 * @tc.desc: Test when node is valid then CalcWhereTmpSize returns correct size
 */
TEST_F(CalcWhereTmpSizeTest, CalcWhereTmpSize_ShouldReturnCorrectSize_WhenNodelsValid) {
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
  af::ascir_op::Where where("where");
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

  where.x1 = load1.y;
  where.x2 = load1.y;
  where.x3 = load1.y;
  where.attr.sched.axis = {z0.id, zo_s_0.id};
  where.y.dtype = ge::DT_FLOAT;
  *where.y.axis = {z0.id, zo_s_0.id};
  *where.y.repeats = {s0, s1};
  *where.y.strides = {s1, Symbol(1)};

  store.x = where.y;
  store.attr.sched.axis = {z0.id, zo_s_0.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id, zo_s_0.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, Symbol(1)};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, zo_s_0.id};
  y.y.dtype = ge::DT_FLOAT;
  *y.y.axis = {z0.id, zo_s_0.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, Symbol(1)};

  std::shared_ptr<af::AscNode> node = graph.FindNode("where");
  node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcWhereTmpSize(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, sym::Min(af::Symbol(12) * s1 * s0, MAX_TMP_BUFFER_SIZE));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

class CalcSelectTmpSizeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

/**
 * @tc.name: CalcSelectTmpSize_ShouldReturnCorrectSize_WhenNodelsValid
 * @tc.number: CalcSelectTmpSize_Test_001
 * @tc.desc: Test when node is valid then CalcSelectTmpSize returns correct size
 */
TEST_F(CalcSelectTmpSizeTest, CalcSelectTmpSize_ShouldReturnCorrectSize_WhenNodelsValid) {
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
  af::ascir_op::Select select("select");
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

  select.x1 = load1.y;
  select.x2 = load1.y;
  select.x3 = load1.y;
  select.attr.sched.axis = {z0.id, zo_s_0.id};
  select.y.dtype = ge::DT_FLOAT;
  *select.y.axis = {z0.id, zo_s_0.id};
  *select.y.repeats = {s0, s1};
  *select.y.strides = {s1, Symbol(1)};

  store.x = select.y;
  store.attr.sched.axis = {z0.id, zo_s_0.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id, zo_s_0.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, Symbol(1)};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, zo_s_0.id};
  y.y.dtype = ge::DT_FLOAT;
  *y.y.axis = {z0.id, zo_s_0.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, Symbol(1)};

  std::shared_ptr<af::AscNode> node = graph.FindNode("select");
  node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcSelectTmpSize(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, sym::Min(af::Symbol(12) * s1 * s0, MAX_TMP_BUFFER_SIZE));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcSelectTmpSizeTest, CalcSelectTmpSize_ShouldReturnCorrectSize_Scalar) {
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
  af::ascir_op::Select select("select");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, zo_s_0.id};
  x1.y.dtype = ge::DT_FLOAT;
  *x1.y.axis = {z0.id, zo_s_0.id};
  *x1.y.repeats = {Symbol(1), Symbol(1)};
  *x1.y.strides = {Symbol(0), Symbol(0)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, zo_s_0.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, zo_s_0.id};
  *load1.y.repeats = {Symbol(1), Symbol(1)};
  *load1.y.strides = {Symbol(0), Symbol(0)};
  *load1.y.vectorized_axis = {z0.id, zo_s_0.id};

  select.x1 = load1.y;
  select.x2 = load1.y;
  select.x3 = load1.y;
  select.attr.sched.axis = {z0.id, zo_s_0.id};
  select.y.dtype = ge::DT_FLOAT;
  *select.y.axis = {z0.id, zo_s_0.id};
  *select.y.repeats = {Symbol(1), Symbol(1)};
  *select.y.strides = {Symbol(0), Symbol(0)};

  store.x = select.y;
  store.attr.sched.axis = {z0.id, zo_s_0.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id, zo_s_0.id};
  *store.y.repeats = {Symbol(1), Symbol(1)};
  *store.y.strides = {Symbol(0), Symbol(0)};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, zo_s_0.id};
  y.y.dtype = ge::DT_FLOAT;
  *y.y.axis = {z0.id, zo_s_0.id};
  *y.y.repeats = {Symbol(1), Symbol(1)};
  *y.y.strides = {Symbol(0), Symbol(0)};

  std::shared_ptr<af::AscNode> node = graph.FindNode("select");
  node->inputs[0].attr.vectorized_strides = {};
  node->inputs[1].attr.vectorized_strides = {Symbol(0), Symbol(0)};
  node->inputs[2].attr.vectorized_strides = {Symbol(0), Symbol(0)};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcSelectTmpSize(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(1) * af::Symbol(12));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcSelectTmpSizeTest, CalcSelectTmpSize_ShouldReturnCorrectSize_ConstScalar) {
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
  af::ascir_op::Select select("select");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, zo_s_0.id};
  x1.y.dtype = ge::DT_FLOAT;
  *x1.y.axis = {z0.id, zo_s_0.id};
  *x1.y.repeats = {};
  *x1.y.strides = {};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, zo_s_0.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, zo_s_0.id};
  *load1.y.repeats = {};
  *load1.y.strides = {};
  *load1.y.vectorized_axis = {};

  select.x1 = load1.y;
  select.x2 = load1.y;
  select.x3 = load1.y;
  select.attr.sched.axis = {z0.id, zo_s_0.id};
  select.y.dtype = ge::DT_FLOAT;
  *select.y.axis = {z0.id, zo_s_0.id};
  *select.y.repeats = {};
  *select.y.strides = {};

  store.x = select.y;
  store.attr.sched.axis = {z0.id, zo_s_0.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id, zo_s_0.id};
  *store.y.repeats = {};
  *store.y.strides = {};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, zo_s_0.id};
  y.y.dtype = ge::DT_FLOAT;
  *y.y.axis = {z0.id, zo_s_0.id};
  *y.y.repeats = {};
  *y.y.strides = {};

  std::shared_ptr<af::AscNode> node = graph.FindNode("select");
  node->inputs[0].attr.vectorized_strides = {};
  node->inputs[1].attr.vectorized_strides = {};
  node->inputs[2].attr.vectorized_strides = {};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcSelectTmpSize(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(12));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}
}  // namespace ascir
}  // namespace af
