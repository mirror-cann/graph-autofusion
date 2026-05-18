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

#include "graph/operator_reg.h"
#include "graph_utils_ex.h"
#include "node_utils.h"
#include "op_desc_utils.h"

#include "ascir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"
#include "default_reg_func_v2.h"

namespace ge {
namespace ascir {
std::vector<std::unique_ptr<ge::TmpBufDesc>>
CalcScaledModifiedBesselK0TmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>>
CalcScaledModifiedBesselK1TmpSizeV2(const ge::AscNode &node);
} // namespace ascir
} // namespace ge

namespace af {
namespace ascir {

using namespace testing;
using namespace af::ascir_op;

class CalcScaledModifiedBesselTmpSizeV2Test : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

/**
 * @tc.name: CalcScaledModifiedBesselK0TmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsFLOAT
 * @tc.number: CalcScaledModifiedBesselK0TmpSizeV2_Test_001
 * @tc.desc: Test CalcScaledModifiedBesselK0TmpSizeV2 returns correct size when input is DT_FLOAT
 */
TEST_F(CalcScaledModifiedBesselTmpSizeV2Test, CalcScaledModifiedBesselK0TmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsFLOAT)
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
    af::ascir_op::Sinh op_node("op_node");
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

    op_node.x = load1.y;
    op_node.attr.sched.axis = {z0.id, zo_s_0.id};
    op_node.y.dtype = ge::DT_FLOAT;
    *op_node.y.axis = {z0.id, zo_s_0.id};
    *op_node.y.repeats = {s0, s1};
    *op_node.y.strides = {s1, Symbol(1)};

    store.x = op_node.y;
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

    std::shared_ptr<AscNode> node = graph.FindNode("op_node");
    node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = ge::ascir::CalcScaledModifiedBesselK0TmpSizeV2(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, Symbol(512));  // 256 * 2
}

/**
 * @tc.name: CalcScaledModifiedBesselK1TmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsFLOAT
 * @tc.number: CalcScaledModifiedBesselK1TmpSizeV2_Test_001
 * @tc.desc: Test CalcScaledModifiedBesselK1TmpSizeV2 returns correct size when input is DT_FLOAT
 */
TEST_F(CalcScaledModifiedBesselTmpSizeV2Test, CalcScaledModifiedBesselK1TmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsFLOAT)
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
    af::ascir_op::Sinh op_node("op_node");
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

    op_node.x = load1.y;
    op_node.attr.sched.axis = {z0.id, zo_s_0.id};
    op_node.y.dtype = ge::DT_FLOAT;
    *op_node.y.axis = {z0.id, zo_s_0.id};
    *op_node.y.repeats = {s0, s1};
    *op_node.y.strides = {s1, Symbol(1)};

    store.x = op_node.y;
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

    std::shared_ptr<AscNode> node = graph.FindNode("op_node");
    node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = ge::ascir::CalcScaledModifiedBesselK1TmpSizeV2(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, Symbol(512));  // 256 * 2
}

} // namespace ascir
} // namespace af
