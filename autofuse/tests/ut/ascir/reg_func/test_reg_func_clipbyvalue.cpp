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
extern std::vector<std::unique_ptr<af::TmpBufDesc>> CalcClipByValueTmpSize(const af::AscNode &node);

using namespace testing;

class CalcClipByValueTmpSizeTest:public::testing::Test{
protected:
    void SetUp() override{}
    void TearDown() override{}
};

/**
 * @tc.name:CalcClipByValueTmpSize_ShouldReturnCorrectSize_WhenNodelsValid
 * @tc.number: CalcClipByValueTmpSize_Test_001
 * @tc.desc: Test when node is valid then CalcClipByValueTmpSize returns correct size
 */
TEST_F(CalcClipByValueTmpSizeTest, CalcClipByValueTmpSize_ShouldReturnCorrectSize_WhenNodelsValid)
{
    af::SizeVar s0(af::Symbol("s0"));
    af::SizeVar s1(af::Symbol("s1"));
    af::SizeVar s2(af::Symbol("s2"));

    af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
    af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
    af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

    af::AscGraph graph("test");
    af::ascir_op::Data x("x", graph);
    std::shared_ptr<af::AscNode> node = graph.FindNode("x");
    std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcClipByValueTmpSize(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, af::Symbol(8192));
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}
} // namespace ascir
} // namespace ge