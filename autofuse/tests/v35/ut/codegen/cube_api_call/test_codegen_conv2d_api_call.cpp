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
#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "codegen_kernel.h"
#include "cube_api_call/conv2d/conv2d_api_call.h"
#include "utils/api_call_factory.h"

using namespace af::ops;
using namespace codegen;
using namespace af::ascir_op;

class Conv2DApiCallTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(Conv2DApiCallTest, Conv2DApiCall_Register_Success) {
  auto api_call = ApiCallFactory::Instance().Create("Conv2DApiCall", "conv2d_api");
  EXPECT_NE(api_call, nullptr);
}

TEST_F(Conv2DApiCallTest, Conv2DApiCall_Constructor_Success) {
  Conv2DApiCall conv2d_api_call("conv2d_api");
}

TEST_F(Conv2DApiCallTest, Conv2DApiCall_GenerateMacro_Success) {
  Conv2DApiCall conv2d_api_call("conv2d_api");
  std::string result;

  EXPECT_EQ(conv2d_api_call.GenerateMacro(result), af::SUCCESS);
}

TEST_F(Conv2DApiCallTest, Conv2DApiCall_PreProcess_Success) {
  Conv2DApiCall conv2d_api_call("conv2d_api");
  Tiler tiler;
  TPipe tpipe("tpipe", tiler);
  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> outputs;
  std::string result;

  EXPECT_EQ(conv2d_api_call.PreProcess(tpipe, current_axis, outputs, result), af::SUCCESS);
}

TEST_F(Conv2DApiCallTest, Conv2DApiCall_PostProcess_Success) {
  Conv2DApiCall conv2d_api_call("conv2d_api");
  Tiler tiler;
  TPipe tpipe("tpipe", tiler);
  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> outputs;
  std::string result;

  EXPECT_EQ(conv2d_api_call.PostProcess(tpipe, current_axis, outputs, result), af::SUCCESS);
}

TEST_F(Conv2DApiCallTest, Conv2DApiCall_Generate_Success) {
  Conv2DApiCall conv2d_api_call("conv2d_api");
  Tiler tiler;
  TPipe tpipe("tpipe", tiler);
  std::vector<ascir::AxisId> current_axis;
  std::string result;

  EXPECT_EQ(conv2d_api_call.Generate(tpipe, current_axis, result), af::SUCCESS);
}

TEST_F(Conv2DApiCallTest, Conv2DApiCall_GenerateFuncDefinition_Success) {
  Conv2DApiCall conv2d_api_call("conv2d_api");
  Tiler tiler;
  TPipe tpipe("tpipe", tiler);
  std::stringstream ss;

  EXPECT_EQ(conv2d_api_call.GenerateFuncDefinition(tpipe, tiler, ss), af::SUCCESS);
}
