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
#include "codegen/ascendc_api_registry.h"

using namespace codegen;

class AscendcCubeApiRegisterConv2DTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(AscendcCubeApiRegisterConv2DTest, Register_Conv2DHeaders_Success) {
  auto &registry = AscendCApiRegistry::GetInstance();
  
  auto conv2d_include_headers = registry.GetFileContent("conv2d_include_headers.h");
  EXPECT_NE(conv2d_include_headers, "");
  
  auto conv2d_v2_tilingkey = registry.GetFileContent("conv2d_v2_tilingkey_cv.h");
  EXPECT_NE(conv2d_v2_tilingkey, "");
  
  auto conv_pingpong_basic_atcos = registry.GetFileContent("conv_pingpong_basic_atcos.h");
  EXPECT_NE(conv_pingpong_basic_atcos, "");
  
  auto conv2d = registry.GetFileContent("conv2d.h");
  EXPECT_NE(conv2d, "");
}

TEST_F(AscendcCubeApiRegisterConv2DTest, Register_MatmulDynamicHeaders_Success) {
  auto &registry = AscendCApiRegistry::GetInstance();
  
  auto matmul_dynamic = registry.GetFileContent("matmul_dynamic.h");
  EXPECT_NE(matmul_dynamic, "");
  
  auto mat_mul_tiling_key_dynamic = registry.GetFileContent("mat_mul_tiling_key_dynamic.h");
  EXPECT_NE(mat_mul_tiling_key_dynamic, "");
  
  auto batch_mat_mul_v3_tiling_key_dynamic = registry.GetFileContent("batch_mat_mul_v3_tiling_key_dynamic.h");
  EXPECT_NE(batch_mat_mul_v3_tiling_key_dynamic, "");
  
  auto mat_mul_pingpong_basic_cmct_dynamic = registry.GetFileContent("mat_mul_pingpong_basic_cmct_dynamic.h");
  EXPECT_NE(mat_mul_pingpong_basic_cmct_dynamic, "");
  
  auto batch_matmul_dynamic = registry.GetFileContent("batch_matmul_dynamic.h");
  EXPECT_NE(batch_matmul_dynamic, "");
}

TEST_F(AscendcCubeApiRegisterConv2DTest, Register_ExistingMatmulHeaders_StillAccessible) {
  auto &registry = AscendCApiRegistry::GetInstance();
  
  auto matmul = registry.GetFileContent("matmul.h");
  EXPECT_NE(matmul, "");
  
  auto mat_mul_include_headers = registry.GetFileContent("matmul_include_headers.h");
  EXPECT_NE(mat_mul_include_headers, "");
  
  auto batch_matmul = registry.GetFileContent("batch_matmul.h");
  EXPECT_NE(batch_matmul, "");
}

TEST_F(AscendcCubeApiRegisterConv2DTest, Register_Conv2DHeaders_NotEmpty) {
  auto &registry = AscendCApiRegistry::GetInstance();
  
  auto conv2d_include_headers = registry.GetFileContent("conv2d_include_headers.h");
  EXPECT_GT(conv2d_include_headers.length(), 0);
  
  auto conv2d = registry.GetFileContent("conv2d.h");
  EXPECT_GT(conv2d.length(), 0);
}

TEST_F(AscendcCubeApiRegisterConv2DTest, Register_AllHeadersCount) {
  auto &registry = AscendCApiRegistry::GetInstance();
  
  EXPECT_NE(registry.GetFileContent("conv2d_include_headers.h"), "");
  EXPECT_NE(registry.GetFileContent("conv2d_v2_tilingkey_cv.h"), "");
  EXPECT_NE(registry.GetFileContent("conv_pingpong_basic_atcos.h"), "");
  EXPECT_NE(registry.GetFileContent("conv2d.h"), "");
  EXPECT_NE(registry.GetFileContent("matmul_dynamic.h"), "");
  EXPECT_NE(registry.GetFileContent("mat_mul_tiling_key_dynamic.h"), "");
  EXPECT_NE(registry.GetFileContent("batch_mat_mul_v3_tiling_key_dynamic.h"), "");
  EXPECT_NE(registry.GetFileContent("mat_mul_pingpong_basic_cmct_dynamic.h"), "");
  EXPECT_NE(registry.GetFileContent("batch_matmul_dynamic.h"), "");
}