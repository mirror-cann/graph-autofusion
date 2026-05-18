/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <cmath>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"
extern "C" __global__ __aicore__ void vf_scalar_fusion_comprehensive_test(GM_ADDR x0, GM_ADDR x1, GM_ADDR y,
                                                                          GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(int s0, int s1, AutofuseTilingData *tiling, uint32_t *workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendVfScalarUbScalarFusion_Code : public testing::Test,
                                               public testing::WithParamInterface<std::vector<int>> {};

float MaxFloat(float a, float b) {
  return (a > b) ? a : b;
}

float MinFloat(float a, float b) {
  return (a < b) ? a : b;
}

TEST_P(E2E_BackendVfScalarUbScalarFusion_Code, CalculateCorrect) {
  auto test_shape = GetParam();
  uint64_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  half *x0 = (half *)AscendC::GmAlloc(test_size * sizeof(half) + 32);
  half *x1 = (half *)AscendC::GmAlloc(sizeof(half) + 32);  // scalar input
  half *y = (half *)AscendC::GmAlloc(test_size * sizeof(half) + 32);
  half *expect = (half *)AscendC::GmAlloc(test_size * sizeof(half) + 32);

  // Prepare test and expect data
  srand(1);
  float ubscalar_val = 1.0f;                // UbScalar输入值
  float scalar_const_val = 2.0f;            // Scalar节点常量值
  x1[0] = static_cast<half>(ubscalar_val);  // UbScalar input
  for (int i = 0; i < test_size; i++) {
    // 生成随机值，范围在0.0到10.0之间
    float rand_val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 10.0f;
    x0[i] = static_cast<half>(rand_val);

    // 计算期望输出
    // 图结构（同时包含UbScalar和Scalar）:
    // tensor(x0) -> load1
    //                 |
    //                 |-----> maximum0 = max(load1, ubscalar)      // UbScalar分支
    //                 |-----> minimum1 = min(load1, ubscalar)      // UbScalar分支
    //                 |-----> add2 = add(load1, ubscalar)          // UbScalar分支
    //                 |-----> add_scalar = add(load1, scalar_const) // Scalar分支
    //                 |
    // add3 = add(maximum0, minimum1)
    // minimum4 = min(add3, add2)
    // maximum5 = max(minimum4, add_scalar)  // 使用Scalar分支的结果
    // output = maximum5

    float f0 = rand_val;
    float us = ubscalar_val;      // UbScalar值 = 1.0
    float ss = scalar_const_val;  // Scalar值 = 2.0

    // UbScalar分支计算
    float maximum0_val = MaxFloat(us, f0);
    float minimum1_val = MinFloat(us, f0);
    float add2_val = us + f0;

    // Scalar分支计算
    float add_scalar_val = ss + f0;

    // 融合计算
    float add3_val = maximum0_val + minimum1_val;
    float minimum4_val = MinFloat(add3_val, add2_val);
    float maximum5_val = MaxFloat(minimum4_val, add_scalar_val);

    expect[i] = static_cast<half>(maximum5_val);
  }

  // Launch
  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);
  printf("tiling key: %d, core_num: %d\n", tiling_data.tiling_key, tiling_data.block_dim);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(vf_scalar_fusion_comprehensive_test, tiling_data.block_dim, (uint8_t *)x0, (uint8_t *)x1, (uint8_t *)y,
              nullptr, (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    printf("y[%d]: %f, expect[%d]: %f\n", i, static_cast<float>(y[i]), i, static_cast<float>(expect[i]));
    // 允许一定的误差
    if (fabs(static_cast<float>(y[i]) - static_cast<float>(expect[i])) > 0.001f) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(x0);
  AscendC::GmFree(x1);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendVfScalarUbScalarFusion_Code,
                         ::testing::Values(std::vector<int>{1, 1}));
