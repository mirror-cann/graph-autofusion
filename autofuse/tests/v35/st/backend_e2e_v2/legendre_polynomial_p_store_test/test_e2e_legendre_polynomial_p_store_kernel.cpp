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

#include <cmath>
#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void legendre_polynomial_p_store_test(GM_ADDR x1, GM_ADDR x2, GM_ADDR y1,
                                                                       GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint32_t *workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2EBackendLegendrePolynomialPStoreCode : public testing::Test,
                                               public testing::WithParamInterface<std::vector<int>> {};

static float LegendrePolynomialPReference(float x, int n) {
  if (n == 0) {
    return 1.0F;
  }
  float prev2 = 1.0F;
  float prev1 = x;
  for (int k = 2; k <= n; ++k) {
    const float cur = ((2.0F * k - 1.0F) * x * prev1 - (k - 1.0F) * prev2) / static_cast<float>(k);
    prev2 = prev1;
    prev1 = cur;
  }
  return prev1;
}

TEST_P(E2EBackendLegendrePolynomialPStoreCode, CalculateCorrect) {
  auto legendre_shape = GetParam();
  uint64_t legendre_block_dim = 48;
  int legendre_test_size = legendre_shape[0] * legendre_shape[1];
  AutofuseTilingData tiling_data;
  float *legendre_x = static_cast<float *>(AscendC::GmAlloc(legendre_test_size * sizeof(float) + 32));
  int32_t *legendre_n = static_cast<int32_t *>(AscendC::GmAlloc(legendre_test_size * sizeof(int32_t) + 32));
  float *legendre_y = static_cast<float *>(AscendC::GmAlloc(legendre_test_size * sizeof(float) + 32));
  float *legendre_expect = static_cast<float *>(AscendC::GmAlloc(legendre_test_size * sizeof(float) + 32));

  for (int i = 0; i < legendre_test_size; i++) {
    legendre_x[i] = static_cast<float>((i % 7) - 3) / 4.0F;
    legendre_n[i] = 3;
    legendre_expect[i] = LegendrePolynomialPReference(legendre_x[i], legendre_n[i]);
  }

  uint32_t ws_size = 0;
  AutofuseTiling(legendre_shape[0], legendre_shape[1], &tiling_data, &ws_size, &legendre_block_dim, 48, 192 * 1024);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(legendre_polynomial_p_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(legendre_x),
              reinterpret_cast<uint8_t *>(legendre_n), reinterpret_cast<uint8_t *>(legendre_y), nullptr,
              reinterpret_cast<uint8_t *>(&tiling_data));

  uint32_t legendre_diff_count = 0;
  for (int i = 0; i < legendre_test_size; i++) {
    if (std::fabs(legendre_y[i] - legendre_expect[i]) > 1e-4F) {
      legendre_diff_count++;
    }
  }
  EXPECT_EQ(legendre_diff_count, 0U) << " of " << legendre_test_size;
  AscendC::GmFree(legendre_x);
  AscendC::GmFree(legendre_n);
  AscendC::GmFree(legendre_y);
  AscendC::GmFree(legendre_expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2EBackendLegendrePolynomialPStoreCode,
                         ::testing::Values(std::vector<int>{32, 16}, std::vector<int>{32, 18},
                                           std::vector<int>{512, 15}));
