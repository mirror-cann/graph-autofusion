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

extern "C" __global__ __aicore__ void load_modified_bessel_k1_store_test(GM_ADDR x1, GM_ADDR y1, GM_ADDR workspace,
                                                                          GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData* tiling, uint32_t* workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2EBackendLoadModifiedBesselK1StoreCode : public testing::Test,
                                                public testing::WithParamInterface<std::vector<int>> {};

static float BesselI1ForK1Reference(float x) {
  const float half_x = x * 0.5F;
  float term = half_x;
  float sum = half_x;
  for (int k = 1; k <= 20; ++k) {
    term *= half_x * half_x / (static_cast<float>(k) * static_cast<float>(k + 1));
    sum += term;
  }
  return sum;
}

static float BesselK1Reference(float x) {
  if (x == 0.0F) {
    return 0.0F;
  }
  const float y = x * x / 4.0F;
  const float poly = 1.0F + y * (0.15443144F + y * (-0.67278579F + y * (-0.18156897F +
      y * (-0.01919402F + y * (-0.00110404F + y * -0.00004686F)))));
  return std::log(x / 2.0F) * BesselI1ForK1Reference(x) + poly / x;
}

TEST_P(E2EBackendLoadModifiedBesselK1StoreCode, CalculateCorrect) {
  auto k1_shape = GetParam();
  uint64_t k1_block_dim = 48;
  int k1_test_size = k1_shape[0] * k1_shape[1];

  AutofuseTilingData tiling_data;
  float* k1_x = static_cast<float*>(AscendC::GmAlloc(k1_test_size * sizeof(float) + 32));
  float* k1_y = static_cast<float*>(AscendC::GmAlloc(k1_test_size * sizeof(float) + 32));
  float* k1_expect = static_cast<float*>(AscendC::GmAlloc(k1_test_size * sizeof(float) + 32));

  for (int i = 0; i < k1_test_size; i++) {
    k1_x[i] = static_cast<float>((i % 8) + 1) / 4.0F;
    k1_expect[i] = BesselK1Reference(k1_x[i]);
  }

  uint32_t ws_size = 0;
  AutofuseTiling(k1_shape[0], k1_shape[1], &tiling_data, &ws_size, &k1_block_dim, 48, 192 * 1024);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_modified_bessel_k1_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t*>(k1_x),
              reinterpret_cast<uint8_t*>(k1_y), nullptr, reinterpret_cast<uint8_t*>(&tiling_data));

  uint32_t k1_diff_count = 0;
  for (int i = 0; i < k1_test_size; i++) {
    if (std::fabs(k1_y[i] - k1_expect[i]) > 1e-4F) {
      k1_diff_count++;
    }
  }
  EXPECT_EQ(k1_diff_count, 0U) << " of " << k1_test_size;

  AscendC::GmFree(k1_x);
  AscendC::GmFree(k1_y);
  AscendC::GmFree(k1_expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2EBackendLoadModifiedBesselK1StoreCode,
                         ::testing::Values(std::vector<int>{32, 16}, std::vector<int>{32, 18},
                                           std::vector<int>{512, 15}));
