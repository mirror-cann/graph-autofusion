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

extern "C" __global__ __aicore__ void load_modified_bessel_k0_store_test(GM_ADDR x1, GM_ADDR y1, GM_ADDR workspace,
                                                                         GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint32_t *workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2EBackendLoadModifiedBesselK0StoreCode : public testing::Test,
                                                public testing::WithParamInterface<std::vector<int>> {};

static float BesselI0ForK0Reference(float x) {
  const float k0_half_x = x * 0.5F;
  float k0_series_term = 1.0F;
  float k0_series_sum = 1.0F;
  for (int order = 1; order <= 20; ++order) {
    const float k0_scale = k0_half_x / static_cast<float>(order);
    k0_series_term *= k0_scale * k0_scale;
    k0_series_sum += k0_series_term;
  }
  return k0_series_sum;
}

static float BesselK0Reference(float x) {
  const float y = x * x / 4.0F;
  const float poly =
      -0.57721566F +
      y * (0.42278420F +
           y * (0.23069756F + y * (0.03488590F + y * (0.00262698F + y * (0.00010750F + y * 0.00000740F)))));
  return -std::log(x / 2.0F) * BesselI0ForK0Reference(x) + poly;
}

TEST_P(E2EBackendLoadModifiedBesselK0StoreCode, CalculateCorrect) {
  auto k0_test_shape = GetParam();
  uint64_t k0_block_dim = 48;
  int k0_test_size = k0_test_shape[0] * k0_test_shape[1];

  AutofuseTilingData tiling_data;
  float *k0_x = static_cast<float *>(AscendC::GmAlloc(k0_test_size * sizeof(float) + 32));
  float *k0_y = static_cast<float *>(AscendC::GmAlloc(k0_test_size * sizeof(float) + 32));
  float *k0_expect = static_cast<float *>(AscendC::GmAlloc(k0_test_size * sizeof(float) + 32));

  for (int i = 0; i < k0_test_size; i++) {
    k0_x[i] = static_cast<float>((i % 8) + 1) / 4.0F;
    k0_expect[i] = BesselK0Reference(k0_x[i]);
  }

  uint32_t ws_size = 0;
  AutofuseTiling(k0_test_shape[0], k0_test_shape[1], &tiling_data, &ws_size, &k0_block_dim, 48, 192 * 1024);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_modified_bessel_k0_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(k0_x),
              reinterpret_cast<uint8_t *>(k0_y), nullptr, reinterpret_cast<uint8_t *>(&tiling_data));

  uint32_t k0_diff_count = 0;
  for (int i = 0; i < k0_test_size; i++) {
    if (std::fabs(k0_y[i] - k0_expect[i]) > 1e-4F) {
      k0_diff_count++;
    }
  }
  EXPECT_EQ(k0_diff_count, 0U) << " of " << k0_test_size;

  AscendC::GmFree(k0_x);
  AscendC::GmFree(k0_y);
  AscendC::GmFree(k0_expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2EBackendLoadModifiedBesselK0StoreCode,
                         ::testing::Values(std::vector<int>{32, 16}, std::vector<int>{32, 18},
                                           std::vector<int>{512, 15}));
