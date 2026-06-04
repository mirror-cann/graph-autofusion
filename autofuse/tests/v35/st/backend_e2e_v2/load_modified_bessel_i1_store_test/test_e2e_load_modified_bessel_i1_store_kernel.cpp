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

extern "C" __global__ __aicore__ void load_modified_bessel_i1_store_test(GM_ADDR x1, GM_ADDR y1, GM_ADDR workspace,
                                                                          GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData* tiling, uint32_t* workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2EBackendLoadModifiedBesselI1StoreCode : public testing::Test,
                                                public testing::WithParamInterface<std::vector<int>> {};

static float BesselI1Reference(float x) {
  const float half_x = x * 0.5F;
  float term = half_x;
  float sum = half_x;
  for (int k = 1; k <= 20; ++k) {
    term *= half_x * half_x / (static_cast<float>(k) * static_cast<float>(k + 1));
    sum += term;
  }
  return sum;
}

TEST_P(E2EBackendLoadModifiedBesselI1StoreCode, CalculateCorrect) {
  auto i1_test_shape = GetParam();
  uint64_t i1_block_dim = 48;
  int i1_test_size = i1_test_shape[0] * i1_test_shape[1];

  AutofuseTilingData tiling_data;
  float* i1_x = static_cast<float*>(AscendC::GmAlloc(i1_test_size * sizeof(float) + 32));
  float* i1_y = static_cast<float*>(AscendC::GmAlloc(i1_test_size * sizeof(float) + 32));
  float* i1_expect = static_cast<float*>(AscendC::GmAlloc(i1_test_size * sizeof(float) + 32));

  for (int i = 0; i < i1_test_size; i++) {
    i1_x[i] = static_cast<float>((i % 9) - 4) / 4.0F;
    i1_expect[i] = BesselI1Reference(i1_x[i]);
  }

  uint32_t ws_size = 0;
  AutofuseTiling(i1_test_shape[0], i1_test_shape[1], &tiling_data, &ws_size, &i1_block_dim, 48, 192 * 1024);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_modified_bessel_i1_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t*>(i1_x),
              reinterpret_cast<uint8_t*>(i1_y), nullptr, reinterpret_cast<uint8_t*>(&tiling_data));

  uint32_t i1_diff_count = 0;
  for (int i = 0; i < i1_test_size; i++) {
    if (std::fabs(i1_y[i] - i1_expect[i]) > 1e-4F) {
      i1_diff_count++;
    }
  }
  EXPECT_EQ(i1_diff_count, 0U) << " of " << i1_test_size;

  AscendC::GmFree(i1_x);
  AscendC::GmFree(i1_y);
  AscendC::GmFree(i1_expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2EBackendLoadModifiedBesselI1StoreCode,
                         ::testing::Values(std::vector<int>{32, 16}, std::vector<int>{32, 18},
                                           std::vector<int>{512, 15}));
