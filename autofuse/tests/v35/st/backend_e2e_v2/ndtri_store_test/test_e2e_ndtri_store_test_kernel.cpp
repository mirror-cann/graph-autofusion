/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cmath>
#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void ndtri_store_test(GM_ADDR x1, GM_ADDR y1, GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint32_t *workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2EBackendNdtriStoreCode : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

static float NdtriReference(float p) {
  constexpr float a1 = -3.969683028665376e+01F;
  constexpr float a2 = 2.209460984245205e+02F;
  constexpr float a3 = -2.759285104469687e+02F;
  constexpr float a4 = 1.383577518672690e+02F;
  constexpr float a5 = -3.066479806614716e+01F;
  constexpr float a6 = 2.506628277459239e+00F;
  constexpr float b1 = -5.447609879822406e+01F;
  constexpr float b2 = 1.615858368580409e+02F;
  constexpr float b3 = -1.556989798598866e+02F;
  constexpr float b4 = 6.680131188771972e+01F;
  constexpr float b5 = -1.328068155288572e+01F;
  constexpr float c1 = -7.784894002430293e-03F;
  constexpr float c2 = -3.223964580411365e-01F;
  constexpr float c3 = -2.400758277161838e+00F;
  constexpr float c4 = -2.549732539343734e+00F;
  constexpr float c5 = 4.374664141464968e+00F;
  constexpr float c6 = 2.938163982698783e+00F;
  constexpr float d1 = 7.784695709041462e-03F;
  constexpr float d2 = 3.224671290700398e-01F;
  constexpr float d3 = 2.445134137142996e+00F;
  constexpr float d4 = 3.754408661907416e+00F;
  constexpr float plow = 0.02425F;
  constexpr float phigh = 1.0F - plow;
  if (p < plow) {
    float q = std::sqrt(-2.0F * std::log(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0F);
  }
  if (p > phigh) {
    float q = std::sqrt(-2.0F * std::log(1.0F - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0F);
  }
  float q = p - 0.5F;
  float r = q * q;
  return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
         (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0F);
}

TEST_P(E2EBackendNdtriStoreCode, CalculateCorrect) {
  auto test_shape = GetParam();
  uint64_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  float *x = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));
  float *y = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));
  float *expect = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));

  for (int i = 0; i < test_size; i++) {
    x[i] = static_cast<float>((i % 98) + 1) / 100.0F;
    expect[i] = NdtriReference(x[i]);
  }

  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(ndtri_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(x), reinterpret_cast<uint8_t *>(y),
              nullptr, reinterpret_cast<uint8_t *>(&tiling_data));

  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    if (std::fabs(y[i] - expect[i]) > 2e-3F) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0U) << " of " << test_size;

  AscendC::GmFree(x);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2EBackendNdtriStoreCode,
                         ::testing::Values(std::vector<int>{32, 16}, std::vector<int>{32, 18},
                                           std::vector<int>{512, 15}));
