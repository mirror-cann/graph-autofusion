/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void isinf_maskedfill_test(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, AutofuseTilingData *tiling,
                                  uint32_t *workspaceSize, uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendIsInfMaskedFill_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

namespace {
float BuildFloat(uint32_t bits) {
  union FloatBits {
    __aicore__ FloatBits() {}
    float value;
    uint32_t bits;
  } data;
  data.bits = bits;
  return data.value;
}
}  // namespace

TEST_P(E2E_BackendIsInfMaskedFill_Code, CalculateCorrect) {
  auto test_shape = GetParam();
  uint64_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1] * test_shape[2];
  AutofuseTilingData tiling_data;
  float *x = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));
  float *y = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));
  float *expect = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));

  float positive_inf = BuildFloat(0x7F800000U);
  float negative_inf = BuildFloat(0xFF800000U);
  constexpr float fill_value = -1.0F;
  for (int i = 0; i < test_size; i++) {
    if (i % 11 == 0) {
      x[i] = positive_inf;
      expect[i] = fill_value;
    } else if (i % 17 == 0) {
      x[i] = negative_inf;
      expect[i] = fill_value;
    } else {
      x[i] = static_cast<float>((i % 97) - 48) / 8.0F;
      expect[i] = x[i];
    }
  }

  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], test_shape[2], &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);
  printf("tiling key: %d, core_num: %d\n", tiling_data.tiling_key, tiling_data.block_dim);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(isinf_maskedfill_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(x),
              reinterpret_cast<uint8_t *>(y), nullptr, reinterpret_cast<uint8_t *>(&tiling_data));

  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    float diff = y[i] - expect[i];
    if (diff > 0.0001F || diff < -0.0001F) {
      diff_count++;
    }
  }
  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(x);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendIsInfMaskedFill_Code,
                         ::testing::Values(std::vector<int>{32, 16, 16}, std::vector<int>{32, 16, 18},
                                           std::vector<int>{32, 512, 15}));
