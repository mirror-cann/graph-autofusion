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

extern "C" __global__ __aicore__ void load_airy_ai_store_test(GM_ADDR x1, GM_ADDR y1, GM_ADDR workspace,
                                                              GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint32_t *workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2EBackendLoadAiryAiStoreCode : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

static float AiryAiReference(float x) {
  if (std::fabs(x - 1.0F) < 1e-6F) {
    return 0.135292416F;
  }
  if (std::fabs(x + 1.0F) < 1e-6F) {
    return 0.535560883F;
  }
  return 0.355028054F;
}

TEST_P(E2EBackendLoadAiryAiStoreCode, CalculateCorrect) {
  auto airy_ai_shape = GetParam();
  uint64_t airy_ai_block_dim = 48;
  int airy_ai_test_size = airy_ai_shape[0] * airy_ai_shape[1];
  AutofuseTilingData tiling_data;
  float *airy_ai_x = static_cast<float *>(AscendC::GmAlloc(airy_ai_test_size * sizeof(float) + 32));
  float *airy_ai_y = static_cast<float *>(AscendC::GmAlloc(airy_ai_test_size * sizeof(float) + 32));
  float *airy_ai_expect = static_cast<float *>(AscendC::GmAlloc(airy_ai_test_size * sizeof(float) + 32));

  for (int i = 0; i < airy_ai_test_size; i++) {
    airy_ai_x[i] = static_cast<float>((i % 3) - 1);
    airy_ai_expect[i] = AiryAiReference(airy_ai_x[i]);
  }

  uint32_t ws_size = 0;
  AutofuseTiling(airy_ai_shape[0], airy_ai_shape[1], &tiling_data, &ws_size, &airy_ai_block_dim, 48, 192 * 1024);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_airy_ai_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(airy_ai_x),
              reinterpret_cast<uint8_t *>(airy_ai_y), nullptr, reinterpret_cast<uint8_t *>(&tiling_data));

  uint32_t airy_ai_diff_count = 0;
  for (int i = 0; i < airy_ai_test_size; i++) {
    if (std::fabs(airy_ai_y[i] - airy_ai_expect[i]) > 1e-4F) {
      airy_ai_diff_count++;
    }
  }
  EXPECT_EQ(airy_ai_diff_count, 0U) << " of " << airy_ai_test_size;
  AscendC::GmFree(airy_ai_x);
  AscendC::GmFree(airy_ai_y);
  AscendC::GmFree(airy_ai_expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2EBackendLoadAiryAiStoreCode,
                         ::testing::Values(std::vector<int>{32, 16}, std::vector<int>{32, 18},
                                           std::vector<int>{512, 15}));
