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

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void remainder_int32_store_test(GM_ADDR x1, GM_ADDR x2, GM_ADDR y1,
                                                                 GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData* tiling, uint32_t* workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2EBackendRemainderInt32StoreCode : public testing::Test,
                                          public testing::WithParamInterface<std::vector<int>> {};

static int32_t RemainderFloorDivReference(int32_t dividend, int32_t divisor) {
  if (divisor == 0) {
    return 0;
  }
  int32_t floor_quotient = dividend / divisor;
  const int32_t floor_remainder = dividend % divisor;
  if (floor_remainder != 0 && ((floor_remainder > 0) != (divisor > 0))) {
    floor_quotient--;
  }
  return floor_quotient;
}

TEST_P(E2EBackendRemainderInt32StoreCode, CalculateCorrect) {
  auto test_shape = GetParam();
  uint64_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  int32_t* x1 = static_cast<int32_t*>(AscendC::GmAlloc(test_size * sizeof(int32_t) + 32));
  int32_t* x2 = static_cast<int32_t*>(AscendC::GmAlloc(test_size * sizeof(int32_t) + 32));
  int32_t* y = static_cast<int32_t*>(AscendC::GmAlloc(test_size * sizeof(int32_t) + 32));
  int32_t* expect = static_cast<int32_t*>(AscendC::GmAlloc(test_size * sizeof(int32_t) + 32));

  for (int i = 0; i < test_size; i++) {
    x1[i] = static_cast<int32_t>((i % 31) - 15);
    x2[i] = static_cast<int32_t>((i % 7) + 1);
    if ((i % 5) == 0) {
      x2[i] = -x2[i];
    }
    expect[i] = x1[i] - x2[i] * RemainderFloorDivReference(x1[i], x2[i]);
  }

  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(remainder_int32_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t*>(x1),
              reinterpret_cast<uint8_t*>(x2), reinterpret_cast<uint8_t*>(y), nullptr,
              reinterpret_cast<uint8_t*>(&tiling_data));

  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }
  EXPECT_EQ(diff_count, 0U) << " of " << test_size;

  AscendC::GmFree(x1);
  AscendC::GmFree(x2);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2EBackendRemainderInt32StoreCode,
                         ::testing::Values(std::vector<int>{32, 16}, std::vector<int>{32, 18},
                                           std::vector<int>{512, 15}));
