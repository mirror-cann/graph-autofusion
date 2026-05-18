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

// Kernel函数声明
extern "C" __global__ __aicore__ void mask_reg_chain_test(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR x4, GM_ADDR x5,
                                                          GM_ADDR x6, GM_ADDR x7, GM_ADDR y1, GM_ADDR workspace,
                                                          GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint32_t *workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendMaskRegChain_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_BackendMaskRegChain_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint64_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;

  // VF外部MaskReg的存储格式：
  // 每个uint8_t元素对应一个mask bit：元素值为0表示bit=0，元素值为非0（如1）表示bit=1
  // 不再使用packed bit format（一个字节存储8个bit）

  // 阶段1: UINT8 mask tensor (one element per mask bit)
  // x1: test_size个uint8_t元素，每个元素代表一个mask bit
  uint8_t *x1 = (uint8_t *)AscendC::GmAlloc(test_size * sizeof(uint8_t) + 32);

  // Where0 输入 (INT64)
  int64_t *x2 = (int64_t *)AscendC::GmAlloc(test_size * sizeof(int64_t) + 32);
  int64_t *x3 = (int64_t *)AscendC::GmAlloc(test_size * sizeof(int64_t) + 32);

  // 阶段2: INT64 Compare -> Where (INT64→INT16)
  int64_t *x4 = (int64_t *)AscendC::GmAlloc(test_size * sizeof(int64_t) + 32);

  // Where1 输入 (INT16)
  int16_t *x5 = (int16_t *)AscendC::GmAlloc(test_size * sizeof(int16_t) + 32);
  int16_t *x6 = (int16_t *)AscendC::GmAlloc(test_size * sizeof(int16_t) + 32);

  // 阶段3: INT16 Compare
  int16_t *x7 = (int16_t *)AscendC::GmAlloc(test_size * sizeof(int16_t) + 32);

  // 输出 (one element per output bit)
  // y1: test_size个uint8_t元素，每个元素代表一个output bit
  uint8_t *y1 = (uint8_t *)AscendC::GmAlloc(test_size * sizeof(uint8_t) + 32);
  uint8_t *expect = (uint8_t *)AscendC::GmAlloc(test_size * sizeof(uint8_t) + 32);

  // Prepare test data
  srand(1);

  // 生成所有输入数据
  for (int i = 0; i < test_size; i++) {
    // x1: mask tensor，每个元素对应一个mask bit（0=bit=0, 1=bit=1）
    x1[i] = (rand() % 2) == 1 ? 1 : 0;
    x2[i] = static_cast<int64_t>(rand() % 10000);
    x3[i] = static_cast<int64_t>(rand() % 10000);
    x4[i] = static_cast<int64_t>(rand() % 10000);
    x5[i] = rand() % 100;
    x6[i] = rand() % 100;
    x7[i] = rand() % 100;  // INT16范围
    y1[i] = 0;
    expect[i] = 0;
  }

  // 计算期望输出
  for (int i = 0; i < test_size; i++) {
    // 阶段1: mask0直接从x1读取（x1[i]为0表示bit=0，非0表示bit=1）
    bool mask0 = (x1[i] != 0);
    int64_t result0 = mask0 ? x2[i] : x3[i];

    // 阶段2: INT64 Ge0(result0>=x4) -> Where1(x5, x6)
    bool mask1 = result0 >= x4[i];
    int16_t result1 = mask1 ? x5[i] : x6[i];

    // 阶段3: INT16 Ge1(result1>=x7) -> Store
    bool output_bit = (result1 >= x7[i]);

    // 输出：expect[i]为0表示bit=0，为1表示bit=1
    expect[i] = output_bit ? 1 : 0;
  }

  // Launch kernel
  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(mask_reg_chain_test, tiling_data.block_dim, (uint8_t *)x1, (uint8_t *)x2, (uint8_t *)x3, (uint8_t *)x4,
              (uint8_t *)x5, (uint8_t *)x6, (uint8_t *)x7, (uint8_t *)y1, nullptr, (uint8_t *)&tiling_data);

  // Count difference - one element per bit format
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    // y1[i]为0表示bit=0，非0（如1）表示bit=1
    uint8_t y1_bit = y1[i] != 0 ? 1 : 0;
    uint8_t expect_bit = expect[i] != 0 ? 1 : 0;
    if (y1_bit != expect_bit) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(x1);
  AscendC::GmFree(x2);  // int64_t
  AscendC::GmFree(x3);  // int64_t
  AscendC::GmFree(x4);  // int64_t
  AscendC::GmFree(x5);
  AscendC::GmFree(x6);
  AscendC::GmFree(x7);
  AscendC::GmFree(y1);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendMaskRegChain_Code,
                         ::testing::Values(std::vector<int>{3, 77}));