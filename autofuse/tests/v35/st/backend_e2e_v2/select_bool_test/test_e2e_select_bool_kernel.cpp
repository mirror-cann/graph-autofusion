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
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"
#include "../bool_select_kernel_common.h"

extern "C" __global__ __aicore__ void select_bool_test(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR x4, GM_ADDR y1,
                                                       GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint32_t *workspaceSize,
                                  uint32_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2EBackendSelectBoolCode : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2EBackendSelectBoolCode, CalculateCorrect) {
  GTEST_SKIP() << "CPU simulator does not support DT_BOOL VF instructions (scalar_pv: unknown instr->name())";
  auto test_shape = GetParam();
  uint32_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  auto data = PrepareBoolSelectKernelData(test_size);

  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(select_bool_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(data.x1),
              reinterpret_cast<uint8_t *>(data.x2), reinterpret_cast<uint8_t *>(data.x3),
              reinterpret_cast<uint8_t *>(data.x4), reinterpret_cast<uint8_t *>(data.y1), nullptr,
              reinterpret_cast<uint8_t *>(&tiling_data));

  uint32_t diff_count = CountBoolSelectKernelDiff(data, test_size);
  EXPECT_EQ(diff_count, 0) << " of " << test_size;
  FreeBoolSelectKernelData(data);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2EBackendSelectBoolCode, ::testing::Values(std::vector<int>{8, 8}));
