/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_TESTS_V35_ST_BACKEND_E2E_V2_BOOL_SELECT_KERNEL_COMMON_H_
#define AUTOFUSE_TESTS_V35_ST_BACKEND_E2E_V2_BOOL_SELECT_KERNEL_COMMON_H_

#include "tikicpulib.h"

struct BoolSelectKernelData {
  float *x1{nullptr};
  float *x2{nullptr};
  float *x3{nullptr};
  float *x4{nullptr};
  float *y1{nullptr};
};

inline BoolSelectKernelData PrepareBoolSelectKernelData(int test_size) {
  BoolSelectKernelData data;
  data.x1 = reinterpret_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));
  data.x2 = reinterpret_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));
  data.x3 = reinterpret_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));
  data.x4 = reinterpret_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));
  data.y1 = reinterpret_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));
  for (int i = 0; i < test_size; i++) {
    data.x1[i] = static_cast<float>((i % 5) + 1);
    data.x2[i] = static_cast<float>((i % 3) - 1);
    data.x3[i] = static_cast<float>(i + 1);
    data.x4[i] = static_cast<float>(-i - 1);
    data.y1[i] = -12345.0F;
  }
  return data;
}

inline uint32_t CountBoolSelectKernelDiff(const BoolSelectKernelData &data, int test_size) {
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    float expected = (data.x1[i] > data.x2[i]) ? data.x3[i] : data.x4[i];
    if (data.y1[i] != expected) {
      diff_count++;
    }
  }
  return diff_count;
}

inline void FreeBoolSelectKernelData(const BoolSelectKernelData &data) {
  AscendC::GmFree(data.x1);
  AscendC::GmFree(data.x2);
  AscendC::GmFree(data.x3);
  AscendC::GmFree(data.x4);
  AscendC::GmFree(data.y1);
}

#endif  // AUTOFUSE_TESTS_V35_ST_BACKEND_E2E_V2_BOOL_SELECT_KERNEL_COMMON_H_
