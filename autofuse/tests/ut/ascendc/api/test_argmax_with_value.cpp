/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "kernel_operator.h"
#include "tikicpulib.h"
using namespace AscendC;

#include "argmax_with_value.h"
#include "test_api_utils.h"
#include "utils.h"
#include "where.h"

class TestApiArgmaxWithValue
    : public testing::Test,
      public testing::WithParamInterface<std::vector<int>> {};

TEST_P(TestApiArgmaxWithValue, Test_argmax_with_value_float_ar) {
  int a = this->GetParam()[0];
  int b = this->GetParam()[1];
  // 构造测试输入和预期结果
  auto *x = (float *)AscendC::GmAlloc(sizeof(float) * a * b);
  auto *y = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * a);
  auto *z = (float *)AscendC::GmAlloc(sizeof(float) * a);
  int64_t expect_index[a];
  float expect_value[a];

  for (int i = 0; i < a; i++) {
    float max_value = -1e30f;
    int64_t max_index = 0;
    for (int j = 0; j < b; j++) {
      x[i * b + j] = (float)(i * b + j);
      if (x[i * b + j] > max_value) {
        max_value = x[i * b + j];
        max_index = j;
      }
    }
    expect_index[i] = max_index;
    expect_value[i] = max_value;
  }

  auto kernel = [](int32_t a, int32_t b, float *x, int64_t *y, float *z) {
    // 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x_buf, y_buf, z_buf, tmp;
    tpipe.InitBuffer(x_buf, sizeof(float) * a * b);
    tpipe.InitBuffer(y_buf, sizeof(int64_t) * a);
    tpipe.InitBuffer(z_buf, sizeof(float) * a);
    tpipe.InitBuffer(tmp, 16 * 1024);

    auto l_x = x_buf.Get<float>();
    auto l_y = y_buf.Get<int64_t>();
    auto l_z = z_buf.Get<float>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);
    GmToUb(l_y, y, a);
    GmToUb(l_z, z, a);

    uint32_t shape[] = {static_cast<uint32_t>(a), static_cast<uint32_t>(b)};

    ArgMaxWithValueExtend<int64_t, float, AscendC::Pattern::Reduce::AR>(
        l_y, l_z, l_x, l_tmp, shape, true);

    UbToGm(y, l_y, a);
    UbToGm(z, l_z, a);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y, z);

  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    if (y[i] != expect_index[i] || std::abs(z[i] - expect_value[i]) > 1e-5f) {
      diff_count++;
    }
  }

  // 验证结果
  EXPECT_EQ(diff_count, 0);
}

TEST_P(TestApiArgmaxWithValue, Test_argmax_with_value_int32_ar) {
  int a = this->GetParam()[0];
  int b = this->GetParam()[1];
  auto *x = (int32_t *)AscendC::GmAlloc(sizeof(int32_t) * a * b);
  auto *y = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * a);
  auto *z = (int32_t *)AscendC::GmAlloc(sizeof(int32_t) * a);
  int64_t expect_index[a];
  int32_t expect_value[a];

  for (int i = 0; i < a; i++) {
    int32_t max_value = -2147483647;
    int64_t max_index = 0;
    for (int j = 0; j < b; j++) {
      x[i * b + j] = (int32_t)(i * b + j);
      if (x[i * b + j] > max_value) {
        max_value = x[i * b + j];
        max_index = j;
      }
    }
    expect_index[i] = max_index;
    expect_value[i] = max_value;
  }

  auto kernel = [](int32_t a, int32_t b, int32_t *x, int64_t *y, int32_t *z) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x_buf, y_buf, z_buf, tmp;
    tpipe.InitBuffer(x_buf, sizeof(int32_t) * a * b);
    tpipe.InitBuffer(y_buf, sizeof(int64_t) * a);
    tpipe.InitBuffer(z_buf, sizeof(int32_t) * a);
    tpipe.InitBuffer(tmp, 16 * 1024);

    auto l_x = x_buf.Get<int32_t>();
    auto l_y = y_buf.Get<int64_t>();
    auto l_z = z_buf.Get<int32_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);
    GmToUb(l_y, y, a);
    GmToUb(l_z, z, a);

    uint32_t shape[] = {static_cast<uint32_t>(a), static_cast<uint32_t>(b)};

    ArgMaxWithValueExtend<int64_t, int32_t, AscendC::Pattern::Reduce::AR>(
        l_y, l_z, l_x, l_tmp, shape, true);

    UbToGm(y, l_y, a);
    UbToGm(z, l_z, a);
  };

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y, z);

  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    if (y[i] != expect_index[i] || z[i] != expect_value[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, TestApiArgmaxWithValue,
                         ::testing::Values(std::vector<int>{16, 32},
                                           std::vector<int>{32, 64},
                                           std::vector<int>{8, 128},
                                           std::vector<int>{4, 72}));

TEST(TestApiArgmaxWithValueRA, Test_argmax_with_value_float_ra) {
  uint32_t a = 16, b = 64;
  auto *x = (float *)AscendC::GmAlloc(sizeof(float) * a * b);
  auto *y = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * b);
  auto *z = (float *)AscendC::GmAlloc(sizeof(float) * b);
  int64_t expect_index[b];
  float expect_value[b];

  for (uint32_t i = 0; i < a * b; i++) {
    x[i] = (float)i;
  }

  for (uint32_t j = 0; j < b; j++) {
    float max_value = -1e30f;
    int64_t max_index = 0;
    for (uint32_t i = 0; i < a; i++) {
      uint32_t idx = i * b + j;
      if (x[idx] > max_value) {
        max_value = x[idx];
        max_index = i;
      }
    }
    expect_index[j] = max_index;
    expect_value[j] = max_value;
  }

  auto kernel = [](uint32_t a, uint32_t b, float *x, int64_t *y, float *z) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x_buf, y_buf, z_buf, tmp;
    tpipe.InitBuffer(x_buf, sizeof(float) * a * b);
    tpipe.InitBuffer(y_buf, sizeof(int64_t) * b);
    tpipe.InitBuffer(z_buf, sizeof(float) * b);
    tpipe.InitBuffer(tmp, 16 * 1024);

    auto l_x = x_buf.Get<float>();
    auto l_y = y_buf.Get<int64_t>();
    auto l_z = z_buf.Get<float>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);
    GmToUb(l_y, y, b);
    GmToUb(l_z, z, b);

    uint32_t shape[] = {a, b};

    ArgMaxWithValueExtend<int64_t, float, AscendC::Pattern::Reduce::RA>(
        l_y, l_z, l_x, l_tmp, shape, true);

    UbToGm(y, l_y, b);
    UbToGm(z, l_z, b);
  };

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y, z);

  uint32_t diff_count = 0;
  for (uint32_t i = 0; i < b; i++) {
    if (y[i] != expect_index[i] || std::abs(z[i] - expect_value[i]) > 1e-5f) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
}

TEST(TestApiArgmaxWithValueRA, Test_argmax_with_value_int32_ra) {
  uint32_t a = 16, b = 64;
  auto *x = (int32_t *)AscendC::GmAlloc(sizeof(int32_t) * a * b);
  auto *y = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * b);
  auto *z = (int32_t *)AscendC::GmAlloc(sizeof(int32_t) * b);
  int64_t expect_index[b];
  int32_t expect_value[b];

  for (uint32_t i = 0; i < a * b; i++) {
    x[i] = (int32_t)i;
  }

  for (uint32_t j = 0; j < b; j++) {
    int32_t max_value = -2147483647;
    int64_t max_index = 0;
    for (uint32_t i = 0; i < a; i++) {
      uint32_t idx = i * b + j;
      if (x[idx] > max_value) {
        max_value = x[idx];
        max_index = i;
      }
    }
    expect_index[j] = max_index;
    expect_value[j] = max_value;
  }

  auto kernel = [](uint32_t a, uint32_t b, int32_t *x, int64_t *y, int32_t *z) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x_buf, y_buf, z_buf, tmp;
    tpipe.InitBuffer(x_buf, sizeof(int32_t) * a * b);
    tpipe.InitBuffer(y_buf, sizeof(int64_t) * b);
    tpipe.InitBuffer(z_buf, sizeof(int32_t) * b);
    tpipe.InitBuffer(tmp, 16 * 1024);

    auto l_x = x_buf.Get<int32_t>();
    auto l_y = y_buf.Get<int64_t>();
    auto l_z = z_buf.Get<int32_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);
    GmToUb(l_y, y, b);
    GmToUb(l_z, z, b);

    uint32_t shape[] = {a, b};

    ArgMaxWithValueExtend<int64_t, int32_t, AscendC::Pattern::Reduce::RA>(
        l_y, l_z, l_x, l_tmp, shape, true);

    UbToGm(y, l_y, b);
    UbToGm(z, l_z, b);
  };

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y, z);

  uint32_t diff_count = 0;
  for (uint32_t i = 0; i < b; i++) {
    if (y[i] != expect_index[i] || z[i] != expect_value[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
}

TEST(TestApiArgmaxWithValueLE64, Test_argmax_with_value_float_le64) {
  uint32_t a = 32, b = 32;
  auto *x = (float *)AscendC::GmAlloc(sizeof(float) * a * b);
  auto *y = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * a);
  auto *z = (float *)AscendC::GmAlloc(sizeof(float) * a);
  int64_t expect_index[a];
  float expect_value[a];

  for (uint32_t i = 0; i < a; i++) {
    float max_value = -1e30f;
    int64_t max_index = 0;
    for (uint32_t j = 0; j < b; j++) {
      x[i * b + j] = (float)(i * b + j);
      if (x[i * b + j] > max_value) {
        max_value = x[i * b + j];
        max_index = j;
      }
    }
    expect_index[i] = max_index;
    expect_value[i] = max_value;
  }

  auto kernel = [](uint32_t a, uint32_t b, float *x, int64_t *y, float *z) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x_buf, y_buf, z_buf, tmp;
    tpipe.InitBuffer(x_buf, sizeof(float) * a * b);
    tpipe.InitBuffer(y_buf, sizeof(int64_t) * a);
    tpipe.InitBuffer(z_buf, sizeof(float) * a);
    tpipe.InitBuffer(tmp, 16 * 1024);

    auto l_x = x_buf.Get<float>();
    auto l_y = y_buf.Get<int64_t>();
    auto l_z = z_buf.Get<float>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);
    GmToUb(l_y, y, a);
    GmToUb(l_z, z, a);

    uint32_t shape[] = {a, b};

    ArgMaxWithValueExtend<int64_t, float, AscendC::Pattern::Reduce::AR>(
        l_y, l_z, l_x, l_tmp, shape, true);

    UbToGm(y, l_y, a);
    UbToGm(z, l_z, a);
  };

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y, z);

  uint32_t diff_count = 0;
  for (uint32_t i = 0; i < a; i++) {
    if (y[i] != expect_index[i] || std::abs(z[i] - expect_value[i]) > 1e-5f) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
}

TEST(TestApiArgmaxWithValueLE64, Test_argmax_with_value_int32_le64) {
  uint32_t a = 32, b = 32;
  auto *x = (int32_t *)AscendC::GmAlloc(sizeof(int32_t) * a * b);
  auto *y = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * a);
  auto *z = (int32_t *)AscendC::GmAlloc(sizeof(int32_t) * a);
  int64_t expect_index[a];
  int32_t expect_value[a];

  for (uint32_t i = 0; i < a; i++) {
    int32_t max_value = -2147483647;
    int64_t max_index = 0;
    for (uint32_t j = 0; j < b; j++) {
      x[i * b + j] = (int32_t)(i * b + j);
      if (x[i * b + j] > max_value) {
        max_value = x[i * b + j];
        max_index = j;
      }
    }
    expect_index[i] = max_index;
    expect_value[i] = max_value;
  }

  auto kernel = [](uint32_t a, uint32_t b, int32_t *x, int64_t *y, int32_t *z) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x_buf, y_buf, z_buf, tmp;
    tpipe.InitBuffer(x_buf, sizeof(int32_t) * a * b);
    tpipe.InitBuffer(y_buf, sizeof(int64_t) * a);
    tpipe.InitBuffer(z_buf, sizeof(int32_t) * a);
    tpipe.InitBuffer(tmp, 16 * 1024);

    auto l_x = x_buf.Get<int32_t>();
    auto l_y = y_buf.Get<int64_t>();
    auto l_z = z_buf.Get<int32_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);
    GmToUb(l_y, y, a);
    GmToUb(l_z, z, a);

    uint32_t shape[] = {a, b};

    ArgMaxWithValueExtend<int64_t, int32_t, AscendC::Pattern::Reduce::AR>(
        l_y, l_z, l_x, l_tmp, shape, true);

    UbToGm(y, l_y, a);
    UbToGm(z, l_z, a);
  };

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y, z);

  uint32_t diff_count = 0;
  for (uint32_t i = 0; i < a; i++) {
    if (y[i] != expect_index[i] || z[i] != expect_value[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
}

TEST(TestApiArgmaxWithValue, Test_argmax_with_value_float_negative) {
  uint32_t a = 8, b = 64;
  auto *x = (float *)AscendC::GmAlloc(sizeof(float) * a * b);
  auto *y = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * a);
  auto *z = (float *)AscendC::GmAlloc(sizeof(float) * a);
  int64_t expect_index[a];
  float expect_value[a];

  for (uint32_t i = 0; i < a; i++) {
    float max_value = -1e30f;
    int64_t max_index = 0;
    for (uint32_t j = 0; j < b; j++) {
      x[i * b + j] = (float)(-(int)(i * b + j));
      if (x[i * b + j] > max_value) {
        max_value = x[i * b + j];
        max_index = j;
      }
    }
    expect_index[i] = max_index;
    expect_value[i] = max_value;
  }

  auto kernel = [](uint32_t a, uint32_t b, float *x, int64_t *y, float *z) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x_buf, y_buf, z_buf, tmp;
    tpipe.InitBuffer(x_buf, sizeof(float) * a * b);
    tpipe.InitBuffer(y_buf, sizeof(int64_t) * a);
    tpipe.InitBuffer(z_buf, sizeof(float) * a);
    tpipe.InitBuffer(tmp, 16 * 1024);

    auto l_x = x_buf.Get<float>();
    auto l_y = y_buf.Get<int64_t>();
    auto l_z = z_buf.Get<float>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);
    GmToUb(l_y, y, a);
    GmToUb(l_z, z, a);

    uint32_t shape[] = {a, b};

    ArgMaxWithValueExtend<int64_t, float, AscendC::Pattern::Reduce::AR>(
        l_y, l_z, l_x, l_tmp, shape, true);

    UbToGm(y, l_y, a);
    UbToGm(z, l_z, a);
  };

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y, z);

  uint32_t diff_count = 0;
  for (uint32_t i = 0; i < a; i++) {
    if (y[i] != expect_index[i] || std::abs(z[i] - expect_value[i]) > 1e-5f) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
}

TEST(TestApiArgmaxWithValue, Test_argmax_with_value_int32_negative) {
  uint32_t a = 8, b = 64;
  auto *x = (int32_t *)AscendC::GmAlloc(sizeof(int32_t) * a * b);
  auto *y = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * a);
  auto *z = (int32_t *)AscendC::GmAlloc(sizeof(int32_t) * a);
  int64_t expect_index[a];
  int32_t expect_value[a];

  for (uint32_t i = 0; i < a; i++) {
    int32_t max_value = -2147483647;
    int64_t max_index = 0;
    for (uint32_t j = 0; j < b; j++) {
      x[i * b + j] = -(int32_t)(i * b + j);
      if (x[i * b + j] > max_value) {
        max_value = x[i * b + j];
        max_index = j;
      }
    }
    expect_index[i] = max_index;
    expect_value[i] = max_value;
  }

  auto kernel = [](uint32_t a, uint32_t b, int32_t *x, int64_t *y, int32_t *z) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x_buf, y_buf, z_buf, tmp;
    tpipe.InitBuffer(x_buf, sizeof(int32_t) * a * b);
    tpipe.InitBuffer(y_buf, sizeof(int64_t) * a);
    tpipe.InitBuffer(z_buf, sizeof(int32_t) * a);
    tpipe.InitBuffer(tmp, 16 * 1024);

    auto l_x = x_buf.Get<int32_t>();
    auto l_y = y_buf.Get<int64_t>();
    auto l_z = z_buf.Get<int32_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);
    GmToUb(l_y, y, a);
    GmToUb(l_z, z, a);

    uint32_t shape[] = {a, b};

    ArgMaxWithValueExtend<int64_t, int32_t, AscendC::Pattern::Reduce::AR>(
        l_y, l_z, l_x, l_tmp, shape, true);

    UbToGm(y, l_y, a);
    UbToGm(z, l_z, a);
  };

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y, z);

  uint32_t diff_count = 0;
  for (uint32_t i = 0; i < a; i++) {
    if (y[i] != expect_index[i] || z[i] != expect_value[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
}
