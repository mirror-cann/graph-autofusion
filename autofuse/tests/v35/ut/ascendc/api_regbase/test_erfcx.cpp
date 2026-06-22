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
#include <random>
#include <algorithm>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/erfcx.h"

using namespace AscendC;

namespace ge {

template <typename T>
struct ErfcxInputParam {
  T *dst{};
  T *exp{};
  T *src{};
  int32_t size{0};
};

class TestApiErfcx : public testing::Test {
 protected:
  template <typename T>
  static void InvokeTensorKernel(ErfcxInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> srcBuf, dstBuf, tmpBuf;
    tpipe.InitBuffer(srcBuf, sizeof(T) * param.size);
    tpipe.InitBuffer(dstBuf, sizeof(T) * param.size);
    tpipe.InitBuffer(tmpBuf, 2048 * sizeof(uint8_t));

    LocalTensor<T> l_src = srcBuf.Get<T>();
    LocalTensor<T> l_dst = dstBuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmpBuf.Get<uint8_t>();

    GmToUb(l_src, param.src, param.size);
    ErfcxExtend<T>(l_dst, l_src, l_tmp, param.size);
    UbToGm(param.dst, l_dst, param.size);
  }

  template <typename T>
  static void CreateTensorInput(ErfcxInputParam<T> &param) {
    param.dst = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    std::mt19937 eng(1);

    auto distr = []() {
      if constexpr (std::is_same_v<T, float>) {
        return std::uniform_real_distribution<float>(-4.0f, 8.0f);
      }
    }();

    for (int i = 0; i < param.size; i++) {
      T input = static_cast<T>(distr(eng));
      param.src[i] = input;
      param.exp[i] = std::exp(input * input) * std::erfc(input);
    }
  }

  template <typename T>
  static void CreateSpecialTensorInput(ErfcxInputParam<T> &param) {
    param.dst = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    T nan_val = std::numeric_limits<T>::quiet_NaN();
    T inf_val = std::numeric_limits<T>::infinity();

    param.src[0] = 0.0f;
    param.exp[0] = 1.0f;

    param.src[1] = -0.0f;
    param.exp[1] = 1.0f;

    param.src[2] = nan_val;
    param.exp[2] = nan_val;

    param.src[3] = inf_val;
    param.exp[3] = 0.0f;

    param.src[4] = -inf_val;
    param.exp[4] = inf_val;

    param.src[5] = 5.0f;
    param.exp[5] = 0.1107046f;

    param.src[6] = -3.0f;
    param.exp[6] = 16205.989f;
  }

  template <typename T>
  static uint32_t Valid(T *dst, T *exp, size_t comp_size) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < comp_size; i++) {
      if (std::isnan(exp[i])) {
        if (!std::isnan(dst[i])) {
          diff_count++;
        }
      } else if (std::isinf(exp[i])) {
        if (std::signbit(dst[i]) != std::signbit(exp[i])) {
          diff_count++;
        }
      } else {
        T abs_diff = std::abs(dst[i] - exp[i]);
        T rel_den = std::max(std::abs(exp[i]), T(1e-10));
        T rel_err = abs_diff / rel_den;
        if (rel_err > T(1e-4)) {
          diff_count++;
        }
      }
    }
    return diff_count;
  }

  template <typename T>
  static void MainTest(const int32_t size) {
    ErfcxInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);

    auto kernel = [&param] { InvokeTensorKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid<T>(param.dst, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
  }

  template <typename T>
  static void MainSpecialTest(const int32_t size) {
    ErfcxInputParam<T> param{};
    param.size = size;
    CreateSpecialTensorInput(param);

    auto kernel = [&param] { InvokeTensorKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid<T>(param.dst, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
  }
};

TEST_F(TestApiErfcx, Erfcx_Special_Success) {
  MainSpecialTest<float>(7);
}

TEST_F(TestApiErfcx, Erfcx_Float_256_Success) {
  MainTest<float>(256);
}

}  // namespace ge
