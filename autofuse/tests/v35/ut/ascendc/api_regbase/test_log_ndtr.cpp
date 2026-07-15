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
#include "api_regbase/log_ndtr.h"

using namespace AscendC;

namespace af {

template <typename T>
struct LogNdtrInputParam {
  T *dst{};
  T *exp{};
  T *src{};
  int32_t size{0};
};

// Reference implementation using double precision
template <typename T>
static T calcRefLogNdtr(T x) {
  double dx = static_cast<double>(x);
  double t = dx * LOG_NDTR::LOG_NDTR_INV_SQRT_2;
  double result;
  if (dx < -1.0) {
    // Left tail: log(erfcx(-t)/2) - t^2
    double erfcx_val = std::exp(t * t) * std::erfc(-t);
    result = std::log(erfcx_val * 0.5) - t * t;
  } else {
    // Right tail: log1p(-erfc(t)/2)
    double erfc_val = std::erfc(t);
    result = std::log1p(-erfc_val * 0.5);
  }
  return static_cast<T>(result);
}

class TestApiLogNdtr : public testing::Test {
 protected:
  template <typename T>
  static void InvokeTensorKernel(LogNdtrInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> srcBuf, dstBuf, tmpBuf;
    tpipe.InitBuffer(srcBuf, sizeof(T) * param.size);
    tpipe.InitBuffer(dstBuf, sizeof(T) * param.size);
    tpipe.InitBuffer(tmpBuf, 2048 * sizeof(uint8_t));

    LocalTensor<T> l_src = srcBuf.Get<T>();
    LocalTensor<T> l_dst = dstBuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmpBuf.Get<uint8_t>();

    GmToUb(l_src, param.src, param.size);
    LogNdtrExtend<T>(l_dst, l_src, l_tmp, param.size);
    UbToGm(param.dst, l_dst, param.size);
  }

  // Create random input covering both left-tail and right-tail branches
  template <typename T>
  static void CreateTensorInput(LogNdtrInputParam<T> &param) {
    param.dst = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    std::mt19937 eng(1);

    // Mix of left tail [-30, -1.0) and right tail [-1.0, 20]
    std::uniform_real_distribution<float> distrLeft(-30.0f, -1.001f);
    std::uniform_real_distribution<float> distrRight(-1.0f, 20.0f);

    for (int i = 0; i < param.size; i++) {
      T input;
      if (i % 2 == 0) {
        input = static_cast<T>(distrLeft(eng));
      } else {
        input = static_cast<T>(distrRight(eng));
      }
      param.src[i] = input;
      param.exp[i] = calcRefLogNdtr(input);
    }
  }

  // Create special value inputs
  template <typename T>
  static void CreateSpecialInput(LogNdtrInputParam<T> &param) {
    param.dst = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    T nan_val = std::numeric_limits<T>::quiet_NaN();
    T inf_val = std::numeric_limits<T>::infinity();

    // [0] NaN → NaN
    param.src[0] = nan_val;
    param.exp[0] = nan_val;

    // [1] +inf → -0.0
    param.src[1] = inf_val;
    param.exp[1] = -0.0f;

    // [2] -inf → -inf
    param.src[2] = -inf_val;
    param.exp[2] = -inf_val;

    // [3] 0.0 → -ln(2) ≈ -0.693147
    param.src[3] = 0.0f;
    param.exp[3] = static_cast<T>(-0.6931471805599453);

    // [4] -0.0 → -ln(2)
    param.src[4] = -0.0f;
    param.exp[4] = static_cast<T>(-0.6931471805599453);

    // [5] x = -1.0 (branch boundary) → ≈ -1.841
    param.src[5] = -1.0f;
    param.exp[5] = calcRefLogNdtr(-1.0f);

    // [6] x = 1.0 → ≈ -0.17275
    param.src[6] = 1.0f;
    param.exp[6] = calcRefLogNdtr(1.0f);

    // [7] x = -5.0 (deep left tail) → ≈ -13.42
    param.src[7] = -5.0f;
    param.exp[7] = calcRefLogNdtr(-5.0f);

    // [8] x = 5.0 (right tail, small result) → ≈ -2.87e-7
    param.src[8] = 5.0f;
    param.exp[8] = calcRefLogNdtr(5.0f);

    // [9] x = -20.0 (very deep left tail)
    param.src[9] = -20.0f;
    param.exp[9] = calcRefLogNdtr(-20.0f);

    // [10] x = 10.0 (large positive, near underflow)
    param.src[10] = 10.0f;
    param.exp[10] = calcRefLogNdtr(10.0f);

    // [11] x = -30.0 (maximum negative test value)
    param.src[11] = -30.0f;
    param.exp[11] = calcRefLogNdtr(-30.0f);

    // [12] x = 20.0 (extreme positive, should underflow)
    param.src[12] = 20.0f;
    param.exp[12] = calcRefLogNdtr(20.0f);

    // [13] x = -1.5 (left tail near boundary)
    param.src[13] = -1.5f;
    param.exp[13] = calcRefLogNdtr(-1.5f);

    // [14] x = -0.5 (right tail near boundary)
    param.src[14] = -0.5f;
    param.exp[14] = calcRefLogNdtr(-0.5f);

    // [15] x = 2.0
    param.src[15] = 2.0f;
    param.exp[15] = calcRefLogNdtr(2.0f);
  }

  // Validation function with branch-aware tolerances
  template <typename T>
  static uint32_t Valid(T *dst, T *exp, T *src, size_t comp_size) {
    uint32_t diff_count = 0;

    for (uint32_t i = 0; i < comp_size; i++) {
      bool is_diff = false;
      T abs_diff = 0;

      if (std::isnan(exp[i])) {
        if (!std::isnan(dst[i])) {
          is_diff = true;
        }
      } else if (std::isinf(exp[i])) {
        if (dst[i] != exp[i]) {
          is_diff = true;
        }
      } else {
        // Combined absolute/relative error: denominator max(1, |exp|) uses
        // absolute error for small values and relative error for large ones
        abs_diff = std::abs(dst[i] - exp[i]);
        T rel_err = abs_diff / std::max(std::abs(exp[i]), T(1));
        if (rel_err > T(1e-5)) {
          is_diff = true;
        }
      }

      if (is_diff) {
        diff_count++;
      }
    }

    return diff_count;
  }

  template <typename T>
  static void MainTest(const int32_t size) {
    LogNdtrInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);

    auto kernel = [&param] { InvokeTensorKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid<T>(param.dst, param.exp, param.src, param.size);
    EXPECT_EQ(diff_count, 0);

    AscendC::GmFree(param.dst);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.src);
  }

  template <typename T>
  static void MainSpecialTest(const int32_t size) {
    LogNdtrInputParam<T> param{};
    param.size = size;
    CreateSpecialInput(param);

    auto kernel = [&param] { InvokeTensorKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid<T>(param.dst, param.exp, param.src, param.size);
    EXPECT_EQ(diff_count, 0);

    AscendC::GmFree(param.dst);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.src);
  }
};

TEST_F(TestApiLogNdtr, LogNdtr_Special_Success) {
  MainSpecialTest<float>(16);
}

TEST_F(TestApiLogNdtr, LogNdtr_Float_32_Success) {
  MainTest<float>(32);
}

TEST_F(TestApiLogNdtr, LogNdtr_Float_64_Success) {
  MainTest<float>(64);
}

TEST_F(TestApiLogNdtr, LogNdtr_Float_128_Success) {
  MainTest<float>(128);
}

TEST_F(TestApiLogNdtr, LogNdtr_Float_256_Success) {
  MainTest<float>(256);
}

TEST_F(TestApiLogNdtr, LogNdtr_Float_1000_Success) {
  MainTest<float>(1000);
}

}  // namespace af
