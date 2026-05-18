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

#include <boost/math/special_functions/laguerre.hpp>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "laguerre_polynomial_l.h"

using namespace AscendC;

namespace af {

template <typename T>
struct LaguerrePolynomialLInputParam : public UnaryInputParam<T> {
  int32_t n{};
};

class TestRegbaseApiLaguerrePolynomialLUT : public testing::Test {
 protected:
  static constexpr double kDefaultTolerance = 1e-6;
  static constexpr double kRecursiveTolerance = 2e-5;

  template <typename T>
  static void InvokeTensorTensorKernel(LaguerrePolynomialLInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf;
    TBuf<TPosition::VECCALC> ybuf;
    tpipe.InitBuffer(xbuf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));

    LocalTensor<T> l_x = xbuf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();

    GmToUb(l_x, param.x1, param.size);
    LaguerrePolynomialLExtend(l_y, l_x, param.n, param.size);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void InvokeTensorTensorTensorKernel(LaguerrePolynomialLInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf;
    TBuf<TPosition::VECCALC> ybuf;
    TBuf<TPosition::VECCALC> nbuf;
    tpipe.InitBuffer(xbuf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));
    tpipe.InitBuffer(nbuf, sizeof(int32_t) * AlignUp(1U, ONE_BLK_SIZE / sizeof(int32_t)));

    LocalTensor<T> l_x = xbuf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<int32_t> l_n = nbuf.Get<int32_t>();

    GmToUb(l_x, param.x1, param.size);
    l_n.SetValue(0, param.n);
    LaguerrePolynomialLExtend(l_y, l_x, l_n, param.size);
    UbToGm(param.y, l_y, param.size);
  }

  static double CalcExpectedValue(double x, int32_t n) {
    return boost::math::laguerre(static_cast<unsigned>(n), x);
  }

  template <typename T>
  static T GetInputValue(uint32_t index, uint32_t size, int32_t n) {
    if (n == 1 && size == 9U) {
      static const T kInputs[] = {
          static_cast<T>(-2.0e-6f), static_cast<T>(-1.0e-6f), static_cast<T>(-9.0e-7f),
          static_cast<T>(-1.0e-7f), static_cast<T>(0.0f),     static_cast<T>(1.0e-7f),
          static_cast<T>(9.0e-7f),  static_cast<T>(1.0e-6f),  static_cast<T>(2.0e-6f),
      };
      return kInputs[index];
    }

    if (n == 6 && size == 6U) {
      static const T kInputs[] = {
          static_cast<T>(-3.0f), static_cast<T>(-0.5f), static_cast<T>(0.25f),
          static_cast<T>(1.0f),  static_cast<T>(2.0f),  static_cast<T>(7.0f),
      };
      return kInputs[index];
    }

    const double start = (n == 4 && size == 65U) ? -3.0 : -5.0;
    const double end = (n == 4 && size == 65U) ? 3.0 : 5.0;
    const double denominator = (size > 1U) ? static_cast<double>(size - 1U) : 1.0;
    return static_cast<T>(start + (end - start) * static_cast<double>(index) / denominator);
  }

  template <typename T>
  static void CreateTensorInput(LaguerrePolynomialLInputParam<T> &param) {
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    for (uint32_t i = 0; i < param.size; ++i) {
      param.x1[i] = GetInputValue<T>(i, param.size, param.n);
      param.exp[i] = static_cast<T>(CalcExpectedValue(static_cast<double>(param.x1[i]), param.n));
    }
  }

  template <typename T>
  static uint32_t Valid(LaguerrePolynomialLInputParam<T> &param) {
    uint32_t diffCount = 0;
    const double tolerance = param.n > 1 ? kRecursiveTolerance : kDefaultTolerance;
    for (uint32_t i = 0; i < param.size; ++i) {
      const double diff = std::fabs(static_cast<double>(param.y[i]) - static_cast<double>(param.exp[i]));
      if (diff > tolerance) {
        ++diffCount;
        printf("diff at index %u: x: %.20e, y: %.20e, expect: %.20e, abs_err: %.20e\n", i,
               static_cast<double>(param.x1[i]), static_cast<double>(param.y[i]), static_cast<double>(param.exp[i]),
               diff);
      }
    }
    return diffCount;
  }

  template <typename T>
  static void LaguerrePolynomialLTest(uint32_t size, int32_t n) {
    LaguerrePolynomialLInputParam<T> param{};
    param.size = size;
    param.n = n;
    CreateTensorInput(param);

    auto kernel = [&param] { InvokeTensorTensorKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    const uint32_t diffCount = Valid(param);
    EXPECT_EQ(diffCount, 0);
  }

  template <typename T>
  static void LaguerrePolynomialLTensorNTest(uint32_t size, int32_t n) {
    LaguerrePolynomialLInputParam<T> param{};
    param.size = size;
    param.n = n;
    CreateTensorInput(param);

    auto kernel = [&param] { InvokeTensorTensorTensorKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    const uint32_t diffCount = Valid(param);
    EXPECT_EQ(diffCount, 0);
  }
};

TEST_F(TestRegbaseApiLaguerrePolynomialLUT, LaguerrePolynomialLZeroOrder_TensorTensor_Test) {
  LaguerrePolynomialLTest<float>(32U, 0);
}

TEST_F(TestRegbaseApiLaguerrePolynomialLUT, LaguerrePolynomialLLinear_TensorTensor_Test) {
  LaguerrePolynomialLTest<float>(9U, 1);
}

TEST_F(TestRegbaseApiLaguerrePolynomialLUT, LaguerrePolynomialLOrder4_TensorTensor_Test) {
  LaguerrePolynomialLTest<float>(65U, 4);
}

TEST_F(TestRegbaseApiLaguerrePolynomialLUT, LaguerrePolynomialLOrder6_TensorTensor_Test) {
  LaguerrePolynomialLTest<float>(6U, 6);
}

TEST_F(TestRegbaseApiLaguerrePolynomialLUT, LaguerrePolynomialLOrder4_TensorTensorTensor_Test) {
  LaguerrePolynomialLTensorNTest<float>(65U, 4);
}

}  // namespace ge
