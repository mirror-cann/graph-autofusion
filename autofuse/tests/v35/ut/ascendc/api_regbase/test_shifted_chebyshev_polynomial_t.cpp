/**
* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <cmath>
#include <random>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "shifted_chebyshev_polynomial_t.h"
#include "boost/math/special_functions/chebyshev.hpp"

using namespace AscendC;

namespace ge {

class TestRegbaseApiShiftedChebyshevPolynomialTUT : public testing::Test {
 protected:
  // Tensor - Tensor 场景
  template <typename T, int64_t N>
  static void InvokeTensorTensorKernel(UnaryInputParam<T> &t_param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, temp;
    tpipe.InitBuffer(xbuf, sizeof(T) * t_param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * t_param.size);
    tpipe.InitBuffer(temp, TMP_UB_SIZE);

    LocalTensor<T> t_x = xbuf.Get<T>();
    LocalTensor<T> t_y = ybuf.Get<T>();
    LocalTensor<uint8_t> t_tmp = temp.Get<uint8_t>();

    GmToUb(t_x, t_param.x1, t_param.size);
    ShiftedChebyshevPolynomialTExtend<float, N>(t_y, t_x, t_tmp, t_param.size);
    UbToGm(t_param.y, t_y, t_param.size);
  }

  template <typename T, int64_t N>
  static void CreateTensorInput(UnaryInputParam<T> &t_param) {
    t_param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * t_param.size));
    t_param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * t_param.size));
    t_param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * t_param.size));

    std::mt19937 eng(1);

    for (uint32_t i = 0; i < t_param.size; i++) {
        std::uniform_real_distribution<float> distr(-5.0f, 5.0f);
        t_param.x1[i] = static_cast<T>(distr(eng));
        t_param.exp[i] = static_cast<T>(boost::math::chebyshev_t(N, (double)(2.0 * t_param.x1[i]) - 1.0));
    }
  }

  template <typename T>
  static uint32_t Valid(UnaryInputParam<T> &t_param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < t_param.size; i++) {
      double y_val = static_cast<double>(t_param.y[i]);
      double exp_val = static_cast<double>(t_param.exp[i]);
      double rel_err = std::abs(y_val - exp_val) / std::max(std::abs(exp_val), 1.0);
      if (rel_err > 1e-2) {
        diff_count++;
        printf("diff at index %d: x: %.20e, y: %.20e, expect: %.20e, rel_err: %f\n", i, static_cast<float>(t_param.x1[i]),
               static_cast<float>(t_param.y[i]), static_cast<float>(t_param.exp[i]), static_cast<float>(rel_err));
      }
    }
    return diff_count;
  }

  template <typename T, int64_t N>
  static void ShiftedChebyshevPolynomialTTensorTensorTest(uint32_t size) {
    UnaryInputParam<T> t_param{};
    t_param.size = size;
    CreateTensorInput<T, N>(t_param);

    // 构造Api调用函数
    auto t_kernel = [&t_param] { InvokeTensorTensorKernel<T, N>(t_param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(t_kernel, 1);

    uint32_t diff_count = Valid(t_param);
    EXPECT_EQ(diff_count, 0);
  }
};

TEST_F(TestRegbaseApiShiftedChebyshevPolynomialTUT, ShiftedChebyshevPolynomialT_TensorTensor_Test_N0) {
  ShiftedChebyshevPolynomialTTensorTensorTest<float, 0>(ONE_BLK_SIZE / sizeof(float));
  ShiftedChebyshevPolynomialTTensorTensorTest<float, 0>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  ShiftedChebyshevPolynomialTTensorTensorTest<float, 0>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  ShiftedChebyshevPolynomialTTensorTensorTest<float, 0>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  ShiftedChebyshevPolynomialTTensorTensorTest<float, 0>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  ShiftedChebyshevPolynomialTTensorTensorTest<float, 0>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(float))) / 2 /sizeof(float));
}

TEST_F(TestRegbaseApiShiftedChebyshevPolynomialTUT, ShiftedChebyshevPolynomialT_TensorTensor_Test_N1) {
  ShiftedChebyshevPolynomialTTensorTensorTest<float, 1>(ONE_BLK_SIZE / sizeof(float));
  ShiftedChebyshevPolynomialTTensorTensorTest<float, 1>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  ShiftedChebyshevPolynomialTTensorTensorTest<float, 1>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  ShiftedChebyshevPolynomialTTensorTensorTest<float, 1>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  ShiftedChebyshevPolynomialTTensorTensorTest<float, 1>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  ShiftedChebyshevPolynomialTTensorTensorTest<float, 1>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(float))) / 2 /sizeof(float));
}

TEST_F(TestRegbaseApiShiftedChebyshevPolynomialTUT, ShiftedChebyshevPolynomialT_TensorTensor_Test_N7) {
  ShiftedChebyshevPolynomialTTensorTensorTest<float, 7>(ONE_BLK_SIZE / sizeof(float));
  ShiftedChebyshevPolynomialTTensorTensorTest<float, 7>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  ShiftedChebyshevPolynomialTTensorTensorTest<float, 7>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  ShiftedChebyshevPolynomialTTensorTensorTest<float, 7>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  ShiftedChebyshevPolynomialTTensorTensorTest<float, 7>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  ShiftedChebyshevPolynomialTTensorTensorTest<float, 7>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(float))) / 2 /sizeof(float));
}
}