/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
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
#include "api_regbase/signbit.h"

using namespace AscendC;

namespace af {

template <typename InT>
struct TensorSignBitInputParam {
  bool *y{};
  bool *exp{};
  InT *src{};
  int32_t size{0};
};

class TestApiSignBit : public testing::Test {
 protected:
  template <typename InT>
  static void InvokeKernel(TensorSignBitInputParam<InT> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmpbuf;
    tpipe.InitBuffer(xbuf, sizeof(InT) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(bool) * param.size);
    tpipe.InitBuffer(tmpbuf, ONE_BLK_SIZE);  // Default buffer, not used internally
    LocalTensor<InT> l_x = xbuf.Get<InT>();
    LocalTensor<bool> l_y = ybuf.Get<bool>();
    LocalTensor<uint8_t> l_tmp = tmpbuf.Get<uint8_t>();

    GmToUb(l_x, param.src, param.size);
    SignBitExtend<InT>(l_y, l_x, l_tmp, param.size);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename InT>
  static void CreateTensorInput(TensorSignBitInputParam<InT> &param) {
    param.y = static_cast<bool *>(AscendC::GmAlloc(sizeof(bool) * param.size));
    param.exp = static_cast<bool *>(AscendC::GmAlloc(sizeof(bool) * param.size));
    param.src = static_cast<InT *>(AscendC::GmAlloc(sizeof(InT) * param.size));

    std::mt19937 eng(1);

    auto distr = []() {
      if constexpr (std::is_integral_v<InT>) {
        return std::uniform_int_distribution<int32_t>(-10, 10);
      } else {
        return std::uniform_real_distribution<float>(-10.0f, 10.0f);
      }
    }();

    for (int i = 0; i < param.size; i++) {
      InT input = static_cast<InT>(distr(eng));
      param.src[i] = input;
      param.exp[i] = std::signbit(param.src[i]) ? true : false;
    }
  }

  template <typename InT>
  static void CreateBoundaryInput(TensorSignBitInputParam<InT> &param, const std::vector<InT> &boundaryValues) {
    param.size = boundaryValues.size();
    param.y = static_cast<bool *>(AscendC::GmAlloc(sizeof(bool) * param.size));
    param.exp = static_cast<bool *>(AscendC::GmAlloc(sizeof(bool) * param.size));
    param.src = static_cast<InT *>(AscendC::GmAlloc(sizeof(InT) * param.size));

    for (size_t i = 0; i < boundaryValues.size(); i++) {
      param.src[i] = boundaryValues[i];
      param.exp[i] = std::signbit(param.src[i]) ? true : false;
    }
  }

  static uint32_t Valid(bool *y, bool *exp, size_t comp_size) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < comp_size; i++) {
      if (y[i] != exp[i]) {
        diff_count++;
      }
    }
    return diff_count;
  }

  template <typename InT>
  static void FreeTensorInput(TensorSignBitInputParam<InT> &param) {
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.src);
  }

  template <typename InT>
  static void SignBitTest(const int32_t size) {
    TensorSignBitInputParam<InT> param{};
    param.size = size;
    CreateTensorInput(param);

    auto kernel = [&param] { InvokeKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);

    FreeTensorInput(param);
  }

  template <typename InT>
  static void SignBitBoundaryTest(const std::vector<InT> &boundaryValues) {
    TensorSignBitInputParam<InT> param{};
    CreateBoundaryInput(param, boundaryValues);

    auto kernel = [&param] { InvokeKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);

    FreeTensorInput(param);
  }
};

TEST_F(TestApiSignBit, SignBit_Test_Int32) {
  SignBitTest<int32_t>(ONE_BLK_SIZE / sizeof(int32_t));
}

TEST_F(TestApiSignBit, SignBit_Test_Float) {
  SignBitTest<float>(ONE_BLK_SIZE / sizeof(float));
}

TEST_F(TestApiSignBit, SignBit_Test_Float_Boundary) {
  // Boundary values for float:
  // -0.0: sign bit is 1, should return true
  // 0.0: sign bit is 0, should return false
  SignBitBoundaryTest<float>({-0.0f, 0.0f});
}

TEST_F(TestApiSignBit, SignBit_Test_Int32_Boundary) {
  // Boundary values for int32:
  // 0: sign bit is 0, should return false
  // -0 (same as 0 in int32): sign bit is 0, should return false
  SignBitBoundaryTest<int32_t>({0, -0});
}

}  // namespace af
