/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <utility>
#include <vector>
#include <securec.h>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/next_after.h"

using namespace AscendC;

namespace af {

template <typename T>
struct NextAfterInputParam {
  T *y{};
  T *exp{};
  T *src{};
  T *other{};
  T otherScalar{};
  uint32_t size{0};
  bool isScalar{false};
};

class TestRegbaseApiNextAfter : public testing::Test {
 protected:
  template <typename T>
  static void InvokeTensorTensorKernel(NextAfterInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> srcBuf, otherBuf, dstBuf, tmpBuf;
    tpipe.InitBuffer(srcBuf, sizeof(T) * param.size);
    tpipe.InitBuffer(otherBuf, sizeof(T) * param.size);
    tpipe.InitBuffer(dstBuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));
    tpipe.InitBuffer(tmpBuf, TMP_UB_SIZE);

    LocalTensor<T> l_src = srcBuf.Get<T>();
    LocalTensor<T> l_other = otherBuf.Get<T>();
    LocalTensor<T> l_dst = dstBuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmpBuf.Get<uint8_t>();

    GmToUb(l_src, param.src, param.size);
    GmToUb(l_other, param.other, param.size);
    NextAfterExtend(l_dst, l_src, l_other, l_tmp, static_cast<int32_t>(param.size));
    UbToGm(param.y, l_dst, param.size);
  }

  template <typename T>
  static void InvokeTensorScalarKernel(NextAfterInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> srcBuf, dstBuf, tmpBuf;
    tpipe.InitBuffer(srcBuf, sizeof(T) * param.size);
    tpipe.InitBuffer(dstBuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));
    tpipe.InitBuffer(tmpBuf, TMP_UB_SIZE);

    LocalTensor<T> l_src = srcBuf.Get<T>();
    LocalTensor<T> l_dst = dstBuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmpBuf.Get<uint8_t>();

    GmToUb(l_src, param.src, param.size);
    NextAfterExtend(l_dst, l_src, param.otherScalar, l_tmp, static_cast<int32_t>(param.size));
    UbToGm(param.y, l_dst, param.size);
  }

  template <typename T>
  static void CreateRandomInput(NextAfterInputParam<T> &param, uint32_t size, bool isScalar) {
    param.isScalar = isScalar;
    param.size = size;
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    std::mt19937 eng(1);
    std::uniform_real_distribution<float> distr(-100.0f, 100.0f);

    if (isScalar) {
      param.other = nullptr;
      param.otherScalar = static_cast<T>(distr(eng));
      for (uint32_t i = 0; i < param.size; i++) {
        param.src[i] = static_cast<T>(distr(eng));
        param.exp[i] = std::nextafter(param.src[i], param.otherScalar);
      }
    } else {
      param.other = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
      for (uint32_t i = 0; i < param.size; i++) {
        param.src[i] = static_cast<T>(distr(eng));
        param.other[i] = static_cast<T>(distr(eng));
        param.exp[i] = std::nextafter(param.src[i], param.other[i]);
      }
    }
  }

  template <typename T>
  static void CreateBoundaryInput(NextAfterInputParam<T> &param, const std::vector<std::pair<T, T>> &cases) {
    param.isScalar = false;
    param.size = static_cast<uint32_t>(cases.size());
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.other = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    for (uint32_t i = 0; i < param.size; i++) {
      param.src[i] = cases[i].first;
      param.other[i] = cases[i].second;
      param.exp[i] = std::nextafter(cases[i].first, cases[i].second);
    }
  }

  template <typename T>
  static void CreateScalarBoundaryInput(NextAfterInputParam<T> &param, const std::vector<T> &xValues, T scalarY) {
    param.isScalar = true;
    param.size = static_cast<uint32_t>(xValues.size());
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.other = nullptr;
    param.otherScalar = scalarY;

    for (uint32_t i = 0; i < param.size; i++) {
      param.src[i] = xValues[i];
      param.exp[i] = std::nextafter(xValues[i], scalarY);
    }
  }

  template <typename T>
  static uint32_t Valid(T *y, T *exp, uint32_t size) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < size; i++) {
      bool both_nan = std::isnan(static_cast<double>(y[i])) && std::isnan(static_cast<double>(exp[i]));
      bool both_zero = (y[i] == static_cast<T>(0.0)) && (exp[i] == static_cast<T>(0.0));
      if (!both_nan && !both_zero && memcmp(&y[i], &exp[i], sizeof(T)) != 0) {
        diff_count++;
        uint32_t got_bits = 0;
        uint32_t exp_bits = 0;
        if (memcpy_s(&got_bits, sizeof(uint32_t), &y[i], sizeof(uint32_t)) != EOK) {
          continue;
        }
        if (memcpy_s(&exp_bits, sizeof(uint32_t), &exp[i], sizeof(uint32_t)) != EOK) {
          continue;
        }
        printf("diff at index %u: got=0x%08x, exp=0x%08x\n", i, got_bits, exp_bits);
      }
    }
    return diff_count;
  }

  template <typename T>
  static void FreeInput(NextAfterInputParam<T> &param) {
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.src);
    if (!param.isScalar && param.other != nullptr) {
      AscendC::GmFree(param.other);
    }
  }

  template <typename T>
  static void NextAfterTensorTensorTest(uint32_t size) {
    NextAfterInputParam<T> param{};
    CreateRandomInput(param, size, false);

    auto kernel = [&param] { InvokeTensorTensorKernel(param); };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
    FreeInput(param);
  }

  template <typename T>
  static void NextAfterTensorScalarTest(uint32_t size) {
    NextAfterInputParam<T> param{};
    CreateRandomInput(param, size, true);

    auto kernel = [&param] { InvokeTensorScalarKernel(param); };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
    FreeInput(param);
  }

  template <typename T>
  static void NextAfterBoundaryTensorTensorTest() {
    const std::vector<std::pair<T, T>> cases = {
        {static_cast<T>(1.0f), static_cast<T>(2.0f)},
        {static_cast<T>(-2.0f), static_cast<T>(0.0f)},
        {static_cast<T>(-2.0f), static_cast<T>(-1.0f)},
        {static_cast<T>(1.0f), static_cast<T>(0.0f)},
        {static_cast<T>(1.0f), static_cast<T>(-1.0f)},
        {static_cast<T>(-2.0f), static_cast<T>(-3.0f)},
        {static_cast<T>(1.5f), static_cast<T>(1.5f)},
        {static_cast<T>(+0.0f), static_cast<T>(1.0f)},
        {static_cast<T>(+0.0f), static_cast<T>(-1.0f)},
        {static_cast<T>(-0.0f), static_cast<T>(1.0f)},
        {static_cast<T>(-0.0f), static_cast<T>(-1.0f)},
        {static_cast<T>(+0.0f), static_cast<T>(+0.0f)},
        {static_cast<T>(-0.0f), static_cast<T>(-0.0f)},
        {static_cast<T>(INFINITY), static_cast<T>(0.0f)},
        {static_cast<T>(-INFINITY), static_cast<T>(0.0f)},
        {static_cast<T>(INFINITY), static_cast<T>(INFINITY)},
        {static_cast<T>(-INFINITY), static_cast<T>(-INFINITY)},
        {static_cast<T>(NAN), static_cast<T>(1.0f)},
        {static_cast<T>(1.0f), static_cast<T>(NAN)},
        {static_cast<T>(NAN), static_cast<T>(NAN)},
        {static_cast<T>(FLT_MAX), static_cast<T>(INFINITY)},
        {static_cast<T>(-FLT_MAX), static_cast<T>(-INFINITY)},
        {static_cast<T>(FLT_TRUE_MIN), static_cast<T>(0.0f)},
        {static_cast<T>(-FLT_TRUE_MIN), static_cast<T>(0.0f)},
    };

    NextAfterInputParam<T> param{};
    CreateBoundaryInput(param, cases);

    auto kernel = [&param] { InvokeTensorTensorKernel(param); };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
    FreeInput(param);
  }

  template <typename T>
  static void NextAfterBoundaryTensorScalarTest(T scalarY) {
    const std::vector<T> xValues = {
        static_cast<T>(1.0f),     static_cast<T>(-2.0f),        static_cast<T>(+0.0f),         static_cast<T>(-0.0f),
        static_cast<T>(INFINITY), static_cast<T>(-INFINITY),    static_cast<T>(NAN),           static_cast<T>(FLT_MAX),
        static_cast<T>(-FLT_MAX), static_cast<T>(FLT_TRUE_MIN), static_cast<T>(-FLT_TRUE_MIN),
    };

    NextAfterInputParam<T> param{};
    CreateScalarBoundaryInput(param, xValues, scalarY);

    auto kernel = [&param] { InvokeTensorScalarKernel(param); };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
    FreeInput(param);
  }
};

TEST_F(TestRegbaseApiNextAfter, NextAfter_TensorTensor_Test) {
  NextAfterTensorTensorTest<float>(ONE_BLK_SIZE / sizeof(float));
  NextAfterTensorTensorTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  NextAfterTensorTensorTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  NextAfterTensorTensorTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  NextAfterTensorTensorTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  NextAfterTensorTensorTest<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                    (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(float))) /
                                   2 / sizeof(float));
}

TEST_F(TestRegbaseApiNextAfter, NextAfter_TensorScalar_Test) {
  NextAfterTensorScalarTest<float>(ONE_BLK_SIZE / sizeof(float));
  NextAfterTensorScalarTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  NextAfterTensorScalarTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  NextAfterTensorScalarTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  NextAfterTensorScalarTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  NextAfterTensorScalarTest<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                    (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(float))) /
                                   2 / sizeof(float));
}

TEST_F(TestRegbaseApiNextAfter, NextAfter_Boundary_TensorTensor_Test) {
  NextAfterBoundaryTensorTensorTest<float>();
}

TEST_F(TestRegbaseApiNextAfter, NextAfter_Boundary_TensorScalar_Test) {
  NextAfterBoundaryTensorScalarTest<float>(1.0f);
  NextAfterBoundaryTensorScalarTest<float>(-1.0f);
  NextAfterBoundaryTensorScalarTest<float>(0.0f);
  NextAfterBoundaryTensorScalarTest<float>(NAN);
  NextAfterBoundaryTensorScalarTest<float>(INFINITY);
  NextAfterBoundaryTensorScalarTest<float>(-INFINITY);
}

}  // namespace af
