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
#include <limits>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/ndtri.h"
#include "boost/math/distributions/normal.hpp"

using namespace AscendC;

namespace af {

class TestRegbaseApiNdtriUT : public testing::Test {
 protected:
  template <typename T>
  static void InvokeTensorKernel(UnaryInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, temp;
    tpipe.InitBuffer(xbuf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * param.size);
    tpipe.InitBuffer(temp, TMP_UB_SIZE);

    LocalTensor<T> l_x = xbuf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = temp.Get<uint8_t>();

    GmToUb(l_x, param.x1, param.size);
    NdtriExtend(l_y, l_x, l_tmp, param.size);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void CreateTensorInput(UnaryInputParam<T> &param) {
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    std::mt19937 eng(1);

    for (uint32_t i = 0; i < param.size; i++) {
        std::uniform_real_distribution<float> distr(0.001f, 0.999f);
        param.x1[i] = static_cast<T>(distr(eng));
        param.exp[i] = static_cast<T>(boost::math::quantile(boost::math::normal_distribution<>(0.0, 1.0), param.x1[i]));
    }
  }

  template <typename T>
  static void CreateSpecialTensorInput(UnaryInputParam<T> &param) {
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    T nan_val = std::numeric_limits<T>::quiet_NaN();
    T inf_val = std::numeric_limits<T>::infinity();
    T denorm_min = std::numeric_limits<T>::denorm_min();

    uint32_t idx = 0;

    param.x1[idx] = 0.5f;
    param.exp[idx++] = 0.0f;

    param.x1[idx] = -0.0f;
    param.exp[idx++] = -inf_val;

    param.x1[idx] = 0.0f;
    param.exp[idx++] = -inf_val;

    param.x1[idx] = 1.0f;
    param.exp[idx++] = inf_val;

    param.x1[idx] = -inf_val;
    param.exp[idx++] = nan_val;

    param.x1[idx] = inf_val;
    param.exp[idx++] = nan_val;

    param.x1[idx] = nan_val;
    param.exp[idx++] = nan_val;

    param.x1[idx] = std::numeric_limits<T>::signaling_NaN();
    param.exp[idx++] = nan_val;

    param.x1[idx] = -1.0f;
    param.exp[idx++] = nan_val;

    param.x1[idx] = -0.001f;
    param.exp[idx++] = nan_val;

    param.x1[idx] = 1.001f;
    param.exp[idx++] = nan_val;

    param.x1[idx] = 2.0f;
    param.exp[idx++] = nan_val;

    param.x1[idx] = -2.0f;
    param.exp[idx++] = nan_val;

    param.x1[idx] = 0.25f;
    param.exp[idx++] = static_cast<T>(boost::math::quantile(boost::math::normal_distribution<>(0.0, 1.0), 0.25f));

    param.x1[idx] = 0.75f;
    param.exp[idx++] = static_cast<T>(boost::math::quantile(boost::math::normal_distribution<>(0.0, 1.0), 0.75f));

    param.x1[idx] = 0.001f;
    param.exp[idx++] = static_cast<T>(boost::math::quantile(boost::math::normal_distribution<>(0.0, 1.0), 0.001f));

    param.x1[idx] = 0.999f;
    param.exp[idx++] = static_cast<T>(boost::math::quantile(boost::math::normal_distribution<>(0.0, 1.0), 0.999f));

    param.x1[idx] = denorm_min;
    param.exp[idx++] = static_cast<T>(-6.36134f);

    param.x1[idx] = -denorm_min;
    param.exp[idx++] = nan_val;

    param.x1[idx] = static_cast<T>(0.9999999f);
    param.exp[idx++] = static_cast<T>(boost::math::quantile(boost::math::normal_distribution<>(0.0, 1.0), static_cast<T>(0.9999999f)));
    param.size = idx;
  }

  template <typename T>
  static void CreateExtremeTensorInput(UnaryInputParam<T> &param) {
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    T denorm_min = std::numeric_limits<T>::denorm_min();
    uint32_t idx = 0;

    param.x1[idx] = 1e-6f;
    param.exp[idx++] = static_cast<T>(boost::math::quantile(boost::math::normal_distribution<>(0.0, 1.0), 1e-6f));

    param.x1[idx] = static_cast<T>(1.0f) - 1e-5f;
    param.exp[idx++] = static_cast<T>(boost::math::quantile(boost::math::normal_distribution<>(0.0, 1.0), static_cast<T>(1.0f) - 1e-5f));

    param.x1[idx] = static_cast<T>(1.0f) - 1e-6f;
    param.exp[idx++] = static_cast<T>(boost::math::quantile(boost::math::normal_distribution<>(0.0, 1.0), static_cast<T>(1.0f) - 1e-6f));

    param.x1[idx] = 0.1f;
    param.exp[idx++] = static_cast<T>(boost::math::quantile(boost::math::normal_distribution<>(0.0, 1.0), 0.1f));

    param.x1[idx] = 0.01f;
    param.exp[idx++] = static_cast<T>(boost::math::quantile(boost::math::normal_distribution<>(0.0, 1.0), 0.01f));

    param.x1[idx] = 0.9f;
    param.exp[idx++] = static_cast<T>(boost::math::quantile(boost::math::normal_distribution<>(0.0, 1.0), 0.9f));

    param.x1[idx] = 0.99f;
    param.exp[idx++] = static_cast<T>(boost::math::quantile(boost::math::normal_distribution<>(0.0, 1.0), 0.99f));
    param.size = idx;
  }

  template <typename T>
  static void CreateSubnormalTensorInput(UnaryInputParam<T> &param) {
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    T denorm_min = std::numeric_limits<T>::denorm_min();
    T nan_val = std::numeric_limits<T>::quiet_NaN();
    uint32_t idx = 0;

    param.x1[idx] = denorm_min;
    param.exp[idx++] = static_cast<T>(-6.36134f);

    param.x1[idx] = denorm_min * 2;
    param.exp[idx++] = static_cast<T>(-6.36134f);

    param.x1[idx] = denorm_min * 10;
    param.exp[idx++] = static_cast<T>(-6.36134f);

    param.x1[idx] = denorm_min * 100;
    param.exp[idx++] = static_cast<T>(-6.36134f);

    param.x1[idx] = -denorm_min;
    param.exp[idx++] = nan_val;

    param.x1[idx] = -denorm_min * 2;
    param.exp[idx++] = nan_val;

    param.x1[idx] = static_cast<T>(0.999999f);
    param.exp[idx++] = static_cast<T>(boost::math::quantile(boost::math::normal_distribution<>(0.0, 1.0), static_cast<T>(0.999999f)));

    param.size = idx;
  }

  template <typename T>
  static uint32_t Valid(T *y, T *exp, uint32_t size) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < size; i++) {
      if (std::isnan(exp[i])) {
        if (!std::isnan(y[i])) {
          diff_count++;
        }
      } else if (std::isinf(exp[i])) {
        if (std::isinf(y[i]) && std::signbit(y[i]) == std::signbit(exp[i])) {
          continue;
        } else {
          diff_count++;
        }
      } else if (std::abs(y[i] - exp[i]) > 1e-3 && std::abs(y[i] - exp[i]) / std::abs(exp[i]) > 1e-3) {
        diff_count++;
      }
    }
    return diff_count;
  }

  template <typename T>
  static void RunNdtriTest(UnaryInputParam<T> &param) {
    auto kernel = [&param] { InvokeTensorKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.x1);
  }

  template <typename T>
  static void NdtriTensorTest(uint32_t size) {
    UnaryInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);
    RunNdtriTest(param);
  }

  template <typename T>
  static void NdtriSpecialTest(uint32_t size) {
    UnaryInputParam<T> param{};
    param.size = size;
    CreateSpecialTensorInput(param);
    RunNdtriTest(param);
  }

  template <typename T>
  static void NdtriExtremeTest(uint32_t size) {
    UnaryInputParam<T> param{};
    param.size = size;
    CreateExtremeTensorInput(param);
    RunNdtriTest(param);
  }

  template <typename T>
  static void NdtriSubnormalTest(uint32_t size) {
    UnaryInputParam<T> param{};
    param.size = size;
    CreateSubnormalTensorInput(param);
    RunNdtriTest(param);
  }
};

TEST_F(TestRegbaseApiNdtriUT, Ndtri_Special_Test) {
    NdtriSpecialTest<float>(30);
}

TEST_F(TestRegbaseApiNdtriUT, Ndtri_Extreme_Test) {
    NdtriExtremeTest<float>(10);
}

TEST_F(TestRegbaseApiNdtriUT, Ndtri_Subnormal_Test) {
    NdtriSubnormalTest<float>(20);
}

TEST_F(TestRegbaseApiNdtriUT, Ndtri_Float_256_Test) {
    NdtriTensorTest<float>(256);
}

TEST_F(TestRegbaseApiNdtriUT, Ndtri_Float_512_Test) {
    NdtriTensorTest<float>(512);
}

TEST_F(TestRegbaseApiNdtriUT, Ndtri_Float_1024_Test) {
    NdtriTensorTest<float>(1024);
}

TEST_F(TestRegbaseApiNdtriUT, Ndtri_Float_2048_Test) {
    NdtriTensorTest<float>(2048);
}

TEST_F(TestRegbaseApiNdtriUT, Ndtri_Float_OddSize_Test) {
    NdtriTensorTest<float>(257);
    NdtriTensorTest<float>(511);
    NdtriTensorTest<float>(1023);
}
}