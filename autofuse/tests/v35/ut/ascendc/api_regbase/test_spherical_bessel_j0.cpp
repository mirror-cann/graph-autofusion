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
#include <limits>
#include <random>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "bessel_j_utils.h"
#include "trigonometric_function_utils.h"
#include "spherical_bessel_j0.h"

using namespace AscendC;

namespace ge {

class TestRegbaseApiSphericalBesselJ0UT : public testing::Test {
 protected:
  template <typename T>
  static void InvokeTensorTensorKernel(UnaryInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<T> l_x = xbuf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, param.x1, param.size);
    SphericalBesselJ0Extend(l_y, l_x, l_tmp, param.size);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void CreateTensorInputRandom(UnaryInputParam<T> &param, float lo, float hi) {
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    std::mt19937 eng(1);
    std::uniform_real_distribution<float> distr(lo, hi);

    for (uint32_t i = 0; i < param.size; i++) {
      param.x1[i] = static_cast<T>(distr(eng));
      double absx = std::fabs(static_cast<double>(param.x1[i]));
      if (absx > 1e30) {
        param.exp[i] = static_cast<T>(0.0);
      } else if (absx == 0.0) {
        param.exp[i] = static_cast<T>(1.0);
      } else {
        param.exp[i] = static_cast<T>(std::sin(absx) / absx);
      }
    }
  }

  template <typename T>
  static void CreateTensorInputSpecial(UnaryInputParam<T> &param, const std::vector<float> &values) {
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    for (uint32_t i = 0; i < param.size; i++) {
      float v = values[i % values.size()];
      param.x1[i] = static_cast<T>(v);
      if (std::isnan(v)) {
        param.exp[i] = std::numeric_limits<T>::quiet_NaN();
      } else if (std::isinf(v)) {
        param.exp[i] = static_cast<T>(0.0);
      } else {
        double absx = std::fabs(static_cast<double>(v));
        if (absx > 1e30) {
          param.exp[i] = static_cast<T>(0.0);
        } else if (absx == 0.0) {
          param.exp[i] = static_cast<T>(1.0);
        } else {
          param.exp[i] = static_cast<T>(std::sin(absx) / absx);
        }
      }
    }
  }

  template <typename T>
  static uint32_t Valid(UnaryInputParam<T> &param, double tol = 1e-2) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      double y_val = static_cast<double>(param.y[i]);
      double exp_val = static_cast<double>(param.exp[i]);

      if (std::isnan(exp_val)) {
        if (!std::isnan(y_val)) {
          diff_count++;
          printf("diff at index %d: x: %.20e, y: %.20e, expect: NaN\n", i, static_cast<float>(param.x1[i]),
                 static_cast<float>(param.y[i]));
        }
        continue;
      }

      double rel_err = std::abs(y_val - exp_val) / std::max(std::abs(exp_val), 1.0);
      if (rel_err > tol) {
        diff_count++;
        printf("diff at index %d: x: %.20e, y: %.20e, expect: %.20e, rel_err: %f\n", i, static_cast<float>(param.x1[i]),
               static_cast<float>(param.y[i]), static_cast<float>(param.exp[i]), static_cast<float>(rel_err));
      }
    }
    return diff_count;
  }

  template <typename T>
  static void SphericalBesselJ0Test(uint32_t size, float lo = -30.0f, float hi = 30.0f) {
    UnaryInputParam<T> param{};
    param.size = size;
    CreateTensorInputRandom(param, lo, hi);

    auto kernel = [&param] { InvokeTensorTensorKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0);
  }

  template <typename T>
  static void SphericalBesselJ0SpecialTest(uint32_t size, const std::vector<float> &values) {
    UnaryInputParam<T> param{};
    param.size = size;
    CreateTensorInputSpecial(param, values);

    auto kernel = [&param] { InvokeTensorTensorKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0);
  }
};

TEST_F(TestRegbaseApiSphericalBesselJ0UT, Aligned_OneBlk) {
  SphericalBesselJ0Test<float>(ONE_BLK_SIZE / sizeof(float));
}

TEST_F(TestRegbaseApiSphericalBesselJ0UT, Aligned_OneRepeat) {
  SphericalBesselJ0Test<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
}

TEST_F(TestRegbaseApiSphericalBesselJ0UT, Aligned_MaxRepeat) {
  SphericalBesselJ0Test<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
}

TEST_F(TestRegbaseApiSphericalBesselJ0UT, Unaligned_OneBlkMinusOne) {
  SphericalBesselJ0Test<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
}

TEST_F(TestRegbaseApiSphericalBesselJ0UT, Unaligned_OneRepeatMinusOneBlk) {
  SphericalBesselJ0Test<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
}

TEST_F(TestRegbaseApiSphericalBesselJ0UT, Unaligned_MaxRepeatMinusOne) {
  SphericalBesselJ0Test<float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
}

TEST_F(TestRegbaseApiSphericalBesselJ0UT, Unaligned_Combined) {
  SphericalBesselJ0Test<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                (ONE_BLK_SIZE - sizeof(float))) /
                               2 / sizeof(float));
}

TEST_F(TestRegbaseApiSphericalBesselJ0UT, Performance_LargeSize) {
  SphericalBesselJ0Test<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(float));
}

TEST_F(TestRegbaseApiSphericalBesselJ0UT, Performance_LargeSizeRandom) {
  SphericalBesselJ0Test<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
}

TEST_F(TestRegbaseApiSphericalBesselJ0UT, Exception_NaN) {
  std::vector<float> nan_values = {
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::signaling_NaN(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN(),
  };
  SphericalBesselJ0SpecialTest<float>(ONE_BLK_SIZE / sizeof(float), nan_values);
}

TEST_F(TestRegbaseApiSphericalBesselJ0UT, Boundary_Zero) {
  std::vector<float> zero_values = {0.0f, -0.0f, 0.0f, 0.0f};
  SphericalBesselJ0SpecialTest<float>(ONE_BLK_SIZE / sizeof(float), zero_values);
}

TEST_F(TestRegbaseApiSphericalBesselJ0UT, Boundary_SmallValues) {
  std::vector<float> small_values = {
      1e-7f, -1e-7f, 1e-5f, -1e-5f, 0.01f, -0.01f, 0.1f, -0.1f, 0.3f, -0.3f, 0.49f, -0.49f,
  };
  SphericalBesselJ0SpecialTest<float>(ONE_BLK_SIZE / sizeof(float), small_values);
}

TEST_F(TestRegbaseApiSphericalBesselJ0UT, Boundary_NearThreshold) {
  std::vector<float> threshold_values = {
      0.499f, 0.5f, 0.501f, -0.499f, -0.5f, -0.501f,
  };
  SphericalBesselJ0SpecialTest<float>(ONE_BLK_SIZE / sizeof(float), threshold_values);
}

TEST_F(TestRegbaseApiSphericalBesselJ0UT, Boundary_LargeValues) {
  std::vector<float> large_values = {
      1e31f, -1e31f, 1e35f, -1e35f, std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
  };
  SphericalBesselJ0SpecialTest<float>(ONE_BLK_SIZE / sizeof(float), large_values);
}

TEST_F(TestRegbaseApiSphericalBesselJ0UT, Boundary_MixedSpecial) {
  std::vector<float> mixed_values = {
      0.0f, 1.0f, -1.0f, std::numeric_limits<float>::quiet_NaN(), 0.49f, 0.51f, 25.0f, -25.0f,
  };
  SphericalBesselJ0SpecialTest<float>(ONE_BLK_SIZE / sizeof(float), mixed_values);
}

}  // namespace ge
