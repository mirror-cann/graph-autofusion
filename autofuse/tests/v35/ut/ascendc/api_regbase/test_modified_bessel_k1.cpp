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
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "modified_bessel_k1.h"

using namespace AscendC;

namespace af {

class TestRegbaseApiModifiedBesselK1UT : public testing::Test {
 protected:
  // Tensor - Tensor 场景
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
    ModifiedBesselK1Extend(l_y, l_x, l_tmp, param.size);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void CreateTensorInput(UnaryInputParam<T> &param) {
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    std::mt19937 eng(1);

    for (uint32_t i = 0; i < param.size; i++) {
      // K1 only defined for x > 0, generate range [0.0, 20.0]
      std::uniform_real_distribution distr(0.0f, 20.0f);
      param.x1[i] = static_cast<T>(distr(eng));
      if (param.x1[i] == 0.0f) {
        param.exp[i] = std::numeric_limits<T>::infinity();
      } else {
        param.exp[i] = static_cast<T>(std::cyl_bessel_k(1, param.x1[i]));
      }
    }
  }

  template <typename T>
  static uint32_t Valid(UnaryInputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      // Skip x <= 0 cases (NaN/Inf) in comparison
      if (param.x1[i] <= 0.0f) {
        continue;
      }
      double y_val = static_cast<double>(param.y[i]);
      double exp_val = static_cast<double>(param.exp[i]);
      double rel_err = std::abs(y_val - exp_val) / std::max(std::abs(exp_val), 1.0);
      if (rel_err > 1e-2) {
        diff_count++;
        printf("diff at index %d: x: %.20e, y: %.20e, expect: %.20e, rel_err: %f\n", i, static_cast<float>(param.x1[i]),
               static_cast<float>(param.y[i]), static_cast<float>(param.exp[i]),
               static_cast<float>(rel_err));
      }
    }
    return diff_count;
  }

  // Tensor - Tensor 测试
  template <typename T>
  static void ModifiedBesselK1Test(uint32_t size) {
    UnaryInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeTensorTensorKernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0);
  }
};

TEST_F(TestRegbaseApiModifiedBesselK1UT, ModifiedBesselK1_TensorTensor_Test) {
  ModifiedBesselK1Test<float>(ONE_BLK_SIZE / sizeof(float));
  ModifiedBesselK1Test<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  ModifiedBesselK1Test<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  ModifiedBesselK1Test<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  ModifiedBesselK1Test<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  ModifiedBesselK1Test<float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  ModifiedBesselK1Test<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                              (ONE_BLK_SIZE - sizeof(float))) / 2 / sizeof(float));
}

} // namespace ge
