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
#include <cstdint>
#include <vector>
#include <algorithm>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/zeta.h"

using namespace AscendC;

namespace af {

class TestApiZeta : public testing::Test {
 protected:
  static void InvokeZetaKernel(BinaryInputParam<float> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xBuf, qBuf, dstBuf, tmpBuf;
    tpipe.InitBuffer(xBuf, sizeof(float) * param.size);
    tpipe.InitBuffer(qBuf, sizeof(float) * param.size);
    tpipe.InitBuffer(dstBuf, sizeof(float) * param.size);
    tpipe.InitBuffer(tmpBuf, TMP_UB_SIZE);
    LocalTensor<float> xTensor = xBuf.Get<float>();
    LocalTensor<float> qTensor = qBuf.Get<float>();
    LocalTensor<float> dstTensor = dstBuf.Get<float>();
    LocalTensor<uint8_t> tmpTensor = tmpBuf.Get<uint8_t>();

    GmToUb(xTensor, param.x1, param.size);
    GmToUb(qTensor, param.x2, param.size);
    Zeta<float>(dstTensor, xTensor, qTensor, tmpTensor, param.size);
    UbToGm(param.y, dstTensor, param.size);
  }

  static BinaryInputParam<float> BuildZetaTestData() {
    std::vector<float> x_list, q_list, exp_list;

    x_list.push_back(0.5f);
    q_list.push_back(1.0f);
    exp_list.push_back(NAN);
    x_list.push_back(0.0f);
    q_list.push_back(2.0f);
    exp_list.push_back(NAN);
    x_list.push_back(-1.0f);
    q_list.push_back(3.0f);
    exp_list.push_back(NAN);

    x_list.push_back(1.0f);
    q_list.push_back(1.0f);
    exp_list.push_back(INFINITY);
    x_list.push_back(1.0f);
    q_list.push_back(2.5f);
    exp_list.push_back(INFINITY);
    x_list.push_back(1.0f);
    q_list.push_back(10.0f);
    exp_list.push_back(INFINITY);

    x_list.push_back(2.0f);
    q_list.push_back(-1.0f);
    exp_list.push_back(INFINITY);
    x_list.push_back(3.0f);
    q_list.push_back(0.0f);
    exp_list.push_back(INFINITY);
    x_list.push_back(2.5f);
    q_list.push_back(-2.5f);
    exp_list.push_back(NAN);

    x_list.push_back(-INFINITY);
    q_list.push_back(1.0f);
    exp_list.push_back(NAN);
    x_list.push_back((1.0f - 1e-4f));
    q_list.push_back(1.0f);
    exp_list.push_back(NAN);
    x_list.push_back((1.0f + 1e-4f));
    q_list.push_back(1.0f);
    exp_list.push_back(9998.917969f);
    x_list.push_back(10.0f);
    q_list.push_back(1.0f);
    exp_list.push_back(1.000995f);
    x_list.push_back(INFINITY);
    q_list.push_back(1.0f);
    exp_list.push_back(1.0f);

    x_list.push_back(-INFINITY);
    q_list.push_back(-1.0f);
    exp_list.push_back(NAN);
    x_list.push_back(0.0f);
    q_list.push_back(-1.0f);
    exp_list.push_back(NAN);
    x_list.push_back((1.0f - 1e-4f));
    q_list.push_back(-1.0f);
    exp_list.push_back(NAN);
    x_list.push_back(1.0f);
    q_list.push_back(-1.0f);
    exp_list.push_back(INFINITY);
    x_list.push_back((1.0f + 1e-4f));
    q_list.push_back(-1.0f);
    exp_list.push_back(INFINITY);
    x_list.push_back(10.0f);
    q_list.push_back(-1.0f);
    exp_list.push_back(INFINITY);
    x_list.push_back(INFINITY);
    q_list.push_back(-1.0f);
    exp_list.push_back(INFINITY);

    x_list.push_back(-INFINITY);
    q_list.push_back(-1.1f);
    exp_list.push_back(NAN);
    x_list.push_back(0.0f);
    q_list.push_back(-1.1f);
    exp_list.push_back(NAN);
    x_list.push_back((1.0f - 1e-4f));
    q_list.push_back(-1.1f);
    exp_list.push_back(NAN);
    x_list.push_back(1.0f);
    q_list.push_back(-1.1f);
    exp_list.push_back(INFINITY);
    x_list.push_back((1.0f + 1e-4f));
    q_list.push_back(-1.1f);
    exp_list.push_back(NAN);
    x_list.push_back(2.0f);
    q_list.push_back(-1.1f);
    exp_list.push_back(102.748940f);
    x_list.push_back(10.0f);
    q_list.push_back(-1.1f);
    exp_list.push_back(9.9999764e9f);
    x_list.push_back(INFINITY);
    q_list.push_back(-1.1f);
    exp_list.push_back(INFINITY);

    x_list.push_back(2.0f);
    q_list.push_back(1.5f);
    exp_list.push_back(0.934802f);
    x_list.push_back(3.0f);
    q_list.push_back(2.5f);
    exp_list.push_back(0.118102f);
    x_list.push_back(4.0f);
    q_list.push_back(3.0f);
    exp_list.push_back(0.019823f);

    x_list.push_back(2.0f);
    q_list.push_back(1.0f);
    exp_list.push_back(1.644934f);
    x_list.push_back(3.0f);
    q_list.push_back(1.0f);
    exp_list.push_back(1.202057f);
    x_list.push_back(2.5f);
    q_list.push_back(1.0f);
    exp_list.push_back(1.341487f);

    x_list.push_back(2.0f);
    q_list.push_back(2.0f);
    exp_list.push_back(0.644934f);
    x_list.push_back(3.0f);
    q_list.push_back(2.0f);
    exp_list.push_back(0.202057f);
    x_list.push_back(2.5f);
    q_list.push_back(2.0f);
    exp_list.push_back(0.341487f);

    x_list.push_back(2.0f);
    q_list.push_back(3.0f);
    exp_list.push_back(0.394934f);
    x_list.push_back(3.0f);
    q_list.push_back(3.0f);
    exp_list.push_back(0.077057f);
    x_list.push_back(2.5f);
    q_list.push_back(3.0f);
    exp_list.push_back(0.164711f);

    x_list.push_back(5.0f);
    q_list.push_back(1.0f);
    exp_list.push_back(1.036928f);
    x_list.push_back(20.0f);
    q_list.push_back(1.0f);
    exp_list.push_back(1.000001f);

    x_list.push_back(2.0f);
    q_list.push_back(10.0f);
    exp_list.push_back(0.105166f);
    x_list.push_back(3.0f);
    q_list.push_back(10.0f);
    exp_list.push_back(0.005525f);
    x_list.push_back(5.0f);
    q_list.push_back(10.0f);
    exp_list.push_back(3.0414e-5f);

    BinaryInputParam<float> param{};
    uint32_t n = (uint32_t)x_list.size();
    param.size = n;
    param.y = static_cast<float *>(GmAlloc(sizeof(float) * param.size));
    param.exp = static_cast<float *>(GmAlloc(sizeof(float) * param.size));
    param.x1 = static_cast<float *>(GmAlloc(sizeof(float) * param.size));
    param.x2 = static_cast<float *>(GmAlloc(sizeof(float) * param.size));
    for (uint32_t i = 0; i < n; ++i) {
      param.x1[i] = x_list[i];
      param.x2[i] = q_list[i];
      param.exp[i] = exp_list[i];
    }
    return param;
  }

  static uint32_t Valid(float *dst, float *exp, size_t comp_size) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < comp_size; i++) {
      bool match = false;
      if (std::isnan(exp[i])) {
        match = std::isnan(dst[i]);
      } else if (std::isinf(exp[i])) {
        match = dst[i] == exp[i];
      } else {
        match = std::abs(dst[i] - exp[i]) <= 1e-5 || std::abs(dst[i] - exp[i]) <= std::abs(exp[i]) * 1e-4;
      }
      if (!match) {
        diff_count++;
        printf("Test[%u] FAIL: exp=%.9g, act=%.9g\n", i, exp[i], dst[i]);
      }
    }
    return diff_count;
  }

  static void ZetaTest() {
    auto param = BuildZetaTestData();

    auto kernel = [&param] { InvokeZetaKernel(param); };

    SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
  }
};

TEST_F(TestApiZeta, zeta_test) {
  ZetaTest();
}

}  // namespace af
