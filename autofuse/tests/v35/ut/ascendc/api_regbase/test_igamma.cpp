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
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "boost/math/special_functions/gamma.hpp"
#include "api_regbase/igamma.h"

using namespace AscendC;

namespace af {

class TestApiIgamma : public testing::Test {
 protected:
  static void InvokeIgammaKernel(BinaryInputParam<float> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> aBuf, xBuf, dstBuf, tmpBuf;
    tpipe.InitBuffer(aBuf, sizeof(float) * param.size);
    tpipe.InitBuffer(xBuf, sizeof(float) * param.size);
    tpipe.InitBuffer(dstBuf, sizeof(float) * param.size);
    tpipe.InitBuffer(tmpBuf, 2048 * sizeof(uint8_t));
    LocalTensor<float> aTensor = aBuf.Get<float>();
    LocalTensor<float> xTensor = xBuf.Get<float>();
    LocalTensor<float> dstTensor = dstBuf.Get<float>();
    LocalTensor<uint8_t> tmpTensor = tmpBuf.Get<uint8_t>();

    GmToUb(aTensor, param.x1, param.size);
    GmToUb(xTensor, param.x2, param.size);
    AscendC::IgammaExtend<float>(dstTensor, aTensor, xTensor, tmpTensor, param.size);
    UbToGm(param.y, dstTensor, param.size);
  }

  // ==================== calc_igamma_float_cpu 测试数据 ====================
  static BinaryInputParam<float> BuildIgammaTestData() {
    std::vector<float> a_list, x_list, exp_list;

    // ----- 边界0: a<0 或 x<0 (返回 NAN) -----
    a_list.push_back(-1.0f);
    x_list.push_back(0.5f);
    exp_list.push_back(NAN);
    a_list.push_back(0.5f);
    x_list.push_back(-1.0f);
    exp_list.push_back(NAN);
    a_list.push_back(-2.0f);
    x_list.push_back(-3.0f);
    exp_list.push_back(NAN);

    // ----- 边界1: a==0.0f && x>0 (返回 1.0f) -----
    a_list.push_back(0.0f);
    x_list.push_back(1.0f);
    exp_list.push_back(1.0f);
    a_list.push_back(0.0f);
    x_list.push_back(5.0f);
    exp_list.push_back(1.0f);
    a_list.push_back(0.0f);
    x_list.push_back(100.0f);
    exp_list.push_back(1.0f);

    // ----- 边界2: a==0.0f && x==0.0f (返回 0.0f) -----
    a_list.push_back(0.0f);
    x_list.push_back(0.0f);
    exp_list.push_back(0.0f);

    // ----- 边界3: x==0.0f && a!=0.0f (返回 0.0f) -----
    a_list.push_back(1.0f);
    x_list.push_back(0.0f);
    exp_list.push_back(0.0f);
    a_list.push_back(5.5f);
    x_list.push_back(0.0f);
    exp_list.push_back(0.0f);
    a_list.push_back(0.1f);
    x_list.push_back(0.0f);
    exp_list.push_back(0.0f);

    // ----- 边界4: isinf(a) && !isinf(x) (返回 0.0f) -----
    a_list.push_back(INFINITY);
    x_list.push_back(1.0f);
    exp_list.push_back(0.0f);
    a_list.push_back(INFINITY);
    x_list.push_back(0.5f);
    exp_list.push_back(0.0f);
    a_list.push_back(INFINITY);
    x_list.push_back(100.0f);
    exp_list.push_back(0.0f);

    // ----- 边界5: isinf(a) && isinf(x) (返回 NAN) -----
    a_list.push_back(INFINITY);
    x_list.push_back(INFINITY);
    exp_list.push_back(NAN);

    // ----- 边界6: isinf(x) (返回 1.0f) -----
    a_list.push_back(1.0f);
    x_list.push_back(INFINITY);
    exp_list.push_back(1.0f);
    a_list.push_back(0.5f);
    x_list.push_back(INFINITY);
    exp_list.push_back(1.0f);
    a_list.push_back(10.0f);
    x_list.push_back(INFINITY);
    exp_list.push_back(1.0f);

    // ----- 渐近分支1: 20<a<200, |x-a|/a < 0.3 (DLMF 8.12.3, igamma=true) -----
    a_list.push_back(30.0f);
    x_list.push_back(31.0f);
    exp_list.push_back(boost::math::gamma_p(a_list.back(), x_list.back()));
    a_list.push_back(50.0f);
    x_list.push_back(55.0f);
    exp_list.push_back(boost::math::gamma_p(a_list.back(), x_list.back()));
    a_list.push_back(150.0f);
    x_list.push_back(170.0f);
    exp_list.push_back(boost::math::gamma_p(a_list.back(), x_list.back()));

    // ----- 渐近分支2: a>200, |x-a|/a < 4.5/sqrt(a) -----
    a_list.push_back(400.0f);
    x_list.push_back(410.0f);
    exp_list.push_back(boost::math::gamma_p(a_list.back(), x_list.back()));
    a_list.push_back(900.0f);
    x_list.push_back(930.0f);
    exp_list.push_back(boost::math::gamma_p(a_list.back(), x_list.back()));
    a_list.push_back(1600.0f);
    x_list.push_back(1650.0f);
    exp_list.push_back(boost::math::gamma_p(a_list.back(), x_list.back()));

    // ----- 互补分支: (x>1.0) && (x>a) -> 返回 1 - igammac(a,x) -----
    a_list.push_back(2.0f);
    x_list.push_back(5.0f);
    exp_list.push_back(boost::math::gamma_p(a_list.back(), x_list.back()));
    a_list.push_back(1.5f);
    x_list.push_back(3.0f);
    exp_list.push_back(boost::math::gamma_p(a_list.back(), x_list.back()));
    a_list.push_back(0.8f);
    x_list.push_back(2.0f);
    exp_list.push_back(boost::math::gamma_p(a_list.back(), x_list.back()));

    // ----- 默认级数分支: 其余所有情况 (直接使用 P 级数) -----
    a_list.push_back(5.0f);
    x_list.push_back(2.0f);
    exp_list.push_back(boost::math::gamma_p(a_list.back(), x_list.back()));  // x<a 且 x>1? 不满足互补条件
    a_list.push_back(0.5f);
    x_list.push_back(0.3f);
    exp_list.push_back(boost::math::gamma_p(a_list.back(), x_list.back()));  // x<1
    a_list.push_back(10.0f);
    x_list.push_back(0.8f);
    exp_list.push_back(boost::math::gamma_p(a_list.back(), x_list.back()));  // x<1

    BinaryInputParam<float> param{};
    uint32_t n = (uint32_t)a_list.size();
    param.size = n;
    param.y = static_cast<float *>(GmAlloc(sizeof(float) * param.size));
    param.exp = static_cast<float *>(GmAlloc(sizeof(float) * param.size));
    param.x1 = static_cast<float *>(GmAlloc(sizeof(float) * param.size));
    param.x2 = static_cast<float *>(GmAlloc(sizeof(float) * param.size));
    for (uint32_t i = 0; i < n; ++i) {
      param.x1[i] = a_list[i];
      param.x2[i] = x_list[i];
      param.exp[i] = exp_list[i];
    }
    return param;
  }

  static uint32_t Valid(float *dst, float *exp, size_t comp_size) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < comp_size; i++) {
      if (std::isnan(exp[i])) {
        if (!std::isnan(dst[i])) {
          diff_count++;
        }
      } else if (std::isinf(exp[i])) {
        if (dst[i] != exp[i]) {
          diff_count++;
        }
      } else if (std::abs(dst[i] - exp[i]) > 1e-5) {
        diff_count++;
      }
    }
    return diff_count;
  }

  static void IgammaTest() {
    auto param = BuildIgammaTestData();

    // 构造Api调用函数
    auto kernel = [&param] { InvokeIgammaKernel(param); };

    // 调用kernel
    SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
  }
};

TEST_F(TestApiIgamma, igamma_test) {
  IgammaTest();
}

}  // namespace af
