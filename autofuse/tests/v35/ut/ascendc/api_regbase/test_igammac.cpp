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
#include "boost/math/special_functions/gamma.hpp"
#include "api_regbase/igammac_helper/series.h"
#include "api_regbase/igammac_helper/continued_fraction.h"
#include "api_regbase/igammac_helper/asymptotic_series.h"
#include "api_regbase/igammac_helper/series_complement.h"
#include "api_regbase/igammac.h"

using namespace AscendC;

namespace af {

class TestApiIgammac : public testing::Test {
 protected:
  static void InvokeIgammaCKernel(BinaryInputParam<float> &param) {
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
    AscendC::IgammacExtend<float>(dstTensor, aTensor, xTensor, tmpTensor, param.size);
    UbToGm(param.y, dstTensor, param.size);
  }

  // ==================== calc_igammac_float_cpu 测试数据 ====================
  static BinaryInputParam<float> BuildIgammaCTestData() {
    std::vector<float> a_list, x_list, exp_list;

    // ----- 边界0: a<0 或 x<0 (返回 NAN) -----
    a_list.push_back(-1.0f);
    x_list.push_back(0.5f);
    exp_list.push_back(NAN);  // a<0
    a_list.push_back(0.5f);
    x_list.push_back(-1.0f);
    exp_list.push_back(NAN);  // x<0
    a_list.push_back(-2.0f);
    x_list.push_back(-3.0f);
    exp_list.push_back(NAN);  // both <0

    // ----- 边界1: a==0.0f && x>0 (返回 0.0f) -----
    a_list.push_back(0.0f);
    x_list.push_back(1.0f);
    exp_list.push_back(0.0f);
    a_list.push_back(0.0f);
    x_list.push_back(5.0f);
    exp_list.push_back(0.0f);
    a_list.push_back(0.0f);
    x_list.push_back(100.0f);
    exp_list.push_back(0.0f);

    // ----- 边界2: a==0.0f && x==0.0f (返回 1.0f) -----
    a_list.push_back(0.0f);
    x_list.push_back(0.0f);
    exp_list.push_back(1.0f);

    // ----- 边界3: x==0.0f && a!=0.0f (返回 1.0f) -----
    a_list.push_back(1.0f);
    x_list.push_back(0.0f);
    exp_list.push_back(1.0f);
    a_list.push_back(5.5f);
    x_list.push_back(0.0f);
    exp_list.push_back(1.0f);
    a_list.push_back(0.1f);
    x_list.push_back(0.0f);
    exp_list.push_back(1.0f);

    // ----- 边界4: isinf(a) && !isinf(x) (返回 1.0f) -----
    a_list.push_back(INFINITY);
    x_list.push_back(1.0f);
    exp_list.push_back(1.0f);
    a_list.push_back(INFINITY);
    x_list.push_back(0.5f);
    exp_list.push_back(1.0f);
    a_list.push_back(INFINITY);
    x_list.push_back(100.0f);
    exp_list.push_back(1.0f);

    // ----- 边界5: isinf(a) && isinf(x) (返回 NAN) -----
    a_list.push_back(INFINITY);
    x_list.push_back(INFINITY);
    exp_list.push_back(NAN);

    // ----- 边界6: isinf(x) (返回 0.0f) -----
    a_list.push_back(1.0f);
    x_list.push_back(INFINITY);
    exp_list.push_back(0.0f);
    a_list.push_back(0.5f);
    x_list.push_back(INFINITY);
    exp_list.push_back(0.0f);
    a_list.push_back(10.0f);
    x_list.push_back(INFINITY);
    exp_list.push_back(0.0f);

    // ----- 渐近分支1: 20<a<200, |x-a|/a < 0.3 (DLMF 8.12.4) -----
    a_list.push_back(30.0f);
    x_list.push_back(31.0f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));  // 需要真值
    a_list.push_back(50.0f);
    x_list.push_back(55.0f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));
    a_list.push_back(150.0f);
    x_list.push_back(170.0f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));

    // ----- 渐近分支2: a>200, |x-a|/a < 4.5/sqrt(a) (DLMF 8.12.4) -----
    a_list.push_back(400.0f);
    x_list.push_back(410.0f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));
    a_list.push_back(900.0f);
    x_list.push_back(930.0f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));
    a_list.push_back(1600.0f);
    x_list.push_back(1650.0f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));

    // ----- 主分支: x>1.1 && x<a (返回 1 - P(a,x), 用级数) -----
    a_list.push_back(5.0f);
    x_list.push_back(2.0f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));
    a_list.push_back(10.0f);
    x_list.push_back(5.0f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));
    a_list.push_back(3.0f);
    x_list.push_back(1.2f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));

    // ----- 主分支: x>1.1 && x>=a (返回 连分式 Q(a,x)) -----
    a_list.push_back(2.0f);
    x_list.push_back(5.0f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));
    a_list.push_back(1.5f);
    x_list.push_back(1.6f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));
    a_list.push_back(10.0f);
    x_list.push_back(20.0f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));

    // ----- 小x分支 (x<=0.5) 且 -0.4/log(x) < a : 返回 1-P (级数)-----
    a_list.push_back(1.5f);
    x_list.push_back(0.1f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));  // log(0.1)=-2.3026, 0.1737<1.5
    a_list.push_back(2.0f);
    x_list.push_back(0.2f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));  // 0.2485<2.0
    a_list.push_back(0.8f);
    x_list.push_back(0.05f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));  // 0.1335<0.8

    // ----- 小x分支 (x<=0.5) 且 -0.4/log(x) >= a : 返回 series_complement -----
    a_list.push_back(0.1f);
    x_list.push_back(0.1f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));  // 0.1737>=0.1
    a_list.push_back(0.05f);
    x_list.push_back(0.01f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));  // 0.0869>=0.05
    a_list.push_back(0.3f);
    x_list.push_back(0.3f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));  // 0.332>=0.3

    // ----- 中间x分支 (0.5<x<=1.1) 且 1.1*x < a : 返回 1-P (级数)-----
    a_list.push_back(2.0f);
    x_list.push_back(1.0f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));  // 1.1*1.0=1.1<2
    a_list.push_back(3.0f);
    x_list.push_back(0.8f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));  // 0.88<3
    a_list.push_back(1.5f);
    x_list.push_back(0.9f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));  // 0.99<1.5

    // ----- 中间x分支 (0.5<x<=1.1) 且 1.1*x >= a : 返回 series_complement -----
    a_list.push_back(0.8f);
    x_list.push_back(0.9f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));  // 0.99 >= 0.8
    a_list.push_back(1.0f);
    x_list.push_back(0.95f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));  // 1.045 >= 1.0
    a_list.push_back(0.6f);
    x_list.push_back(0.7f);
    exp_list.push_back(boost::math::gamma_q(a_list.back(), x_list.back()));  // 0.77 >= 0.6

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

  static void IgammaCTest() {
    auto param = BuildIgammaCTestData();

    // 构造Api调用函数
    auto kernel = [&param] { InvokeIgammaCKernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
  }
};

TEST_F(TestApiIgammac, igammac_test) {
  IgammaCTest();
}
}  // namespace af
