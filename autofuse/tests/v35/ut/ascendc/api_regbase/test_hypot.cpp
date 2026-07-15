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
#include "api_regbase/hypot.h"

using namespace AscendC;

namespace af {

struct HypotInputParam {
  float *y{};
  float *exp{};
  float *src0{};
  float *src1{};
  uint32_t size{0};
};

class TestRegbaseApiHypot : public testing::Test {
 protected:
  static void InvokeKernel(HypotInputParam &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x0Buf, x1Buf, yBuf, tmpBuf;
    tpipe.InitBuffer(x0Buf, sizeof(float) * param.size);
    tpipe.InitBuffer(x1Buf, sizeof(float) * param.size);
    tpipe.InitBuffer(yBuf, sizeof(float) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(float)));
    tpipe.InitBuffer(tmpBuf, TMP_UB_SIZE);

    LocalTensor<float> l_x0 = x0Buf.Get<float>();
    LocalTensor<float> l_x1 = x1Buf.Get<float>();
    LocalTensor<float> l_y = yBuf.Get<float>();
    LocalTensor<uint8_t> l_tmp = tmpBuf.Get<uint8_t>();

    GmToUb(l_x0, param.src0, param.size);
    GmToUb(l_x1, param.src1, param.size);
    HypotExtend(l_y, l_x0, l_x1, l_tmp, param.size);
    UbToGm(param.y, l_y, param.size);
  }

  static void CreateRandomInput(HypotInputParam &param, uint32_t size) {
    param.size = size;
    param.y = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * param.size));
    param.exp = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * param.size));
    param.src0 = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * param.size));
    param.src1 = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * param.size));

    std::mt19937 eng(1);
    std::uniform_real_distribution<float> distr(-100.0f, 100.0f);

    for (uint32_t i = 0; i < param.size; i++) {
      param.src0[i] = distr(eng);
      param.src1[i] = distr(eng);
      param.exp[i] = std::hypot(param.src0[i], param.src1[i]);
    }
  }

  static void CreateBoundaryInput(HypotInputParam &param, const std::vector<std::pair<float, float>> &cases) {
    param.size = static_cast<uint32_t>(cases.size());
    param.y = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * param.size));
    param.exp = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * param.size));
    param.src0 = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * param.size));
    param.src1 = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * param.size));

    for (uint32_t i = 0; i < param.size; i++) {
      param.src0[i] = cases[i].first;
      param.src1[i] = cases[i].second;
      param.exp[i] = std::hypot(cases[i].first, cases[i].second);
    }
  }

  static uint32_t Valid(float *y, float *exp, uint32_t size) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < size; i++) {
      bool both_nan = std::isnan(y[i]) && std::isnan(exp[i]);
      bool both_inf = std::isinf(y[i]) && std::isinf(exp[i]);
      bool close = false;
      if (!both_nan && !both_inf) {
        float rel_err = std::abs(y[i] - exp[i]) / std::max(std::abs(exp[i]), 1e-30f);
        close = rel_err < 1e-5f;
      }
      if (!both_nan && !both_inf && !close) {
        diff_count++;
        printf("diff at index %u: got=%.9f, exp=%.9f\n", i, y[i], exp[i]);
      }
    }
    return diff_count;
  }

  static void FreeInput(HypotInputParam &param) {
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.src0);
    AscendC::GmFree(param.src1);
  }

  static void HypotRandomTest(uint32_t size) {
    HypotInputParam param{};
    CreateRandomInput(param, size);

    auto kernel = [&param] { InvokeKernel(param); };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
    FreeInput(param);
  }

  static void HypotBoundaryTest() {
    const std::vector<std::pair<float, float>> cases = {
        {0.0f, 0.0f},
        {-0.0f, 0.0f},
        {0.0f, -0.0f},
        {3.0f, 4.0f},
        {-3.0f, 4.0f},
        {3.0f, -4.0f},
        {-3.0f, -4.0f},
        {1e38f, 1e38f},
        {1e-20f, 1e-20f},
        {1e38f, 1e-20f},
        {1e-20f, 1e38f},
        {1e38f, 0.0f},
        {0.0f, 1e38f},
        {1e-20f, 0.0f},
        {0.0f, 1e-20f},
        {INFINITY, 3.0f},
        {3.0f, INFINITY},
        {-INFINITY, 3.0f},
        {3.0f, -INFINITY},
        {INFINITY, INFINITY},
        {-INFINITY, INFINITY},
        {INFINITY, -INFINITY},
        {-INFINITY, -INFINITY},
        {INFINITY, 0.0f},
        {0.0f, INFINITY},
        {NAN, 3.0f},
        {3.0f, NAN},
        {NAN, NAN},
        {NAN, 0.0f},
        {0.0f, NAN},
        {NAN, INFINITY},
        {INFINITY, NAN},
        {FLT_MAX, FLT_MAX},
        {1.0f, FLT_MAX},
    };

    HypotInputParam param{};
    CreateBoundaryInput(param, cases);

    auto kernel = [&param] { InvokeKernel(param); };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
    FreeInput(param);
  }
};

TEST_F(TestRegbaseApiHypot, Hypot_Random_Test) {
  HypotRandomTest(ONE_BLK_SIZE / sizeof(float));
  HypotRandomTest(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  HypotRandomTest(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  HypotRandomTest((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  HypotRandomTest((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  HypotRandomTest(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                   (ONE_BLK_SIZE - sizeof(float))) /
                  2 / sizeof(float));
}

TEST_F(TestRegbaseApiHypot, Hypot_Boundary_Test) {
  HypotBoundaryTest();
}

}  // namespace af
