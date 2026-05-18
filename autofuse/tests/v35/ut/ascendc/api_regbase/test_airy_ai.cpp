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

/**
 * test_add.cpp
 */

#include <cmath>
#include <random>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "airy_ai.h"
#include "boost/math/special_functions/airy.hpp"

using namespace AscendC;

namespace af {

class TestRegbaseApiAiryAiUT : public testing::Test {
 protected:
  // Tensor - Tensor 场景
  template <typename T>
  static void InvokeTensorTensorKernel(UnaryInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, temp;
    tpipe.InitBuffer(xbuf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * param.size);
    tpipe.InitBuffer(temp, TMP_UB_SIZE);

    LocalTensor<T> l_x = xbuf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = temp.Get<uint8_t>();

    GmToUb(l_x, param.x1, param.size);
    AiryAiExtend(l_y, l_x, l_tmp, param.size);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void CreateTensorInput(UnaryInputParam<T> &param) {
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    std::mt19937 eng(1);

    for (uint32_t i = 0; i < param.size; i++) {
        std::uniform_real_distribution<float> distr(-20.0f, 20.0f);
        param.x1[i] = static_cast<T>(distr(eng));
        param.exp[i] = static_cast<T>(boost::math::airy_ai((double)(param.x1[i])));
    }
  }

  template <typename T>
  static void AiryAiTensorTensorTest(uint32_t size) {
    UnaryInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeTensorTensorKernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
  }
};

TEST_F(TestRegbaseApiAiryAiUT, AiryAi_TensorTensor_Test) {
    AiryAiTensorTensorTest<float>(ONE_BLK_SIZE / sizeof(float));
    AiryAiTensorTensorTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
    AiryAiTensorTensorTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
    AiryAiTensorTensorTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
    AiryAiTensorTensorTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
    AiryAiTensorTensorTest<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(float))) / 2 /sizeof(float));
}
}