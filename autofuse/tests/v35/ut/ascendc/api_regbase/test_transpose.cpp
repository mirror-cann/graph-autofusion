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
#include "api_regbase/transpose.h"

using namespace AscendC;

template <typename T, typename U>
struct TensorTransposeInputParam {
  T *x{};
  T *y{};
  U *idx{};
  T *y_exp{};
  U *idx_exp{};
  std::vector<U> dst_dims;
  std::vector<U> src_strides;
  std::vector<U> dst_strides;
  U size;
};

class TestRegbaseApiTranspose :public testing::Test {
 protected:
  template <typename T, typename U>
  static void InvokeKernelTestIndex(TensorTransposeInputParam<T, U> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> idx_buf;

    tpipe.InitBuffer(idx_buf, sizeof(U) * param.size);
    LocalTensor<U> l_idx = idx_buf.Get<U>();
    if (param.dst_dims.size() == 1) {
      GenTransposeIndex<U, 1>((__ubuf__ U*)l_idx.GetPhyAddr(), {param.dst_dims[0]}, {param.src_strides[0]}, param.size);
    } else if (param.dst_dims.size() == 2) {
      GenTransposeIndex<U, 2>((__ubuf__ U*)l_idx.GetPhyAddr(), {param.dst_dims[0], param.dst_dims[1]},
        {param.src_strides[0], param.src_strides[1]}, param.size);
    } else if (param.dst_dims.size() == 3) {
      GenTransposeIndex<U, 3>((__ubuf__ U*)l_idx.GetPhyAddr(), {param.dst_dims[0], param.dst_dims[1], param.dst_dims[2]},
        {param.src_strides[0], param.src_strides[1], param.src_strides[2]}, param.size);
    } else if (param.dst_dims.size() == 4) {
      GenTransposeIndex<U, 4>((__ubuf__ U*)l_idx.GetPhyAddr(),
        {param.dst_dims[0], param.dst_dims[1], param.dst_dims[2], param.dst_dims[3]},
        {param.src_strides[0], param.src_strides[1], param.src_strides[2], param.src_strides[3]}, param.size);
    } else if (param.dst_dims.size() == 5) {
      GenTransposeIndex<U, 5>((__ubuf__ U*)l_idx.GetPhyAddr(),
        {param.dst_dims[0], param.dst_dims[1], param.dst_dims[2], param.dst_dims[3], param.dst_dims[4]},
        {param.src_strides[0], param.src_strides[1], param.src_strides[2], param.src_strides[3], param.src_strides[4]}, param.size);
    }
    UbToGm(param.idx, l_idx, param.size);
  }

  template <typename T, typename U>
  static void CreateDimIndex(TensorTransposeInputParam<T, U> &param) {
    // 构造测试输入和预期结果
    param.idx = static_cast<U *>(AscendC::GmAlloc(sizeof(U) * param.size));
    param.idx_exp = static_cast<U *>(AscendC::GmAlloc(sizeof(U) * param.size));

    std::vector<U> idx(param.dst_dims.size());

    for (int i = 0; i < param.size; i++) {
      param.idx_exp[i] = 0;
      for (int j = 0; j < param.dst_dims.size(); j++) {
        idx[j] = i / param.dst_strides[j];
        idx[j] = idx[j] % param.dst_dims[j];
        param.idx_exp[i] += idx[j] * param.src_strides[j];
      }
    }
  }

  template <typename T, typename U>
  static void FreeIndexTensor(TensorTransposeInputParam<T, U> &param) {
    AscendC::GmFree(param.idx);
    AscendC::GmFree(param.idx_exp);
  }

  template <typename T>
  static uint32_t Valid(T *y, T *exp, size_t comp_size) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < comp_size; i++) {
      if (y[i] != exp[i]) {
        diff_count++;
      }
    }
    return diff_count;
  }

  template <typename T, typename U>
  static void InitTransposeInputParam(TensorTransposeInputParam<T, U> &param,
                                      std::pair<std::vector<U>, std::vector<U>> input_params) {
    param.size = 1;
    param.dst_dims.resize(input_params.first.size());
    param.src_strides.resize(input_params.first.size());
    param.dst_strides.resize(input_params.first.size());
    std::vector<U> tmp(input_params.first.size());
    for (int i = input_params.first.size() - 1; i >= 0; i--) {
      param.size *= input_params.first[i];
      param.dst_dims[i] = input_params.first[input_params.second[i]];
      if (i == input_params.first.size() - 1) {
        tmp[i] = 1;
        param.dst_strides[i] = 1;
      } else {
        tmp[i] = input_params.first[i + 1] * tmp[i + 1];
        param.dst_strides[i] = param.dst_dims[i + 1] * param.dst_strides[i + 1];
      }
    }
    for (int i = 0; i < input_params.first.size(); i++) {
      param.src_strides[i] = tmp[input_params.second[i]];
    }
  }

  template <typename T, typename U>
  static void TransposeIndexTest(std::pair<std::vector<U>, std::vector<U>> input_params) {
    if (input_params.first.size() != input_params.second.size()) {
      return;
    }
    TensorTransposeInputParam<T, U> param{};
    InitTransposeInputParam(param, input_params);
    CreateDimIndex(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeKernelTestIndex(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = Valid(param.idx, param.idx_exp, param.size);
    EXPECT_EQ(diff_count, 0);
    // 释放内存
    FreeIndexTensor(param);
  }

  template <typename T, typename U>
  static void CreateTransposeTensor(TensorTransposeInputParam<T, U> &param) {
    // 构造测试输入和预期结果
    param.x = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.idx = static_cast<U *>(AscendC::GmAlloc(sizeof(U) * param.size));
    param.y_exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    CreateDimIndex(param);
    for (int i = 0; i < param.size; i++) {
      param.x[i] = i;
    }
    for (int i = 0; i < param.size; i++) {
      param.y_exp[i] = param.x[param.idx_exp[i]];
    }
  }

  template <typename T, typename U>
  static void InvokeKernelTest(TensorTransposeInputParam<T, U> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x_buf;
    TBuf<TPosition::VECCALC> y_buf;
    TBuf<TPosition::VECCALC> idx_buf;

    tpipe.InitBuffer(x_buf, sizeof(T) * param.size);
    tpipe.InitBuffer(y_buf, sizeof(T) * param.size);
    tpipe.InitBuffer(idx_buf, sizeof(U) * param.size);

    LocalTensor<T> l_x = x_buf.Get<T>();
    LocalTensor<T> l_y = y_buf.Get<T>();
    LocalTensor<U> l_idx = idx_buf.Get<U>();

    GmToUb(l_x, param.x, param.size);
    GmToUb(l_idx, param.idx, param.size);

    if (param.dst_dims.size() == 1) {
      TransposeExtend<T, 1>(l_y, l_x, l_idx, {param.dst_dims[0]}, {param.src_strides[0]}, {param.dst_strides[0]});
    } else if (param.dst_dims.size() == 2) {
      TransposeExtend<T, 2>(l_y, l_x, l_idx, {param.dst_dims[0], param.dst_dims[1]},
        {param.src_strides[0], param.src_strides[1]}, {param.dst_strides[0], param.dst_strides[1]});
    } else if (param.dst_dims.size() == 3) {
      TransposeExtend<T, 3>(l_y, l_x, l_idx, {param.dst_dims[0], param.dst_dims[1], param.dst_dims[2]},
        {param.src_strides[0], param.src_strides[1], param.src_strides[2]},
        {param.dst_strides[0], param.dst_strides[1], param.dst_strides[2]});
    } else if (param.dst_dims.size() == 4) {
      TransposeExtend<T, 4>(l_y, l_x, l_idx,
        {param.dst_dims[0], param.dst_dims[1], param.dst_dims[2], param.dst_dims[3]},
        {param.src_strides[0], param.src_strides[1], param.src_strides[2], param.src_strides[3]},
        {param.dst_strides[0], param.dst_strides[1], param.dst_strides[2], param.dst_strides[3]});
    } else if (param.dst_dims.size() == 5) {
      TransposeExtend<T, 5>(l_y, l_x, l_idx,
        {param.dst_dims[0], param.dst_dims[1], param.dst_dims[2], param.dst_dims[3], param.dst_dims[4]},
        {param.src_strides[0], param.src_strides[1], param.src_strides[2], param.src_strides[3], param.src_strides[4]},
        {param.dst_strides[0], param.dst_strides[1], param.dst_strides[2], param.dst_strides[3], param.dst_strides[4]});
    } else if (param.dst_dims.size() == 6) {
      TransposeExtend<T, 6>(l_y, l_x, l_idx,
        {param.dst_dims[0], param.dst_dims[1], param.dst_dims[2], param.dst_dims[3], param.dst_dims[4], param.dst_dims[5]},
        {param.src_strides[0], param.src_strides[1], param.src_strides[2], param.src_strides[3], param.src_strides[4], param.src_strides[5]},
        {param.dst_strides[0], param.dst_strides[1], param.dst_strides[2], param.dst_strides[3], param.dst_strides[4], param.dst_strides[5]});
    } else if (param.dst_dims.size() == 7) {
      TransposeExtend<T, 7>(l_y, l_x, l_idx,
        {param.dst_dims[0], param.dst_dims[1], param.dst_dims[2], param.dst_dims[3], param.dst_dims[4], param.dst_dims[5], param.dst_dims[6]},
        {param.src_strides[0], param.src_strides[1], param.src_strides[2], param.src_strides[3], param.src_strides[4], param.src_strides[5], param.src_strides[6]},
        {param.dst_strides[0], param.dst_strides[1], param.dst_strides[2], param.dst_strides[3], param.dst_strides[4], param.dst_strides[5], param.dst_strides[6]});
    } else if (param.dst_dims.size() == 8) {
      TransposeExtend<T, 8>(l_y, l_x, l_idx,
        {param.dst_dims[0], param.dst_dims[1], param.dst_dims[2], param.dst_dims[3], param.dst_dims[4], param.dst_dims[5], param.dst_dims[6], param.dst_dims[7]},
        {param.src_strides[0], param.src_strides[1], param.src_strides[2], param.src_strides[3], param.src_strides[4], param.src_strides[5], param.src_strides[6], param.src_strides[7]},
        {param.dst_strides[0], param.dst_strides[1], param.dst_strides[2], param.dst_strides[3], param.dst_strides[4], param.dst_strides[5], param.dst_strides[6], param.dst_strides[7]});
    }
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T, typename U>
  static void TransposeTest(std::pair<std::vector<U>, std::vector<U>> input_params) {
    if (input_params.first.size() != input_params.second.size()) {
      return;
    }
    TensorTransposeInputParam<T, U> param{};
    InitTransposeInputParam(param, input_params);
    CreateTransposeTensor(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeKernelTest(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = Valid(param.y, param.y_exp, param.size);
    EXPECT_EQ(diff_count, 0);
    // 释放内存
    FreeIndexTensor(param);
  }
};

TEST_F(TestRegbaseApiTranspose, Transpose_Index_Test) {
  TransposeIndexTest<float, int32_t>({{8}, {0}});
  TransposeIndexTest<float, int32_t>({{5, 8}, {1, 0}});
  TransposeIndexTest<float, int32_t>({{5, 8, 4}, {2, 1, 0}});
  TransposeIndexTest<float, int32_t>({{5, 8, 4, 3}, {3, 2, 1, 0}});
  TransposeIndexTest<float, int32_t>({{3, 2, 4, 3, 2}, {4, 3, 2, 1, 0}});
}

TEST_F(TestRegbaseApiTranspose, Transpose_Test) {
  TransposeTest<float, int32_t>({{8}, {0}});
  TransposeTest<float, int32_t>({{5, 8}, {1, 0}});
  TransposeTest<float, int32_t>({{5, 8, 4}, {2, 1, 0}});
  TransposeTest<float, int32_t>({{5, 8, 4, 3}, {3, 2, 1, 0}});
  TransposeTest<float, int32_t>({{3, 2, 4, 3, 2}, {4, 3, 2, 1, 0}});
  TransposeTest<float, int32_t>({{2, 2, 3, 4, 3, 3}, {5, 4, 3, 2, 1, 0}});
  TransposeTest<float, int32_t>({{1, 2, 2, 3, 4, 3, 2}, {6, 5, 4, 3, 2, 1, 0}});
  TransposeTest<float, int32_t>({{1, 2, 2, 3, 4, 3, 2, 4}, {7, 6, 5, 4, 3, 2, 1, 0}});
}
