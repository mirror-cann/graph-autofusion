/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * test_bucketize.cpp
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/bucketize.h"

using namespace AscendC;

namespace af {

template <typename T>
struct BucketizeInputParam {
  int32_t *y{};
  int32_t *exp{};
  T *src{};
  T *boundaries{};
  uint32_t calCount{};
  uint32_t boundCount{};
  bool right{false};
};

class TestRegbaseApiBucketizeUT : public testing::Test {
 protected:
  template <typename T>
  static void InvokeKernel(BucketizeInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> srcBuf, boundariesBuf, dstBuf, tmpBuf;
    tpipe.InitBuffer(srcBuf, sizeof(T) * param.calCount);
    // InitBuffer rejects 0; when boundCount==0 the buffer is unused (API
    // returns early via Duplicate), but still needs a non-zero allocation.
    tpipe.InitBuffer(boundariesBuf, std::max<uint32_t>(sizeof(T) * param.boundCount, 1u));
    tpipe.InitBuffer(dstBuf, sizeof(int32_t) * AlignUp(param.calCount, ONE_BLK_SIZE / sizeof(int32_t)));
    tpipe.InitBuffer(tmpBuf, TMP_UB_SIZE);

    LocalTensor<T> l_src = srcBuf.Get<T>();
    LocalTensor<T> l_boundaries = boundariesBuf.Get<T>();
    LocalTensor<int32_t> l_dst = dstBuf.Get<int32_t>();
    LocalTensor<uint8_t> l_tmp = tmpBuf.Get<uint8_t>();

    GmToUb(l_src, param.src, param.calCount);
    GmToUb(l_boundaries, param.boundaries, param.boundCount);
    BucketizeExtend(l_dst, l_src, l_boundaries, l_tmp, param.calCount, param.boundCount, param.right);
    UbToGm(param.y, l_dst, param.calCount);
  }

  // CPU reference: binary search matching the device algorithm.
  // right=false → lower_bound: smallest i such that boundaries[i] >= value.
  // right=true  → upper_bound: smallest i such that boundaries[i] >  value.
  template <typename T>
  static int32_t CpuBucketize(const T *boundaries, uint32_t boundCount, T value, bool right) {
    int32_t low = 0, high = static_cast<int32_t>(boundCount);
    while (low < high) {
      int32_t mid = (low + high) / 2;
      bool hit;
      if (right) {
        hit = boundaries[mid] > value;
      } else {
        hit = boundaries[mid] >= value;
      }
      if (hit) {
        high = mid;
      } else {
        low = mid + 1;
      }
    }
    // NaN → boundCount (PyTorch semantics; floating-point inputs only)
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
      if (std::isnan(static_cast<float>(value))) {
        return static_cast<int32_t>(boundCount);
      }
    }
    return low;
  }

  // Helper: generate sorted boundaries and src/exp for floating-point family.
  template <typename T>
  static void CreateRandomInputFloat(BucketizeInputParam<T> &param, std::mt19937 &eng) {
    std::uniform_real_distribution<float> distr(-100.0f, 100.0f);
    std::vector<float> tmp(param.boundCount);
    for (uint32_t i = 0; i < param.boundCount; i++) {
      tmp[i] = distr(eng);
    }
    std::sort(tmp.begin(), tmp.end());
    for (uint32_t i = 0; i < param.boundCount; i++) {
      param.boundaries[i] = static_cast<T>(tmp[i]);
    }
    std::uniform_real_distribution<float> srcDistr(-150.0f, 150.0f);
    for (uint32_t i = 0; i < param.calCount; i++) {
      param.src[i] = static_cast<T>(srcDistr(eng));
      param.exp[i] = CpuBucketize(param.boundaries, param.boundCount, param.src[i], param.right);
    }
  }

  // Helper: generate sorted boundaries and src/exp for signed integer family.
  template <typename T, typename IntT>
  static void CreateRandomInputSigned(BucketizeInputParam<T> &param, std::mt19937 &eng, IntT valRange) {
    std::uniform_int_distribution<IntT> distr(-valRange / 2, valRange / 2 - 1);
    std::vector<IntT> tmp(param.boundCount);
    for (uint32_t i = 0; i < param.boundCount; i++) {
      tmp[i] = distr(eng);
    }
    std::sort(tmp.begin(), tmp.end());
    for (uint32_t i = 0; i < param.boundCount; i++) {
      param.boundaries[i] = static_cast<T>(tmp[i]);
    }
    std::uniform_int_distribution<IntT> srcDistr(-valRange, valRange);
    for (uint32_t i = 0; i < param.calCount; i++) {
      param.src[i] = static_cast<T>(srcDistr(eng));
      param.exp[i] = CpuBucketize(param.boundaries, param.boundCount, param.src[i], param.right);
    }
  }

  // Helper: generate sorted boundaries and src/exp for unsigned integer family.
  template <typename T, typename UIntT>
  static void CreateRandomInputUnsigned(BucketizeInputParam<T> &param, std::mt19937 &eng, UIntT valRange,
                                        UIntT offset) {
    std::uniform_int_distribution<UIntT> distr(offset, valRange - offset);
    std::vector<UIntT> tmp(param.boundCount);
    for (uint32_t i = 0; i < param.boundCount; i++) {
      tmp[i] = distr(eng);
    }
    std::sort(tmp.begin(), tmp.end());
    for (uint32_t i = 0; i < param.boundCount; i++) {
      param.boundaries[i] = static_cast<T>(tmp[i]);
    }
    std::uniform_int_distribution<UIntT> srcDistr(0, valRange);
    for (uint32_t i = 0; i < param.calCount; i++) {
      param.src[i] = static_cast<T>(srcDistr(eng));
      param.exp[i] = CpuBucketize(param.boundaries, param.boundCount, param.src[i], param.right);
    }
  }

  template <typename T>
  static void CreateRandomInput(BucketizeInputParam<T> &param) {
    uint32_t alignedDstCount = AlignUp(param.calCount, ONE_BLK_SIZE / sizeof(int32_t));
    param.y = static_cast<int32_t *>(AscendC::GmAlloc(sizeof(int32_t) * alignedDstCount));
    param.exp = static_cast<int32_t *>(AscendC::GmAlloc(sizeof(int32_t) * param.calCount));
    param.src = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.calCount));
    param.boundaries = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.boundCount));

    std::mt19937 eng(1);

    // Dispatch to the appropriate helper based on T.
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
      CreateRandomInputFloat<T>(param, eng);
    } else if constexpr (sizeof(T) == 1) {
      if constexpr (std::is_signed_v<T>) {
        CreateRandomInputSigned<T, int>(param, eng, 100);
      } else {
        CreateRandomInputUnsigned<T, int>(param, eng, 200, 10);
      }
    } else if constexpr (sizeof(T) == 2) {
      if constexpr (std::is_signed_v<T>) {
        CreateRandomInputSigned<T, int>(param, eng, 10000);
      } else {
        CreateRandomInputUnsigned<T, int>(param, eng, 50000, 100);
      }
    } else if constexpr (sizeof(T) == 4) {
      if constexpr (std::is_signed_v<T>) {
        CreateRandomInputSigned<T, int64_t>(param, eng, 100000);
      } else {
        CreateRandomInputUnsigned<T, uint64_t>(param, eng, 200000, 100);
      }
    } else if constexpr (sizeof(T) == 8) {
      if constexpr (std::is_signed_v<T>) {
        CreateRandomInputSigned<T, int64_t>(param, eng, 1000000);
      } else {
        CreateRandomInputUnsigned<T, uint64_t>(param, eng, 2000000, 100);
      }
    }
  }

  template <typename T>
  static uint32_t Valid(BucketizeInputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.calCount; i++) {
      if (param.y[i] != param.exp[i]) {
        diff_count++;
        printf("bucketize diff at index %u: got=%d, exp=%d, src=", i, param.y[i], param.exp[i]);
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
          printf("%f\n", static_cast<float>(param.src[i]));
        } else if constexpr (std::is_same_v<T, int64_t>) {
          printf("%ld\n", static_cast<int64_t>(param.src[i]));
        } else if constexpr (std::is_same_v<T, uint64_t>) {
          printf("%lu\n", static_cast<uint64_t>(param.src[i]));
        } else {
          printf("%d\n", static_cast<int>(param.src[i]));
        }
      }
    }
    return diff_count;
  }

  template <typename T>
  static void FreeInput(BucketizeInputParam<T> &param) {
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.src);
    AscendC::GmFree(param.boundaries);
  }

  template <typename T>
  static void BucketizeTest(uint32_t calCount, uint32_t boundCount, bool right) {
    BucketizeInputParam<T> param{};
    param.calCount = calCount;
    param.boundCount = boundCount;
    param.right = right;
    CreateRandomInput(param);

    auto kernel = [&param] { InvokeKernel(param); };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0);
    FreeInput(param);
  }

  // Fixed-boundary edge-case test: boundaries are manually specified.
  template <typename T>
  static void BucketizeEdgeTest(const std::vector<T> &boundaries, const std::vector<T> &srcValues,
                                const std::vector<int32_t> &expected, bool right) {
    uint32_t calCount = static_cast<uint32_t>(srcValues.size());
    uint32_t boundCount = static_cast<uint32_t>(boundaries.size());

    BucketizeInputParam<T> param{};
    param.calCount = calCount;
    param.boundCount = boundCount;
    param.right = right;

    uint32_t alignedDstCount = AlignUp(param.calCount, ONE_BLK_SIZE / sizeof(int32_t));
    param.y = static_cast<int32_t *>(AscendC::GmAlloc(sizeof(int32_t) * alignedDstCount));
    param.exp = static_cast<int32_t *>(AscendC::GmAlloc(sizeof(int32_t) * param.calCount));
    param.src = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.calCount));
    param.boundaries = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.boundCount));

    for (uint32_t i = 0; i < boundCount; i++) {
      param.boundaries[i] = boundaries[i];
    }
    for (uint32_t i = 0; i < calCount; i++) {
      param.src[i] = srcValues[i];
      param.exp[i] = expected[i];
    }

    auto kernel = [&param] { InvokeKernel(param); };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0);
    FreeInput(param);
  }
};

// ==================== Integer type tests ====================

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Int8_Left_Test) {
  // right=false (lower_bound)
  BucketizeTest<int8_t>(ONE_BLK_SIZE / sizeof(int8_t), 10, false);
  BucketizeTest<int8_t>((ONE_BLK_SIZE - sizeof(int8_t)) / sizeof(int8_t), 10, false);
  BucketizeTest<int8_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int8_t), 10, false);
  BucketizeTest<int8_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int8_t), 10, false);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Int8_Right_Test) {
  BucketizeTest<int8_t>(ONE_BLK_SIZE / sizeof(int8_t), 10, true);
  BucketizeTest<int8_t>((ONE_BLK_SIZE - sizeof(int8_t)) / sizeof(int8_t), 10, true);
  BucketizeTest<int8_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int8_t), 10, true);
  BucketizeTest<int8_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int8_t), 10, true);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Uint8_Left_Test) {
  BucketizeTest<uint8_t>(ONE_BLK_SIZE / sizeof(uint8_t), 12, false);
  BucketizeTest<uint8_t>((ONE_BLK_SIZE - sizeof(uint8_t)) / sizeof(uint8_t), 12, false);
  BucketizeTest<uint8_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t), 12, false);
  BucketizeTest<uint8_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint8_t), 12, false);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Uint8_Right_Test) {
  BucketizeTest<uint8_t>(ONE_BLK_SIZE / sizeof(uint8_t), 12, true);
  BucketizeTest<uint8_t>((ONE_BLK_SIZE - sizeof(uint8_t)) / sizeof(uint8_t), 12, true);
  BucketizeTest<uint8_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t), 12, true);
  BucketizeTest<uint8_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint8_t), 12, true);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Int16_Left_Test) {
  BucketizeTest<int16_t>(ONE_BLK_SIZE / sizeof(int16_t), 8, false);
  BucketizeTest<int16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 8, false);
  BucketizeTest<int16_t>((ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t), 8, false);
  BucketizeTest<int16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t), 8, false);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Int16_Right_Test) {
  BucketizeTest<int16_t>(ONE_BLK_SIZE / sizeof(int16_t), 8, true);
  BucketizeTest<int16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 8, true);
  BucketizeTest<int16_t>((ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t), 8, true);
  BucketizeTest<int16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t), 8, true);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Uint16_Left_Test) {
  BucketizeTest<uint16_t>(ONE_BLK_SIZE / sizeof(uint16_t), 10, false);
  BucketizeTest<uint16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint16_t), 10, false);
  BucketizeTest<uint16_t>((ONE_BLK_SIZE - sizeof(uint16_t)) / sizeof(uint16_t), 10, false);
  BucketizeTest<uint16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint16_t), 10, false);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Uint16_Right_Test) {
  BucketizeTest<uint16_t>(ONE_BLK_SIZE / sizeof(uint16_t), 10, true);
  BucketizeTest<uint16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint16_t), 10, true);
  BucketizeTest<uint16_t>((ONE_BLK_SIZE - sizeof(uint16_t)) / sizeof(uint16_t), 10, true);
  BucketizeTest<uint16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint16_t), 10, true);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Int32_Left_Test) {
  BucketizeTest<int32_t>(ONE_BLK_SIZE / sizeof(int32_t), 16, false);
  BucketizeTest<int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 16, false);
  BucketizeTest<int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), 16, false);
  BucketizeTest<int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), 16, false);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Int32_Right_Test) {
  BucketizeTest<int32_t>(ONE_BLK_SIZE / sizeof(int32_t), 16, true);
  BucketizeTest<int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 16, true);
  BucketizeTest<int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), 16, true);
  BucketizeTest<int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), 16, true);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Uint32_Left_Test) {
  BucketizeTest<uint32_t>(ONE_BLK_SIZE / sizeof(uint32_t), 14, false);
  BucketizeTest<uint32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint32_t), 14, false);
  BucketizeTest<uint32_t>((ONE_BLK_SIZE - sizeof(uint32_t)) / sizeof(uint32_t), 14, false);
  BucketizeTest<uint32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint32_t), 14, false);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Uint32_Right_Test) {
  BucketizeTest<uint32_t>(ONE_BLK_SIZE / sizeof(uint32_t), 14, true);
  BucketizeTest<uint32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint32_t), 14, true);
  BucketizeTest<uint32_t>((ONE_BLK_SIZE - sizeof(uint32_t)) / sizeof(uint32_t), 14, true);
  BucketizeTest<uint32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint32_t), 14, true);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Int64_Left_Test) {
  BucketizeTest<int64_t>(ONE_BLK_SIZE / sizeof(int64_t), 10, false);
  BucketizeTest<int64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 10, false);
  BucketizeTest<int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), 10, false);
  BucketizeTest<int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t), 10, false);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Int64_Right_Test) {
  BucketizeTest<int64_t>(ONE_BLK_SIZE / sizeof(int64_t), 10, true);
  BucketizeTest<int64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 10, true);
  BucketizeTest<int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), 10, true);
  BucketizeTest<int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t), 10, true);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Uint64_Left_Test) {
  BucketizeTest<uint64_t>(ONE_BLK_SIZE / sizeof(uint64_t), 8, false);
  BucketizeTest<uint64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint64_t), 8, false);
  BucketizeTest<uint64_t>((ONE_BLK_SIZE - sizeof(uint64_t)) / sizeof(uint64_t), 8, false);
  BucketizeTest<uint64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint64_t), 8, false);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Uint64_Right_Test) {
  BucketizeTest<uint64_t>(ONE_BLK_SIZE / sizeof(uint64_t), 8, true);
  BucketizeTest<uint64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint64_t), 8, true);
  BucketizeTest<uint64_t>((ONE_BLK_SIZE - sizeof(uint64_t)) / sizeof(uint64_t), 8, true);
  BucketizeTest<uint64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint64_t), 8, true);
}

// ==================== Floating-point type tests ====================

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Float_Left_Test) {
  BucketizeTest<float>(ONE_BLK_SIZE / sizeof(float), 16, false);
  BucketizeTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float), 16, false);
  BucketizeTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float), 16, false);
  BucketizeTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float), 16, false);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Float_Right_Test) {
  BucketizeTest<float>(ONE_BLK_SIZE / sizeof(float), 16, true);
  BucketizeTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float), 16, true);
  BucketizeTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float), 16, true);
  BucketizeTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float), 16, true);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Half_Left_Test) {
  BucketizeTest<half>(ONE_BLK_SIZE / sizeof(half), 10, false);
  BucketizeTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half), 10, false);
  BucketizeTest<half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half), 10, false);
  BucketizeTest<half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half), 10, false);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Half_Right_Test) {
  BucketizeTest<half>(ONE_BLK_SIZE / sizeof(half), 10, true);
  BucketizeTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half), 10, true);
  BucketizeTest<half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half), 10, true);
  BucketizeTest<half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half), 10, true);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_BFloat16_Left_Test) {
  BucketizeTest<bfloat16_t>(ONE_BLK_SIZE / sizeof(bfloat16_t), 10, false);
  BucketizeTest<bfloat16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t), 10, false);
  BucketizeTest<bfloat16_t>((ONE_BLK_SIZE - sizeof(bfloat16_t)) / sizeof(bfloat16_t), 10, false);
  BucketizeTest<bfloat16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(bfloat16_t), 10, false);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_BFloat16_Right_Test) {
  BucketizeTest<bfloat16_t>(ONE_BLK_SIZE / sizeof(bfloat16_t), 10, true);
  BucketizeTest<bfloat16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t), 10, true);
  BucketizeTest<bfloat16_t>((ONE_BLK_SIZE - sizeof(bfloat16_t)) / sizeof(bfloat16_t), 10, true);
  BucketizeTest<bfloat16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(bfloat16_t), 10, true);
}

// ==================== Large-size tests ====================

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_LargeSize_Test) {
  BucketizeTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float), 20, false);
  BucketizeTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float), 20, true);
}

// ==================== Edge-case tests ====================

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Edge_SingleBoundary_Left) {
  // boundary = {0}; src values below, at, and above the boundary.
  // right=false (lower_bound): boundaries[i] >= value
  // value=-1: boundaries[0]=0 >= -1 → hit → high=0 → result=0 ✓
  // value=0:  boundaries[0]=0 >= 0  → hit → high=0 → result=0
  // value=1:  boundaries[0]=0 < 1   → miss → low=1 → result=1 (boundCount)
  BucketizeEdgeTest<float>({0.0f}, {-1.0f, 0.0f, 1.0f}, {0, 0, 1}, false);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Edge_SingleBoundary_Right) {
  // right=true (upper_bound): boundaries[i] > value
  // value=-1: boundaries[0]=0 > -1 → hit → high=0 → result=0
  // value=0:  boundaries[0]=0 NOT > 0 → miss → low=1 → result=1 (boundCount)
  // value=1:  boundaries[0]=0 NOT > 1 → miss → low=1 → result=1
  BucketizeEdgeTest<float>({0.0f}, {-1.0f, 0.0f, 1.0f}, {0, 1, 1}, true);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Edge_EmptyBoundaries) {
  // No boundaries: all values map to index 0 (boundCount = 0)
  BucketizeEdgeTest<int32_t>({}, {5, -3, 100, 0}, {0, 0, 0, 0}, false);
  BucketizeEdgeTest<int32_t>({}, {5, -3, 100, 0}, {0, 0, 0, 0}, true);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Edge_MultiBoundary_Left) {
  // boundaries = {10, 20, 30}; right=false (lower_bound)
  // value=5:  boundaries[1]=10 >= 5 → result=0
  // value=10: boundaries[1]=10 >= 10 → result=0 (10 belongs to bucket 0)
  // value=15: boundaries[1]=10 < 15, boundaries[2]=20 >= 15 → result=1
  // value=20: boundaries[2]=20 >= 20 → result=1
  // value=25: boundaries[3]=30 >= 25 → result=2
  // value=30: boundaries[3]=30 >= 30 → result=2
  // value=35: all < 35 → result=3 (boundCount)
  BucketizeEdgeTest<int32_t>({10, 20, 30}, {5, 10, 15, 20, 25, 30, 35}, {0, 0, 1, 1, 2, 2, 3}, false);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Edge_MultiBoundary_Right) {
  // boundaries = {10, 20, 30}; right=true (upper_bound)
  // value=5:  boundaries[1]=10 > 5 → result=0
  // value=10: boundaries[1]=10 NOT > 10, boundaries[2]=20 > 10 → result=1
  // value=15: boundaries[2]=20 > 15 → result=1
  // value=20: boundaries[3]=30 > 20 → result=2
  // value=25: boundaries[3]=30 > 25 → result=2
  // value=30: boundaries[3]=30 NOT > 30 → result=3 (boundCount)
  // value=35: all ≤ 35 → result=3 (boundCount)
  BucketizeEdgeTest<int32_t>({10, 20, 30}, {5, 10, 15, 20, 25, 30, 35}, {0, 1, 1, 2, 2, 3, 3}, true);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Edge_AllBelowFirstBoundary) {
  // All values below the first boundary → all result = 0
  BucketizeEdgeTest<float>({100.0f, 200.0f, 300.0f}, {-50.0f, 0.0f, 50.0f, 99.0f}, {0, 0, 0, 0}, false);
  BucketizeEdgeTest<float>({100.0f, 200.0f, 300.0f}, {-50.0f, 0.0f, 50.0f, 99.0f}, {0, 0, 0, 0}, true);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Edge_AllAboveLastBoundary) {
  // All values above the last boundary → all result = boundCount
  BucketizeEdgeTest<float>({100.0f, 200.0f, 300.0f}, {301.0f, 400.0f, 500.0f}, {3, 3, 3}, false);
  BucketizeEdgeTest<float>({100.0f, 200.0f, 300.0f}, {301.0f, 400.0f, 500.0f}, {3, 3, 3}, true);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Edge_NaN_Float_Left) {
  // NaN values → boundCount
  float nan = std::nanf("");
  BucketizeEdgeTest<float>({0.0f, 10.0f, 20.0f}, {5.0f, nan, 15.0f},
                           // 5.0: 10 >= 5 -> result 1; nan -> boundCount=3; 15.0: 20 >= 15 -> result 2
                           {1, 3, 2}, false);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Edge_NaN_Float_Right) {
  float nan = std::nanf("");
  BucketizeEdgeTest<float>({0.0f, 10.0f, 20.0f}, {5.0f, nan, 15.0f},
                           // 5.0: 0 > 5 -> false, 10 > 5 -> result 1; nan -> boundCount=3; 15.0: 20 > 15 -> result 2
                           {1, 3, 2}, true);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Edge_Int8_Boundary) {
  BucketizeEdgeTest<int8_t>({static_cast<int8_t>(-50), static_cast<int8_t>(0), static_cast<int8_t>(50)},
                            {static_cast<int8_t>(-100), static_cast<int8_t>(-50), static_cast<int8_t>(0),
                             static_cast<int8_t>(25), static_cast<int8_t>(50), static_cast<int8_t>(100)},
                            {0, 0, 1, 2, 2, 3}, false);
}

TEST_F(TestRegbaseApiBucketizeUT, Bucketize_Edge_Uint8_Boundary) {
  BucketizeEdgeTest<uint8_t>(
      {static_cast<uint8_t>(50), static_cast<uint8_t>(100), static_cast<uint8_t>(200)},
      {static_cast<uint8_t>(0), static_cast<uint8_t>(50), static_cast<uint8_t>(75), static_cast<uint8_t>(100),
       static_cast<uint8_t>(150), static_cast<uint8_t>(200), static_cast<uint8_t>(255)},
      {0, 0, 1, 1, 2, 2, 3}, false);
}
}  // namespace af
