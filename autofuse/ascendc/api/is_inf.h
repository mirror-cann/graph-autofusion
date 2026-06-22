/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_IS_INF_H__
#define __ASCENDC_API_IS_INF_H__

constexpr int16_t IS_INF_HALF_NEG_INF_NUM = -0x7c00;
constexpr int32_t IS_INF_FLOAT_NEG_INF_NUM = -0x7f800000;
constexpr int16_t IS_INF_HALF_SIGN_MASK = 0x7fff;
constexpr int32_t IS_INF_FLOAT_SIGN_MASK = 0x7fffffff;

template <typename T>
inline __aicore__ constexpr T GetIsInfNegInfNum() {
  if constexpr (AscendC::IsSameType<T, int16_t>::value) {
    return IS_INF_HALF_NEG_INF_NUM;
  } else {
    return IS_INF_FLOAT_NEG_INF_NUM;
  }
}

template <typename T>
struct GetIsInfCalcType {
  using Type = T;
};

template <>
struct GetIsInfCalcType<half> {
  using Type = int16_t;
};

template <>
struct GetIsInfCalcType<float> {
  using Type = int32_t;
};

template <typename T1, typename T2>
inline __aicore__ constexpr T2 GetIsInfSignMask() {
  if constexpr (AscendC::IsSameType<T1, half>::value) {
    return IS_INF_HALF_SIGN_MASK;
  } else {
    return IS_INF_FLOAT_SIGN_MASK;
  }
}

template <typename T1, typename T2>
inline __aicore__ void InitIsInfLocalTensor(LocalTensor<uint8_t> &tmp_buf, LocalTensor<T2> &sign_mask,
                                             LocalTensor<T2> &calc_buf) {
  uint32_t offset = 0;

  sign_mask = tmp_buf.template ReinterpretCast<T2>();
  sign_mask.SetSize(ONE_BLK_SIZE / sizeof(T2));
  offset += ONE_BLK_SIZE;
  Duplicate(sign_mask, GetIsInfSignMask<T1, T2>(), ONE_BLK_SIZE / sizeof(T2));

  calc_buf = tmp_buf[offset].template ReinterpretCast<T2>();
  calc_buf.SetSize((tmp_buf.GetSize() - offset) / sizeof(T2));
}

template <typename T>
inline __aicore__ void DoIsInf(const AscendC::LocalTensor<uint8_t> &dst, const AscendC::LocalTensor<T> &calc_buf,
                                const int calc_size, const int calCount) {
  // IsInf detection: diff == 0 means infinity
  // After And + Adds: NaN > 0, Inf == 0, finite < 0
  // We need to detect diff == 0
  Abs(calc_buf, calc_buf, calCount);
  Mins(calc_buf, calc_buf, static_cast<T>(1), calCount);
  Adds(calc_buf, calc_buf, static_cast<T>(-1), calCount);
  Abs(calc_buf, calc_buf, calCount);
  Mins(calc_buf, calc_buf, static_cast<T>(1), calCount);
  LocalTensor<half> half_tmp = calc_buf.template ReinterpretCast<half>();
  if constexpr (AscendC::IsSameType<T, int32_t>::value) {
    AscendC::SetDeqScale(half(1.0));
  }
  AscendC::PipeBarrier<PIPE_V>();
  Cast(half_tmp, calc_buf, RoundMode::CAST_NONE, calCount);
  AscendC::PipeBarrier<PIPE_V>();
  Cast(dst[calc_size], half_tmp, RoundMode::CAST_NONE, calCount);
}

template <typename T>
inline __aicore__ void IsInfExtend(const AscendC::LocalTensor<uint8_t> &dst, const AscendC::LocalTensor<T> &src,
                                    LocalTensor<uint8_t> &tmp_buf, const uint32_t size) {
  // Init local tensor from tmp_buf
  using T2 = typename GetIsInfCalcType<T>::Type;
  LocalTensor<T2> sign_mask;
  LocalTensor<T2> calc_buf;
  InitIsInfLocalTensor<T, T2>(tmp_buf, sign_mask, calc_buf);
  LocalTensor<int16_t> sign_mask_int16 = sign_mask.template ReinterpretCast<int16_t>();
  LocalTensor<int16_t> calc_buf_int16 = calc_buf.template ReinterpretCast<int16_t>();
  LocalTensor<int16_t> src_int16 = src.template ReinterpretCast<int16_t>();
  constexpr uint32_t RATIO = sizeof(T) / sizeof(int16_t);
  constexpr uint32_t one_repeat_calc_size = ONE_REPEAT_BYTE_SIZE / sizeof(T);

  // Prepare binary repeat params for and operation
  BinaryRepeatParams and_repeat(1, 1, 0, 8, 8, 0);
  int calc_size = 0;

  // Calc when size > max_repeat
  int max_repeat = calc_buf.GetSize() / one_repeat_calc_size;
  max_repeat = max_repeat > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : max_repeat;
  max_repeat = max_repeat == 0 ? 1 : max_repeat;
  const int max_repeat_calc_size = max_repeat * one_repeat_calc_size;
  for (; calc_size + max_repeat_calc_size < size; calc_size += max_repeat_calc_size) {
    And(calc_buf_int16, src_int16[calc_size * RATIO], sign_mask_int16, ONE_REPEAT_BYTE_SIZE / sizeof(int16_t),
        max_repeat, and_repeat);
    Adds(calc_buf, calc_buf, GetIsInfNegInfNum<T>(), max_repeat_calc_size);
    DoIsInf(dst, calc_buf, calc_size, max_repeat_calc_size);
  }

  // Calc max_repeat > size > one_repeat
  if (calc_size + one_repeat_calc_size <= size) {
    int repeat = (size - calc_size) / one_repeat_calc_size;
    And(calc_buf_int16, src_int16[calc_size * RATIO], sign_mask_int16, ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), repeat,
        and_repeat);
    Adds(calc_buf, calc_buf, GetIsInfNegInfNum<T>(), repeat * one_repeat_calc_size);
    DoIsInf(dst, calc_buf, calc_size, repeat * one_repeat_calc_size);
    calc_size += repeat * one_repeat_calc_size;
  }

  // Calc when one_repeat > size
  if (calc_size < size) {
    And(calc_buf_int16, src_int16[calc_size * RATIO], sign_mask_int16, (size - calc_size) * RATIO, 1, and_repeat);
    Adds(calc_buf, calc_buf, GetIsInfNegInfNum<T>(), size - calc_size);
    DoIsInf(dst, calc_buf, calc_size, size - calc_size);
  }
}

#endif  // __ASCENDC_API_IS_INF_H__
