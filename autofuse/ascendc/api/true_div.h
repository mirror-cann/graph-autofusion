
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_TRUE_DIV_H__
#define __ASCENDC_API_TRUE_DIV_H__

template <typename T, typename U>
inline __aicore__ void TrueDivExtend(const AscendC::LocalTensor<U> &dst, const AscendC::LocalTensor<T> &src1,
                                     const AscendC::LocalTensor<T> &src2, const uint32_t size,
                                     AscendC::LocalTensor<uint8_t> &tmp_buf) {
  static_assert(std::is_same<T, int32_t>::value || std::is_same<T, float>::value || std::is_same<T, half>::value,
              "Unsupported data type for TrueDivExtend");
  if constexpr (AscendC::IsSameType<T, int32_t>::value) {
    constexpr uint32_t cast_dst_align = 32U / sizeof(U);
    uint32_t cast_buf_size = (size + cast_dst_align - 1) / cast_dst_align * cast_dst_align;

    AscendC::LocalTensor<U> cast_src1 = tmp_buf.template ReinterpretCast<U>();
    uint32_t offset = cast_buf_size * sizeof(U);
    AscendC::LocalTensor<U> cast_src2 = tmp_buf[offset].template ReinterpretCast<U>();

    AscendC::Cast(cast_src1, src1, AscendC::RoundMode::CAST_RINT, size);
    AscendC::Cast(cast_src2, src2, AscendC::RoundMode::CAST_RINT, size);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div(dst, cast_src1, cast_src2, size);
  } else {
    AscendC::Div(dst, src1, src2, size);
  }
}

template <typename T, typename U, bool IS_SCALAR_LATTER = true>
inline __aicore__ void TrueDivExtends(const LocalTensor<U> &dst, const LocalTensor<T> &src, const T constant_x,
                            const uint32_t size, LocalTensor<uint8_t> &tmp_buf) {
  static_assert(std::is_same<T, int32_t>::value || std::is_same<T, float>::value || std::is_same<T, half>::value,
              "Unsupported data type for TrueDivExtend");
  if constexpr (AscendC::IsSameType<T, int32_t>::value) {
    constexpr uint32_t cast_dst_align = 32U / sizeof(U);
    uint32_t cast_buf_size = (size + cast_dst_align - 1) / cast_dst_align * cast_dst_align;
    AscendC::LocalTensor<U> cast_src = tmp_buf.template ReinterpretCast<U>();
    uint32_t offset = cast_buf_size * sizeof(U);
    AscendC::LocalTensor<uint8_t> left_tmp_buf = tmp_buf[offset].template ReinterpretCast<uint8_t>();

    AscendC::Cast(cast_src, src, AscendC::RoundMode::CAST_RINT, size);
    AscendC::PipeBarrier<PIPE_V>();
    Divs<U, IS_SCALAR_LATTER>(dst, cast_src, static_cast<U>(constant_x), size, left_tmp_buf);
  } else {
    Divs<T, IS_SCALAR_LATTER>(dst, src, constant_x, size, tmp_buf);
  }
}

template <typename T, typename U>
inline __aicore__ void TrueDivExtends(const LocalTensor<U>& dst, const T x, const T y) {
  U res = static_cast<U>(x) / y;
  AscendC::Duplicate(dst, res, dst.GetSize());
}

#endif  // __ASCENDC_API_TRUE_DIV_H__
