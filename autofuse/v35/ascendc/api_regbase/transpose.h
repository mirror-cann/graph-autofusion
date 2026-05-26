/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_API_REGBASE_TRANSPOSE_H
#define ASCENDC_API_REGBASE_TRANSPOSE_H

using namespace AscendC;

template <typename T>
using RangeType = std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>;

template <typename T>
using IdxType = std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>;

template <typename T>
inline __simd_vf__ void GenOneInnerDimTransposeIndex(__ubuf__ T *dst_idx, const T src_stride0, RangeType<T> count) {
  uint16_t vl_size = static_cast<uint16_t>(GetVecLen() / sizeof(T));
  uint32_t cal_cnt = static_cast<uint32_t>(count);
  uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(cal_cnt, vl_size));
  MicroAPI::RegTensor<T> idx_reg, dst_reg;
  // idx_reg init: 0~(VL-1)
  MicroAPI::Arange(idx_reg, 0);
  MicroAPI::MaskReg mask;
  for (uint16_t i = 0U; i < repeat_time; i++) {
    mask = MicroAPI::UpdateMask<T>(cal_cnt);
    MicroAPI::Muls(dst_reg, idx_reg, src_stride0, mask);
    MicroAPI::DataCopy(dst_idx + i * vl_size, dst_reg, mask);
    MicroAPI::Adds(idx_reg, idx_reg, vl_size, mask);
  }
}

template <typename T>
inline __simd_vf__ void GenTwoInnerDimTransposeIndex(__ubuf__ T *dst_idx, const T dst_dim1, const T src_stride0,
                                                const T src_stride1, RangeType<T> count) {
  uint16_t vl_size = static_cast<uint16_t>(GetVecLen() / sizeof(T));
  T last_dim_inc = static_cast<T>(vl_size % dst_dim1);
  T last_2nd_dim_inc = static_cast<T>(vl_size / dst_dim1);

  uint16_t repeat_time = 0;
  uint32_t left_size = 0;
  if (count > vl_size) {
    left_size = count - vl_size;
    repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(left_size, vl_size));
  }
  MicroAPI::RegTensor<T> idx_reg, dim0_reg, dim1_reg, tmp_reg, dst_reg, zero_reg, one_reg, cmp_reg;
  MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
  // vec_a: VL % a
  MicroAPI::Arange(idx_reg, 0);
  MicroAPI::Duplicate(dim1_reg, dst_dim1);
  MicroAPI::Div(tmp_reg, idx_reg, dim1_reg, mask);
  MicroAPI::Copy(dim0_reg, tmp_reg); // vec_b: VL / a
  MicroAPI::Mul(tmp_reg, tmp_reg, dim1_reg, mask);
  MicroAPI::Sub(dim1_reg, idx_reg, tmp_reg, mask);
  // index: vec_a * a_in_offset + vec_b * b_in_offset
  MicroAPI::Muls(tmp_reg, dim1_reg, src_stride1, mask);
  MicroAPI::Muls(dst_reg, dim0_reg, src_stride0, mask);
  MicroAPI::Add(dst_reg, dst_reg, tmp_reg, mask);
  MicroAPI::DataCopy(dst_idx, dst_reg, mask);

  MicroAPI::MaskReg left_mask, sel_mask;
  MicroAPI::Duplicate(zero_reg, 0);
  MicroAPI::Duplicate(one_reg, 1);
  for (uint16_t i = 0U; i < repeat_time; i++) {
    left_mask = MicroAPI::UpdateMask<T>(left_size);
    /*   vec_a += a_inc
     *   cmp_a = vec_a >= a
     *   vec_a = vec_a - cmp_a * a
     */
    MicroAPI::Adds(dim1_reg, dim1_reg, last_dim_inc, left_mask);
    MicroAPI::CompareScalar<T, CMPMODE::GE>(sel_mask, dim1_reg, dst_dim1, left_mask);
    MicroAPI::Select(cmp_reg, one_reg, zero_reg, sel_mask);
    MicroAPI::Muls(tmp_reg, cmp_reg, dst_dim1, left_mask);
    MicroAPI::Sub(dim1_reg, dim1_reg, tmp_reg, left_mask);
    // vec_b += (b_inc + cmp_a)
    MicroAPI::Adds(dim0_reg, dim0_reg, last_2nd_dim_inc, left_mask);
    MicroAPI::Add(dim0_reg, dim0_reg, cmp_reg, left_mask);
    // index: vec_a * a_in_offset + vec_b * b_in_offset
    MicroAPI::Muls(tmp_reg, dim1_reg, src_stride1, left_mask);
    MicroAPI::Muls(dst_reg, dim0_reg, src_stride0, left_mask);
    MicroAPI::Add(dst_reg, dst_reg, tmp_reg, left_mask);
    MicroAPI::DataCopy(dst_idx + (i + 1) * vl_size, dst_reg, left_mask);
  }
}

template <typename T>
inline __simd_vf__ void GenThreeDimTransposeIndex(__ubuf__ T *dst_idx, const T dst_dim1, const T dst_dim2,
                                                  const T src_stride0, const T src_stride1, const T src_stride2,
                                                  RangeType<T> count) {
  uint16_t vl_size = static_cast<uint16_t>(GetVecLen() / sizeof(T));
  T last_dim_inc = static_cast<T>(vl_size % dst_dim2);
  T last_2nd_dim_inc = static_cast<T>(vl_size / dst_dim2 % dst_dim1);
  T last_3rd_dim_inc = static_cast<T>(vl_size / (dst_dim2 * dst_dim1));

  uint16_t repeat_time = 0;
  uint32_t left_size = 0;
  if (count > vl_size) {
    left_size = count - vl_size;
    repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(left_size, vl_size));
  }
  MicroAPI::RegTensor<T> idx_reg, dim0_reg, dim1_reg, dim2_reg, tmp_reg, tmp1_reg, dst_reg, zero_reg, one_reg, cmpReg;
  MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
  // vec_a: VL % a
  MicroAPI::Arange(idx_reg, 0);
  MicroAPI::Duplicate(dim2_reg, dst_dim2);
  MicroAPI::Copy(dim0_reg, dim2_reg); // backup a
  MicroAPI::Div(tmp_reg, idx_reg, dim2_reg, mask);
  MicroAPI::Copy(dim1_reg, tmp_reg); // backup VL / a
  MicroAPI::Mul(tmp_reg, tmp_reg, dim2_reg, mask);
  MicroAPI::Sub(dim2_reg, idx_reg, tmp_reg, mask);
  // vec_b: VL / a % b
  MicroAPI::Duplicate(tmp1_reg, dst_dim1);
  MicroAPI::Mul(dim0_reg, dim0_reg, tmp1_reg, mask); // backup b
  MicroAPI::Div(tmp_reg, dim1_reg, tmp1_reg, mask);
  MicroAPI::Mul(tmp_reg, tmp_reg, tmp1_reg, mask);
  MicroAPI::Sub(dim1_reg, dim1_reg, tmp_reg, mask);
  // vec_c: VL / (a * b)
  MicroAPI::Div(dim0_reg, idx_reg, dim0_reg, mask);
  // index: vec_a * a_in_offset + vec_b * b_in_offset + vec_c * c_in_offset
  MicroAPI::Muls(tmp_reg, dim2_reg, src_stride2, mask);
  MicroAPI::Muls(tmp1_reg, dim1_reg, src_stride1, mask);
  MicroAPI::Muls(dst_reg, dim0_reg, src_stride0, mask);
  MicroAPI::Add(dst_reg, dst_reg, tmp_reg, mask);
  MicroAPI::Add(dst_reg, dst_reg, tmp1_reg, mask);
  MicroAPI::DataCopy(dst_idx, dst_reg, mask);

  MicroAPI::MaskReg left_mask, sel_mask;
  MicroAPI::Duplicate(zero_reg, 0);
  MicroAPI::Duplicate(one_reg, 1);
  for (uint16_t i = 0U; i < repeat_time; i++) {
    left_mask = MicroAPI::UpdateMask<T>(left_size);
    /*   vec_a += a_inc
     *   cmp_a = vec_a >= a
     *   vec_a = vec_a - cmp_a * a
     */
    MicroAPI::Adds(dim2_reg, dim2_reg, last_dim_inc, left_mask);
    MicroAPI::CompareScalar<T, CMPMODE::GE>(sel_mask, dim2_reg, dst_dim2, left_mask);
    MicroAPI::Select(cmpReg, one_reg, zero_reg, sel_mask);
    MicroAPI::Muls(tmp_reg, cmpReg, dst_dim2, left_mask);
    MicroAPI::Sub(dim2_reg, dim2_reg, tmp_reg, left_mask);
    /*   vec_b += (b_inc + cmp_a)
     *   cmp_b = vec_b >= b
     *   vec_b = vec_b - cmp_b * b
     */
    MicroAPI::Adds(cmpReg, cmpReg, last_2nd_dim_inc, left_mask);
    MicroAPI::Add(dim1_reg, dim1_reg, cmpReg, left_mask);
    MicroAPI::CompareScalar<T, CMPMODE::GE>(sel_mask, dim1_reg, dst_dim1, left_mask);
    MicroAPI::Select(cmpReg, one_reg, zero_reg, sel_mask);
    MicroAPI::Muls(tmp_reg, cmpReg, dst_dim1, left_mask);
    MicroAPI::Sub(dim1_reg, dim1_reg, tmp_reg, left_mask);
    // vec_c += (c_inc + cmp_b)
    MicroAPI::Adds(dim0_reg, dim0_reg, last_3rd_dim_inc, left_mask);
    MicroAPI::Add(dim0_reg, dim0_reg, cmpReg, left_mask);
    // index: vec_a * a_in_offset + vec_b * b_in_offset + vec_c * c_in_offset
    MicroAPI::Muls(tmp_reg, dim2_reg, src_stride2, left_mask);
    MicroAPI::Muls(tmp1_reg, dim1_reg, src_stride1, left_mask);
    MicroAPI::Muls(dst_reg, dim0_reg, src_stride0, left_mask);
    MicroAPI::Add(dst_reg, dst_reg, tmp_reg, left_mask);
    MicroAPI::Add(dst_reg, dst_reg, tmp1_reg, left_mask);
    MicroAPI::DataCopy(dst_idx + (i + 1) * vl_size, dst_reg, left_mask);
  }
}

template <typename T, uint8_t dim>
__aicore__ inline void GenTransposeIndex(__ubuf__ T* dst_idx, const T (&dst_dims)[dim], const T (&src_strides)[dim],
                                         RangeType<T> cal_cnt) {
  if constexpr (dim == 1) {
    GenOneInnerDimTransposeIndex(dst_idx, src_strides[0], cal_cnt);
  } else if constexpr (dim == 2) {
    GenTwoInnerDimTransposeIndex(dst_idx, dst_dims[1], src_strides[0], src_strides[1], cal_cnt);
  }
}

template <typename T>
inline __simd_vf__ void TransposeExtendImpl(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ RangeType<T>* index,
                                            RangeType<T> count) {
  uint16_t vl_size = static_cast<uint16_t>(GetVecLen() / sizeof(T));
  uint32_t cal_cnt = static_cast<uint32_t>(count);
  uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(cal_cnt, vl_size));

  MicroAPI::RegTensor<RangeType<T>> idx_reg;
  MicroAPI::RegTensor<T> dst_reg;
  MicroAPI::MaskReg mask;
  for (uint16_t i = 0U; i < repeat_time; i++) {
    mask = MicroAPI::UpdateMask<T>(cal_cnt);
    MicroAPI::LoadAlign(idx_reg, index + i * vl_size);
    MicroAPI::Gather(dst_reg, src, (MicroAPI::RegTensor<IdxType<T>>&)idx_reg, mask);
    MicroAPI::StoreAlign(dst + i * vl_size, dst_reg, mask);
  }
}

template <typename T>
inline __simd_vf__ void Transpose2Dim1InnerDimExtendImpl(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ RangeType<T>* index,
                                                         RangeType<T> dst_dim0, RangeType<T> src_stride0,
                                                         RangeType<T> dst_stride0, RangeType<T> count) {
  uint16_t vl_size = static_cast<uint16_t>(GetVecLen() / sizeof(T));
  uint32_t cal_cnt = static_cast<uint32_t>(count);
  uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(cal_cnt, vl_size));

  MicroAPI::RegTensor<RangeType<T>> idx_reg;
  MicroAPI::RegTensor<T> dst_reg;
  MicroAPI::MaskReg mask;
  for (uint16_t j = 0U; j < repeat_time; j++) {
    mask = MicroAPI::UpdateMask<T>(cal_cnt);
    for (uint16_t i = 0U; i < dst_dim0; i++) {
      MicroAPI::LoadAlign(idx_reg, index + j * vl_size);
      MicroAPI::Gather(dst_reg, src + i * src_stride0, (MicroAPI::RegTensor<IdxType<T>>&)idx_reg, mask);
      MicroAPI::StoreAlign(dst + i * dst_stride0 + j * vl_size, dst_reg, mask);
    }
  }
}

template <typename T>
inline __simd_vf__ void TransposeOneOuterDimExtendImpl(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ RangeType<T>* index,
                                                       RangeType<T> dst_dim0, RangeType<T> src_stride0,
                                                       RangeType<T> dst_stride0, RangeType<T> count) {
  uint16_t vl_size = static_cast<uint16_t>(GetVecLen() / sizeof(T));
  uint32_t cal_cnt = static_cast<uint32_t>(count);
  uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(cal_cnt, vl_size));

  MicroAPI::RegTensor<RangeType<T>> idx_reg;
  MicroAPI::RegTensor<T> dst_reg;
  MicroAPI::MaskReg mask;
  for (uint16_t i = 0U; i < repeat_time; i++) {
    mask = MicroAPI::UpdateMask<T>(cal_cnt);
    for (uint16_t j = 0U; j < dst_dim0; j++) {
      MicroAPI::LoadAlign(idx_reg, index + i * vl_size);
      MicroAPI::Gather(dst_reg, src + j * src_stride0, (MicroAPI::RegTensor<IdxType<T>>&)idx_reg, mask);
      MicroAPI::StoreAlign(dst + j * dst_stride0 + i * vl_size, dst_reg, mask);
    }
  }
}

template <typename T>
inline __simd_vf__ void TransposeTwoOuterDimExtendImpl(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ RangeType<T>* index,
                                                       RangeType<T> dst_dim0, RangeType<T> src_stride0,
                                                       RangeType<T> dst_stride0, RangeType<T> dst_dim1,
                                                       RangeType<T> src_stride1, RangeType<T> dst_stride1,
                                                       RangeType<T> count) {
  uint16_t vl_size = static_cast<uint16_t>(GetVecLen() / sizeof(T));
  uint32_t cal_cnt = static_cast<uint32_t>(count);
  uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(cal_cnt, vl_size));

  MicroAPI::RegTensor<RangeType<T>> idx_reg;
  MicroAPI::RegTensor<T> dst_reg;
  MicroAPI::MaskReg mask;
  for (uint16_t i = 0U; i < repeat_time; i++) {
    mask = MicroAPI::UpdateMask<T>(cal_cnt);
    for (uint16_t j = 0U; j < dst_dim0; j++) {
      for (uint16_t k = 0U; k < dst_dim1; k++) {
        MicroAPI::LoadAlign(idx_reg, index + i * vl_size);
        MicroAPI::Gather(dst_reg, src + j * src_stride0 + k * src_stride1, (MicroAPI::RegTensor<IdxType<T>>&)idx_reg,
                         mask);
        MicroAPI::StoreAlign(dst + i * vl_size + j * dst_stride0 + k * dst_stride1, dst_reg, mask);
      }
    }
  }
}

template <typename T>
inline __simd_vf__ void TransposeThreeOuterDimExtendImpl(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ RangeType<T>* index,
                                                         RangeType<T> dst_dim0, RangeType<T> src_stride0,
                                                         RangeType<T> dst_stride0, RangeType<T> dst_dim1,
                                                         RangeType<T> src_stride1, RangeType<T> dst_stride1,
                                                         RangeType<T> dst_dim2, RangeType<T> src_stride2,
                                                         RangeType<T> dst_stride2, RangeType<T> count) {
  uint32_t cal_cnt = static_cast<uint32_t>(count);
  uint16_t vl_size = static_cast<uint16_t>(GetVecLen() / sizeof(T));
  uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(cal_cnt, vl_size));

  MicroAPI::RegTensor<RangeType<T>> idx_reg;
  MicroAPI::RegTensor<T> dst_reg;
  MicroAPI::MaskReg mask;
  for (uint16_t i = 0U; i < repeat_time; i++) {
    mask = MicroAPI::UpdateMask<T>(cal_cnt);
    for (uint16_t j = 0U; j < dst_dim0; j++) {
      for (uint16_t k = 0U; k < dst_dim1; k++) {
        for (uint16_t m = 0U; m < dst_dim2; m++) {
          MicroAPI::LoadAlign(idx_reg, index + i * vl_size);
          MicroAPI::Gather(dst_reg, src + j * src_stride0 + k * src_stride1 + m * src_stride2,
                          (MicroAPI::RegTensor<IdxType<T>>&)idx_reg, mask);
          MicroAPI::StoreAlign(dst + i * vl_size + j * dst_stride0 + k * dst_stride1 + m * dst_stride2, dst_reg, mask);
        }
      }
    }
  }
}

template <uint8_t inner_dim, uint8_t dim, typename T>
__aicore__ inline void TransposeExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                       const LocalTensor<uint8_t> &tmp_buf, const RangeType<T> (&dst_dims)[dim],
                                       const RangeType<T> (&src_strides)[dim], const RangeType<T> (&dst_strides)[dim]) {
  __ubuf__ T* src_buf = (__ubuf__ T*)src.GetPhyAddr();
  __ubuf__ T* dst_buf = (__ubuf__ T*)dst.GetPhyAddr();
  __ubuf__ RangeType<T>* index_buf = (__ubuf__ RangeType<T>*)tmp_buf.GetPhyAddr();
  RangeType<T> cal_cnt = 1;
  for (uint8_t i = dim - inner_dim; i < dim; i++) {
    cal_cnt *= dst_dims[i];
  }
  if constexpr (dim == 2U && inner_dim == 1U) {
    GenOneInnerDimTransposeIndex(index_buf, src_strides[1], cal_cnt);
    TransposeOneOuterDimExtendImpl(dst_buf, src_buf, index_buf, dst_dims[0], src_strides[0], dst_strides[0], cal_cnt);
  } else if constexpr (dim == 3U && inner_dim == 1U) {
    GenOneInnerDimTransposeIndex(index_buf, src_strides[2], cal_cnt);
    TransposeTwoOuterDimExtendImpl(dst_buf, src_buf, index_buf, dst_dims[0], src_strides[0], dst_strides[0],
                                   dst_dims[1], src_strides[1], dst_strides[1], cal_cnt);
  } else if constexpr (dim == 3U && inner_dim == 2U) {
    GenTwoInnerDimTransposeIndex(index_buf, dst_dims[2], src_strides[1], src_strides[2], cal_cnt);
    TransposeOneOuterDimExtendImpl(dst_buf, src_buf, index_buf, dst_dims[0], src_strides[0], dst_strides[0], cal_cnt);
  } else if constexpr (dim == 4U && inner_dim == 1U) {
    GenOneInnerDimTransposeIndex(index_buf, src_strides[3], cal_cnt);
    TransposeThreeOuterDimExtendImpl(dst_buf, src_buf, index_buf, dst_dims[0], src_strides[0], dst_strides[0],
                                     dst_dims[1], src_strides[1], dst_strides[1], dst_dims[2], src_strides[2],
                                     dst_strides[2], cal_cnt);
  } else if constexpr (dim == 4U && inner_dim == 2U) {
    GenTwoInnerDimTransposeIndex(index_buf, dst_dims[3], src_strides[2], src_strides[3], cal_cnt);
    TransposeTwoOuterDimExtendImpl(dst_buf, src_buf, index_buf, dst_dims[0], src_strides[0], dst_strides[0],
                                   dst_dims[1], src_strides[1], dst_strides[1], cal_cnt);
  } else {
    GenTransposeIndex<RangeType<T>, dim>(index_buf, dst_dims, src_strides, cal_cnt);
    TransposeExtendImpl(dst_buf, src_buf, index_buf, cal_cnt);
  }
}
#endif
