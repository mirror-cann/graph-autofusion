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

template <typename T, uint8_t dim>
__aicore__ inline void GenOneDimTransposeIndex(__ubuf__ T *dst_idx, const T (&dst_dims)[dim],
                                               const T (&src_strides)[dim], const RangeType<T>& count) {
  uint16_t vl_size = static_cast<uint16_t>(GetVecLen() / sizeof(T));
  uint32_t cal_cnt = static_cast<uint32_t>(count);
  uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(cal_cnt, vl_size));
  MicroAPI::RegTensor<T> idx_reg, dst_reg;
  // idx_reg init: 0~(VL-1)
  MicroAPI::Arange(idx_reg, 0);
  MicroAPI::MaskReg mask;
  for (uint16_t i = 0U; i < repeat_time; i++) {
    mask = MicroAPI::UpdateMask<T>(cal_cnt);
    MicroAPI::Muls(dst_reg, idx_reg, src_strides[0], mask);
    MicroAPI::DataCopy(dst_idx + i * vl_size, dst_reg, mask);
    MicroAPI::Adds(idx_reg, idx_reg, vl_size, mask);
  }
}

template <typename T, uint8_t dim>
__aicore__ inline void GenTwoDimTransposeIndex(__ubuf__ T *dst_idx, const T (&dst_dims)[dim],
                                               const T (&src_strides)[dim], const RangeType<T>& count) {
  uint16_t vl_size = static_cast<uint16_t>(GetVecLen() / sizeof(T));
  uint32_t cal_cnt = static_cast<uint32_t>(count);
  uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(cal_cnt, vl_size));

  MicroAPI::RegTensor<T> idx_reg, dst_reg, dim0_reg, dim1_reg;
  // idx_reg init: 0~(VL-1)
  MicroAPI::Arange(idx_reg, 0);
  MicroAPI::MaskReg mask;
  for (uint16_t i = 0U; i < repeat_time; i++) {
    mask = MicroAPI::UpdateMask<T>(cal_cnt);
    MicroAPI::Duplicate(dim1_reg, dst_dims[1]);
    MicroAPI::Div(dim1_reg, idx_reg, dim1_reg, mask);
    MicroAPI::Copy(dim0_reg, dim1_reg);
    MicroAPI::Muls(dim1_reg, dim1_reg, dst_dims[1], mask);
    MicroAPI::Sub(dim1_reg, idx_reg, dim1_reg, mask);
    // index: index_dim0 * dim0_in_offset + index_dim1 * dim1_in_offset
    MicroAPI::Muls(dim1_reg, dim1_reg, src_strides[1], mask);
    MicroAPI::Muls(dim0_reg, dim0_reg, src_strides[0], mask);
    MicroAPI::Add(dst_reg, dim1_reg, dim0_reg, mask);
    MicroAPI::DataCopy(dst_idx + i * vl_size, dst_reg, mask);
    MicroAPI::Adds(idx_reg, idx_reg, vl_size, mask);
  }
}

template <typename T, uint8_t dim>
__aicore__ inline void GenThreeDimTransposeIndex(__ubuf__ T *dst_idx, const T (&dst_dims)[dim],
                                                 const T (&src_strides)[dim], const RangeType<T>& count) {
  uint16_t vl_size = static_cast<uint16_t>(GetVecLen() / sizeof(T));
  uint32_t cal_cnt = static_cast<uint32_t>(count);
  uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(cal_cnt, vl_size));
  auto last_two_dim_size = dst_dims[1] * dst_dims[2];

  MicroAPI::RegTensor<T> idx_reg, dst_reg, dim0_reg, dim1_reg, dim2_reg, tmp_reg;
  // idx_reg init: 0~(VL-1)
  MicroAPI::Arange(idx_reg, 0);
  MicroAPI::MaskReg mask;
  for (uint16_t i = 0U; i < repeat_time; i++) {
    mask = MicroAPI::UpdateMask<T>(cal_cnt);
    MicroAPI::Duplicate(dim2_reg, dst_dims[2]);
    MicroAPI::Div(dim2_reg, idx_reg, dim2_reg, mask);
    MicroAPI::Copy(dim1_reg, dim2_reg);
    MicroAPI::Muls(dim2_reg, dim2_reg, dst_dims[2], mask);
    MicroAPI::Sub(dim2_reg, idx_reg, dim2_reg, mask);

    MicroAPI::Duplicate(tmp_reg, dst_dims[1]);
    MicroAPI::Div(tmp_reg, dim1_reg, tmp_reg, mask);
    MicroAPI::Muls(tmp_reg, tmp_reg, dst_dims[1], mask);
    MicroAPI::Sub(dim1_reg, dim1_reg, tmp_reg, mask);

    MicroAPI::Duplicate(dim0_reg, last_two_dim_size);
    MicroAPI::Div(dim0_reg, idx_reg, dim0_reg, mask);
    // index: index_dim0 * dim0_in_offset + index_dim1 * dim1_in_offset + index_dim2 * dim2_in_offset
    MicroAPI::Muls(dim2_reg, dim2_reg, src_strides[2], mask);
    MicroAPI::Muls(dim1_reg, dim1_reg, src_strides[1], mask);
    MicroAPI::Muls(dim0_reg, dim0_reg, src_strides[0], mask);
    MicroAPI::Add(dst_reg, dim2_reg, dim1_reg, mask);
    MicroAPI::Add(dst_reg, dst_reg, dim0_reg, mask);
    MicroAPI::DataCopy(dst_idx + i * vl_size, dst_reg, mask);
    MicroAPI::Adds(idx_reg, idx_reg, vl_size, mask);
  }
}

template <typename T, uint8_t dim>
__aicore__ inline void GenFourDimTransposeIndex(__ubuf__ T *dst_idx, const T (&dst_dims)[dim],
                                                const T (&src_strides)[dim], const RangeType<T>& count) {
  uint16_t vl_size = static_cast<uint16_t>(GetVecLen() / sizeof(T));
  uint32_t cal_cnt = static_cast<uint32_t>(count);
  uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(cal_cnt, vl_size));
  auto last_three_dim_size = dst_dims[1] * dst_dims[2] * dst_dims[3];

  MicroAPI::RegTensor<T> idx_reg, dst_reg, dim0_reg, dim1_reg, dim2_reg, dim3_reg, tmp_reg;
  // idx_reg init: 0~(VL-1)
  MicroAPI::Arange(idx_reg, 0);
  MicroAPI::MaskReg mask;
  for (uint16_t i = 0U; i < repeat_time; i++) {
    mask = MicroAPI::UpdateMask<T>(cal_cnt);
    MicroAPI::Duplicate(dim3_reg, dst_dims[3]);
    MicroAPI::Div(dim3_reg, idx_reg, dim3_reg, mask);
    MicroAPI::Copy(dim2_reg, dim3_reg);
    MicroAPI::Muls(dim3_reg, dim3_reg, dst_dims[3], mask);
    MicroAPI::Sub(dim3_reg, idx_reg, dim3_reg, mask);

    MicroAPI::Duplicate(tmp_reg, dst_dims[2]);
    MicroAPI::Div(tmp_reg, dim2_reg, tmp_reg, mask);
    MicroAPI::Copy(dim1_reg, tmp_reg);
    MicroAPI::Muls(tmp_reg, tmp_reg, dst_dims[2], mask);
    MicroAPI::Sub(dim2_reg, dim2_reg, tmp_reg, mask);

    MicroAPI::Duplicate(tmp_reg, dst_dims[1]);
    MicroAPI::Div(tmp_reg, dim1_reg, tmp_reg, mask);
    MicroAPI::Muls(tmp_reg, tmp_reg, dst_dims[1], mask);
    MicroAPI::Sub(dim1_reg, dim1_reg, tmp_reg, mask);

    MicroAPI::Duplicate(dim0_reg, last_three_dim_size);
    MicroAPI::Div(dim0_reg, idx_reg, dim0_reg, mask);
    // index: index_dim0 * dim0_in_offset + index_dim1 * dim1_in_offset + index_dim2 * dim2_in_offset
    MicroAPI::Muls(dim3_reg, dim3_reg, src_strides[3], mask);
    MicroAPI::Muls(dim2_reg, dim2_reg, src_strides[2], mask);
    MicroAPI::Muls(dim1_reg, dim1_reg, src_strides[1], mask);
    MicroAPI::Muls(dim0_reg, dim0_reg, src_strides[0], mask);
    MicroAPI::Add(dst_reg, dim3_reg, dim2_reg, mask);
    MicroAPI::Add(tmp_reg, dim1_reg, dim0_reg, mask);
    MicroAPI::Add(dst_reg, dst_reg, tmp_reg, mask);
    MicroAPI::DataCopy(dst_idx + i * vl_size, dst_reg, mask);
    MicroAPI::Adds(idx_reg, idx_reg, vl_size, mask);
  }
}

template <typename T, uint8_t dim>
__aicore__ inline void GenFiveDimTransposeIndex(__ubuf__ T *dst_idx, const T (&dst_dims)[dim],
                                                const T (&src_strides)[dim], const RangeType<T>& count) {
  uint16_t vl_size = static_cast<uint16_t>(GetVecLen() / sizeof(T));
  uint32_t cal_cnt = static_cast<uint32_t>(count);
  uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(cal_cnt, vl_size));
  auto last_four_dim_size = dst_dims[1] * dst_dims[2] * dst_dims[3] * dst_dims[4];

  MicroAPI::RegTensor<T> idx_reg, dst_reg, dim0_reg, dim1_reg, dim2_reg, dim3_reg, dim4_reg, tmp_reg;
  // idx_reg init: 0~(VL-1)
  MicroAPI::Arange(idx_reg, 0);
  MicroAPI::MaskReg mask;
  for (uint16_t i = 0U; i < repeat_time; i++) {
    mask = MicroAPI::UpdateMask<T>(cal_cnt);
    MicroAPI::Duplicate(dim4_reg, dst_dims[4]);
    MicroAPI::Div(dim4_reg, idx_reg, dim4_reg, mask);
    MicroAPI::Copy(dim3_reg, dim4_reg);
    MicroAPI::Muls(dim4_reg, dim4_reg, dst_dims[4], mask);
    MicroAPI::Sub(dim4_reg, idx_reg, dim4_reg, mask);

    MicroAPI::Duplicate(tmp_reg, dst_dims[3]);
    MicroAPI::Div(tmp_reg, dim3_reg, tmp_reg, mask);
    MicroAPI::Copy(dim2_reg, tmp_reg);
    MicroAPI::Muls(tmp_reg, tmp_reg, dst_dims[3], mask);
    MicroAPI::Sub(dim3_reg, dim3_reg, tmp_reg, mask);

    MicroAPI::Duplicate(tmp_reg, dst_dims[2]);
    MicroAPI::Div(tmp_reg, dim2_reg, tmp_reg, mask);
    MicroAPI::Copy(dim1_reg, tmp_reg);
    MicroAPI::Muls(tmp_reg, tmp_reg, dst_dims[2], mask);
    MicroAPI::Sub(dim2_reg, dim2_reg, tmp_reg, mask);

    MicroAPI::Duplicate(tmp_reg, dst_dims[1]);
    MicroAPI::Div(tmp_reg, dim1_reg, tmp_reg, mask);
    MicroAPI::Copy(dim0_reg, tmp_reg);
    MicroAPI::Muls(tmp_reg, tmp_reg, dst_dims[1], mask);
    MicroAPI::Sub(dim1_reg, dim1_reg, tmp_reg, mask);

    MicroAPI::Duplicate(dim0_reg, last_four_dim_size);
    MicroAPI::Div(dim0_reg, idx_reg, dim0_reg, mask);
    // index: index_dim0 * dim0_in_offset + index_dim1 * dim1_in_offset + index_dim2 * dim2_in_offset
    MicroAPI::Muls(dim4_reg, dim4_reg, src_strides[4], mask);
    MicroAPI::Muls(dim3_reg, dim3_reg, src_strides[3], mask);
    MicroAPI::Muls(dim2_reg, dim2_reg, src_strides[2], mask);
    MicroAPI::Muls(dim1_reg, dim1_reg, src_strides[1], mask);
    MicroAPI::Muls(dim0_reg, dim0_reg, src_strides[0], mask);
    MicroAPI::Add(dst_reg, dim4_reg, dim3_reg, mask);
    MicroAPI::Add(tmp_reg, dim2_reg, dim1_reg, mask);
    MicroAPI::Add(dst_reg, dst_reg, tmp_reg, mask);
    MicroAPI::Add(dst_reg, dst_reg, dim0_reg, mask);
    MicroAPI::DataCopy(dst_idx + i * vl_size, dst_reg, mask);
    MicroAPI::Adds(idx_reg, idx_reg, vl_size, mask);
  }
}

template <typename T, uint8_t dim>
__aicore__ inline void GenTransposeIndex(__ubuf__ T* dst_idx, const T (&dst_dims)[dim], const T (&src_strides)[dim],
                                         const RangeType<T>& cal_cnt) {
  if constexpr (dim == 1) {
    GenOneDimTransposeIndex<T, dim>(dst_idx, dst_dims, src_strides, cal_cnt);
  } else if constexpr (dim == 2) {
    GenTwoDimTransposeIndex<T, dim>(dst_idx, dst_dims, src_strides, cal_cnt);
  } else if constexpr (dim == 3) {
    GenThreeDimTransposeIndex<T, dim>(dst_idx, dst_dims, src_strides, cal_cnt);
  } else if constexpr (dim == 4) {
    GenFourDimTransposeIndex<T, dim>(dst_idx, dst_dims, src_strides, cal_cnt);
  } else if constexpr (dim == 5) {
    GenFiveDimTransposeIndex<T, dim>(dst_idx, dst_dims, src_strides, cal_cnt);
  }
}

template <typename T>
__aicore__ inline void TransposeExtendImpl(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ RangeType<T>* index,
                                           RangeType<T>& count) {
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
template <typename T, uint8_t dim>
__aicore__ inline void TransposeExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                       const LocalTensor<RangeType<T>> &tmp_buf, const RangeType<T> (&dst_dims)[dim],
                                       const RangeType<T> (&src_strides)[dim], const RangeType<T> (&dst_strides)[dim]) {
  __ubuf__ T* src_buf = (__ubuf__ T*)src.GetPhyAddr();
  __ubuf__ T* dst_buf = (__ubuf__ T*)dst.GetPhyAddr();
  __ubuf__ RangeType<T>* index_buf = (__ubuf__ RangeType<T>*)tmp_buf.GetPhyAddr();
  RangeType<T> cal_cnt = 1;
  uint8_t inner_dim_start = (dim > 5) ? dim - 5 : 0;
  for (uint8_t i = inner_dim_start; i < dim; i++) {
    cal_cnt *= dst_dims[i];
  }
  if constexpr (dim == 6) {
    RangeType<T> out_dims[5] = {dst_dims[1], dst_dims[2], dst_dims[3], dst_dims[4], dst_dims[5]};
    RangeType<T> in_strides[5] = {src_strides[1], src_strides[2], src_strides[3], src_strides[4], src_strides[5]};
    GenFiveDimTransposeIndex<RangeType<T>, 5>(index_buf, out_dims, in_strides, cal_cnt);
    for (RangeType<T> i = 0; i < dst_dims[0]; i++) {
      TransposeExtendImpl<T>(dst_buf + i * dst_strides[0], src_buf + i * src_strides[0], index_buf, cal_cnt);
    }
  } else if constexpr (dim == 7) {
    RangeType<T> out_dims[5] = {dst_dims[2], dst_dims[3], dst_dims[4], dst_dims[5], dst_dims[6]};
    RangeType<T> in_strides[5] = {src_strides[2], src_strides[3], src_strides[4], src_strides[5], src_strides[6]};
    GenFiveDimTransposeIndex<RangeType<T>, 5>(index_buf, out_dims, in_strides, cal_cnt);
    for (RangeType<T> i = 0; i < dst_dims[0]; i++) {
      for (RangeType<T> j = 0; j < dst_dims[1]; j++) {
        TransposeExtendImpl<T>(dst_buf + i * dst_strides[0] + j * dst_strides[1],
                               src_buf + i * src_strides[0] + j * src_strides[1], index_buf, cal_cnt);
      }
    }
  } else if constexpr (dim == 8) {
    RangeType<T> out_dims[5] = {dst_dims[3], dst_dims[4], dst_dims[5], dst_dims[6], dst_dims[7]};
    RangeType<T> in_strides[5] = {src_strides[3], src_strides[4], src_strides[5], src_strides[6], src_strides[7]};
    GenFiveDimTransposeIndex<RangeType<T>, 5>(index_buf, out_dims, in_strides, cal_cnt);
    for (RangeType<T> i = 0; i < dst_dims[0]; i++) {
      for (RangeType<T> j = 0; j < dst_dims[1]; j++) {
        for (RangeType<T> k = 0; k < dst_dims[2]; k++) {
          TransposeExtendImpl<T>(dst_buf + i * dst_strides[0] + j * dst_strides[1] + k * dst_strides[2],
                                 src_buf + i * src_strides[0] + j * src_strides[1] + k * src_strides[2],
                                 index_buf, cal_cnt);
        }
      }
    }
  } else {
    GenTransposeIndex<RangeType<T>, dim>(index_buf, dst_dims, src_strides, cal_cnt);
    TransposeExtendImpl<T>(dst_buf, src_buf, index_buf, cal_cnt);
  }
}
#endif
