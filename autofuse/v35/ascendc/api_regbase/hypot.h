/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_HYPOT_H__
#define __ASCENDC_API_REGBASE_HYPOT_H__

/*
 * hypot(x, y) = sqrt(x^2 + y^2)
 *
 * Naive computation overflows for large inputs and loses precision for small inputs.
 * Numerically stable algorithm:
 *   a = |x|, b = |y|
 *   m = max(a, b), n = min(a, b)
 *   hypot(x, y) = m * sqrt(1 + (n/m)^2)
 *
 * Boundary handling (applied after normal computation, in priority order):
 *   1. m == 0 (both inputs are 0): ratio = 0/0 = NaN, override result to 0
 *   2. Either input is NaN: override result to NaN
 *      (needed because max(NaN, x) returns x in IEEE 754)
 *   3. Either |input| is inf: override result to inf
 *      (per C standard: inf dominates NaN, hypot(inf, nan) = inf)
 */

template <typename T>
__simd_vf__ inline void HypotImplVF(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint32_t calc_cnt) {
  uint32_t vl_size = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
  uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(calc_cnt, vl_size));

  AscendC::Reg::RegTensor<T> src0_reg, src1_reg, dst_reg;
  AscendC::Reg::RegTensor<T> abs0_reg, abs1_reg, max_reg, min_reg, ratio_reg, ratio_sq_reg;
  AscendC::Reg::RegTensor<T> one_plus_reg, sqrt_reg, special_reg;
  AscendC::Reg::MaskReg mask, cmp_mask;

  for (uint16_t i = 0U; i < repeat_time; ++i) {
    mask = AscendC::Reg::UpdateMask<T>(calc_cnt);
    AscendC::Reg::LoadAlign(src0_reg, src0 + i * vl_size);
    AscendC::Reg::LoadAlign(src1_reg, src1 + i * vl_size);

    // a = |x|, b = |y|
    AscendC::Reg::Abs(abs0_reg, src0_reg, mask);
    AscendC::Reg::Abs(abs1_reg, src1_reg, mask);

    // m = max(a, b), n = min(a, b)
    AscendC::Reg::Compare<T, AscendC::CMPMODE::GE>(cmp_mask, abs0_reg, abs1_reg, mask);
    AscendC::Reg::Select(max_reg, abs0_reg, abs1_reg, cmp_mask);
    AscendC::Reg::Select(min_reg, abs1_reg, abs0_reg, cmp_mask);

    // ratio = n / m
    AscendC::Reg::Div(ratio_reg, min_reg, max_reg, mask);

    // ratio_sq = ratio * ratio
    AscendC::Reg::Mul(ratio_sq_reg, ratio_reg, ratio_reg, mask);

    // val = 1 + ratio_sq
    AscendC::Reg::Adds(one_plus_reg, ratio_sq_reg, (T)1.0, mask);

    // sqrt_val = sqrt(val)
    AscendC::Reg::Sqrt(sqrt_reg, one_plus_reg, mask);

    // result = m * sqrt_val
    AscendC::Reg::Mul(dst_reg, max_reg, sqrt_reg, mask);

    // 1. Both inputs are 0 (m == 0): ratio = 0/0 = NaN, override result to 0
    AscendC::Reg::Duplicate<T>(special_reg, (T)0.0, mask);
    AscendC::Reg::Compares<T, AscendC::CMPMODE::EQ>(cmp_mask, max_reg, (T)0.0, mask);
    AscendC::Reg::Select(dst_reg, special_reg, dst_reg, cmp_mask);

    // 2. Either input is NaN: override result to NaN
    //    (needed because max(NaN, x) returns x in IEEE 754)
    AscendC::Reg::Compare<T, AscendC::CMPMODE::NE>(cmp_mask, src0_reg, src0_reg, mask);
    AscendC::Reg::Select(dst_reg, src0_reg, dst_reg, cmp_mask);
    AscendC::Reg::Compare<T, AscendC::CMPMODE::NE>(cmp_mask, src1_reg, src1_reg, mask);
    AscendC::Reg::Select(dst_reg, src1_reg, dst_reg, cmp_mask);

    // 3. Either |input| is inf: override result to inf
    //    (per C standard, inf dominates NaN: hypot(inf, nan) = inf)
    AscendC::Reg::Duplicate<T>(special_reg, (float &)AscendC::F32_INF, mask);
    AscendC::Reg::Compares<T, AscendC::CMPMODE::EQ>(cmp_mask, abs0_reg, (float &)AscendC::F32_INF, mask);
    AscendC::Reg::Select(dst_reg, special_reg, dst_reg, cmp_mask);
    AscendC::Reg::Compares<T, AscendC::CMPMODE::EQ>(cmp_mask, abs1_reg, (float &)AscendC::F32_INF, mask);
    AscendC::Reg::Select(dst_reg, special_reg, dst_reg, cmp_mask);

    AscendC::Reg::StoreAlign(dst + i * vl_size, dst_reg, mask);
  }
}

template <typename T>
__aicore__ inline void HypotExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src0,
                                   const AscendC::LocalTensor<T> &src1, const LocalTensor<uint8_t> &tmpBuffer,
                                   const uint32_t calc_cnt) {
  static_assert(std::is_same<T, float>::value, "Hypot currently only supports float");
  ASCENDC_ASSERT(calc_cnt > 0, { KERNEL_LOG(KERNEL_ERROR, "calc_cnt must be positive, got %u", calc_cnt); });
  if ASCEND_IS_AIC {
    return;
  }

  HypotImplVF<T>((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src0.GetPhyAddr(), (__ubuf__ T *)src1.GetPhyAddr(),
                 calc_cnt);
}

#endif  // __ASCENDC_API_REGBASE_HYPOT_H__
