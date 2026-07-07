/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_NDTR_H__
#define __ASCENDC_API_REGBASE_NDTR_H__

using namespace AscendC;

namespace NDTR {

constexpr float NDTR_INV_SQRT_2 = 0.7071067811865475f;
constexpr float NDTR_LARGE_THRESHOLD = 3.92f;
constexpr uint32_t NDTR_NAN_UINT = 0x7fc00000;
constexpr uint32_t NDTR_POS_INF_UINT = 0x7f800000;

constexpr float ERF_P0 = 0.29639384698e5f;
constexpr float ERF_P1_COEF = 0.50637915060e4f;
constexpr float ERF_P2_COEF = 0.13938061484e4f;
constexpr float ERF_P3_COEF = 0.10162808918e3f;
constexpr float ERF_P4_COEF = 0.75517016694e1f;
constexpr float ERF_P5_COEF = 0.053443748819f;

constexpr float ERF_Q0 = 0.26267224157e5f;
constexpr float ERF_Q1 = 0.13243365831e5f;
constexpr float ERF_Q2 = 0.30231248150e4f;
constexpr float ERF_Q3 = 0.39856963806e3f;
constexpr float ERF_Q4 = 0.31212858877e2f;

__simd_callee__ inline void NdtrComputeCore(Reg::RegTensor<float> &dstReg, Reg::RegTensor<float> &srcReg,
                                            Reg::MaskReg &mask) {
  Reg::RegTensor<float> scaledReg, clippedReg, tmpReg, pReg, qReg, erfReg;

  Reg::Muls(scaledReg, srcReg, NDTR_INV_SQRT_2, mask);

  Reg::Mins(clippedReg, scaledReg, 3.92f, mask);
  Reg::Maxs(clippedReg, clippedReg, -3.92f, mask);

  Reg::Mul(tmpReg, clippedReg, clippedReg, mask);

  Reg::Muls(pReg, tmpReg, ERF_P5_COEF, mask);
  Reg::Adds(pReg, pReg, ERF_P4_COEF, mask);
  Reg::Mul(pReg, pReg, tmpReg, mask);
  Reg::Adds(pReg, pReg, ERF_P3_COEF, mask);
  Reg::Mul(pReg, pReg, tmpReg, mask);
  Reg::Adds(pReg, pReg, ERF_P2_COEF, mask);
  Reg::Mul(pReg, pReg, tmpReg, mask);
  Reg::Adds(pReg, pReg, ERF_P1_COEF, mask);
  Reg::Mul(pReg, pReg, tmpReg, mask);
  Reg::Adds(pReg, pReg, ERF_P0, mask);
  Reg::Mul(pReg, pReg, clippedReg, mask);

  Reg::Adds(qReg, tmpReg, ERF_Q4, mask);
  Reg::Mul(qReg, qReg, tmpReg, mask);
  Reg::Adds(qReg, qReg, ERF_Q3, mask);
  Reg::Mul(qReg, qReg, tmpReg, mask);
  Reg::Adds(qReg, qReg, ERF_Q2, mask);
  Reg::Mul(qReg, qReg, tmpReg, mask);
  Reg::Adds(qReg, qReg, ERF_Q1, mask);
  Reg::Mul(qReg, qReg, tmpReg, mask);
  Reg::Adds(qReg, qReg, ERF_Q0, mask);

  Reg::Div(erfReg, pReg, qReg, mask);

  Reg::Adds(dstReg, erfReg, 1.0f, mask);
  Reg::Muls(dstReg, dstReg, 0.5f, mask);
}

__simd_callee__ inline void NdtrHandleLargeInput(Reg::RegTensor<float> &dstReg, Reg::RegTensor<float> &srcReg,
                                                 Reg::MaskReg &mask) {
  Reg::RegTensor<float> absReg, zeroReg, oneReg;
  Reg::MaskReg largeMask, negMask, largeNegMask, largePosMask;

  Reg::Abs(absReg, srcReg, mask);
  Reg::Compares<float, CMPMODE::GE>(largeMask, absReg, NDTR_LARGE_THRESHOLD, mask);

  Reg::Compares<float, CMPMODE::LT>(negMask, srcReg, 0.0f, mask);

  Reg::And(largeNegMask, largeMask, negMask, mask);

  Reg::Not(largePosMask, negMask, mask);
  Reg::And(largePosMask, largeMask, largePosMask, mask);

  Reg::Duplicate(zeroReg, 0.0f, mask);
  Reg::Duplicate(oneReg, 1.0f, mask);

  Reg::Copy<float, Reg::MaskMergeMode::MERGING>(dstReg, zeroReg, largeNegMask);
  Reg::Copy<float, Reg::MaskMergeMode::MERGING>(dstReg, oneReg, largePosMask);
}

__simd_callee__ inline void NdtrHandleSpecialCases(Reg::RegTensor<float> &dstReg, Reg::RegTensor<float> &srcReg,
                                                   Reg::MaskReg &mask) {
  Reg::RegTensor<float> nanReg, constReg, oneReg, zeroReg;
  Reg::MaskReg nanMask, posInfMask, negInfMask;

  Reg::Compare<float, CMPMODE::NE>(nanMask, srcReg, srcReg, mask);
  Reg::Duplicate(nanReg, (float &)NDTR_NAN_UINT, mask);
  Reg::Select(dstReg, nanReg, dstReg, nanMask);

  Reg::Duplicate(constReg, (float &)NDTR_POS_INF_UINT, mask);
  Reg::Compare<float, CMPMODE::EQ>(posInfMask, srcReg, constReg, mask);
  Reg::Duplicate(oneReg, 1.0f, posInfMask);
  Reg::Select(dstReg, oneReg, dstReg, posInfMask);

  Reg::Neg(constReg, constReg, mask);
  Reg::Compare<float, CMPMODE::EQ>(negInfMask, srcReg, constReg, mask);
  Reg::Duplicate(zeroReg, 0.0f, negInfMask);
  Reg::Select(dstReg, zeroReg, dstReg, negInfMask);
}

template <typename T>
__simd_vf__ inline void NdtrCoreImpl(__ubuf__ T *dstUb, __ubuf__ T *srcUb, uint32_t calCount, uint16_t repeatTimes) {
  static_assert((std::is_same_v<T, float>), "Ndtr only supports float on current device!");
  constexpr uint32_t oneRepSize = static_cast<uint32_t>(GetVecLen() / sizeof(float));

  for (uint16_t i = 0; i < repeatTimes; ++i) {
    uint32_t sreg = calCount;
    Reg::MaskReg mask = Reg::UpdateMask<float>(sreg);
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    Reg::RegTensor<float> srcReg, dstReg, resultReg;
    Reg::MaskReg largeMask;

    Reg::LoadAlign(srcReg, srcUb + i * oneRepSize);

    Reg::Duplicate(resultReg, 0.5f, fullMask);

    NdtrComputeCore(dstReg, srcReg, mask);
    Reg::Copy<float, Reg::MaskMergeMode::MERGING>(resultReg, dstReg, mask);

    NdtrHandleLargeInput(resultReg, srcReg, mask);
    NdtrHandleSpecialCases(resultReg, srcReg, mask);

    Reg::StoreAlign(dstUb + i * oneRepSize, resultReg, mask);
  }
}

}  // namespace NDTR

template <typename T>
__aicore__ inline void NdtrExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                  const LocalTensor<uint8_t> &tmpBuffer, const uint32_t calCount) {
  static_assert((std::is_same_v<T, float>), "Ndtr only supports float on current device!");
  if ASCEND_IS_AIC {
    return;
  }

  __ubuf__ T *dstUb = (__ubuf__ T *)dst.GetPhyAddr();
  __ubuf__ T *srcUb = (__ubuf__ T *)src.GetPhyAddr();

  constexpr uint32_t oneRepSize = static_cast<uint32_t>(GetVecLen() / sizeof(float));
  uint16_t repeatTimes = CeilDivision(calCount, oneRepSize);

  NDTR::NdtrCoreImpl<T>(dstUb, srcUb, calCount, repeatTimes);
}

#endif  // __ASCENDC_API_REGBASE_NDTR_H__
