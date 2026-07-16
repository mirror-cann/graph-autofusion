/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_LOG_NDTR_H__
#define __ASCENDC_API_REGBASE_LOG_NDTR_H__

using namespace AscendC;

namespace LOG_NDTR {

constexpr float LOG_NDTR_INV_SQRT_2 = 0.7071067811865475f;
constexpr float LOG_NDTR_BRANCH_THRESHOLD = -1.0f;
// |t| threshold for the erf Pade approximation; beyond it the result saturates to sign(t)
constexpr float LOG_NDTR_LARGE_THRESHOLD = 3.92f;
// Float bit patterns (AscendC TIK compiler does not provide memcpy/bit_cast;
// pointer-punning via reference is the established convention in api_regbase)
constexpr uint32_t LOG_NDTR_NAN_UINT = 0x7fc00000;
constexpr uint32_t LOG_NDTR_POS_INF_UINT = 0x7f800000;
constexpr uint32_t LOG_NDTR_NEG_INF_UINT = 0xff800000;

// Pade coefficients for erf(t) rational approximation (from ndtr.h)
constexpr float LOG_NDTR_ERF_P0 = 0.29639'384698e5f;
constexpr float LOG_NDTR_ERF_P1 = 0.50637'915060e4f;
constexpr float LOG_NDTR_ERF_P2 = 0.13938'061484e4f;
constexpr float LOG_NDTR_ERF_P3 = 0.10162'808918e3f;
constexpr float LOG_NDTR_ERF_P4 = 0.75517'016694e1f;
constexpr float LOG_NDTR_ERF_P5 = 0.05344'3748819f;
constexpr float LOG_NDTR_ERF_Q0 = 0.26267'224157e5f;
constexpr float LOG_NDTR_ERF_Q1 = 0.13243'365831e5f;
constexpr float LOG_NDTR_ERF_Q2 = 0.30231'248150e4f;
constexpr float LOG_NDTR_ERF_Q3 = 0.39856'963806e3f;
constexpr float LOG_NDTR_ERF_Q4 = 0.31212'858877e2f;

// Erfcx low-path polynomial coefficients (from erfcx.h)
constexpr float LOG_NDTR_P_COEFF0 = 0.0008912171f;
constexpr float LOG_NDTR_P_COEFF1 = 0.007045788f;
constexpr float LOG_NDTR_P_COEFF2 = -0.0158668961f;
constexpr float LOG_NDTR_P_COEFF3 = 0.036429625f;
constexpr float LOG_NDTR_P_COEFF4 = -0.06664343f;
constexpr float LOG_NDTR_P_COEFF5 = 0.09381453f;
constexpr float LOG_NDTR_P_COEFF6 = -0.100990564f;
constexpr float LOG_NDTR_P_COEFF7 = 0.068094f;
constexpr float LOG_NDTR_P_COEFF8 = 0.0153773874f;
constexpr float LOG_NDTR_P_COEFF9 = -0.139621079f;
constexpr float LOG_NDTR_P_COEFF10 = 1.23299515f;

// ---------------------------------------------------------------------------
// ErfcxPolyLite: simplified low-path erfcx polynomial evaluation
// Input: axReg = |t| (non-negative)
// Output: erfcx(ax) = p(t_mapped) / (2*ax + 1), where t_mapped = (ax-4)/(ax+4)
// ---------------------------------------------------------------------------
__simd_callee__ inline void ErfcxPolyLite(Reg::RegTensor<float> &dstReg, Reg::RegTensor<float> &axReg,
                                          Reg::MaskReg &mask) {
  Reg::RegTensor<float> invAxP4Reg, tMappedReg, tmpReg, pReg, invDenomReg;

  // inv_ax_plus_4 = 1 / (ax + 4)
  Reg::Adds(invAxP4Reg, axReg, 4.0f, mask);
  Reg::Duplicate(tmpReg, 1.0f, mask);
  Reg::Div(invAxP4Reg, tmpReg, invAxP4Reg, mask);

  // t_mapped = (ax - 4) / (ax + 4) = (ax - 4) * inv_ax_plus_4
  Reg::Adds(tMappedReg, axReg, -4.0f, mask);
  Reg::Mul(tMappedReg, tMappedReg, invAxP4Reg, mask);

  // Horner evaluation of polynomial P(t) = LOG_NDTR_P_COEFF0 + t*(LOG_NDTR_P_COEFF1 + t*(... + t*LOG_NDTR_P_COEFF10))
  Reg::Duplicate(pReg, LOG_NDTR_P_COEFF0, mask);
  Reg::Duplicate(tmpReg, LOG_NDTR_P_COEFF1, mask);
  Reg::FusedMulDstAdd(pReg, tMappedReg, tmpReg, mask);
  Reg::Duplicate(tmpReg, LOG_NDTR_P_COEFF2, mask);
  Reg::FusedMulDstAdd(pReg, tMappedReg, tmpReg, mask);
  Reg::Duplicate(tmpReg, LOG_NDTR_P_COEFF3, mask);
  Reg::FusedMulDstAdd(pReg, tMappedReg, tmpReg, mask);
  Reg::Duplicate(tmpReg, LOG_NDTR_P_COEFF4, mask);
  Reg::FusedMulDstAdd(pReg, tMappedReg, tmpReg, mask);
  Reg::Duplicate(tmpReg, LOG_NDTR_P_COEFF5, mask);
  Reg::FusedMulDstAdd(pReg, tMappedReg, tmpReg, mask);
  Reg::Duplicate(tmpReg, LOG_NDTR_P_COEFF6, mask);
  Reg::FusedMulDstAdd(pReg, tMappedReg, tmpReg, mask);
  Reg::Duplicate(tmpReg, LOG_NDTR_P_COEFF7, mask);
  Reg::FusedMulDstAdd(pReg, tMappedReg, tmpReg, mask);
  Reg::Duplicate(tmpReg, LOG_NDTR_P_COEFF8, mask);
  Reg::FusedMulDstAdd(pReg, tMappedReg, tmpReg, mask);
  Reg::Duplicate(tmpReg, LOG_NDTR_P_COEFF9, mask);
  Reg::FusedMulDstAdd(pReg, tMappedReg, tmpReg, mask);
  Reg::Duplicate(tmpReg, LOG_NDTR_P_COEFF10, mask);
  Reg::FusedMulDstAdd(pReg, tMappedReg, tmpReg, mask);

  // inv_denom = 1 / (2*ax + 1)
  Reg::Muls(invDenomReg, axReg, 2.0f, mask);
  Reg::Adds(invDenomReg, invDenomReg, 1.0f, mask);
  Reg::Duplicate(tmpReg, 1.0f, mask);
  Reg::Div(invDenomReg, tmpReg, invDenomReg, mask);

  // result = p * inv_denom
  Reg::Mul(dstReg, pReg, invDenomReg, mask);
}

// ---------------------------------------------------------------------------
// ErfCompute: compute erf(t) using Pade rational approximation
// For |t| < 3.92: Pade approximation; for |t| >= 3.92: sign(t)
// ---------------------------------------------------------------------------
__simd_callee__ inline void ErfCompute(Reg::RegTensor<float> &dstReg, Reg::RegTensor<float> &tReg, Reg::MaskReg &mask) {
  Reg::RegTensor<float> clippedReg, tmpReg, pReg, qReg;
  Reg::RegTensor<float> absReg, negOneReg, oneReg;
  Reg::MaskReg maskLarge, maskNeg, maskLargeNeg, maskLargePos;

  // Clip t to [-3.92, 3.92] for safe Pade evaluation
  Reg::Mins(clippedReg, tReg, LOG_NDTR_LARGE_THRESHOLD, mask);
  Reg::Maxs(clippedReg, clippedReg, -LOG_NDTR_LARGE_THRESHOLD, mask);

  // tmp = clipped^2
  Reg::Mul(tmpReg, clippedReg, clippedReg, mask);

  // P polynomial (degree 5): clipped * sum_i P_i * tmp^i
  Reg::Muls(pReg, tmpReg, LOG_NDTR_ERF_P5, mask);
  Reg::Adds(pReg, pReg, LOG_NDTR_ERF_P4, mask);
  Reg::Mul(pReg, pReg, tmpReg, mask);
  Reg::Adds(pReg, pReg, LOG_NDTR_ERF_P3, mask);
  Reg::Mul(pReg, pReg, tmpReg, mask);
  Reg::Adds(pReg, pReg, LOG_NDTR_ERF_P2, mask);
  Reg::Mul(pReg, pReg, tmpReg, mask);
  Reg::Adds(pReg, pReg, LOG_NDTR_ERF_P1, mask);
  Reg::Mul(pReg, pReg, tmpReg, mask);
  Reg::Adds(pReg, pReg, LOG_NDTR_ERF_P0, mask);
  Reg::Mul(pReg, pReg, clippedReg, mask);

  // Q polynomial (degree 5 with leading coefficient 1): sum_i Q_i * tmp^i + tmp^5
  Reg::Adds(qReg, tmpReg, LOG_NDTR_ERF_Q4, mask);
  Reg::Mul(qReg, qReg, tmpReg, mask);
  Reg::Adds(qReg, qReg, LOG_NDTR_ERF_Q3, mask);
  Reg::Mul(qReg, qReg, tmpReg, mask);
  Reg::Adds(qReg, qReg, LOG_NDTR_ERF_Q2, mask);
  Reg::Mul(qReg, qReg, tmpReg, mask);
  Reg::Adds(qReg, qReg, LOG_NDTR_ERF_Q1, mask);
  Reg::Mul(qReg, qReg, tmpReg, mask);
  Reg::Adds(qReg, qReg, LOG_NDTR_ERF_Q0, mask);

  // erf = p / q
  Reg::Div(dstReg, pReg, qReg, mask);

  // For |t| >= 3.92: override with sign(t)
  Reg::Abs(absReg, tReg, mask);
  Reg::Compares<float, CMPMODE::GE>(maskLarge, absReg, LOG_NDTR_LARGE_THRESHOLD, mask);
  Reg::Compares<float, CMPMODE::LT>(maskNeg, tReg, 0.0f, mask);

  Reg::And(maskLargeNeg, maskLarge, maskNeg, mask);
  Reg::Not(maskLargePos, maskNeg, mask);
  Reg::And(maskLargePos, maskLarge, maskLargePos, mask);

  Reg::Duplicate(negOneReg, -1.0f, mask);
  Reg::Duplicate(oneReg, 1.0f, mask);

  Reg::Copy<float, Reg::MaskMergeMode::MERGING>(dstReg, negOneReg, maskLargeNeg);
  Reg::Copy<float, Reg::MaskMergeMode::MERGING>(dstReg, oneReg, maskLargePos);
}

// ---------------------------------------------------------------------------
// Left tail computation: result = log(erfcx(-t) / 2) - t^2
// Used when x < -1.0 (t < -0.707)
// ---------------------------------------------------------------------------
__simd_callee__ inline void LogNdtrComputeLeftTail(Reg::RegTensor<float> &resultReg, Reg::RegTensor<float> &tReg,
                                                   Reg::MaskReg &maskLeft) {
  Reg::RegTensor<float> negTReg, erfcxReg, halfErfcxReg, logReg, tSqReg;

  // neg_t = -t (positive since t < 0 in left tail)
  Reg::Neg(negTReg, tReg, maskLeft);

  // erfcx_val = erfcx(neg_t)
  ErfcxPolyLite(erfcxReg, negTReg, maskLeft);

  // log(erfcx_val / 2) = log(erfcx_val * 0.5)
  Reg::Muls(halfErfcxReg, erfcxReg, 0.5f, maskLeft);
  Reg::Log(logReg, halfErfcxReg, maskLeft);

  // result = log(erfcx/2) - t^2
  Reg::Mul(tSqReg, tReg, tReg, maskLeft);
  Reg::Sub(resultReg, logReg, tSqReg, maskLeft);
}

// ---------------------------------------------------------------------------
// Right tail computation: result = log(1 - erfc(t)/2) = log(0.5 + erf(t)/2)
// Used when x >= -1.0 (t >= -0.707)
// ---------------------------------------------------------------------------
__simd_callee__ inline void LogNdtrComputeRightTail(Reg::RegTensor<float> &resultReg, Reg::RegTensor<float> &tReg,
                                                    Reg::MaskReg &maskRight) {
  Reg::RegTensor<float> erfReg, erfcReg, argReg;

  // erf(t)
  ErfCompute(erfReg, tReg, maskRight);

  // erfc = 1.0 - erf
  Reg::RegTensor<float> oneReg;
  Reg::Duplicate(oneReg, 1.0f, maskRight);
  Reg::Sub(erfcReg, oneReg, erfReg, maskRight);

  // arg = 1.0 - erfc * 0.5
  Reg::Muls(argReg, erfcReg, -0.5f, maskRight);
  Reg::Adds(argReg, argReg, 1.0f, maskRight);

  // result = log(arg) = log1p(-erfc/2)
  Reg::Log(resultReg, argReg, maskRight);
}

// ---------------------------------------------------------------------------
// Special cases: NaN, +inf, -inf
// ---------------------------------------------------------------------------
__simd_callee__ inline void LogNdtrHandleSpecialCases(Reg::RegTensor<float> &dstReg, Reg::RegTensor<float> &srcReg,
                                                      Reg::MaskReg &mask) {
  Reg::RegTensor<float> nanReg, constReg, negZeroReg;
  Reg::MaskReg nanMask, posInfMask, negInfMask, posUnderMask;

  // NaN detection: x != x
  Reg::Compare<float, CMPMODE::NE>(nanMask, srcReg, srcReg, mask);
  // AscendC convention: float-from-bits via reference (TIK compiler lacks memcpy/bit_cast)
  Reg::Duplicate(nanReg, (float &)LOG_NDTR_NAN_UINT, mask);
  Reg::Select(dstReg, nanReg, dstReg, nanMask);

  // +inf: result = -0.0 (compute via IEEE 754: 0.0 * -1.0 = -0.0)
  Reg::Duplicate(constReg, (float &)LOG_NDTR_POS_INF_UINT, mask);
  Reg::Compare<float, CMPMODE::EQ>(posInfMask, srcReg, constReg, mask);
  Reg::Duplicate(negZeroReg, 0.0f, mask);
  Reg::Muls(negZeroReg, negZeroReg, -1.0f, mask);  // +0.0 * -1.0 → -0.0
  Reg::Select(dstReg, negZeroReg, dstReg, posInfMask);

  // -inf: result = -inf
  Reg::Duplicate(constReg, (float &)LOG_NDTR_NEG_INF_UINT, mask);
  Reg::Compare<float, CMPMODE::EQ>(negInfMask, srcReg, constReg, mask);
  Reg::Select(dstReg, constReg, dstReg, negInfMask);

  // Large positive x underflows to -0.0 (log_ndtr approaches 0 from below)
  // Detect: result is +0.0 but input > 0 → force to -0.0
  Reg::Compares<float, CMPMODE::GT>(posUnderMask, srcReg, 0.0f, mask);
  Reg::RegTensor<float> zeroReg;
  Reg::Duplicate(zeroReg, 0.0f, mask);
  Reg::MaskReg resZeroMask;
  Reg::Compare<float, CMPMODE::EQ>(resZeroMask, dstReg, zeroReg, mask);
  Reg::And(posUnderMask, posUnderMask, resZeroMask, mask);
  Reg::Select(dstReg, negZeroReg, dstReg, posUnderMask);
}

// ---------------------------------------------------------------------------
// LogNdtrCoreImpl: per-tile SIMD core loop
// ---------------------------------------------------------------------------
template <typename T>
__simd_vf__ inline void LogNdtrCoreImpl(__ubuf__ T *dstUb, __ubuf__ T *srcUb, uint32_t calCount, uint16_t repeatTimes) {
  static_assert((std::is_same_v<T, float>), "LogNdtr only supports float on current device!");
  constexpr uint32_t oneRepSize = static_cast<uint32_t>(GetVecLen() / sizeof(float));
  uint32_t sreg = calCount;
  for (uint16_t i = 0; i < repeatTimes; ++i) {
    Reg::MaskReg mask = Reg::UpdateMask<float>(sreg);
    Reg::RegTensor<float> srcReg, tReg, resultReg, resultLeft, resultRight;
    Reg::MaskReg maskLeft, maskRight;

    Reg::LoadAlign(srcReg, srcUb + i * oneRepSize);

    // t = x / sqrt(2)
    Reg::Muls(tReg, srcReg, LOG_NDTR_INV_SQRT_2, mask);

    // Branch: x < -1.0 → left tail (erfcx path), else → right tail (erfc+log1p path)
    Reg::Compares<float, CMPMODE::LT>(maskLeft, srcReg, LOG_NDTR_BRANCH_THRESHOLD, mask);
    Reg::Not(maskRight, maskLeft, mask);

    // Compute both branches
    Reg::Duplicate(resultLeft, 0.0f, mask);
    LogNdtrComputeLeftTail(resultLeft, tReg, maskLeft);

    Reg::Duplicate(resultRight, 0.0f, mask);
    LogNdtrComputeRightTail(resultRight, tReg, maskRight);

    // Merge results: select left or right based on maskLeft
    Reg::Select(resultReg, resultLeft, resultRight, maskLeft);

    // Handle special values (NaN, +/-inf)
    LogNdtrHandleSpecialCases(resultReg, srcReg, mask);

    Reg::StoreAlign(dstUb + i * oneRepSize, resultReg, mask);
  }
}

}  // namespace LOG_NDTR

// ---------------------------------------------------------------------------
// __aicore__ entry function
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void LogNdtrExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                     const LocalTensor<uint8_t> &tmpBuffer, const uint32_t calCount) {
  static_assert((std::is_same_v<T, float>), "LogNdtr only supports float on current device!");
  if ASCEND_IS_AIC {
    return;
  }

  __ubuf__ T *dstUb = (__ubuf__ T *)dst.GetPhyAddr();
  __ubuf__ T *srcUb = (__ubuf__ T *)src.GetPhyAddr();

  constexpr uint32_t oneRepSize = static_cast<uint32_t>(GetVecLen() / sizeof(T));
  uint16_t repeatTimes = CeilDivision(calCount, oneRepSize);

  LOG_NDTR::LogNdtrCoreImpl<T>(dstUb, srcUb, calCount, repeatTimes);
}

#endif  // __ASCENDC_API_REGBASE_LOG_NDTR_H__
