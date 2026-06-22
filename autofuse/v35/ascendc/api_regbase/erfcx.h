/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_ERFCX_H__
#define __ASCENDC_API_REGBASE_ERFCX_H__

using namespace AscendC;

namespace ERFCX {

constexpr float K_THRESHOLD = 10.055f;
constexpr float K_INV_SQRT_PI = 0.5641896f;
constexpr float K_FLOAT_MAX = 0x1.fffffep+127f;

// Low-path polynomial p(t) coefficients (degree 10)
constexpr float P_COEFF_0 = 0.0008912171f;
constexpr float P_COEFF_1 = 0.007045788f;
constexpr float P_COEFF_2 = -0.0158668961f;
constexpr float P_COEFF_3 = 0.036429625f;
constexpr float P_COEFF_4 = -0.06664343f;
constexpr float P_COEFF_5 = 0.09381453f;
constexpr float P_COEFF_6 = -0.100990564f;
constexpr float P_COEFF_7 = 0.068094f;
constexpr float P_COEFF_8 = 0.0153773874f;
constexpr float P_COEFF_9 = -0.139621079f;
constexpr float P_COEFF_10 = 1.23299515f;

// High-path asymptotic polynomial coefficients (degree 4)
constexpr float A_COEFF_0 = 6.5625f;
constexpr float A_COEFF_1 = -1.875f;
constexpr float A_COEFF_2 = 0.75f;
constexpr float A_COEFF_3 = -0.5f;
constexpr float A_COEFF_4 = 1.0f;

constexpr uint32_t U32_POS_INF = 0x7F800000u;

// ---------------------------------------------------------------------------
// __simd_callee__ helpers
// ---------------------------------------------------------------------------

__simd_callee__ inline void ErfcxPolyP(Reg::RegTensor<float> &pReg, Reg::RegTensor<float> &tReg, Reg::MaskReg &mask) {
  Reg::RegTensor<float> tmpReg;
  Reg::Duplicate(pReg, P_COEFF_0, mask);
  Reg::Duplicate(tmpReg, P_COEFF_1, mask);
  Reg::FusedMulDstAdd(pReg, tReg, tmpReg, mask);
  Reg::Duplicate(tmpReg, P_COEFF_2, mask);
  Reg::FusedMulDstAdd(pReg, tReg, tmpReg, mask);
  Reg::Duplicate(tmpReg, P_COEFF_3, mask);
  Reg::FusedMulDstAdd(pReg, tReg, tmpReg, mask);
  Reg::Duplicate(tmpReg, P_COEFF_4, mask);
  Reg::FusedMulDstAdd(pReg, tReg, tmpReg, mask);
  Reg::Duplicate(tmpReg, P_COEFF_5, mask);
  Reg::FusedMulDstAdd(pReg, tReg, tmpReg, mask);
  Reg::Duplicate(tmpReg, P_COEFF_6, mask);
  Reg::FusedMulDstAdd(pReg, tReg, tmpReg, mask);
  Reg::Duplicate(tmpReg, P_COEFF_7, mask);
  Reg::FusedMulDstAdd(pReg, tReg, tmpReg, mask);
  Reg::Duplicate(tmpReg, P_COEFF_8, mask);
  Reg::FusedMulDstAdd(pReg, tReg, tmpReg, mask);
  Reg::Duplicate(tmpReg, P_COEFF_9, mask);
  Reg::FusedMulDstAdd(pReg, tReg, tmpReg, mask);
  Reg::Duplicate(tmpReg, P_COEFF_10, mask);
  Reg::FusedMulDstAdd(pReg, tReg, tmpReg, mask);
}

__simd_callee__ inline void ErfcxPolyA(Reg::RegTensor<float> &polyReg, Reg::RegTensor<float> &zReg,
                                       Reg::MaskReg &mask) {
  Reg::RegTensor<float> tmpReg;
  Reg::Duplicate(polyReg, A_COEFF_0, mask);
  Reg::Duplicate(tmpReg, A_COEFF_1, mask);
  Reg::FusedMulDstAdd(polyReg, zReg, tmpReg, mask);
  Reg::Duplicate(tmpReg, A_COEFF_2, mask);
  Reg::FusedMulDstAdd(polyReg, zReg, tmpReg, mask);
  Reg::Duplicate(tmpReg, A_COEFF_3, mask);
  Reg::FusedMulDstAdd(polyReg, zReg, tmpReg, mask);
  Reg::Duplicate(tmpReg, A_COEFF_4, mask);
  Reg::FusedMulDstAdd(polyReg, zReg, tmpReg, mask);
}

// ---------------------------------------------------------------------------
// __simd_vf__ core per-tile implementation
// ---------------------------------------------------------------------------

template <typename T>
__simd_vf__ inline void ErfcxCoreImpl(__ubuf__ T *dstUb, __ubuf__ T *srcUb, uint32_t calCount, uint16_t repeatTimes) {
  static_assert((std::is_same_v<T, float>), "Erfcx only supports float on current device!");
  constexpr uint32_t oneRepSize = static_cast<uint32_t>(GetVecLen() / sizeof(float));

  for (uint16_t i = 0; i < repeatTimes; ++i) {
    uint32_t sreg = calCount;
    Reg::MaskReg mask = Reg::UpdateMask<float>(sreg);
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    Reg::RegTensor<float> srcReg, axReg, resultReg;
    Reg::RegTensor<float> tmpReg, tmp2Reg, tmpFmaReg;
    Reg::MaskReg maskLo, maskHi;
    Reg::MaskReg maskNeg, maskNegNoInf, maskInf;

    Reg::LoadAlign(srcReg, srcUb + i * oneRepSize);

    Reg::Abs(axReg, srcReg, mask);

    // ---- ax < kThreshold: rational approximation ----
    Reg::Compares<float, CMPMODE::LT>(maskLo, axReg, K_THRESHOLD, mask);

    // inv_ax_plus_4 = 1 / (ax + 4)
    Reg::RegTensor<float> invAxP4Reg;
    Reg::Adds(tmpReg, axReg, 4.0f, maskLo);
    Reg::Duplicate(invAxP4Reg, 1.0f, maskLo);
    Reg::Div(invAxP4Reg, invAxP4Reg, tmpReg, maskLo);

    // t = (ax - 4) * inv_ax_plus_4
    Reg::RegTensor<float> tReg;
    Reg::Adds(tReg, axReg, -4.0f, maskLo);
    Reg::Mul(tReg, tReg, invAxP4Reg, maskLo);

    // residual_t = fmaf(-t, ax, fmaf(-4, t+1, ax))
    //            = ax - 4(t+1)  +  (-t)*ax  =  ax - 4t - 4 - t*ax
    Reg::Adds(tmpReg, tReg, 1.0f, maskLo);
    Reg::Muls(tmpReg, tmpReg, -4.0f, maskLo);
    Reg::Add(tmpReg, tmpReg, axReg, maskLo);
    Reg::Neg(tmp2Reg, tReg, maskLo);
    Reg::Mul(tmpFmaReg, tmp2Reg, axReg, maskLo);
    Reg::Add(tmpReg, tmpReg, tmpFmaReg, maskLo);

    // t = fmaf(inv_ax_plus_4, residual_t, t)
    Reg::Mul(tmpFmaReg, invAxP4Reg, tmpReg, maskLo);
    Reg::Add(tReg, tReg, tmpFmaReg, maskLo);

    // p(t)  polynomial evaluation
    Reg::RegTensor<float> pReg;
    ErfcxPolyP(pReg, tReg, maskLo);

    // inv_denom = 1 / (2*ax + 1),  q = p * inv_denom
    Reg::RegTensor<float> invDenomReg, qReg;
    Reg::Muls(tmpReg, axReg, 2.0f, maskLo);
    Reg::Adds(tmpReg, tmpReg, 1.0f, maskLo);
    Reg::Duplicate(invDenomReg, 1.0f, maskLo);
    Reg::Div(invDenomReg, invDenomReg, tmpReg, maskLo);
    Reg::Mul(qReg, pReg, invDenomReg, maskLo);

    // residual_q = fmaf(ax, -2*q, p) - q  =  p - 2*ax*q - q
    Reg::Muls(tmpReg, qReg, -2.0f, maskLo);
    Reg::Mul(tmpReg, tmpReg, axReg, maskLo);
    Reg::Add(tmpReg, tmpReg, pReg, maskLo);
    Reg::Sub(tmpReg, tmpReg, qReg, maskLo);

    // result = fmaf(residual_q, inv_denom, q)
    Reg::Mul(tmpFmaReg, tmpReg, invDenomReg, maskLo);
    Reg::Add(qReg, qReg, tmpFmaReg, maskLo);

    // Save low-path result into resultReg
    Reg::Duplicate(resultReg, 0.0f, fullMask);
    Reg::Copy<float, Reg::MaskMergeMode::MERGING>(resultReg, qReg, maskLo);

    // ---- ax >= kThreshold: asymptotic expansion ----
    Reg::Not(maskHi, maskLo, fullMask);

    // inv_x = 1 / ax
    Reg::RegTensor<float> invXReg, zReg, polyAReg;
    Reg::Duplicate(invXReg, 1.0f, maskHi);
    Reg::Div(invXReg, invXReg, axReg, maskHi);

    // z = inv_x * inv_x
    Reg::Mul(zReg, invXReg, invXReg, maskHi);

    // poly(z)  polynomial evaluation
    ErfcxPolyA(polyAReg, zReg, maskHi);

    // result = kInvSqrtPi * inv_x * poly
    Reg::Mul(polyAReg, polyAReg, invXReg, maskHi);
    Reg::Muls(polyAReg, polyAReg, K_INV_SQRT_PI, maskHi);

    // Merge high-path result (MERGING keeps low-path elements intact)
    Reg::Copy<float, Reg::MaskMergeMode::MERGING>(resultReg, polyAReg, maskHi);

    // ---- Post-branch: x < 0 handling ----
    Reg::Compares<float, CMPMODE::LT>(maskNeg, srcReg, 0.0f, mask);

    // axSq = ax * ax,  exp_term = exp(axSq)
    Reg::RegTensor<float> axSqReg, expReg;
    Reg::Mul(axSqReg, axReg, axReg, maskNeg);
    Reg::Exp(expReg, axSqReg, maskNeg);

    // Check isinf(exp_term): compare with K_FLOAT_MAX to detect overflow
    Reg::Compares<float, CMPMODE::GT>(maskInf, expReg, K_FLOAT_MAX, maskNeg);

    // maskInf: result = +inf
    Reg::RegTensor<float> infReg;
    Reg::Duplicate(infReg, (float &)F32_INF, fullMask);
    Reg::Copy<float, Reg::MaskMergeMode::MERGING>(resultReg, infReg, maskInf);

    // maskNeg & !maskInf: result = 2 * exp_term - result
    Reg::Not(maskNegNoInf, maskInf, fullMask);
    Reg::And(maskNegNoInf, maskNegNoInf, maskNeg, fullMask);
    Reg::Muls(expReg, expReg, 2.0f, maskNegNoInf);
    Reg::Sub(expReg, expReg, resultReg, maskNegNoInf);
    Reg::Copy<float, Reg::MaskMergeMode::MERGING>(resultReg, expReg, maskNegNoInf);

    Reg::StoreAlign(dstUb + i * oneRepSize, resultReg, mask);
  }
}

}  // namespace ERFCX

// ---------------------------------------------------------------------------
// __aicore__ entry function
// ---------------------------------------------------------------------------

template <typename T>
__aicore__ inline void ErfcxExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                   AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t calc_cnt) {
  static_assert((std::is_same_v<T, float>), "Erfcx only supports float on current device!");
  __ubuf__ T *dstUb = (__ubuf__ T *)dst.GetPhyAddr();
  __ubuf__ T *srcUb = (__ubuf__ T *)src.GetPhyAddr();
  uint16_t repeatTimes = CeilDivision(calc_cnt, static_cast<uint32_t>(GetVecLen() / sizeof(float)));
  ERFCX::ErfcxCoreImpl<T>(dstUb, srcUb, calc_cnt, repeatTimes);
}
#endif
