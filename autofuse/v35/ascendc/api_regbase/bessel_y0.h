/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_BESSEL_Y0_H__
#define __ASCENDC_API_REGBASE_BESSEL_Y0_H__

#include "bessel_j0.h"
#include "bessel_j_utils.h"
#include "trigonometric_function_utils.h"
// Coefficients for small argument: 0 < x <= 5.0 (from PyTorch Math.cuh)
constexpr float Y0_YP[8] = {
    +1.55924367855235737965e+04, -1.46639295903971606143e+07, +5.43526477051876500413e+09, -9.82136065717911466409e+11,
    +8.75906394395366999549e+13, -3.46628303384729719441e+15, +4.42733268572569800351e+16, -1.84950800436986690637e+16,
};

constexpr float Y0_YQ[7] = {
    +1.04128353664259848412e+03, +6.26107330137134956842e+05, +2.68919633393814121987e+08, +8.64002487103935000337e+10,
    +2.02979612750105546709e+13, +3.17157752842975028269e+15, +2.50596256172653059228e+17,
};

constexpr float Y0_PHASE = -0.785398163397448309615660845819875721;  // -pi/4
constexpr float Y0_TWO_OVER_PI = 0.636619772367581343075535053490057448;
// 设计文档中安全钳位设置为1e-10
// 1e-10 将 B4 分支（1e-15~1e-10）的所有微小值钳位为同一值，导致 log(x) 全部输出 log(1e-10)=-23.02
// 改为 1e-38（float32最小正正规值附近），允许微小正值正常计算
constexpr float Y0_SAFE_MIN = 1e-38;
constexpr uint32_t BESSEL_Y0_NEG_INF = 0xff800000;

template <typename T>
__simd_callee__ inline void BesselY0SmallCompute(AscendC::Reg::RegTensor<T> &validXReg,
                                                 AscendC::Reg::RegTensor<T> &xSqReg,
                                                 AscendC::Reg::RegTensor<T> &smallDstReg,
                                                 AscendC::Reg::MaskReg &branchMask, __ubuf__ T *dst, uint32_t offSet) {
  AscendC::Reg::RegTensor<T> pReg, qReg, lnReg, j0Reg;

  AscendC::Reg::Duplicate<T>(pReg, (T)0.0, branchMask);
  AscendC::Reg::Duplicate<T>(qReg, (T)0.0, branchMask);

  HornerPoly<T, 0, 7, Y0_YP>(pReg, xSqReg, branchMask);

  HornerPoly<T, 0, 6, Y0_YQ>(qReg, xSqReg, branchMask);

  AscendC::Reg::Div(pReg, pReg, qReg, branchMask);

  AscendC::Reg::LoadAlign(j0Reg, dst + offSet);

  AscendC::Reg::Log(lnReg, validXReg, branchMask);

  AscendC::Reg::Muls(lnReg, lnReg, (T)Y0_TWO_OVER_PI, branchMask);
  AscendC::Reg::Mul(lnReg, lnReg, j0Reg, branchMask);

  AscendC::Reg::Add(smallDstReg, pReg, lnReg, branchMask);
}

template <typename T>
__simd_callee__ inline void BesselY0LargeCompute(AscendC::Reg::RegTensor<T> &validXReg,
                                                 AscendC::Reg::RegTensor<T> &xSqReg,
                                                 AscendC::Reg::RegTensor<T> &bigDstReg,
                                                 AscendC::Reg::MaskReg &branchMask) {
  AscendC::Reg::RegTensor<T> zReg, pReg, qReg, p2Reg, q2Reg, tmpReg, sinReg, cosReg, phaseReg;

  // z = 25.0 / x^2
  AscendC::Reg::Duplicate(zReg, (T)25.0, branchMask);
  AscendC::Reg::Div(zReg, zReg, xSqReg, branchMask);

  AscendC::Reg::Duplicate<T>(pReg, (T)0.0, branchMask);
  AscendC::Reg::Duplicate<T>(qReg, (T)0.0, branchMask);

  HornerPoly<T, 0, 6, J0_PP>(pReg, zReg, branchMask);
  HornerPoly<T, 0, 6, J0_PQ>(qReg, zReg, branchMask);
  AscendC::Reg::Div(pReg, pReg, qReg, branchMask);

  AscendC::Reg::Adds(phaseReg, validXReg, (T)Y0_PHASE, branchMask);

  AutofuseSin<T>(sinReg, phaseReg, branchMask);

  AscendC::Reg::Mul(tmpReg, pReg, sinReg, branchMask);

  // z = 25.0 / x^2
  AscendC::Reg::Duplicate(zReg, (T)25.0, branchMask);
  AscendC::Reg::Div(zReg, zReg, xSqReg, branchMask);

  AscendC::Reg::Duplicate<T>(p2Reg, (T)0.0, branchMask);
  AscendC::Reg::Duplicate<T>(q2Reg, (T)0.0, branchMask);

  HornerPoly<T, 0, 7, J0_QP>(p2Reg, zReg, branchMask);
  HornerPoly<T, 0, 6, J0_QQ>(q2Reg, zReg, branchMask);
  AscendC::Reg::Div(p2Reg, p2Reg, q2Reg, branchMask);
  AscendC::Reg::Muls(p2Reg, p2Reg, (T)5.0, branchMask);
  AscendC::Reg::Div(p2Reg, p2Reg, validXReg, branchMask);

  AutofuseCos<T>(cosReg, phaseReg, branchMask);

  AscendC::Reg::Mul(p2Reg, p2Reg, cosReg, branchMask);

  // result = (R1 * sin(T) + R2 * (5/x) * cos(T)) * sqrt(2/pi) / sqrt(x)
  AscendC::Reg::Add(bigDstReg, tmpReg, p2Reg, branchMask);

  AscendC::Reg::Sqrt(tmpReg, validXReg, branchMask);

  AscendC::Reg::Muls(bigDstReg, bigDstReg, (T)BESSEL_J_SQRT_2_OVER_PI, branchMask);
  AscendC::Reg::Div(bigDstReg, bigDstReg, tmpReg, branchMask);
}

template <typename T>
__simd_vf__ inline void BesselY0ImplVF(__ubuf__ T *dst, __ubuf__ T *src, uint32_t calCount) {
  uint32_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(T));
  uint16_t repeatTime = static_cast<uint16_t>(AscendC::CeilDivision(calCount, vlSize));

  AscendC::Reg::RegTensor<T> srcReg, validXReg, xSqReg, smallDstReg, bigDstReg, dstReg;
  AscendC::Reg::RegTensor<T> nanReg, ninfReg;
  AscendC::Reg::MaskReg mask, branchMask, maskNeg, maskZero, nanMask;

  for (uint16_t i = 0U; i < repeatTime; ++i) {
    mask = AscendC::Reg::UpdateMask<T>(calCount);
    AscendC::Reg::LoadAlign(srcReg, src + i * vlSize);

    AscendC::Reg::Compares<T, CMPMODE::LT>(maskNeg, srcReg, (T)0.0, mask);
    AscendC::Reg::Compares<T, CMPMODE::EQ>(maskZero, srcReg, (T)0.0, mask);
    AscendC::Reg::Compare<T, CMPMODE::NE>(nanMask, srcReg, srcReg, mask);

    AscendC::Reg::Maxs(validXReg, srcReg, (T)Y0_SAFE_MIN, mask);

    AscendC::Reg::Mul(xSqReg, validXReg, validXReg, mask);

    // ===== Small branch: validX <= 5.0 =====
    AscendC::Reg::Compares<T, CMPMODE::LE>(branchMask, validXReg, (T)5.0, mask);
    BesselY0SmallCompute<T>(validXReg, xSqReg, smallDstReg, branchMask, dst, i * vlSize);

    // ===== Large branch: validX > 5.0 =====
    AscendC::Reg::Compares<T, CMPMODE::GT>(branchMask, validXReg, (T)5.0, mask);
    BesselY0LargeCompute<T>(validXReg, xSqReg, bigDstReg, branchMask);

    // Merge
    AscendC::Reg::Select(dstReg, bigDstReg, smallDstReg, branchMask);

    // Handle exceptions: x < 0 -> NaN
    AscendC::Reg::Duplicate(nanReg, (float &)BESSEL_FLOAT_NAN, mask);
    AscendC::Reg::Select(dstReg, nanReg, dstReg, maskNeg);

    // Handle NaN input -> NaN
    AscendC::Reg::Select(dstReg, nanReg, dstReg, nanMask);

    // Handle x == 0 -> -Inf
    AscendC::Reg::Duplicate(ninfReg, (float &)BESSEL_Y0_NEG_INF, mask);
    AscendC::Reg::Select(dstReg, ninfReg, dstReg, maskZero);

    // Store output
    AscendC::Reg::StoreAlign(dst + i * vlSize, dstReg, mask);
  }
}

template <typename T>
__aicore__ inline void BesselY0Extend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                      const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount) {
  static_assert(SupportType<T, float>(), "Current data type is not supported on current device!");
  // BesselJ0Extend 对应 gendata.py bessel_j0_impl（J0 Step0~F），结果写入 dst
  BesselJ0Extend<T>(dst, src, sharedTmpBuffer, calCount);
  BesselY0ImplVF<T>((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), calCount);
}

#endif  // __ASCENDC_API_REGBASE_BESSEL_Y0_H__
