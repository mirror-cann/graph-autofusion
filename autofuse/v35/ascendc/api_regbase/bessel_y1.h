/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_BESSEL_Y1_H__
#define __ASCENDC_API_REGBASE_BESSEL_Y1_H__

#include "bessel_j1.h"
#include "bessel_j_utils.h"
#include "trigonometric_function_utils.h"

// Coefficients for small argument: 0 < x <= 5.0 (from PyTorch aten/src/ATen/native/cuda/Math.cuh bessel_y1)
constexpr float Y1_YP[6] = {
    +1.26320474790178026440e+09, -6.47355876379160291031e+11,
    +1.14509511541823727583e+14, -8.12770255501325109621e+15,
    +2.02439475713594898196e+17, -7.78877196265950026825e+17,
};

constexpr float Y1_YQ[8] = {
    +5.94301592346128195359e+02, +2.35564092943068577943e+05,
    +7.34811944459721705660e+07, +1.87601316108706159478e+10,
    +3.88231277496238566008e+12, +6.20557727146953693363e+14,
    +6.87141087355300489866e+16, +3.97270608116560655612e+18,
};

constexpr float Y1_TWO_OVER_PI = 0.636619772367581343075535053490057448;
// 1e-38 to allow subnormal/tiny positive values for log to behave, matching y0 design choice
constexpr float Y1_SAFE_MIN = 1e-38;
constexpr uint32_t BESSEL_Y1_NEG_INF = 0xff800000;

template <typename T>
__simd_callee__ inline void BesselY1SmallCompute(AscendC::Reg::RegTensor<T>& validXReg,
                                                   AscendC::Reg::RegTensor<T>& xSqReg,
                                                   AscendC::Reg::RegTensor<T>& smallDstReg,
                                                   AscendC::Reg::MaskReg& branchMask,
                                                   __ubuf__ T* dst, uint32_t offSet) {
    AscendC::Reg::RegTensor<T> pReg, qReg, lnReg, j1Reg, recReg, onesReg;

    AscendC::Reg::Duplicate<T>(pReg, (T)0.0, branchMask);
    AscendC::Reg::Duplicate<T>(qReg, (T)0.0, branchMask);
    AscendC::Reg::Duplicate<T>(onesReg, (T)1.0, branchMask);

    HornerPoly<T, 0, 5, Y1_YP>(pReg, xSqReg, branchMask);

    HornerPoly<T, 0, 7, Y1_YQ>(qReg, xSqReg, branchMask);

    AscendC::Reg::Div(pReg, pReg, qReg, branchMask);

    AscendC::Reg::LoadAlign(j1Reg, dst + offSet);

    AscendC::Reg::Log(lnReg, validXReg, branchMask);

    AscendC::Reg::Muls(lnReg, lnReg, (T)Y1_TWO_OVER_PI, branchMask);
    AscendC::Reg::Mul(lnReg, lnReg, j1Reg, branchMask);

    AscendC::Reg::Div(recReg, onesReg, validXReg, branchMask);
    AscendC::Reg::Muls(recReg, recReg, (T)Y1_TWO_OVER_PI, branchMask);
    AscendC::Reg::Sub(lnReg, lnReg, recReg, branchMask);

    AscendC::Reg::Mul(pReg, pReg, validXReg, branchMask);
    AscendC::Reg::Add(smallDstReg, pReg, lnReg, branchMask);
}

template <typename T>
__simd_callee__ inline void BesselY1LargeCompute(AscendC::Reg::RegTensor<T>& validXReg,
                                                   AscendC::Reg::RegTensor<T>& xSqReg,
                                                   AscendC::Reg::RegTensor<T>& bigDstReg,
                                                   AscendC::Reg::MaskReg& branchMask) {
    AscendC::Reg::RegTensor<T> zReg, pReg, qReg, p2Reg, q2Reg, tmpReg, sinReg, cosReg, phaseReg;

    // z = 25.0 / x^2
    AscendC::Reg::Duplicate(zReg, (T)25.0, branchMask);
    AscendC::Reg::Div(zReg, zReg, xSqReg, branchMask);

    AscendC::Reg::Duplicate<T>(pReg, (T)0.0, branchMask);
    AscendC::Reg::Duplicate<T>(qReg, (T)0.0, branchMask);

    HornerPoly<T, 0, 6, J1_PP>(pReg, zReg, branchMask);
    HornerPoly<T, 0, 6, J1_PQ>(qReg, zReg, branchMask);
    AscendC::Reg::Div(pReg, pReg, qReg, branchMask);

    AscendC::Reg::Adds(phaseReg, validXReg, (T)J1_PHASE, branchMask);

    AutofuseSin<T>(sinReg, phaseReg, branchMask);

    AscendC::Reg::Mul(tmpReg, pReg, sinReg, branchMask);

    // z = 25.0 / x^2 (recompute as in y0 style)
    AscendC::Reg::Duplicate(zReg, (T)25.0, branchMask);
    AscendC::Reg::Div(zReg, zReg, xSqReg, branchMask);

    AscendC::Reg::Duplicate<T>(p2Reg, (T)0.0, branchMask);
    AscendC::Reg::Duplicate<T>(q2Reg, (T)0.0, branchMask);

    HornerPoly<T, 0, 7, J1_QP>(p2Reg, zReg, branchMask);
    HornerPoly<T, 0, 6, J1_QQ>(q2Reg, zReg, branchMask);
    AscendC::Reg::Div(p2Reg, p2Reg, q2Reg, branchMask);
    AscendC::Reg::Muls(p2Reg, p2Reg, (T)5.0, branchMask);
    AscendC::Reg::Div(p2Reg, p2Reg, validXReg, branchMask);

    AutofuseCos<T>(cosReg, phaseReg, branchMask);

    AscendC::Reg::Mul(p2Reg, p2Reg, cosReg, branchMask);

    // result = (R3 * sin(T) + R4 * (5.0/x) * cos(T)) * sqrt(2/pi) / sqrt(x)
    AscendC::Reg::Add(bigDstReg, tmpReg, p2Reg, branchMask);

    AscendC::Reg::Sqrt(tmpReg, validXReg, branchMask);

    AscendC::Reg::Muls(bigDstReg, bigDstReg, (T)BESSEL_J_SQRT_2_OVER_PI, branchMask);
    AscendC::Reg::Div(bigDstReg, bigDstReg, tmpReg, branchMask);
}

template <typename T>
__simd_vf__ inline void BesselY1ImplVF(__ubuf__ T* dst, __ubuf__ T* src, uint32_t calCount) {
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

        AscendC::Reg::Maxs(validXReg, srcReg, (T)Y1_SAFE_MIN, mask);

        AscendC::Reg::Mul(xSqReg, validXReg, validXReg, mask);

        // ===== Small branch: validX <= 5.0 =====
        AscendC::Reg::Compares<T, CMPMODE::LE>(branchMask, validXReg, (T)5.0, mask);
        BesselY1SmallCompute<T>(validXReg, xSqReg, smallDstReg, branchMask, dst, i * vlSize);

        // ===== Large branch: validX > 5.0 =====
        AscendC::Reg::Compares<T, CMPMODE::GT>(branchMask, validXReg, (T)5.0, mask);
        BesselY1LargeCompute<T>(validXReg, xSqReg, bigDstReg, branchMask);

        // Merge
        AscendC::Reg::Select(dstReg, bigDstReg, smallDstReg, branchMask);

        // Handle exceptions: x < 0 -> NaN
        AscendC::Reg::Duplicate(nanReg, (float&)BESSEL_FLOAT_NAN, mask);
        AscendC::Reg::Select(dstReg, nanReg, dstReg, maskNeg);

        // Handle NaN input -> NaN
        AscendC::Reg::Select(dstReg, nanReg, dstReg, nanMask);

        // Handle x == 0 -> -Inf
        AscendC::Reg::Duplicate(ninfReg, (float&)BESSEL_Y1_NEG_INF, mask);
        AscendC::Reg::Select(dstReg, ninfReg, dstReg, maskZero);

        // Store output
        AscendC::Reg::StoreAlign(dst + i * vlSize, dstReg, mask);
    }
}

template <typename T>
__aicore__ inline void BesselY1Extend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                       const LocalTensor<uint8_t>& sharedTmpBuffer,
                                       const uint32_t calCount) {
    static_assert(SupportType<T, float>(), "Current data type is not supported on current device!");
    // BesselJ1Extend populates J1 values into dst for Y1 small region reuse (same pattern as Y0)
    BesselJ1Extend<T>(dst, src, sharedTmpBuffer, calCount);
    BesselY1ImplVF<T>((__ubuf__ T*)dst.GetPhyAddr(),
                       (__ubuf__ T*)src.GetPhyAddr(),
                       calCount);
}

#endif  // __ASCENDC_API_REGBASE_BESSEL_Y1_H__
