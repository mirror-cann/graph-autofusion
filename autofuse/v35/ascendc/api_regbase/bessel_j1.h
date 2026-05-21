/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_BESSEL_J1_H__
#define __ASCENDC_API_REGBASE_BESSEL_J1_H__

#include "bessel_j_utils.h"
#include "trigonometric_function_utils.h"

// Coefficients for small argument: 0 <= x <= 5.0
constexpr float J1_RP[4] = {
    -8.99971225705559398224e+08, +4.52228297998194034323e+11,
    -7.27494245221818276015e+13, +3.68295732863852883286e+15,
};

constexpr float J1_RQ[8] = {
    +6.20836478118054335476e+02, +2.56987256757748830383e+05,
    +8.35146791431949253037e+07, +2.21511595479792499675e+10,
    +4.74914122079991414898e+12, +7.84369607876235854894e+14,
    +8.95222336184627338078e+16, +5.32278620332680085395e+18,
};

// Coefficients for large argument: x > 5.0
constexpr float J1_PP[7] = {
    +7.62125616208173112003e-04, +7.31397056940917570436e-02,
    +1.12719608129684925192e+00, +5.11207951146807644818e+00,
    +8.42404590141772420927e+00, +5.21451598682361504063e+00,
    +1.00000000000000000254e+00,
};

constexpr float J1_PQ[7] = {
    +5.71323128072548699714e-04, +6.88455908754495404082e-02,
    +1.10514232634061696926e+00, +5.07386386128601488557e+00,
    +8.39985554327604159757e+00, +5.20982848682361821619e+00,
    +9.99999999999999997461e-01,
};

constexpr float J1_QP[8] = {
    +5.10862594750176621635e-02, +4.98213872951233449420e+00,
    +7.58238284132545283818e+01, +3.66779609360150777800e+02,
    +7.10856304998926107277e+02, +5.97489612400613639965e+02,
    +2.11688757100572135698e+02, +2.52070205858023719784e+01,
};

constexpr float J1_QQ[7] = {
    +7.42373277035675149943e+01, +1.05644886038262816351e+03,
    +4.98641058337653607651e+03, +9.56231892404756170795e+03,
    +7.99704160447350683650e+03, +2.82619278517639096600e+03,
    +3.36093607810698293419e+02,
};

constexpr float J1_ZERO1 = -1.46819706421238932572e+01;
constexpr float J1_ZERO2 = -4.92184563216946036703e+01;
constexpr float J1_PHASE = -2.356194490192344928846982537459627163;  // 3*pi/4

template <typename T>
__simd_callee__ inline void BesselJ1SmallCompute(AscendC::Reg::RegTensor<T>& absXReg,
                                                   AscendC::Reg::RegTensor<T>& xSqReg,
                                                   AscendC::Reg::RegTensor<T>& smallDstReg,
                                                   AscendC::Reg::MaskReg& branchMask) {
    AscendC::Reg::RegTensor<T> rpReg, rqReg, tmpReg;

    AscendC::Reg::Duplicate<T>(rpReg, (T)0.0, branchMask);
    AscendC::Reg::Duplicate<T>(rqReg, (T)0.0, branchMask);

    // rp = Horner(x^2, J1_RP, 4 terms)
    HornerPoly<T, 0, 3, J1_RP>(rpReg, xSqReg, branchMask);

    // rq = Horner(x^2, J1_RQ, 8 terms)
    HornerPoly<T, 0, 7, J1_RQ>(rqReg, xSqReg, branchMask);

    // result = rp / rq * |x| * (x^2 - zero1) * (x^2 - zero2)
    AscendC::Reg::Adds(tmpReg, xSqReg, (T)J1_ZERO1, branchMask);
    AscendC::Reg::Adds(smallDstReg, xSqReg, (T)J1_ZERO2, branchMask);
    AscendC::Reg::Mul(smallDstReg, smallDstReg, tmpReg, branchMask);
    AscendC::Reg::Mul(smallDstReg, smallDstReg, rpReg, branchMask);
    AscendC::Reg::Div(smallDstReg, smallDstReg, rqReg, branchMask);
    AscendC::Reg::Mul(smallDstReg, smallDstReg, absXReg, branchMask);
}

template <typename T>
__simd_callee__ inline void BesselJ1LargeCompute(AscendC::Reg::RegTensor<T>& absXReg,
                                                   AscendC::Reg::RegTensor<T>& xSqReg,
                                                   AscendC::Reg::RegTensor<T>& bigDstReg,
                                                   AscendC::Reg::MaskReg& branchMask) {
    AscendC::Reg::RegTensor<T> zReg, pReg, qReg, tmpReg;

    // z = 25.0 / x^2
    AscendC::Reg::Duplicate(zReg, (T)25.0, branchMask);
    AscendC::Reg::Div(zReg, zReg, xSqReg, branchMask);

    AscendC::Reg::Duplicate<T>(pReg, (T)0.0, branchMask);
    AscendC::Reg::Duplicate<T>(qReg, (T)0.0, branchMask);

    HornerPoly<T, 0, 6, J1_PP>(pReg, zReg, branchMask);
    HornerPoly<T, 0, 6, J1_PQ>(qReg, zReg, branchMask);
    AscendC::Reg::Div(pReg, pReg, qReg, branchMask);
    
    AscendC::Reg::Adds(zReg, absXReg, (T)J1_PHASE, branchMask);
    AutofuseCos<T>(zReg, zReg, branchMask);
    AscendC::Reg::Mul(tmpReg, pReg, zReg, branchMask);

    // z = 25.0 / x^2
    AscendC::Reg::Duplicate(zReg, (T)25.0, branchMask);
    AscendC::Reg::Div(zReg, zReg, xSqReg, branchMask);

    AscendC::Reg::Duplicate<T>(pReg, (T)0.0, branchMask);
    AscendC::Reg::Duplicate<T>(qReg, (T)0.0, branchMask);

    HornerPoly<T, 0, 7, J1_QP>(pReg, zReg, branchMask);
    HornerPoly<T, 0, 6, J1_QQ>(qReg, zReg, branchMask);
    AscendC::Reg::Div(pReg, pReg, qReg, branchMask);
    AscendC::Reg::Muls(pReg, pReg, (T)5.0, branchMask);
    AscendC::Reg::Div(pReg, pReg, absXReg, branchMask);
    
    AscendC::Reg::Adds(zReg, absXReg, (T)J1_PHASE, branchMask);
    AutofuseSin<T>(zReg, zReg, branchMask);
    AscendC::Reg::Mul(pReg, pReg, zReg, branchMask);

    // result = (pp * cos(phase) - qp * sin(phase)) * sqrt(2/pi) / sqrt(x)
    AscendC::Reg::Sub(bigDstReg, tmpReg, pReg, branchMask);
    AscendC::Reg::Sqrt(tmpReg, absXReg, branchMask);
    AscendC::Reg::Muls(bigDstReg, bigDstReg, (T)BESSEL_J_SQRT_2_OVER_PI, branchMask);
    AscendC::Reg::Div(bigDstReg, bigDstReg, tmpReg, branchMask);
}

template <typename T>
__simd_vf__ inline void BesselJ1ImplVF(__ubuf__ T* dst, __ubuf__ T* src, uint32_t calCount) {
    uint32_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(T));
    uint16_t repeatTime = static_cast<uint16_t>(AscendC::CeilDivision(calCount, vlSize));

    AscendC::Reg::RegTensor<T> srcReg, absXReg, xSqReg, smallDstReg, bigDstReg, dstReg, nanReg, negReg;
    AscendC::Reg::MaskReg mask, branchMask;

    for (uint16_t i = 0U; i < repeatTime; ++i) {
        mask = AscendC::Reg::UpdateMask<T>(calCount);
        AscendC::Reg::LoadAlign(srcReg, src + i * vlSize);
        AscendC::Reg::Abs(absXReg, srcReg, mask);

        // x^2
        AscendC::Reg::Mul(xSqReg, absXReg, absXReg, mask);

        // ===== Small branch: |x| <= 5.0 =====
        AscendC::Reg::Compares<T, CMPMODE::LE>(branchMask, absXReg, (T)5.0, mask);
        BesselJ1SmallCompute<T>(absXReg, xSqReg, smallDstReg, branchMask);

        // ===== Large branch: |x| > 5.0 =====
        AscendC::Reg::Compares<T, CMPMODE::GT>(branchMask, absXReg, (T)5.0, mask);
        BesselJ1LargeCompute<T>(absXReg, xSqReg, bigDstReg, branchMask);

        // Merge: select small for |x| <= 5.0
        AscendC::Reg::Select(dstReg, bigDstReg, smallDstReg, branchMask);

        // J1 is odd: negate for x < 0
        AscendC::Reg::Compares<T, CMPMODE::LT>(branchMask, srcReg, (T)0.0, mask);
        AscendC::Reg::Muls(negReg, dstReg, (T)(-1.0), branchMask);
        AscendC::Reg::Select(dstReg, negReg, dstReg, branchMask);

        // Handle NaN input
        AscendC::Reg::Compare<T, CMPMODE::NE>(branchMask, srcReg, srcReg, mask);
        AscendC::Reg::Duplicate(nanReg, (float&)BESSEL_FLOAT_NAN, mask);
        AscendC::Reg::Select(dstReg, nanReg, dstReg, branchMask);

        // Store output
        AscendC::Reg::StoreAlign(dst + i * vlSize, dstReg, mask);
    }
}

template <typename T>
__aicore__ inline void BesselJ1Extend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                       const LocalTensor<uint8_t>& sharedTmpBuffer,
                                       const uint32_t calCount) {
    static_assert(SupportType<T, float>(), "Current data type is not supported on current device!");
    BesselJ1ImplVF<T>((__ubuf__ T*)dst.GetPhyAddr(),
                        (__ubuf__ T*)src.GetPhyAddr(),
                        calCount);
}

#endif  // __ASCENDC_API_REGBASE_BESSEL_J1_H__
