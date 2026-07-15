/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_BESSEL_J0_H__
#define __ASCENDC_API_REGBASE_BESSEL_J0_H__

// Coefficients for small argument: |x| <= 5.0
constexpr float J0_RP[4] = {
    -4.79443220978201773821e+09,
    +1.95617491946556577543e+12,
    -2.49248344360967716204e+14,
    +9.70862251047306323952e+15,
};

constexpr float J0_RQ[8] = {
    +4.99563147152651017219e+02, +1.73785401676374683123e+05, +4.84409658339962045305e+07, +1.11855537045356834862e+10,
    +2.11277520115489217587e+12, +3.10518229857422583814e+14, +3.18121955943204943306e+16, +1.71086294081043136091e+18,
};

// Coefficients for large argument: |x| > 5.0
constexpr float J0_PP[7] = {
    +7.96936729297347051624e-04, +8.28352392107440799803e-02, +1.23953371646414299388e+00, +5.44725003058768775090e+00,
    +8.74716500199817011941e+00, +5.30324038235394892183e+00, +9.99999999999999997821e-01,
};

constexpr float J0_PQ[7] = {
    +9.24408810558863637013e-04, +8.56288474354474431428e-02, +1.25352743901058953537e+00, +5.47097740330417105182e+00,
    +8.76190883237069594232e+00, +5.30605288235394617618e+00, +1.00000000000000000218e+00,
};

constexpr float J0_QP[8] = {
    -1.13663838898469149931e-02, -1.28252718670509318512e+00, -1.95539544257735972385e+01, -9.32060152123768231369e+01,
    -1.77681167980488050595e+02, -1.47077505154951170175e+02, -5.14105326766599330220e+01, -6.05014350600728481186e+00,
};

constexpr float J0_QQ[7] = {
    +6.43178256118178023184e+01, +8.56430025976980587198e+02, +3.88240183605401609683e+03, +7.24046774195652478189e+03,
    +5.93072701187316984827e+03, +2.06209331660327847417e+03, +2.42005740240291393179e+02,
};

constexpr float J0_ZERO1 = -5.78318596294678452118e+00;
constexpr float J0_ZERO2 = -3.04712623436620863991e+01;
constexpr float J0_PHASE = -0.785398163397448309615660845819875721;  // pi/4

template <typename T>
__simd_callee__ inline void BesselJ0SmallCompute(AscendC::Reg::RegTensor<T> &xSqReg,
                                                 AscendC::Reg::RegTensor<T> &smallDstReg,
                                                 AscendC::Reg::MaskReg &branchMask) {
  AscendC::Reg::RegTensor<T> rpReg, rqReg, tmpReg;

  AscendC::Reg::Duplicate<T>(rpReg, (T)0.0, branchMask);
  AscendC::Reg::Duplicate<T>(rqReg, (T)0.0, branchMask);

  // rp = Horner(x^2, J0_RP, 4 terms)
  HornerPoly<T, 0, 3, J0_RP>(rpReg, xSqReg, branchMask);
  // rq = Horner(x^2, J0_RQ, 8 terms)
  HornerPoly<T, 0, 7, J0_RQ>(rqReg, xSqReg, branchMask);

  // result = (x^2 - zero1) * (x^2 - zero2) * rp / rq
  AscendC::Reg::Adds(tmpReg, xSqReg, (T)J0_ZERO1, branchMask);
  AscendC::Reg::Adds(smallDstReg, xSqReg, (T)J0_ZERO2, branchMask);
  AscendC::Reg::Mul(smallDstReg, smallDstReg, tmpReg, branchMask);
  AscendC::Reg::Mul(smallDstReg, smallDstReg, rpReg, branchMask);
  AscendC::Reg::Div(smallDstReg, smallDstReg, rqReg, branchMask);
}

template <typename T>
__simd_callee__ inline void BesselJ0LargeCompute(AscendC::Reg::RegTensor<T> &absXReg,
                                                 AscendC::Reg::RegTensor<T> &xSqReg,
                                                 AscendC::Reg::RegTensor<T> &bigDstReg,
                                                 AscendC::Reg::MaskReg &branchMask) {
  AscendC::Reg::RegTensor<T> zReg, pReg, qReg, tmpReg;

  // z = 25.0 / x^2
  AscendC::Reg::Duplicate(zReg, (T)25.0, branchMask);
  AscendC::Reg::Div(zReg, zReg, xSqReg, branchMask);

  AscendC::Reg::Duplicate<T>(pReg, (T)0.0, branchMask);
  AscendC::Reg::Duplicate<T>(qReg, (T)0.0, branchMask);

  HornerPoly<T, 0, 6, J0_PP>(pReg, zReg, branchMask);
  HornerPoly<T, 0, 6, J0_PQ>(qReg, zReg, branchMask);
  AscendC::Reg::Div(pReg, pReg, qReg, branchMask);

  AscendC::Reg::Adds(zReg, absXReg, (T)J0_PHASE, branchMask);
  AutofuseCos<T>(zReg, zReg, branchMask);
  AscendC::Reg::Mul(tmpReg, pReg, zReg, branchMask);

  // z = 25.0 / x^2
  AscendC::Reg::Duplicate(zReg, (T)25.0, branchMask);
  AscendC::Reg::Div(zReg, zReg, xSqReg, branchMask);

  AscendC::Reg::Duplicate<T>(pReg, (T)0.0, branchMask);
  AscendC::Reg::Duplicate<T>(qReg, (T)0.0, branchMask);

  HornerPoly<T, 0, 7, J0_QP>(pReg, zReg, branchMask);
  HornerPoly<T, 0, 6, J0_QQ>(qReg, zReg, branchMask);
  AscendC::Reg::Div(pReg, pReg, qReg, branchMask);
  AscendC::Reg::Muls(pReg, pReg, (T)5.0, branchMask);
  AscendC::Reg::Div(pReg, pReg, absXReg, branchMask);

  AscendC::Reg::Adds(zReg, absXReg, (T)J0_PHASE, branchMask);
  AutofuseSin<T>(zReg, zReg, branchMask);
  AscendC::Reg::Mul(pReg, pReg, zReg, branchMask);

  // result = (pp * cos(phase) - qp * sin(phase)) * sqrt(2/pi) / sqrt(x)
  AscendC::Reg::Sub(bigDstReg, tmpReg, pReg, branchMask);
  AscendC::Reg::Sqrt(tmpReg, absXReg, branchMask);
  AscendC::Reg::Muls(bigDstReg, bigDstReg, (T)BESSEL_J_SQRT_2_OVER_PI, branchMask);
  AscendC::Reg::Div(bigDstReg, bigDstReg, tmpReg, branchMask);
}

template <typename T>
__simd_vf__ inline void BesselJ0ImplVF(__ubuf__ T *dst, __ubuf__ T *src, uint32_t calCount) {
  uint32_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(T));
  uint16_t repeatTime = static_cast<uint16_t>(AscendC::CeilDivision(calCount, vlSize));

  AscendC::Reg::RegTensor<T> srcReg, absXReg, xSqReg, smallDstReg, bigDstReg, dstReg, nanReg, nearZeroReg;
  AscendC::Reg::MaskReg mask, branchMask;

  for (uint16_t i = 0U; i < repeatTime; ++i) {
    mask = AscendC::Reg::UpdateMask<T>(calCount);
    AscendC::Reg::LoadAlign(srcReg, src + i * vlSize);
    AscendC::Reg::Abs(absXReg, srcReg, mask);

    // x^2
    AscendC::Reg::Mul(xSqReg, absXReg, absXReg, mask);

    // ===== Small branch: |x| <= 5.0 =====
    AscendC::Reg::Compares<T, CMPMODE::LE>(branchMask, absXReg, (T)5.0, mask);
    BesselJ0SmallCompute<T>(xSqReg, smallDstReg, branchMask);

    // Override for near-zero: |x| < 0.00001 -> 1.0 - x^2/4
    AscendC::Reg::Compares<T, CMPMODE::LT>(branchMask, absXReg, (T)0.00001, branchMask);
    AscendC::Reg::Muls(nearZeroReg, xSqReg, (T)(-0.25), branchMask);
    AscendC::Reg::Adds(nearZeroReg, nearZeroReg, (T)1.0, branchMask);
    AscendC::Reg::Select(smallDstReg, nearZeroReg, smallDstReg, branchMask);

    // ===== Large branch: |x| > 5.0 =====
    AscendC::Reg::Compares<T, CMPMODE::GT>(branchMask, absXReg, (T)5.0, mask);
    BesselJ0LargeCompute<T>(absXReg, xSqReg, bigDstReg, branchMask);

    // Merge: select small for |x| <= 5.0
    AscendC::Reg::Select(dstReg, bigDstReg, smallDstReg, branchMask);

    // Handle NaN input
    AscendC::Reg::Compare<T, CMPMODE::NE>(branchMask, srcReg, srcReg, mask);
    AscendC::Reg::Duplicate(nanReg, (float &)BESSEL_FLOAT_NAN, mask);
    AscendC::Reg::Select(dstReg, nanReg, dstReg, branchMask);

    // Store output
    AscendC::Reg::StoreAlign(dst + i * vlSize, dstReg, mask);
  }
}

template <typename T>
__aicore__ inline void BesselJ0Extend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                      const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount) {
  static_assert(SupportType<T, float>(), "Current data type is not supported on current device!");
  BesselJ0ImplVF<T>((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), calCount);
}

#endif  // __ASCENDC_API_REGBASE_BESSEL_J0_H__
