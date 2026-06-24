/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_SPHERICAL_BESSEL_J0_H__
#define __ASCENDC_API_REGBASE_SPHERICAL_BESSEL_J0_H__

#include "bessel_j_utils.h"
#include "trigonometric_function_utils.h"

constexpr float SBESJ0_TAYLOR_COEFFS[6] = {
    1.0f / 6227020800.0f, -1.0f / 39916800.0f, 1.0f / 362880.0f, -1.0f / 5040.0f, 1.0f / 120.0f, -1.0f / 6.0f,
};

template <typename T>
__simd_vf__ inline void SphericalBesselJ0ImplVF(__ubuf__ T *dst, __ubuf__ T *src, uint32_t calCount) {
  uint32_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(T));
  uint16_t repeatTime = static_cast<uint16_t>(AscendC::CeilDivision(calCount, vlSize));

  AscendC::Reg::RegTensor<T> srcReg, absXReg, x2Reg, tempReg, dstReg, nanReg, zeroReg;
  AscendC::Reg::MaskReg mask, smallMask, infMask, nanMask;

  for (uint16_t i = 0U; i < repeatTime; ++i) {
    mask = AscendC::Reg::UpdateMask<T>(calCount);
    AscendC::Reg::LoadAlign(srcReg, src + i * vlSize);
    AscendC::Reg::Abs(absXReg, srcReg, mask);
    AscendC::Reg::Mul(x2Reg, srcReg, srcReg, mask);

    AutofuseSin<T>(dstReg, srcReg, mask);
    AscendC::Reg::Div(dstReg, dstReg, srcReg, mask);

    AscendC::Reg::Duplicate<T>(tempReg, (T)0.0, mask);
    HornerPoly<T, 0, 5, SBESJ0_TAYLOR_COEFFS>(tempReg, x2Reg, mask);
    AscendC::Reg::Mul(tempReg, tempReg, x2Reg, mask);
    AscendC::Reg::Adds(tempReg, tempReg, (T)1.0, mask);

    AscendC::Reg::Compares<T, CMPMODE::LT>(smallMask, absXReg, (T)0.5, mask);
    AscendC::Reg::Select(dstReg, tempReg, dstReg, smallMask);

    AscendC::Reg::Compares<T, CMPMODE::GT>(infMask, absXReg, (T)1e30, mask);
    AscendC::Reg::Duplicate<T>(zeroReg, (T)0.0, mask);
    AscendC::Reg::Select(dstReg, zeroReg, dstReg, infMask);

    AscendC::Reg::Compare<T, CMPMODE::NE>(nanMask, srcReg, srcReg, mask);
    AscendC::Reg::Duplicate(nanReg, (float &)BESSEL_FLOAT_NAN, mask);
    AscendC::Reg::Select(dstReg, nanReg, dstReg, nanMask);

    AscendC::Reg::StoreAlign(dst + i * vlSize, dstReg, mask);
  }
}

template <typename T>
__aicore__ inline void SphericalBesselJ0Extend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                               const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount) {
  static_assert(SupportType<T, float>(), "Current data type is not supported on current device!");
  SphericalBesselJ0ImplVF<T>((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), calCount);
}

#endif
