/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_BESSEL_J_UTILS_H__
#define __ASCENDC_API_REGBASE_BESSEL_J_UTILS_H__

// sqrt(2/PI)
constexpr float BESSEL_J_SQRT_2_OVER_PI = 0.797884560802865355879892119868763737;
constexpr uint32_t BESSEL_FLOAT_NAN = 0x7fc00000;

// Horner polynomial evaluation: result = ((...(coeffs[0] * z + coeffs[1]) * z + coeffs[2]) * z + ... + coeffs[N])
template <typename T, uint32_t currentIdx, uint32_t endIdx, const float *coeffs>
__simd_callee__ inline void HornerPoly(AscendC::Reg::RegTensor<T> &result, AscendC::Reg::RegTensor<T> &z,
                                       AscendC::Reg::MaskReg &mask) {
  AscendC::Reg::Mul(result, result, z, mask);
  AscendC::Reg::Adds(result, result, (T)coeffs[currentIdx], mask);
  if constexpr (currentIdx < endIdx) {
    HornerPoly<T, currentIdx + 1, endIdx, coeffs>(result, z, mask);
  }
}

#endif  // __ASCENDC_API_REGBASE_BESSEL_J_UTILS_H__
