/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_SHIFTED_CHEBYSHEV_POLYNOMIAL_UTILS_H__
#define __ASCENDC_API_REGBASE_SHIFTED_CHEBYSHEV_POLYNOMIAL_UTILS_H__

template <typename T, int64_t N, int64_t curN = 2>
__simd_callee__ inline void ShiftedChebyshevPolynomialCalByTemplateExpansion(AscendC::Reg::RegTensor<T> &res,
                                                                             AscendC::Reg::RegTensor<T> &coef,
                                                                             AscendC::Reg::RegTensor<T> &temp1,
                                                                             AscendC::Reg::RegTensor<T> &temp2,
                                                                             AscendC::Reg::MaskReg &mask) {
  AscendC::Reg::Move(temp2, res);
  AscendC::Reg::Mul(res, coef, res, mask);
  AscendC::Reg::Sub(res, res, temp1, mask);
  AscendC::Reg::Move(temp1, temp2);

  if constexpr (curN < N) {
    ShiftedChebyshevPolynomialCalByTemplateExpansion<T, N, curN + 1>(res, coef, temp1, temp2, mask);
  }
}

template <typename T, int64_t N>
__simd_callee__ inline void ShiftedChebyshevPolynomialCalByLoop(AscendC::Reg::RegTensor<T> &res,
                                                                AscendC::Reg::RegTensor<T> &coef,
                                                                AscendC::Reg::RegTensor<T> &temp1,
                                                                AscendC::Reg::RegTensor<T> &temp2,
                                                                AscendC::Reg::MaskReg &mask) {
  for (int k = 2; k <= N; k++) {
    AscendC::Reg::Move(temp2, res);
    AscendC::Reg::Mul(res, coef, res, mask);
    AscendC::Reg::Sub(res, res, temp1, mask);
    AscendC::Reg::Move(temp1, temp2);
  }
}

template <typename T, int64_t N>
__simd_callee__ inline void ShiftedChebyshevPolynomialCal(AscendC::Reg::RegTensor<T> &res,
                                                          AscendC::Reg::RegTensor<T> &coef,
                                                          AscendC::Reg::RegTensor<T> &temp1,
                                                          AscendC::Reg::RegTensor<T> &temp2,
                                                          AscendC::Reg::MaskReg &mask) {
  // 模板展开栈过深编译过程中会爆栈
  if constexpr (N <= 1000) {
    ShiftedChebyshevPolynomialCalByTemplateExpansion<T, N>(res, coef, temp1, temp2, mask);
  } else {
    ShiftedChebyshevPolynomialCalByLoop<T, N>(res, coef, temp1, temp2, mask);
  }
}

#endif
