/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_HERMITE_POLYNOMIAL_H_H__
#define __ASCENDC_API_REGBASE_HERMITE_POLYNOMIAL_H_H__

constexpr int32_t HERMITE_POLYNOMIAL_H_LIMIT = 128;

template <typename T>
__simd_vf__ inline void HermitePolynomialHFillVF(__ubuf__ T *dstUb, uint32_t calCount, T fillValue) {
  const uint32_t repeatElem = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
  const uint16_t repeatTime = static_cast<uint16_t>(AscendC::CeilDivision(calCount, repeatElem));
  AscendC::Reg::MaskReg activeMask;
  AscendC::Reg::RegTensor<T> yReg;
  AscendC::Reg::Duplicate(yReg, fillValue);
  for (uint16_t i = 0U; i < repeatTime; ++i) {
    activeMask = AscendC::Reg::UpdateMask<T>(calCount);
    AscendC::Reg::StoreAlign(dstUb + i * repeatElem, yReg, activeMask);
  }
}

/**
 * H_1(x) = 2x。
 */
template <typename T>
__simd_vf__ inline void HermitePolynomialHLinearVF(__ubuf__ T *srcUb, __ubuf__ T *dstUb, uint32_t calCount) {
  const uint32_t repeatElem = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
  const uint16_t repeatTime = static_cast<uint16_t>(AscendC::CeilDivision(calCount, repeatElem));

  AscendC::Reg::MaskReg activeMask;
  AscendC::Reg::RegTensor<T> xReg;
  AscendC::Reg::RegTensor<T> outReg;
  for (uint16_t i = 0U; i < repeatTime; ++i) {
    activeMask = AscendC::Reg::UpdateMask<T>(calCount);
    AscendC::Reg::LoadAlign(xReg, srcUb + i * repeatElem);
    AscendC::Reg::Add(outReg, xReg, xReg, activeMask);
    AscendC::Reg::StoreAlign(dstUb + i * repeatElem, outReg, activeMask);
  }
}

/**
 * n >= 2 且 n <= 128场景使用。
 * 递推公式：H_m = 2x · H_{m-1} - 2(m-1) · H_{m-2}
 */
template <typename T>
__simd_vf__ inline void HermitePolynomialHImplVF(__ubuf__ T *srcUb, __ubuf__ T *dstUb, int32_t n, uint32_t calCount) {
  const uint32_t repeatElem = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
  const uint16_t repeatTime = static_cast<uint16_t>(AscendC::CeilDivision(calCount, repeatElem));

  AscendC::Reg::MaskReg activeMask;
  AscendC::Reg::MaskReg nonNanMask;
  AscendC::Reg::RegTensor<T> xReg;
  AscendC::Reg::RegTensor<T> twoXReg;
  AscendC::Reg::RegTensor<T> twoReg;
  AscendC::Reg::RegTensor<T> kReg;
  AscendC::Reg::RegTensor<T> negKReg;
  AscendC::Reg::RegTensor<T> pReg;
  AscendC::Reg::RegTensor<T> qReg;
  AscendC::Reg::RegTensor<T> kpReg;
  AscendC::Reg::Duplicate(twoReg, static_cast<T>(2.0f));

  for (uint16_t i = 0U; i < repeatTime; ++i) {
    activeMask = AscendC::Reg::UpdateMask<T>(calCount);
    AscendC::Reg::LoadAlign(xReg, srcUb + i * repeatElem);
    AscendC::Reg::Add(twoXReg, xReg, xReg, activeMask);
    AscendC::Reg::Duplicate(pReg, static_cast<T>(1.0f), activeMask);  // pReg = H_0 = 1
    AscendC::Reg::Add(qReg, xReg, xReg, activeMask);                  // qReg = H_1 = 2x
    AscendC::Reg::Compare<T, AscendC::CMPMODE::EQ>(nonNanMask, xReg, xReg, activeMask);
    AscendC::Reg::Duplicate(kReg, static_cast<T>(2.0f), activeMask);
    AscendC::Reg::Neg(negKReg, kReg, activeMask);

    for (uint8_t k = 0U; k < static_cast<uint8_t>(n - 1); ++k) {
      AscendC::Reg::Mul(kpReg, negKReg, pReg, nonNanMask);        // kp = -k · H_{m-2}
      AscendC::Reg::MulAddDst(kpReg, twoXReg, qReg, nonNanMask);  // kp = -k·H_{m-2} + 2x·H_{m-1} = H_m
      AscendC::Reg::Select(pReg, qReg, pReg, nonNanMask);
      AscendC::Reg::Select(qReg, kpReg, qReg, nonNanMask);
      AscendC::Reg::Add(kReg, kReg, twoReg, activeMask);
      AscendC::Reg::Neg(negKReg, kReg, activeMask);
    }
    AscendC::Reg::StoreAlign(dstUb + i * repeatElem, qReg, activeMask);
  }
}

/**
 * 对外接口（标量 n 版）：计算物理学版埃尔米特多项式 H_n(x)。
 * @tparam T  数据类型（当前仅支持 float）
 * @tparam U  阶数 n 的类型（int32_t 等）
 * @param dst      输出 LocalTensor
 * @param src      输入 LocalTensor（x 值）
 * @param n                阶数标量
 * @param sharedTmpBuffer  共享临时 buffer（当前未使用，仅便于后续拓展）
 * @param calCount         需要计算的元素总数
 */
template <typename T, typename U>
__aicore__ inline void HermitePolynomialHExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                                U n, const AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
                                                const uint32_t calCount) {
  if ASCEND_IS_AIC {
    return;
  }
  static_assert(std::is_same<T, float>::value, "HermitePolynomialH currently only supports float");
  static_assert(std::is_integral<U>::value, "n must be an integer type");
  const int32_t order = static_cast<int32_t>(n);

  __ubuf__ T *dstUb = (__ubuf__ T *)dst.GetPhyAddr();

  if (order < 0) {
    HermitePolynomialHFillVF<T>(dstUb, calCount, static_cast<T>(0.0f));
    return;
  }

  if (order == 0) {
    HermitePolynomialHFillVF<T>(dstUb, calCount, static_cast<T>(1.0f));
    return;
  }

  if (order > HERMITE_POLYNOMIAL_H_LIMIT) {
    HermitePolynomialHFillVF<T>(dstUb, calCount, AscendC::NumericLimits<T>::QuietNaN());
    return;
  }

  __ubuf__ T *srcUb = (__ubuf__ T *)src.GetPhyAddr();
  if (order == 1) {
    HermitePolynomialHLinearVF<T>(srcUb, dstUb, calCount);
    return;
  }

  HermitePolynomialHImplVF<T>(srcUb, dstUb, order, calCount);
}

/**
 * 对外接口（tensor n 版）：阶数 n 从 LocalTensor 中读取。
 * 用于图编译场景，n 来自前序算子输出。
 */
template <typename T, typename U>
__aicore__ inline void HermitePolynomialHExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                                const AscendC::LocalTensor<U> &n,
                                                const AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
                                                const uint32_t calCount) {
  HermitePolynomialHExtend(dst, src, n.GetValue(0), sharedTmpBuffer, calCount);
}

#endif  // __ASCENDC_API_REGBASE_HERMITE_POLYNOMIAL_H_H__
