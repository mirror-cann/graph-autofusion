/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_ZETA_H__
#define __ASCENDC_API_REGBASE_ZETA_H__

#include "kernel_operator.h"

namespace AscendC {
template <typename T>
__simd_vf__ inline void ZetaImplVF(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint32_t calCount)
{
  constexpr T ZERO = static_cast<T>(0.0);
  constexpr T HALF = static_cast<T>(0.5);
  constexpr T ONE = static_cast<T>(1.0);
  constexpr T NEG_ONE = static_cast<T>(-1.0);

  uint32_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(T));
  uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, vlSize));

  Reg::MaskReg mask;

  Reg::RegTensor<T> src0Reg, src1Reg, dstReg;
  Reg::RegTensor<T> aTensor, bTensor, wTensor, tTensor, lnTmp, mulTmp, xMinusOne, kPlusX, nanTensor;
  Reg::MaskReg mask0, mask1, mask2, mask3;

  for (uint16_t i = 0U; i < repeatTimes; ++i) {
    mask = Reg::UpdateMask<T>(calCount);

    Reg::LoadAlign(src0Reg, src0 + i * vlSize);
    Reg::LoadAlign(src1Reg, src1 + i * vlSize);

    Reg::Duplicate(nanTensor, __builtin_nanf(""), mask);

    Reg::Compares<T, CMPMODE::LT>(mask0, src0Reg, ONE, mask);
    Reg::Compares<T, CMPMODE::EQ>(mask1, src0Reg, ONE, mask);
    Reg::Compares<T, CMPMODE::LE>(mask2, src1Reg, ZERO, mask);

    Reg::Truncate<T, RoundMode::CAST_FLOOR, Reg::MaskMergeMode::ZEROING>(aTensor, src1Reg, mask);
    Reg::Compare<T, CMPMODE::EQ>(mask3, aTensor, src1Reg, mask);

    Reg::Muls(mulTmp, src0Reg, NEG_ONE, mask);

    Reg::Abs(lnTmp, src1Reg, mask);
    Reg::Log(tTensor, lnTmp, mask);
    Reg::Mul(tTensor, mulTmp, tTensor, mask);
    Reg::Exp(dstReg, tTensor, mask);

    Reg::Compares<T, CMPMODE::EQ>(mask2, src1Reg, ONE, mask);
    Reg::Duplicate(wTensor, ONE, mask);
    Reg::Select(dstReg, wTensor, dstReg, mask2);

    Reg::Muls(aTensor, src1Reg, ONE, mask);

    Reg::Adds(aTensor, aTensor, ONE, mask);
    Reg::Abs(lnTmp, aTensor, mask);
    Reg::Log(tTensor, lnTmp, mask);
    Reg::Mul(tTensor, mulTmp, tTensor, mask);
    Reg::Exp(bTensor, tTensor, mask);
    Reg::Add(dstReg, dstReg, bTensor, mask);

    for (uint16_t iter = 0; iter < 8; ++iter) {
      Reg::Adds(aTensor, aTensor, ONE, mask);
      Reg::Abs(lnTmp, aTensor, mask);
      Reg::Log(tTensor, lnTmp, mask);
      Reg::Mul(tTensor, mulTmp, tTensor, mask);
      Reg::Exp(bTensor, tTensor, mask);
      Reg::Add(dstReg, dstReg, bTensor, mask);
    }

    Reg::Adds(xMinusOne, src0Reg, NEG_ONE, mask);
    Reg::Mul(mulTmp, bTensor, aTensor, mask);
    Reg::Div(mulTmp, mulTmp, xMinusOne, mask);
    Reg::Add(dstReg, dstReg, mulTmp, mask);

    Reg::Muls(lnTmp, bTensor, HALF, mask);
    Reg::Sub(dstReg, dstReg, lnTmp, mask);

    Reg::Muls(xMinusOne, dstReg, ONE, mask);

    Reg::Duplicate(lnTmp, ZERO, mask);
    Reg::Adds(lnTmp, lnTmp, ONE, mask);
    Reg::Div(lnTmp, lnTmp, aTensor, mask);
    Reg::Duplicate(aTensor, ONE, mask);

    // Bernoulli series — 12 terms with compile-time constants
    // Term  0: B_2  / 2!  =  1/12
    {
      T kVal = ZERO;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Muls(tTensor, mulTmp, 1.0f / 12.0f, mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
      kVal += ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
    }
    // Term  1: B_4  / 4!  =  -1/720
    {
      T kVal = 2 * ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Muls(tTensor, mulTmp, -1.0f / 720.0f, mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
      kVal += ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
    }
    // Term  2: B_6  / 6!  =  1/30240
    {
      T kVal = 4 * ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Muls(tTensor, mulTmp, (1.0f / 30240.0f), mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
      kVal += ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
    }
    // Term  3: B_8  / 8!  =  -1/1209600
    {
      T kVal = 6 * ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Muls(tTensor, mulTmp, (-1.0f / 1209600.0f), mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
      kVal += ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
    }
    // Term  4: B_10 / 10! =  1/47900160
    {
      T kVal = 8 * ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Muls(tTensor, mulTmp, (1.0f / 47900160.0f), mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
      kVal += ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
    }
    // Term  5
    {
      T kVal = 10 * ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Muls(tTensor, mulTmp, (-1.0f / 1.8924375803183791606e9f), mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
      kVal += ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
    }
    // Term  6
    {
      T kVal = 12 * ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Muls(tTensor, mulTmp, (1.0f / 7.47242496e10f), mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
      kVal += ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
    }
    // Term  7
    {
      T kVal = 14 * ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Muls(tTensor, mulTmp, (-1.0f / 2.950130727918164224e12f), mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
      kVal += ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
    }
    // Term  8
    {
      T kVal = 16 * ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Muls(tTensor, mulTmp, (1.0f / 1.1646782814350067249e14f), mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
      kVal += ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
    }
    // Term  9
    {
      T kVal = 18 * ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Muls(tTensor, mulTmp, (-1.0f / 4.5979787224074726105e15f), mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
      kVal += ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
    }
    // Term 10
    {
      T kVal = 20 * ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Muls(tTensor, mulTmp, (1.0f / 1.8152105401943546773e17f), mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
    }

    Reg::Compare<T, CMPMODE::NE>(mask2, dstReg, dstReg, mask);
    Reg::Select(dstReg, xMinusOne, dstReg, mask2);

    Reg::Duplicate(tTensor, ZERO, mask);
    Reg::Duplicate(mulTmp, ZERO, mask);
    Reg::Adds(tTensor, tTensor, ONE, mask);
    Reg::Div(mulTmp, tTensor, mulTmp, mask);

    Reg::Duplicate(nanTensor, __builtin_nanf(""), mask);

    Reg::Select(dstReg, nanTensor, dstReg, mask0);
    Reg::Select(dstReg, mulTmp, dstReg, mask1);

    Reg::Compares<T, CMPMODE::LE>(mask2, src1Reg, ZERO, mask);

    Reg::Duplicate(bTensor, ZERO, mask);
    Reg::Adds(bTensor, bTensor, ONE, mask);
    Reg::Duplicate(wTensor, ZERO, mask);

    Reg::Compares<T, CMPMODE::GT>(mask0, src0Reg, ONE, mask);
    Reg::Select(kPlusX, bTensor, wTensor, mask0);

    Reg::Truncate<T, RoundMode::CAST_FLOOR, Reg::MaskMergeMode::ZEROING>(tTensor, src0Reg, mask);
    Reg::Compare<T, CMPMODE::NE>(mask1, tTensor, src0Reg, mask);
    Reg::Select(aTensor, bTensor, wTensor, mask1);

    Reg::Select(lnTmp, bTensor, wTensor, mask2);
    Reg::Select(xMinusOne, bTensor, wTensor, mask3);

    Reg::Mul(tTensor, lnTmp, xMinusOne, mask);
    Reg::Mul(tTensor, tTensor, kPlusX, mask);
    Reg::Compares<T, CMPMODE::GT>(mask3, tTensor, ZERO, mask);
    Reg::Mul(tTensor, lnTmp, xMinusOne, mask);

    Reg::Select(dstReg, mulTmp, dstReg, mask3);

    Reg::Muls(tTensor, xMinusOne, NEG_ONE, mask);
    Reg::Adds(tTensor, tTensor, ONE, mask);
    Reg::Mul(tTensor, tTensor, lnTmp, mask);
    Reg::Mul(tTensor, tTensor, aTensor, mask);
    Reg::Mul(tTensor, tTensor, kPlusX, mask);
    Reg::Compares<T, CMPMODE::GT>(mask2, tTensor, ZERO, mask);
    Reg::Select(dstReg, nanTensor, dstReg, mask2);

    Reg::StoreAlign(dst + i * vlSize, dstReg, mask);
  }
}

template <typename T>
__aicore__ inline void Zeta(
  const LocalTensor<T>& dst,
  const LocalTensor<T>& src0,
  const LocalTensor<T>& src1,
  const LocalTensor<uint8_t>& sharedTmpBuffer,
  const uint32_t calCount)
{
  if ASCEND_IS_AIC {
    return;
  }
  static_assert((std::is_same_v<T, float>), "Zeta only supports float on current device!");
  ZetaImplVF<T>((__ubuf__ T*)dst.GetPhyAddr(),
                (__ubuf__ T*)src0.GetPhyAddr(),
                (__ubuf__ T*)src1.GetPhyAddr(),
                 calCount);
}
}  // namespace AscendC

#endif  // __ASCENDC_API_REGBASE_ZETA_H__