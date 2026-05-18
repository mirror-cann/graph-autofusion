/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef __ASCENDC_API_REGBASE_IGAMMAC_HELPER_SERIES_COMPLEMENT_H__
#define __ASCENDC_API_REGBASE_IGAMMAC_HELPER_SERIES_COMPLEMENT_H__

namespace AscendC {
namespace IGammaCInternal {

template <int32_t iterationNum>
__simd_callee__ inline void SeriesComplementIter(
    Reg::RegTensor<float>& facReg, Reg::RegTensor<float>& sumReg,
    Reg::RegTensor<float>& src0Reg, Reg::RegTensor<float>& src1Reg,
    Reg::RegTensor<float>& tmpReg, Reg::MaskReg& mask)
{
    Reg::Muls(tmpReg, src1Reg, -1.0f / iterationNum, mask);
    Reg::Mul(facReg, facReg, tmpReg, mask);
    Reg::Adds(tmpReg, src0Reg, iterationNum, mask);
    Reg::Div(tmpReg, facReg, tmpReg, mask);
    Reg::Add(sumReg, sumReg, tmpReg, mask);

    if constexpr (iterationNum < 25) {
        SeriesComplementIter<iterationNum + 1>(facReg, sumReg, src0Reg, src1Reg, tmpReg, mask);
    }
}

__simd_callee__ inline void Igammac_helper_series_complement_float(
    Reg::RegTensor<float>& dstReg,
    Reg::RegTensor<float>& src0Reg,
    Reg::RegTensor<float>& src1Reg,
    Reg::RegTensor<float>& lgammaReg,
    Reg::MaskReg& mask)
{
    constexpr float machep = 5.9604644775390625e-8f;
    Reg::RegTensor<float> sumReg, tmpReg, tmpReg1;

    Reg::Duplicate(tmpReg, 1.0f, mask); // fac = 1
    Reg::Duplicate(sumReg, 0.0f, mask); // sum = 0

    SeriesComplementIter<1>(tmpReg, sumReg, src0Reg, src1Reg, tmpReg1, mask);

    Reg::Log(tmpReg, src0Reg, mask); // log(a)
    Reg::Add(tmpReg, tmpReg, lgammaReg, mask);  // lgamma(1 + a)

    Reg::Log(tmpReg1, src1Reg, mask); // log(x)
    Reg::Mul(tmpReg1, src0Reg, tmpReg1, mask); // a * log(x)
    Reg::Sub(tmpReg, tmpReg1, tmpReg, mask); // a * log(x) - lgamma(1+a)

    Reg::Exp(tmpReg, tmpReg, mask);
    Reg::Adds(tmpReg, tmpReg, -1.0f, mask);
    Reg::Muls(tmpReg, tmpReg, -1.0f, mask); // -expm1f(a * log(x) - lgamma(1+a))

    Reg::Sub(tmpReg1, tmpReg1, lgammaReg, mask); // a * log(x) - lgamma(a)
    Reg::Exp(tmpReg1, tmpReg1, mask); // exp(a * log(x) - lgamma(a))
    Reg::Mul(tmpReg1, tmpReg1, sumReg, mask); // exp(a * log(x) - lgamma(a)) * sum
    Reg::Sub(dstReg, tmpReg, tmpReg1, mask); // -expm1f(a * log(x) - lgamma(1+a)) -  exp(a * log(x) - lgamma(a)) * sum
}

} // namespace IGammaCInternal
} // namespace AscendC

#endif // __ASCENDC_API_REGBASE_IGAMMAC_HELPER_SERIES_COMPLEMENT_H__