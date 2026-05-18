/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef __ASCENDC_API_REGBASE_IGAMMAC_HELPER_CONTINUED_FRACTION_H__
#define __ASCENDC_API_REGBASE_IGAMMAC_HELPER_CONTINUED_FRACTION_H__

#include "series.h"

namespace AscendC {
namespace IGammaCInternal {

__simd_callee__ inline void Igammac_compute_ans_float(
    Reg::RegTensor<float>& ansReg, Reg::RegTensor<float>& cReg,
    Reg::RegTensor<float>& yReg, Reg::RegTensor<float>& zReg,
    Reg::RegTensor<float>& pkm1Reg, Reg::RegTensor<float>& pkm2Reg,
    Reg::RegTensor<float>& qkm1Reg, Reg::RegTensor<float>& qkm2Reg,
    Reg::MaskReg& mask)
{
    static constexpr float machep = 5.9604644775390625e-8f;
    static constexpr float big = 16777216.0f;
    static constexpr float biginv = 5.9604644775390625e-8f;
    static constexpr int maxiter = 25;
    Reg::RegTensor<float> pkReg, qkReg, tmpReg;
    Reg::MaskReg tmpMask;

    for (int i = 0; i < maxiter; i++) {
        Reg::Adds(cReg, cReg, 1.0f, mask);
        Reg::Adds(yReg, yReg, 1.0f, mask);
        Reg::Adds(zReg, zReg, 2.0f, mask);

        Reg::Mul(pkReg, pkm1Reg, zReg, mask);
        Reg::Mul(tmpReg, yReg, cReg, mask);
        Reg::Mul(tmpReg, pkm2Reg, tmpReg, mask);
        Reg::Sub(pkReg, pkReg, tmpReg, mask);

        Reg::Mul(qkReg, qkm1Reg, zReg, mask);
        Reg::Mul(tmpReg, yReg, cReg, mask);
        Reg::Mul(tmpReg, qkm2Reg, tmpReg, mask);
        Reg::Sub(qkReg, qkReg, tmpReg, mask);

        Reg::Compares<float, CMPMODE::NE>(tmpMask, qkReg, 0.0f, mask);
        Reg::Div(tmpReg, pkReg, qkReg, tmpMask);
        Reg::Copy(ansReg, tmpReg, tmpMask);

        Reg::Copy(pkm2Reg, pkm1Reg, mask);
        Reg::Copy(pkm1Reg, pkReg, mask);
        Reg::Copy(qkm2Reg, qkm1Reg, mask);
        Reg::Copy(qkm1Reg, qkReg, mask);

        Reg::Abs(tmpReg, pkReg, mask);
        Reg::Compares<float, CMPMODE::GT>(tmpMask, tmpReg, big, mask);
        Reg::Muls(tmpReg, pkm2Reg, biginv, tmpMask);
        Reg::Copy(pkm2Reg, tmpReg, tmpMask);
        Reg::Muls(tmpReg, pkm1Reg, biginv, tmpMask);
        Reg::Copy(pkm1Reg, tmpReg, tmpMask);
        Reg::Muls(tmpReg, qkm2Reg, biginv, tmpMask);
        Reg::Copy(qkm2Reg, tmpReg, tmpMask);
        Reg::Muls(tmpReg, qkm1Reg, biginv, tmpMask);
        Reg::Copy(qkm1Reg, tmpReg, tmpMask);
    }
}

__simd_callee__ inline void Igammac_helper_continued_fraction_float(Reg::RegTensor<float>& dstReg, Reg::RegTensor<float>& src0Reg,
                                                                    Reg::RegTensor<float>& src1Reg, Reg::RegTensor<float>& lgammaReg,
                                                                    Reg::RegTensor<float>& powReg, Reg::MaskReg& mask)
{
    Reg::MaskReg tmpMask;
    Reg::RegTensor<float> axReg, ansReg, cReg, yReg, zReg, pkm1Reg, pkm2Reg, qkm1Reg, qkm2Reg;

    Igammac_helper_fac_float(axReg, src0Reg, src1Reg, powReg, lgammaReg, mask);

    Reg::Compares<float, CMPMODE::NE>(tmpMask, axReg, 0.0f, mask);

    Reg::Duplicate(yReg, 1.0f, tmpMask);
    Reg::Sub(yReg, yReg, src0Reg, tmpMask);
    Reg::Add(zReg, src1Reg, yReg, tmpMask);
    Reg::Adds(zReg, zReg, 1.0f, tmpMask);
    Reg::Duplicate(cReg, 0.0f, tmpMask);
    Reg::Duplicate(pkm2Reg, 1.0f, tmpMask);
    Reg::Copy(qkm2Reg, src1Reg, tmpMask);
    Reg::Adds(pkm1Reg, src1Reg, 1.0f, tmpMask);
    Reg::Mul(qkm1Reg, zReg, src1Reg, tmpMask);
    Reg::Div(ansReg, pkm1Reg, qkm1Reg, tmpMask);

    Igammac_compute_ans_float(ansReg, cReg, yReg, zReg, pkm1Reg, pkm2Reg, qkm1Reg, qkm2Reg, tmpMask);
    Reg::Mul(dstReg, ansReg, axReg, tmpMask);
}
} // namespace IGammaCInternal
} // namespace AscendC

#endif // __ASCENDC_API_REGBASE_IGAMMAC_HELPER_CONTINUED_FRACTION_H__