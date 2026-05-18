/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef __ASCENDC_API_REGBASE_IGAMMAC_HELPER_SERIES_H__
#define __ASCENDC_API_REGBASE_IGAMMAC_HELPER_SERIES_H__

namespace AscendC {
namespace {
// Lanczos coefficients
constexpr float kLanczosNum[13] = {
  0.006061842346248906525783753964555936883222f,
  0.5098416655656676188125178644804694509993f,
  19.51992788247617482847860966235652136208f,
  449.9445569063168119446858607650988409623f,
  6955.999602515376140356310115515198987526f,
  75999.29304014542649875303443598909137092f,
  601859.6171681098786670226533699352302507f,
  3481712.15498064590882071018964774556468f,
  14605578.08768506808414169982791359218571f,
  43338889.32467613834773723740590533316085f,
  86363131.28813859145546927288977868422342f,
  103794043.1163445451906271053616070238554f,
  56906521.91347156388090791033559122686859f};

constexpr float kLanczosDen[13] = {
  1.0f,
  66.0f,
  1925.0f,
  32670.0f,
  357423.0f,
  2637558.0f,
  13339535.0f,
  45995730.0f,
  105258076.0f,
  150917976.0f,
  120543840.0f,
  39916800.0f,
  0.0f};
} // anonymous namespace

namespace IGammaCInternal {

template <int32_t iterationNum>
__simd_callee__ inline void LanczosNumIter(
    Reg::RegTensor<float>& numAnsReg, Reg::RegTensor<float>& yReg,
    Reg::MaskReg& mask0, Reg::MaskReg& mask1, Reg::MaskReg& fullMask)
{
    Reg::RegTensor<float> tmpReg;

    // num_ans = num_ans * y + *p
    Reg::Mul(tmpReg, numAnsReg, yReg, fullMask);
    Reg::Adds(numAnsReg, tmpReg, kLanczosNum[13 - iterationNum], mask1);
    Reg::Adds(tmpReg, tmpReg, kLanczosNum[iterationNum-1], mask0);
    Reg::Copy<float, Reg::MaskMergeMode::MERGING>(numAnsReg, tmpReg, mask0);
    if constexpr (iterationNum > 1) {
        LanczosNumIter<iterationNum - 1>(numAnsReg, yReg, mask0, mask1, fullMask);
    }
}

template <int32_t iterationNum>
__simd_callee__ inline void LanczosDenIter(
    Reg::RegTensor<float>& denAnsReg, Reg::RegTensor<float>& yReg,
    Reg::MaskReg& mask0, Reg::MaskReg& mask1, Reg::MaskReg& fullMask)
{
    Reg::RegTensor<float> tmpReg;

    // den_ans = den_ans * y + *p
    Reg::Mul(tmpReg, denAnsReg, yReg, fullMask);
    Reg::Adds(denAnsReg, tmpReg, kLanczosDen[13 - iterationNum], mask1);
    Reg::Adds(tmpReg, tmpReg, kLanczosDen[iterationNum - 1], mask0);
    Reg::Copy<float, Reg::MaskMergeMode::MERGING>(denAnsReg, tmpReg, mask0);

    if constexpr (iterationNum > 1) {
        LanczosDenIter<iterationNum - 1>(denAnsReg, yReg, mask0, mask1, fullMask);
    }
}

__simd_callee__ inline void Igammac_lanczos_sum_expg_scaled_float(Reg::RegTensor<float>& dstReg, Reg::RegTensor<float>& src0Reg, Reg::MaskReg mask)
{
    // return igammac_ratevl_float(a, kLanczosNum, (int64_t)(sizeof(kLanczosNum) / sizeof(kLanczosNum[0] -1)), kLanczosDen, (int64_t)(sizeof(kLanczosNum) / sizeof(kLanczosNum[0] -1)))
    // return igammac_ratevl_float(a, kLanczosNum, (int64_t)(12), kLanczosDen, (int64_t)(12))

    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::RegTensor<float> tmpReg, tmpReg1;

    // if abs(a) > 1.0f
    Reg::RegTensor<float> absReg;
    Reg::RegTensor<float> yReg;
    Reg::MaskReg mask0;
    Reg::Abs(absReg, src0Reg, fullMask);
    Reg::Compares<float, CMPMODE::GT>(mask0, absReg, 1.0f, fullMask);
    // p is a pointer, dir should move a pointer
    Reg::Duplicate<float, Reg::MaskMergeMode::MERGING>(tmpReg, 1.0f, mask0);
    Reg::Div(yReg, tmpReg, src0Reg, mask0);

    // else abs(a) <= 1.0f
    Reg::MaskReg mask1;
    Reg::Not(mask1, mask0, fullMask);
    Reg::Copy<float, Reg::MaskMergeMode::MERGING>(yReg, src0Reg, mask1);

    // num_ans = *p
    // *p在不同的位置被dup进了numAnsReg，所以numAnsReg就是*p
    Reg::RegTensor<float> numAnsReg;
    Reg::Duplicate<float, Reg::MaskMergeMode::MERGING>(numAnsReg, kLanczosNum[12], mask0);
    Reg::Duplicate<float, Reg::MaskMergeMode::MERGING>(numAnsReg, kLanczosNum[0], mask1);

    // Recursively unroll the loop for numerator
    LanczosNumIter<12>(numAnsReg, yReg, mask0, mask1, fullMask);

    // p = (absx > 1.0f) ? (den + n) : den
    // den_ans = *p
    // reuse mask0, absReg
    Reg::RegTensor<float> denAnsReg;
    Reg::Duplicate<float, Reg::MaskMergeMode::MERGING>(denAnsReg, kLanczosDen[12], mask0);
    Reg::Duplicate<float, Reg::MaskMergeMode::MERGING>(denAnsReg, kLanczosDen[0], mask1);

    // Recursively unroll the loop for denominator
    LanczosDenIter<12>(denAnsReg, yReg, mask0, mask1, fullMask);

    // absx > 1.0f
    // return (powf(x ,0.0f) * num_ans / den_ans)
    // replace it with return num_ans / den_ans, if nan input happens to be wrong,look here
    Reg::Div(dstReg, numAnsReg, denAnsReg, fullMask);
}

__simd_callee__ inline void Igammac_helper_fac_float(Reg::RegTensor<float>& dstReg, Reg::RegTensor<float>& src0Reg, Reg::RegTensor<float>& src1Reg,
                                                     Reg::RegTensor<float>& powReg, Reg::RegTensor<float>& lgammaReg, Reg::MaskReg fullMask)
{
    constexpr float maxlog = 88.72283905206835;
    constexpr float exp1 = 2.718281828459045;
    constexpr float lanczos_g = 6.024680040776729583740234375;

    Reg::RegTensor<float> tmpReg, tmpReg1;
    Reg::RegTensor<float> helpReg;

    // 0.4f * fabsf(a)
    Reg::RegTensor<float> absReg0;
    Reg::RegTensor<float> dupReg;
    Reg::RegTensor<float> mulScalarReg;
    Reg::Duplicate(dupReg, 0.4f, fullMask);
    Reg::Abs(absReg0, src0Reg, fullMask);
    Reg::Mul(mulScalarReg, dupReg, absReg0, fullMask);

    // fabsf(a- x)
    Reg::RegTensor<float> absReg1;
    Reg::RegTensor<float> subReg;
    Reg::Sub(subReg, src0Reg, src1Reg, fullMask);
    Reg::Abs(absReg1, subReg, fullMask);

    // if : fabsf(a- x) > 0.4f * fabsf(a)
    Reg::MaskReg mask0;
    Reg::Compare<float, CMPMODE::GT>(mask0, absReg1, mulScalarReg, fullMask);

    // ax = a * logf(x) - x - lgammaf(a), under mask0
    Reg::RegTensor<float> logReg;
    Reg::Log(logReg, src1Reg, mask0);
    Reg::Mul(tmpReg, src0Reg, logReg, mask0);
    Reg::Sub(tmpReg, tmpReg, src1Reg, mask0);
    Reg::Sub(tmpReg, tmpReg, lgammaReg, mask0);

    // if : ax < -maxlog, return 0
    Reg::MaskReg mask1;
    Reg::Compares<float, CMPMODE::LT>(mask1, tmpReg, -maxlog, mask0);
    Reg::Duplicate<float, Reg::MaskMergeMode::MERGING>(dstReg, 0.0f, mask1);

    // else: ax > -maxlog, return expf(ax)
    Reg::MaskReg mask2 = Reg::CreateMask<float, Reg::MaskPattern::ALLF>();
    Reg::Not(mask2, mask1, mask0);
    Reg::Exp(dstReg, tmpReg, mask2);

    // fac = a + lanczos_g - 0.5f
    Reg::RegTensor<float> facReg;
    Reg::Duplicate(tmpReg, lanczos_g - 0.5f, fullMask);
    Reg::Add(facReg, src0Reg, tmpReg, fullMask);

    // res = sqrtf(fac / exp1) / igammac_lanczos_sum_expg_scaled_float(a)
    Reg::RegTensor<float> resReg;
    Igammac_lanczos_sum_expg_scaled_float(helpReg, src0Reg, fullMask);
    Reg::Duplicate(tmpReg, exp1, fullMask);
    Reg::Div(tmpReg, facReg, tmpReg, fullMask);
    Reg::Sqrt(tmpReg, tmpReg, fullMask);
    Reg::Div(resReg, tmpReg, helpReg, fullMask);

    // a < 200.0f && x < 200.0f
    // res *= expf(a - x) * powf(x / fac, a)
    // reuse mask1, mask2
    Reg::Compares<float, CMPMODE::LT>(mask1, src0Reg, 200.0f, fullMask);
    Reg::Compares<float, CMPMODE::LT>(mask2, src1Reg, 200.0f, fullMask);
    Reg::And(mask1, mask1, mask2, fullMask);
    Reg::Not(mask2, mask0, fullMask);
    Reg::And(mask1, mask1, mask2, fullMask);
    Reg::Sub(tmpReg, src0Reg, src1Reg, mask1);
    Reg::Exp(tmpReg, tmpReg, mask1); // expf(a - x)
    // pass powf(x / fac, a) from UB
    Reg::Mul(tmpReg, tmpReg, powReg, mask1);
    Reg::Mul(tmpReg, resReg, tmpReg, mask1); // pass to dstReg at the end
    Reg::Copy(resReg, tmpReg, mask1); // Merging Mode: Zeroing, but can't affect regReg
    Reg::Copy(dstReg, resReg, mask1);

    // else
    Reg::RegTensor<float> numFacReg;
    Reg::Not(mask2, mask1, fullMask);
    Reg::Not(mask1, mask0, fullMask);
    Reg::And(mask2, mask1, mask2, fullMask);
    // num = x - a - lanczos_g + 0.5f
    // numfac = num / fac
    // res *= expf(
    //             a * (log1pf(numfac) - numfac) +
    //             x * (0.5f - lanczos_g) / fac )
    Reg::Duplicate(numFacReg, 0.5f - lanczos_g, fullMask);
    Reg::Sub(tmpReg, src1Reg, src0Reg, mask2);
    Reg::Add(numFacReg, tmpReg, numFacReg, mask2);
    Reg::Div(numFacReg, numFacReg, facReg, mask2);
    // compute res
    Reg::Adds(tmpReg, numFacReg, 1.0f, mask2);
    Reg::Log(tmpReg, tmpReg, mask2);
    Reg::Sub(tmpReg, tmpReg, numFacReg, mask2);
    Reg::Mul(tmpReg, src0Reg, tmpReg, mask2);

    Reg::Muls(tmpReg1, src1Reg, 0.5f - lanczos_g, mask2);
    Reg::Div(tmpReg1, tmpReg1, facReg, mask2);

    Reg::Add(tmpReg, tmpReg, tmpReg1, mask2);
    Reg::Exp(tmpReg, tmpReg, mask2);
    Reg::Mul(tmpReg, resReg, tmpReg, mask2);

    // return res
    Reg::Copy(dstReg, tmpReg, mask2);
}

__simd_callee__ inline void Igammac_helper_series_float(Reg::RegTensor<float>& dstReg, Reg::RegTensor<float>& src0Reg, Reg::RegTensor<float>& src1Reg,
                                                        Reg::RegTensor<float>& lgammaReg, Reg::RegTensor<float>& powReg, Reg::MaskReg mask)
{
    constexpr float machep = 5.9604644775390625e-8f;
    constexpr int maxiter = 25; // minimize from 2000 to 25, guarantee 1e-6 accuracy

    Reg::RegTensor<float> tmpReg;

    Reg::RegTensor<float> axReg;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Igammac_helper_fac_float(axReg, src0Reg, src1Reg, powReg, lgammaReg, fullMask);

    // ax == 0.0f
    Reg::MaskReg mask0, mask1;
    Reg::Compares<float, CMPMODE::EQ>(mask0, axReg, 0.0f, fullMask);
    Reg::Duplicate<float, Reg::MaskMergeMode::MERGING>(dstReg, 0.0f, mask0);

    // r = a
    // c = 1.0f
    // ans = 1.0f
    Reg::RegTensor<float> rReg;
    Reg::RegTensor<float> cReg;
    Reg::RegTensor<float> ansReg;
    Reg::Copy(rReg, src0Reg, fullMask);
    Reg::Duplicate(cReg, 1.0f, fullMask);
    Reg::Duplicate(ansReg, 1.0f, fullMask);

    // for loop
    // r += 1.0f
    // c *= x / r
    // ans += c
    mask1 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    for (uint32_t i = 0; i < maxiter; ++i) {
        Reg::Adds(rReg, rReg, 1.0f, fullMask);
        Reg::Div(tmpReg, src1Reg, rReg, fullMask);
        Reg::Mul(cReg, cReg, tmpReg, fullMask);
        Reg::Add(ansReg, ansReg, cReg, fullMask);

        // c <= machep * ans ; break
        // abort this if branch, 25 times for loop is enough
        Reg::Muls(tmpReg, ansReg, machep, fullMask);
        Reg::Compare<float, CMPMODE::LE>(mask1, cReg, tmpReg, fullMask);
        Reg::And(mask1, mask1, mask0, fullMask);
    }

    // return ans * ax / a
    Reg::Mul(tmpReg, ansReg, axReg, fullMask);
    Reg::Div(dstReg, tmpReg, src0Reg, mask);
}
} // namespace IGammaCInternal
} // namespace AscendC

#endif // __ASCENDC_API_REGBASE_IGAMMAC_HELPER_SERIES_H__