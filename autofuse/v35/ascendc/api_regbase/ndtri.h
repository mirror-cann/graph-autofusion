/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_NDTRI_H__
#define __ASCENDC_API_REGBASE_NDTRI_H__

namespace AscendC {

constexpr float NDTRI_S2PI = 2.50662827463100050242f;
constexpr float NDTRI_M_LN2 = 0.69314718055994530942f;
constexpr float NDTRI_EXP_NEG_2 = 0.13533528323661269189f;
constexpr float NDTRI_TWO_LN2 = 1.38629436111989061883f;
constexpr float NDTRI_MIN_Y = 1.0e-10f;

constexpr float NDTRI_P0[5] = {
    -5.99633501014107895267E1f, 9.80010754185999661536E1f,
    -5.66762857469070293439E1f, 1.39312609387279679503E1f,
    -1.23916583867381258016E0f,
};

constexpr float NDTRI_Q0[8] = {
    1.95448858338141759834E0f, 4.67627912898881538453E0f,
    8.63602421391490525046E1f, -2.25462687854119370527E2f,
    2.00260212380060660359E2f, -8.20372256168333339912E1f,
    1.59056225126211695515E1f, -1.18331621121330003142E0f,
};

constexpr float NDTRI_P1[9] = {
    4.05544892305962419923E0f, 3.15251094599893866154E1f,
    5.71628192246421288162E1f, 4.40805073893200834700E1f,
    1.46849561928858024014E1f, 2.18663306850790267539E0f,
    -1.40256079171354495875E-1f, -3.50424626827848203418E-2f,
    -8.57456785154685413611E-4f,
};

constexpr float NDTRI_Q1[8] = {
    1.57799883256466749731E1f, 4.53907635128879210584E1f,
    4.13172038254672030440E1f, 1.50425385692907503408E1f,
    2.50464946208309415979E0f, -1.42182922854787788574E-1f,
    -3.80806407691578277194E-2f, -9.33259480895457427372E-4f,
};

constexpr float NDTRI_P2[9] = {
    3.23774891776946035970E0f, 6.91522889068984211695E0f,
    3.93881064292473654471E0f, 1.33303460815807582389E0f,
    2.01485389549079005769E-1f, 1.23716634817820021358E-2f,
    3.01581553508235416007E-4f, 2.65806974686737550832E-6f,
    6.23974539184983293730E-9f,
};

constexpr float NDTRI_Q2[8] = {
    6.02427039364742014255E0f, 3.67983563856160859403E0f,
    1.37702099489081330271E0f, 2.16236993594496635890E-1f,
    1.34204006088543172337E-2f, 3.28014464643627710824E-4f,
    2.89247864776880622627E-6f, 6.79019408009981274425E-9f,
};

constexpr uint32_t NDTRI_P0_START_IDX = 1;
constexpr uint32_t NDTRI_P0_END_IDX = sizeof(NDTRI_P0) / sizeof(NDTRI_P0[0]) - 1;
constexpr uint32_t NDTRI_Q0_START_IDX = 0;
constexpr uint32_t NDTRI_Q0_END_IDX = sizeof(NDTRI_Q0) / sizeof(NDTRI_Q0[0]) - 1;
constexpr uint32_t NDTRI_P1_START_IDX = 1;
constexpr uint32_t NDTRI_P1_END_IDX = sizeof(NDTRI_P1) / sizeof(NDTRI_P1[0]) - 1;
constexpr uint32_t NDTRI_Q1_START_IDX = 0;
constexpr uint32_t NDTRI_Q1_END_IDX = sizeof(NDTRI_Q1) / sizeof(NDTRI_Q1[0]) - 1;
constexpr uint32_t NDTRI_P2_START_IDX = 1;
constexpr uint32_t NDTRI_P2_END_IDX = sizeof(NDTRI_P2) / sizeof(NDTRI_P2[0]) - 1;
constexpr uint32_t NDTRI_Q2_START_IDX = 0;
constexpr uint32_t NDTRI_Q2_END_IDX = sizeof(NDTRI_Q2) / sizeof(NDTRI_Q2[0]) - 1;

template <uint32_t currentIdx, uint32_t endIdx, const float* coeffs>
__simd_callee__ inline void NdtriPolyEval(Reg::RegTensor<float>& result,
                                           Reg::RegTensor<float>& x,
                                           Reg::MaskReg& mask) {
    Reg::Mul(result, result, x, mask);
    Reg::Adds(result, result, coeffs[currentIdx], mask);
    if constexpr (currentIdx < endIdx) {
        NdtriPolyEval<currentIdx + 1, endIdx, coeffs>(result, x, mask);
    }
}

__simd_callee__ inline void NdtriComputeMiddle(
    Reg::RegTensor<float>& dstReg, Reg::RegTensor<float>& srcReg,
    Reg::MaskReg& mask)
{
    Reg::RegTensor<float> y2Reg, numReg, denReg, tempReg;

    Reg::Mul(y2Reg, srcReg, srcReg, mask);

    Reg::Duplicate(numReg, NDTRI_P0[0], mask);
    NdtriPolyEval<NDTRI_P0_START_IDX, NDTRI_P0_END_IDX, NDTRI_P0>(numReg, y2Reg, mask);

    Reg::Duplicate(denReg, 1.0f, mask);
    NdtriPolyEval<NDTRI_Q0_START_IDX, NDTRI_Q0_END_IDX, NDTRI_Q0>(denReg, y2Reg, mask);

    Reg::Mul(tempReg, y2Reg, numReg, mask);
    Reg::Div(tempReg, tempReg, denReg, mask);
    Reg::Mul(tempReg, tempReg, srcReg, mask);
    Reg::Add(tempReg, srcReg, tempReg, mask);
    Reg::Muls(dstReg, tempReg, NDTRI_S2PI, mask);
}

__simd_callee__ inline void NdtriComputeTail(
    Reg::RegTensor<float>& dstReg, Reg::RegTensor<float>& srcReg,
    Reg::MaskReg& mask)
{
    Reg::RegTensor<float> xReg, x0Reg, zReg, x1Reg, x1MidReg, x1ExtReg;
    Reg::RegTensor<float> log2YReg, negLog2YReg, log2XReg, tempReg, constReg;
    Reg::MaskReg tailMidMask, tailExtMask;

    Reg::Duplicate(constReg, 8.0f, mask);
    Reg::Log2(log2YReg, srcReg, mask);
    Reg::Neg(negLog2YReg, log2YReg, mask);
    Reg::Muls(tempReg, negLog2YReg, NDTRI_TWO_LN2, mask);
    Reg::Sqrt(xReg, tempReg, mask);

    Reg::Log2(log2XReg, xReg, mask);
    Reg::Muls(tempReg, log2XReg, NDTRI_M_LN2, mask);
    Reg::Div(tempReg, tempReg, xReg, mask);
    Reg::Sub(x0Reg, xReg, tempReg, mask);

    Reg::Duplicate(zReg, 1.0f, mask);
    Reg::Div(zReg, zReg, xReg, mask);

    Reg::Compare<float, CMPMODE::LT>(tailMidMask, xReg, constReg, mask);
    Reg::Compare<float, CMPMODE::GE>(tailExtMask, xReg, constReg, mask);

    Reg::Duplicate(x1MidReg, NDTRI_P1[0], tailMidMask);
    NdtriPolyEval<NDTRI_P1_START_IDX, NDTRI_P1_END_IDX, NDTRI_P1>(x1MidReg, zReg, tailMidMask);
    Reg::Duplicate(tempReg, 1.0f, tailMidMask);
    NdtriPolyEval<NDTRI_Q1_START_IDX, NDTRI_Q1_END_IDX, NDTRI_Q1>(tempReg, zReg, tailMidMask);
    Reg::Div(x1MidReg, x1MidReg, tempReg, tailMidMask);
    Reg::Mul(x1MidReg, x1MidReg, zReg, tailMidMask);

    Reg::Duplicate(x1ExtReg, NDTRI_P2[0], tailExtMask);
    NdtriPolyEval<NDTRI_P2_START_IDX, NDTRI_P2_END_IDX, NDTRI_P2>(x1ExtReg, zReg, tailExtMask);
    Reg::Duplicate(tempReg, 1.0f, tailExtMask);
    NdtriPolyEval<NDTRI_Q2_START_IDX, NDTRI_Q2_END_IDX, NDTRI_Q2>(tempReg, zReg, tailExtMask);
    Reg::Div(x1ExtReg, x1ExtReg, tempReg, tailExtMask);
    Reg::Mul(x1ExtReg, x1ExtReg, zReg, tailExtMask);

    Reg::Select(x1Reg, x1MidReg, x1ExtReg, tailMidMask);
    Reg::Sub(dstReg, x0Reg, x1Reg, mask);
}

__simd_callee__ inline void NdtriOverrideSpecialCases(
    Reg::RegTensor<float>& dstReg, Reg::RegTensor<float>& srcReg,
    Reg::MaskReg& mask)
{
    Reg::RegTensor<float> tempReg, constReg;
    Reg::MaskReg compMask;

    Reg::Duplicate(constReg, 0.0f, mask);
    Reg::Compare<float, CMPMODE::EQ>(compMask, srcReg, constReg, mask);
    Reg::Duplicate(tempReg, (float &)F32_NEG_INF, mask);
    Reg::Select(dstReg, tempReg, dstReg, compMask);

    Reg::Duplicate(constReg, 1.0f, mask);
    Reg::Compare<float, CMPMODE::EQ>(compMask, srcReg, constReg, mask);
    Reg::Duplicate(tempReg, (float &)F32_INF, mask);
    Reg::Select(dstReg, tempReg, dstReg, compMask);

    // Build validMask: elements where 0 <= src <= 1 (excluding NaN, ±inf)
    Reg::MaskReg validMask;
    Reg::Compare<float, CMPMODE::GE>(validMask, srcReg, srcReg, mask);
    Reg::Duplicate(constReg, (float &)F32_INF, mask);
    Reg::Compare<float, CMPMODE::NE>(validMask, srcReg, constReg, validMask);
    Reg::Neg(constReg, constReg, mask);
    Reg::Compare<float, CMPMODE::NE>(validMask, srcReg, constReg, validMask);
    Reg::Duplicate(constReg, 0.0f, mask);
    Reg::Compare<float, CMPMODE::GE>(validMask, srcReg, constReg, validMask);
    Reg::Duplicate(constReg, 1.0f, mask);
    Reg::Compare<float, CMPMODE::LE>(validMask, srcReg, constReg, validMask);
    // Set invalid elements (validMask=0) to NaN
    Reg::Duplicate(tempReg, (float &)F32_NAN, mask);
    Reg::Select(dstReg, dstReg, tempReg, validMask);
}

template <typename T>
__simd_vf__ inline void NdtriCoreImpl(
    __ubuf__ T* dstUb, __ubuf__ T* srcUb, uint32_t calCount, uint16_t repeatTimes)
{
    Reg::RegTensor<float> castReg, yAbsReg, yMidReg, resultReg, tempReg, tempRegNeg, constReg;
    Reg::MaskReg mask, middleMask, tailMask, codeMask, clampMask;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = Reg::UpdateMask<float>(calCount);
        Reg::LoadAlign(castReg, srcUb + i * B32_DATA_NUM_PER_REPEAT);

        Reg::Duplicate(constReg, 0.5f, mask);
        Reg::Compare<float, CMPMODE::GT>(codeMask, castReg, constReg, mask);

        Reg::Duplicate(constReg, 1.0f, mask);
        Reg::Sub(yAbsReg, constReg, castReg, mask);
        Reg::Select(yAbsReg, yAbsReg, castReg, codeMask);

        Reg::Duplicate(constReg, NDTRI_MIN_Y, mask);

        Reg::Compare<float, CMPMODE::LT>(clampMask, yAbsReg, constReg, mask);
        Reg::Select(yAbsReg, constReg, yAbsReg, clampMask);

        Reg::Duplicate(constReg, NDTRI_EXP_NEG_2, mask);
        Reg::Compare<float, CMPMODE::GT>(middleMask, yAbsReg, constReg, mask);
        Reg::Compare<float, CMPMODE::LE>(tailMask, yAbsReg, constReg, mask);

        Reg::Adds(yMidReg, castReg, -0.5f, mask);
        Reg::Duplicate(resultReg, 0.0f, mask);
        NdtriComputeMiddle(resultReg, yMidReg, middleMask);

        Reg::Duplicate(tempReg, 0.0f, mask);
        Reg::Duplicate(tempRegNeg, 0.0f, mask);
        NdtriComputeTail(tempReg, yAbsReg, tailMask);
        Reg::Neg(tempRegNeg, tempReg, tailMask);
        Reg::Select(tempReg, tempReg, tempRegNeg, codeMask);

        Reg::Select(resultReg, resultReg, tempReg, middleMask);

        NdtriOverrideSpecialCases(resultReg, castReg, mask);

        Reg::StoreAlign(dstUb + i * B32_DATA_NUM_PER_REPEAT, resultReg, mask);
    }
}

template <typename T>
__aicore__ inline void NdtriExtend(
    const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<uint8_t>& sharedTmpBuffer,
    const uint32_t calCount)
{
    static_assert((std::is_same_v<T, float>), "Ndtri only supports float on current device!");
    if ASCEND_IS_AIC {
        return;
    }
    __ubuf__ T* dstUb = (__ubuf__ T*)dst.GetPhyAddr();
    __ubuf__ T* srcUb = (__ubuf__ T*)src.GetPhyAddr();
    uint16_t repeatTimes = CeilDivision(calCount, B32_DATA_NUM_PER_REPEAT);
    NdtriCoreImpl<T>(dstUb, srcUb, calCount, repeatTimes);
}

} // namespace AscendC

#endif // __ASCENDC_API_REGBASE_NDTRI_H__