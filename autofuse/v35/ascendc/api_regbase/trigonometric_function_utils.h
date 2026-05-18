/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_TRIGONOMETRIC_FUNCTION_UTILS_H__
#define __ASCENDC_API_REGBASE_TRIGONOMETRIC_FUNCTION_UTILS_H__

using namespace AscendC;

constexpr float PI_FOR_X_TODIV = 0.3183098733425140380859375;
constexpr float PI_0 = 3.140625;
constexpr float KPI_FIRS_PI_MULS = 0.0009670257568359375;
constexpr float KPI_TWI_PI_MULS = 6.2771141529083251953125e-7;
constexpr float KPI_THIR_PI_MULS = 1.21644916362129151821136474609375e-10;
constexpr float RES_MULTI_SCA = 2.604926501e-6;
constexpr float RES_ADDICT_UP = -0.0001980894471;
constexpr float SEC_ADDS = 0.008333049340;
constexpr float THI_ADDS = -0.1666665792;
constexpr float POINT_FIVE = 0.5;
constexpr float M4_SCA = 4.0;
constexpr float K2_SCA = -2.0;
constexpr float SCALAR_ONE = 1.0;
constexpr float COS_KPI_FOR_PI_MULS = -1.0290623200529979163359041220560e-13;
constexpr float COS_PI_DOWN = 1.57079637050628662109375;
constexpr float COS_PI_RESDOWN_ADDS_NEG = -0.00000004371139000189375;

template<typename T>
__simd_callee__ inline void CommonCalInSinCos(
    Reg::RegTensor<T>& dstReg, Reg::RegTensor<T>& x, Reg::RegTensor<T>& round,
    Reg::RegTensor<T>& kpi, Reg::MaskReg mask)
{
    Reg::Mul(kpi, x, x, mask);
    Reg::Muls(dstReg, round, POINT_FIVE, mask);
    Reg::Truncate<float, RoundMode::CAST_FLOOR, Reg::MaskMergeMode::ZEROING>(dstReg, dstReg, mask);
    Reg::Muls(dstReg, dstReg, M4_SCA, mask);
    Reg::Muls(round, round, K2_SCA, mask);
    Reg::Add(dstReg, dstReg, round, mask);
    Reg::Adds(dstReg, dstReg, SCALAR_ONE, mask);
    Reg::Muls(round, kpi, RES_MULTI_SCA, mask);
    Reg::Adds(round, round, RES_ADDICT_UP, mask);
    Reg::Mul(round, round, kpi, mask);
    Reg::Adds(round, round, SEC_ADDS, mask);
    Reg::Mul(round, round, kpi, mask);
    Reg::Adds(round, round, THI_ADDS, mask);
    Reg::Mul(round, round, kpi, mask);
    Reg::Adds(round, round, SCALAR_ONE, mask);
    Reg::Mul(round, round, x, mask);
    Reg::Mul(dstReg, round, dstReg, mask);
}

template<typename T>
__simd_callee__ inline void AutofuseSin(
    Reg::RegTensor<T>& dstReg, Reg::RegTensor<T>& srcReg, Reg::MaskReg mask)
{
    static_assert(SupportType<T, float>(), "Current data type is not support in current function!");
    Reg::RegTensor<T> x, round, kpi;

    Reg::Muls(round, srcReg, PI_FOR_X_TODIV, mask);
    Reg::Truncate<float, RoundMode::CAST_RINT, Reg::MaskMergeMode::ZEROING>(round, round, mask);
    Reg::Muls(kpi, round, PI_0, mask);
    Reg::Sub(x, srcReg, kpi, mask);
    Reg::Muls(kpi, round, KPI_FIRS_PI_MULS, mask);
    Reg::Sub(x, x, kpi, mask);
    Reg::Muls(kpi, round, KPI_TWI_PI_MULS, mask);
    Reg::Sub(x, x, kpi, mask);
    Reg::Muls(kpi, round, KPI_THIR_PI_MULS, mask);
    Reg::Sub(x, x, kpi, mask);

    CommonCalInSinCos<T>(dstReg, x, round, kpi, mask);
}

template<typename T>
__simd_callee__ inline void AutofuseCos(
    Reg::RegTensor<T>& dstReg, Reg::RegTensor<T>& srcReg, Reg::MaskReg mask)
{
    static_assert(SupportType<T, float>(), "Current data type is not support in current function!");
    Reg::RegTensor<T> x, round, kpi;
    
    Reg::Muls(round, srcReg, PI_FOR_X_TODIV, mask);
    Reg::Adds(round, round, POINT_FIVE, mask);
    Reg::Truncate<float, RoundMode::CAST_RINT, Reg::MaskMergeMode::ZEROING>(round, round, mask);
    Reg::Muls(kpi, round, PI_0, mask);
    Reg::Sub(x, srcReg, kpi, mask);
    Reg::Muls(kpi, round, KPI_FIRS_PI_MULS, mask);
    Reg::Sub(x, x, kpi, mask);
    Reg::Adds(x, x, COS_PI_DOWN, mask);
    Reg::Muls(kpi, round, KPI_TWI_PI_MULS, mask);
    Reg::Sub(x, x, kpi, mask);
    Reg::Muls(kpi, round, KPI_THIR_PI_MULS, mask);
    Reg::Sub(x, x, kpi, mask);
    Reg::Muls(kpi, round, COS_KPI_FOR_PI_MULS, mask);
    Reg::Sub(x, x, kpi, mask);
    Reg::Adds(x, x, COS_PI_RESDOWN_ADDS_NEG, mask);

    CommonCalInSinCos<T>(dstReg, x, round, kpi, mask);

    Reg::Mins(dstReg, dstReg, SCALAR_ONE, mask);
    Reg::Maxs(dstReg, dstReg, -SCALAR_ONE, mask);
}

#endif