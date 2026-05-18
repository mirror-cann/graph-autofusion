/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCEND_API_AIRY_AI_H
#define __ASCEND_API_AIRY_AI_H

#include "trigonometric_function_utils.h"

using namespace AscendC;

constexpr float AN[] = {
    +3.46538101525629032477e-01,
    +1.20075952739645805542e+01,
    +7.62796053615234516538e+01,
    +1.68089224934630576269e+02,
    +1.59756391350164413639e+02,
    +7.05360906840444183113e+01,
    +1.40264691163389668864e+01,
    +9.99999999999999995305e-01,
};

constexpr float AD[] = {
    +5.67594532638770212846e-01,
    +1.47562562584847203173e+01,
    +8.45138970141474626562e+01,
    +1.77318088145400459522e+02,
    +1.64234692871529701831e+02,
    +7.14778400825575695274e+01,
    +1.40959135607834029598e+01,
    +1.00000000000000000470e+00,
};

constexpr float AFN[] = {
    -1.31696323418331795333e-01,
    -6.26456544431912369773e-01,
    -6.93158036036933542233e-01,
    -2.79779981545119124951e-01,
    -4.91900132609500318020e-02,
    -4.06265923594885404393e-03,
    -1.59276496239262096340e-04,
    -2.77649108155232920844e-06,
    -1.67787698489114633780e-08,
};

constexpr float AFD[] = {
    +1.33560420706553243746e+01,
    +3.26825032795224613948e+01,
    +2.67367040941499554804e+01,
    +9.18707402907259625840e+00,
    +1.47529146771666414581e+00,
    +1.15687173795188044134e-01,
    +4.40291641615211203805e-03,
    +7.54720348287414296618e-05,
    +4.51850092970580378464e-07,
};

constexpr float AGN[] = {
    +1.97339932091685679179e-02,
    +3.91103029615688277255e-01,
    +1.06579897599595591108e+00,
    +9.39169229816650230044e-01,
    +3.51465656105547619242e-01,
    +6.33888919628925490927e-02,
    +5.85804113048388458567e-03,
    +2.82851600836737019778e-04,
    +6.98793669997260967291e-06,
    +8.11789239554389293311e-08,
    +3.41551784765923618484e-10,
};

constexpr float AGD[] = {
    +9.30892908077441974853e+00,
    +1.98352928718312140417e+01,
    +1.55646628932864612953e+01,
    +5.47686069422975497931e+00,
    +9.54293611618961883998e-01,
    +8.64580826352392193095e-02,
    +4.12656523824222607191e-03,
    +1.01259085116509135510e-04,
    +1.17166733214413521882e-06,
    +4.91834570062930015649e-09,
};

constexpr float FX[] = {
    +5.80371000263110835022e-49,
    +1.66102180275302325571e-45,
    +4.23560559702020900772e-42,
    +9.55552622687759098105e-39,
    +1.89199419292176297248e-35,
    +3.25801400021127561678e-32,
    +4.82837674831311040933e-29,
    +6.08375470287451899020e-26,
    +6.42444496623549256793e-23,
    +5.58926712062487814148e-20,
    +3.92366551867866474662e-17,
    +2.16586336631062305353e-14,
    +9.09662613850461648405e-12,
    +2.78356759838241253103e-09,
    +5.84549195660306672875e-07,
    +7.71604938271604787019e-05,
    +5.55555555555555490022e-03,
    +1.66666666666666657415e-01,
    1.0,
};

constexpr float GX[] = {
    5.47129309017487506784e-50,
    1.62497404778193802201e-46,
    4.30943117471769969038e-43,
    1.01357821229360293531e-39,
    2.09810689944775812928e-36,
    3.78918106040265142003e-33,
    5.91112245422813666410e-30,
    7.87361510903187844295e-27,
    8.83419615233376792007e-24,
    8.21580242167040445954e-21,
    6.21114663078282631910e-18,
    3.72668797846969601332e-15,
    1.72172984605299955342e-12,
    5.88831607350125831115e-10,
    1.41319585764030204431e-07,
    2.20458553791887106206e-05,
    1.98412698412698401684e-03,
    8.33333333333333287074e-02,
    1.0,
};

constexpr float INV_SQRT_PI = 0.564189583547756286948f;
constexpr float PI_OVER_4 = 0.7853981633974483f;
constexpr float AIRY_AI_0 = 0.355028053887817239260;
constexpr float AIRY_AI_PRIME_0 = -0.258819403792806798405;

template<typename T, int32_t iterationNum, int32_t factorLen, const float* factorList>
__simd_callee__ inline void mainIterForAiryAi(
    Reg::RegTensor<T>& iterReg, Reg::RegTensor<T>& xReg, Reg::MaskReg& mask)
{    
    Reg::Mul(iterReg, iterReg, xReg, mask);
    Reg::Adds(iterReg, iterReg, (T)factorList[iterationNum], mask);
    if constexpr (iterationNum < factorLen - 1) {
        mainIterForAiryAi<T, iterationNum + 1, factorLen, factorList>(iterReg, xReg, mask);
    }
}

template<typename T>
__simd_callee__ inline void AiryAiNeg(
    Reg::RegTensor<T>& srcReg, Reg::RegTensor<T>& constReg,
    Reg::RegTensor<T>& tempRes, Reg::MaskReg& mask)
{
    Reg::RegTensor<T> z, z2, t, temp, tempX, afn, afd, agn, agd, sinT, cosT;
    Reg::Abs(temp, srcReg, mask);
    Reg::Sqrt(tempX, temp, mask);
    Reg::Mul(temp, temp, tempX, mask); //temp = (-x)^(3/2)
    Reg::Move(z, temp, mask);
    Reg::Move(t, temp, mask);
    Reg::Duplicate(constReg, (T)1.5, mask);
    Reg::Div(z, constReg, z, mask); // z = 1.5 * (-x)^(-3/2)
    Reg::Mul(z2, z, z, mask); // z^2
    Reg::Muls(t, t, (T)2.0, mask);
    Reg::Duplicate(constReg, (T)3.0, mask);
    Reg::Div(t, t, constReg, mask);
    Reg::Adds(t, t, PI_OVER_4, mask); // t = 2/3 * (-x)^(3/2) + pi/4

    Reg::Duplicate(afn, (T)0.0, mask);
    mainIterForAiryAi<T, 0, 9, AFN>(afn, z2, mask);

    Reg::Duplicate(afd, (T)0.0, mask);
    mainIterForAiryAi<T, 0, 9, AFD>(afd, z2, mask);

    Reg::Duplicate(agn, (T)0.0, mask);
    mainIterForAiryAi<T, 0, 11, AGN>(agn, z2, mask);

    Reg::Duplicate(agd, (T)0.0, mask);
    mainIterForAiryAi<T, 0, 10, AGD>(agd, z2, mask);

    Reg::Div(afn, afn, afd, mask);
    Reg::Mul(afn, afn, z2, mask);
    Reg::Adds(afn, afn, (T)1.0, mask);
    
    AutofuseSin<T>(sinT, t, mask);
    Reg::Mul(sinT, sinT, afn, mask);

    Reg::Div(agn, agn, agd, mask);
    Reg::Mul(agn, z, agn, mask);

    AutofuseCos<T>(cosT, t, mask);
    Reg::Mul(cosT, cosT, agn, mask);

    Reg::Sub(tempRes, sinT, cosT, mask);
    Reg::Sqrt(tempX, tempX, mask);
    Reg::Div(tempRes, tempRes, tempX, mask);
    Reg::Muls(tempRes, tempRes, INV_SQRT_PI, mask);
}

template <typename T>
__simd_callee__ inline void AiryAiMid(
    Reg::RegTensor<T>& srcReg, Reg::RegTensor<T>& tempRes, Reg::MaskReg& mask)
{
    Reg::RegTensor<T> x3, fx, gx;
    Reg::Move(x3, srcReg, mask);
    Reg::Mul(x3, x3, srcReg, mask);
    Reg::Mul(x3, x3, srcReg, mask);

    Reg::Duplicate(fx, (T)0.0, mask);
    mainIterForAiryAi<T, 0, 19, FX>(fx, x3, mask);

    Reg::Duplicate(gx, (T)0.0, mask);
    mainIterForAiryAi<T, 0, 19, GX>(gx, x3, mask);
    Reg::Mul(gx, gx, srcReg, mask);

    Reg::Muls(fx, fx, AIRY_AI_0, mask);
    Reg::Muls(gx, gx, AIRY_AI_PRIME_0, mask);

    Reg::Add(tempRes, fx, gx, mask);
}

template <typename T>
__simd_callee__ inline void AiryAiPos(
    Reg::RegTensor<T>& srcReg, Reg::RegTensor<T>& constReg,
    Reg::RegTensor<T>& tempRes, Reg::MaskReg& mask)
{
    Reg::RegTensor<T> y, zeta, tempX, temp, an, af;
    Reg::Abs(temp, srcReg, mask);
    Reg::Sqrt(tempX, temp, mask); // tempX = x ^ (1/2)
    Reg::Mul(zeta, tempX, srcReg, mask);
    Reg::Muls(zeta, zeta, (T)2.0, mask);
    Reg::Duplicate(constReg, (T)3.0, mask);
    Reg::Div(zeta, zeta, constReg, mask);
    Reg::Duplicate(constReg, (T)1.0, mask);
    Reg::Div(y, constReg, zeta, mask);

    Reg::Duplicate(an, (T)0.0, mask);
    mainIterForAiryAi<T, 0, 8, AN>(an, y, mask);

    Reg::Duplicate(af, (T)0.0, mask);
    mainIterForAiryAi<T, 0, 8, AD>(af, y, mask);

    Reg::Div(tempRes, an, af, mask);
    Reg::Muls(tempRes, tempRes, INV_SQRT_PI, mask);

    Reg::Sqrt(tempX, tempX, mask);
    Reg::Exp(temp, zeta, mask);
    Reg::Div(tempRes, tempRes, tempX, mask);
    Reg::Div(tempRes, tempRes, temp, mask);
    Reg::Duplicate(temp, (T)2.0, mask);
    Reg::Div(tempRes, tempRes, temp, mask);
}

template <typename T>
__aicore__ inline void AiryAiImplVF(__ubuf__ T* dst, __ubuf__ T* src, uint32_t calCount)
{
    uint32_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(T));
    uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, vlSize));

    Reg::RegTensor<T> srcReg, dstReg, constReg, tempRes;
    
    Reg::MaskReg maskNeg, maskMid, maskPos, mask;
    maskNeg = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    maskMid = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    maskPos = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    mask = Reg::UpdateMask<T>(calCount);

    for (uint16_t i = 0U; i < repeatTime; ++i) {
        Reg::LoadAlign(srcReg, src + i * vlSize);

        Reg::Compares<T, CMPMODE::LT>(maskNeg, srcReg, (T)(-2.09), mask);
        Reg::Compares<T, CMPMODE::GE>(maskPos, srcReg, (T)2.09, mask);
        Reg::Or(maskMid, maskNeg, maskPos, mask);
        Reg::Not(maskMid, maskMid, mask);

        //负半轴计算
        AiryAiNeg<T>(srcReg, constReg, tempRes, mask);
        Reg::Select(dstReg, tempRes, dstReg, maskNeg);
        

        //中间区域计算
        AiryAiMid<T>(srcReg, tempRes, mask);
        Reg::Select(dstReg, tempRes, dstReg, maskMid);
        

        //正半轴区域计算
        AiryAiPos<T>(srcReg, constReg, tempRes, mask);
        Reg::Select(dstReg, tempRes, dstReg, maskPos);

        Reg::StoreAlign(dst + i * vlSize, dstReg, mask);
    }
}

template<typename T>
__aicore__ inline void AiryAiExtend(
    const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    static_assert(SupportType<T, float>(), "Current data type is not support on current device!");
    
    VF_CALL<AiryAiImplVF<T>>(
        (__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), calCount);
}

#endif