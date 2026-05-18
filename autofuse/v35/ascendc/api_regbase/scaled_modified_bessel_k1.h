/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_SCALED_MODIFIED_BESSEL_K1_H__
#define __ASCENDC_API_REGBASE_SCALED_MODIFIED_BESSEL_K1_H__

#include "modified_bessel_k1.h"

template <typename T, uint32_t currentIteration, uint32_t endIteration, uint32_t sliceNum>
__simd_callee__ inline void ScaledModifiedBesselK1BigSliceCompute(AscendC::Reg::RegTensor<T>& srcReg,
                                                                   AscendC::Reg::RegTensor<T>& bigDstReg,
                                                                   AscendC::Reg::MaskReg& branchMask,
                                                                   __ubuf__ T* dst, __ubuf__ T* tmpBuf,
                                                                   uint32_t offSet, uint32_t tensorLen) {
    AscendC::Reg::RegTensor<T> pReg, qReg, xFactorReg, constReg, iterReg, factorReg;
    ModifiedBesselImportData<T, sliceNum, K1_B>(pReg, qReg, constReg, branchMask, dst, tmpBuf, offSet, tensorLen);

    // x_factor = 8/x - 2
    AscendC::Reg::Duplicate(xFactorReg, (T)8, branchMask);
    AscendC::Reg::Div(xFactorReg, xFactorReg, srcReg, branchMask);
    AscendC::Reg::Adds(xFactorReg, xFactorReg, (T)(-2.0), branchMask);

    mainIter<T, currentIteration, endIteration, K1_B>(pReg, qReg, constReg, xFactorReg, iterReg, branchMask);
    if constexpr (sliceNum == 1) {
        // result_big = (0.5 * (b - p)) / sqrt(x), no exp(-x)
        AscendC::Reg::Sub(bigDstReg, constReg, pReg, branchMask);
        AscendC::Reg::Muls(bigDstReg, bigDstReg, (T)0.5, branchMask);
        ScaledModifiedBesselKFactorBigCompute<T>(srcReg, bigDstReg, branchMask);
    } else {
        ModifiedBesselExportData<T>(pReg, qReg, constReg, branchMask, dst, tmpBuf, offSet, tensorLen);
    }
}

template <typename T>
__simd_vf__ inline void ScaledModifiedBesselK1ImplVF(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmpBuf, uint32_t calCount) {
    uint32_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(T));
    uint16_t repeatTime = static_cast<uint16_t>(AscendC::CeilDivision(calCount, vlSize));
    uint32_t tensorLen = repeatTime * vlSize;
    uint32_t calCount2 = calCount;

    AscendC::Reg::RegTensor<T> srcReg, dstReg, factorReg, smallDstReg, bigDstReg;
    AscendC::Reg::MaskReg mask, branchMask;

    for (uint16_t i = 0U; i < repeatTime; ++i) {
        mask = AscendC::Reg::UpdateMask<T>(calCount);
        AscendC::Reg::LoadAlign(srcReg, src + i * vlSize);

        // ===== Big branch: x > 2.0 =====
        AscendC::Reg::Compares<T, CMPMODE::GT>(branchMask, srcReg, (T)2.0, mask);
        ScaledModifiedBesselK1BigSliceCompute<T, 1, 12, 0>(srcReg, bigDstReg, branchMask, dst, tmpBuf, i * vlSize, tensorLen);
    }

    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

    for (uint16_t i = 0U; i < repeatTime; ++i) {
        mask = AscendC::Reg::UpdateMask<T>(calCount2);
        AscendC::Reg::LoadAlign(srcReg, src + i * vlSize);

        // ===== Small branch: 0 < x <= 2.0 =====
        AscendC::Reg::Compares<T, CMPMODE::GT>(branchMask, srcReg, (T)0.0, mask);
        AscendC::Reg::Compares<T, CMPMODE::LE>(branchMask, srcReg, (T)2.0, branchMask);
        ModifiedBesselK1SmallCompute<T>(srcReg, smallDstReg, branchMask, dst, i * vlSize);
        // scaled: result_small *= exp(x)
        AscendC::Reg::Exp(factorReg, srcReg, branchMask);
        AscendC::Reg::Mul(smallDstReg, smallDstReg, factorReg, branchMask);

        // ===== Big branch: x > 2.0 =====
        AscendC::Reg::Compares<T, CMPMODE::GT>(branchMask, srcReg, (T)2.0, mask);
        ScaledModifiedBesselK1BigSliceCompute<T, 13, 24, 1>(srcReg, bigDstReg, branchMask, dst, tmpBuf, i * vlSize, tensorLen);

        AscendC::Reg::Select(dstReg, bigDstReg, smallDstReg, branchMask);

        ModifiedBesselKHandleSpecialCases<T>(srcReg, dstReg, mask);

        // Store output
        AscendC::Reg::StoreAlign(dst + i * vlSize, dstReg, mask);
    }
}

template <typename T>
__aicore__ inline void ScaledModifiedBesselK1Extend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                           const LocalTensor<uint8_t>& sharedTmpBuffer,
                                           const uint32_t calCount) {
    static_assert(SupportType<T, float>(), "Current data type is  not supported on current device!");
    ModifiedBesselI1Extend<T>(dst, src, sharedTmpBuffer, calCount);
    auto tmpUB = sharedTmpBuffer.ReinterpretCast<T>();
    ScaledModifiedBesselK1ImplVF<T>((__ubuf__ T*)dst.GetPhyAddr(),
                               (__ubuf__ T*)src.GetPhyAddr(),
                               (__ubuf__ T*)tmpUB.GetPhyAddr(),
                               calCount);
}

#endif  // __ASCENDC_API_REGBASE_SCALED_MODIFIED_BESSEL_K1_H__
