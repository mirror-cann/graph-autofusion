/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_MODIFIED_BESSEL_UTILS_H__
#define __ASCENDC_API_REGBASE_MODIFIED_BESSEL_UTILS_H__

constexpr uint32_t MODIFIED_BESSEL_FLOAT_NAN = 0x7fc00000;

#ifndef INFINITY
#define INFINITY (1.0f / 0.0f)
#endif

template <typename T, uint32_t currentIteration, uint32_t endIteration, const float* factorList>
__simd_callee__ inline void mainIter(AscendC::Reg::RegTensor<T>& pReg, AscendC::Reg::RegTensor<T>& qReg, 
                                    AscendC::Reg::RegTensor<T>& constReg, AscendC::Reg::RegTensor<T>& xFactorReg,
                                    AscendC::Reg::RegTensor<T>& iterReg, AscendC::Reg::MaskReg& branchMask) {
    AscendC::Reg::Move(pReg, qReg, branchMask);
    AscendC::Reg::Move(qReg, constReg, branchMask);
    AscendC::Reg::Mul(iterReg, xFactorReg, qReg, branchMask);
    AscendC::Reg::Sub(iterReg, iterReg, pReg, branchMask);
    AscendC::Reg::Adds(constReg, iterReg, (T)factorList[currentIteration], branchMask);
    if constexpr (currentIteration < endIteration) {
        mainIter<T, currentIteration + 1, endIteration, factorList>(pReg, qReg, constReg, xFactorReg, iterReg, branchMask);
    }
}

template <typename T, uint32_t sliceNum, const float* factorList>
__simd_callee__ inline void ModifiedBesselImportData(AscendC::Reg::RegTensor<T>& pReg, AscendC::Reg::RegTensor<T>& qReg, 
                                    AscendC::Reg::RegTensor<T>& constReg, AscendC::Reg::MaskReg& branchMask, __ubuf__ T* dst,  
                                    __ubuf__ T* tmpBuf, uint32_t offSet, uint32_t tensorLen) {
    if constexpr (sliceNum == 0) {
        AscendC::Reg::Duplicate(pReg, (T)0.0, branchMask);
        AscendC::Reg::Duplicate(qReg, (T)0.0, branchMask);
        AscendC::Reg::Duplicate(constReg, factorList[0], branchMask);
    } else {
        AscendC::Reg::LoadAlign(constReg, dst + offSet);
        AscendC::Reg::LoadAlign(pReg, tmpBuf + offSet);
        AscendC::Reg::LoadAlign(qReg, tmpBuf + tensorLen + offSet);
    }
}

template <typename T>
__simd_callee__ inline void ModifiedBesselExportData(AscendC::Reg::RegTensor<T>& pReg, AscendC::Reg::RegTensor<T>& qReg, 
                                    AscendC::Reg::RegTensor<T>& constReg, AscendC::Reg::MaskReg& branchMask, __ubuf__ T* dst,  
                                    __ubuf__ T* tmpBuf, uint32_t offSet, uint32_t tensorLen) {
    AscendC::Reg::StoreAlign(dst + offSet, constReg, branchMask);
    AscendC::Reg::StoreAlign(tmpBuf + offSet, pReg, branchMask);
    AscendC::Reg::StoreAlign(tmpBuf + tensorLen + offSet, qReg, branchMask);
}

template <typename T>
__simd_callee__ inline void ModifiedBesselKFactorBigCompute(AscendC::Reg::RegTensor<T>& srcReg,
                                                    AscendC::Reg::RegTensor<T>& bigDstReg, AscendC::Reg::MaskReg& branchMask) {
    AscendC::Reg::RegTensor<T> factorReg;
    AscendC::Reg::Muls(factorReg, srcReg, (T)(-1), branchMask);
    AscendC::Reg::Exp(factorReg, factorReg, branchMask);
    AscendC::Reg::Mul(bigDstReg, bigDstReg, factorReg, branchMask);
    AscendC::Reg::Sqrt(factorReg, srcReg, branchMask);
    AscendC::Reg::Div(bigDstReg, bigDstReg, factorReg, branchMask);
}

template <typename T>
__simd_callee__ inline void ScaledModifiedBesselKFactorBigCompute(AscendC::Reg::RegTensor<T>& srcReg,
                                                    AscendC::Reg::RegTensor<T>& bigDstReg, AscendC::Reg::MaskReg& branchMask) {
    AscendC::Reg::RegTensor<T> factorReg;
    AscendC::Reg::Sqrt(factorReg, srcReg, branchMask);
    AscendC::Reg::Div(bigDstReg, bigDstReg, factorReg, branchMask);
}

template <typename T>
__simd_callee__ inline void ModifiedBesselKHandleSpecialCases(AscendC::Reg::RegTensor<T>& srcReg,
                                                    AscendC::Reg::RegTensor<T>& dstReg, AscendC::Reg::MaskReg& mask) {
    // ===== x < 0: NaN =====
    AscendC::Reg::RegTensor<T> nanReg, infReg; 
    AscendC::Reg::MaskReg branchMask;
    AscendC::Reg::Compares<T, CMPMODE::LT>(branchMask, srcReg, (T)0.0, mask);
    AscendC::Reg::Duplicate(nanReg, (float&)MODIFIED_BESSEL_FLOAT_NAN, branchMask);
    AscendC::Reg::Select(dstReg, nanReg, dstReg, branchMask);

    // ===== x == 0: Inf =====
    AscendC::Reg::Compares<T, CMPMODE::EQ>(branchMask, srcReg, (T)0.0, mask);
    AscendC::Reg::Duplicate(infReg, INFINITY, branchMask);
    AscendC::Reg::Select(dstReg, infReg, dstReg, branchMask);

    // handle nan input
    AscendC::Reg::Compare<T, CMPMODE::NE>(branchMask, srcReg, srcReg, mask);
    AscendC::Reg::Duplicate(nanReg, (float&)MODIFIED_BESSEL_FLOAT_NAN, mask);
    AscendC::Reg::Select(dstReg, nanReg, dstReg, branchMask);
}

#endif // __ASCENDC_API_REGBASE_MODIFIED_BESSEL_UTILS_H__