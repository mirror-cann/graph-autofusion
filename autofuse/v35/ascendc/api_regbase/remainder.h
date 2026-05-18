/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_REMAINDER_H
#define __ASCENDC_API_REGBASE_REMAINDER_H

// Remainder(x1, x2) = x1 - x2 * floor(x1/x2)

template <typename T>
__aicore__ inline void RemainderImplVF(__ubuf__ T* dst, __ubuf__ T* src1, __ubuf__ T* src2,
                                        uint32_t count, uint16_t repeat_time) {
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
    AscendC::MicroAPI::RegTensor<T> srcVreg1;
    AscendC::MicroAPI::RegTensor<T> srcVreg2;
    AscendC::MicroAPI::RegTensor<T> dstVreg;
    AscendC::MicroAPI::MaskReg mask;
    AscendC::MicroAPI::RegTensor<T> quotient;
    AscendC::MicroAPI::RegTensor<T> truncatedQuotient;
    AscendC::MicroAPI::RegTensor<T> mulResult;

    for (uint16_t i = 0; i < repeat_time; i++) {
        mask = AscendC::MicroAPI::UpdateMask<T>(count);
        AscendC::MicroAPI::DataCopy(srcVreg1, src1 + i * oneRepElm);
        AscendC::MicroAPI::DataCopy(srcVreg2, src2 + i * oneRepElm);

        // Calculate x1 / x2
        AscendC::MicroAPI::Div(quotient, srcVreg1, srcVreg2, mask);

        // Truncate to floor
        AscendC::MicroAPI::Truncate<T, AscendC::RoundMode::CAST_FLOOR>(truncatedQuotient, quotient, mask);

        // Calculate x2 * floor(x1/x2)
        AscendC::MicroAPI::Mul(mulResult, srcVreg2, truncatedQuotient, mask);

        // Calculate remainder: x1 - x2 * floor(x1/x2)
        AscendC::MicroAPI::Sub(dstVreg, srcVreg1, mulResult, mask);

        AscendC::MicroAPI::DataCopy(dst + i * oneRepElm, dstVreg, mask);
    }
}

template <typename T>
__aicore__ inline void RemainderIntImplVF(__ubuf__ T* dst, __ubuf__ T* src1, __ubuf__ T* src2,
                                        uint32_t count, uint16_t repeatTime) {
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));

    for (uint16_t i = 0; i < repeatTime; i++) {
        AscendC::Reg::RegTensor<T> srcReg1;
        AscendC::Reg::RegTensor<T> srcReg2;
        AscendC::Reg::RegTensor<T> remReg;
        AscendC::Reg::MaskReg mask;
        AscendC::Reg::MaskReg signDiffMask;
        AscendC::Reg::MaskReg src1Mask;

        mask = AscendC::Reg::UpdateMask<T>(count);

        AscendC::Reg::DataCopy(srcReg1, src1 + i * oneRepElm);
        AscendC::Reg::DataCopy(srcReg2, src2 + i * oneRepElm);

        // Calculate x1 mod x2
        AscendC::Reg::Div(remReg, srcReg1, srcReg2, mask);
        AscendC::Reg::Mul(remReg, srcReg2, remReg, mask);
        AscendC::Reg::Sub(remReg, srcReg1, remReg, mask);

        // sign diff mask
        AscendC::Reg::Compares<T, CMPMODE::GT>(src1Mask, srcReg1, 0, mask);
        AscendC::Reg::Compares<T, CMPMODE::GT>(signDiffMask, srcReg2, 0, mask);
        AscendC::Reg::Xor(signDiffMask, src1Mask, signDiffMask, mask);
        AscendC::Reg::Compares<T, CMPMODE::NE>(src1Mask, remReg, 0, mask);
        AscendC::Reg::And(signDiffMask, signDiffMask, src1Mask, mask);

        AscendC::Reg::Add<T, AscendC::Reg::MaskMergeMode::MERGING>(remReg, remReg, srcReg2, signDiffMask);

        AscendC::Reg::DataCopy(dst + i * oneRepElm, remReg, mask);
    }
}

template <typename T>
__aicore__ inline void RemainderExtend(const AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src1,
                                   const AscendC::LocalTensor<T>& src2, const uint32_t size) {
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
    uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(size, oneRepElm));
    VF_CALL<RemainderImplVF<T>>((__ubuf__ T*)dst.GetPhyAddr(), (__ubuf__ T*)src1.GetPhyAddr(),
                                (__ubuf__ T*)src2.GetPhyAddr(), size, repeat_time);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void RemainderExtend(const AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src1,
                                   const AscendC::LocalTensor<T>& src2, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t size)
{
    static_assert(SupportType<T, int32_t, float>(), "RemainderExtend only support int32_t/float data type on current device!");
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
    uint16_t repeatTime = static_cast<uint16_t>(AscendC::CeilDivision(size, oneRepElm));

    if constexpr (IsSameType<T, int32_t>::value) {
        VF_CALL<RemainderIntImplVF<T>>((__ubuf__ T*)dst.GetPhyAddr(), (__ubuf__ T*)src1.GetPhyAddr(),
                                        (__ubuf__ T*)src2.GetPhyAddr(), size, repeatTime);
    } else {
        RemainderExtend(dst, src1, src2, size);
    }
}

#endif  // __ASCENDC_API_REGBASE_REMAINDER_H
