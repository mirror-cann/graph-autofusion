/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef __ASCENDC_API_REGBASE_IGAMMA_H__
#define __ASCENDC_API_REGBASE_IGAMMA_H__

#include "igammac.h"

namespace AscendC {
namespace IGammaInternal{
template <typename T>
__simd_vf__ inline void IgammaImplXgt1p0AndGtA(__ubuf__ T* dstUb, __ubuf__ T* src0Ub, __ubuf__ T* src1Ub,
__ubuf__ float* workUb)
{
    Reg::MaskReg mask, mask1, mask2;
    Reg::RegTensor<T> src0Reg, src1Reg, dstReg, lgammaReg, powReg, tmpReg, oneReg;

    Reg::LoadAlign(src0Reg, src0Ub);
    Reg::LoadAlign(src1Reg, src1Ub);
    Reg::LoadAlign(dstReg, dstUb);
    Reg::LoadAlign(lgammaReg, workUb);
    Reg::LoadAlign(powReg, workUb + 128);
    Reg::LoadAlign(mask, workUb + 256);
    Reg::Duplicate(oneReg, 1.0f, mask);

    // x > 1 && x > a
    Reg::Compares<T, CMPMODE::GT>(mask1, src1Reg, 1.0f, mask);
    Reg::Compare<T, CMPMODE::GT>(mask1, src1Reg, src0Reg, mask1);

    //      x > 1.1     使用  1-igammac_helper_continued_fraction_float(a, x);
    Reg::Compares<T, CMPMODE::GT>(mask2, src1Reg, 1.1f, mask1);
    IGammaCInternal::Igammac_helper_continued_fraction_float(tmpReg, src0Reg, src1Reg, lgammaReg, powReg, mask2);
    Reg::Sub(tmpReg, oneReg, tmpReg, mask2);
    Reg::Copy(dstReg, tmpReg, mask2);

    //      x <= 1.1 使用 1-igammac_helper_series_complement_float(a, x);
    Reg::Not(mask2, mask2, mask1);
    IGammaCInternal::Igammac_helper_series_complement_float(tmpReg, src0Reg, src1Reg, lgammaReg, mask2);
    Reg::Sub(tmpReg, oneReg, tmpReg, mask2);
    Reg::Copy(dstReg, tmpReg, mask2);

    // store dst
    Reg::StoreAlign(dstUb, dstReg, mask);

    // store mask
    Reg::Not(mask1, mask1, mask);
    Reg::StoreAlign(workUb + 256, mask1);
}

template <typename T>
__simd_vf__ inline void IgammaImplXother(__ubuf__ T* dstUb, __ubuf__ T* src0Ub, __ubuf__ T* src1Ub,
    __ubuf__ float* workUb)
{
    Reg::MaskReg mask;
    Reg::RegTensor<T> src0Reg, src1Reg, dstReg, lgammaReg, powReg, tmpReg;

    Reg::LoadAlign(src0Reg, src0Ub);
    Reg::LoadAlign(src1Reg, src1Ub);
    Reg::LoadAlign(dstReg, dstUb);
    Reg::LoadAlign(lgammaReg, workUb);
    Reg::LoadAlign(powReg, workUb + 128);
    Reg::LoadAlign(mask, workUb + 256);

    IGammaCInternal::Igammac_helper_series_float(tmpReg, src0Reg, src1Reg, lgammaReg, powReg, mask);
    Reg::Copy(dstReg, tmpReg, mask);
    Reg::StoreAlign(dstUb, dstReg, mask);
}
} // namespace IGammaInternal

template <typename T>
__aicore__ inline void IgammaExtend(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const LocalTensor<uint8_t>& tmp, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(SupportType<T, float>(), "current data type is not supported on current device!");

    constexpr float lanczos_g = 6.024680040776729583740234375;
    constexpr uint32_t stride = GetVecLen() / sizeof(T);
    uint16_t repeatTime = (calCount + stride - 1) / stride;

    uint16_t tailNum = calCount % stride;
    bool hasTail = tailNum > 0;

    __ubuf__ T* dstUb = (__ubuf__ T*)dst.GetPhyAddr();
    __ubuf__ T* src0Ub = (__ubuf__ T*)src0.GetPhyAddr();
    __ubuf__ T* src1Ub = (__ubuf__ T*)src1.GetPhyAddr();
    __ubuf__ T* workUb = (__ubuf__ T*)tmp.GetPhyAddr();

    LocalTensor<T> LgammaLocal = tmp.ReinterpretCast<T>();
    LocalTensor<T> DivLocal = tmp[256].ReinterpretCast<T>();
    LocalTensor<T> PowLocal = tmp[512].ReinterpretCast<T>();
    LocalTensor<T> tmpLocal = tmp[768].ReinterpretCast<T>();

    for (uint16_t i = 0; i < repeatTime; ++i) {
        uint32_t sliceCnt = stride;
        if (i == repeatTime - 1 && hasTail) {
            sliceCnt = tailNum;
        }
        __ubuf__ T* dstChunkUb = dstUb + i * stride;
        __ubuf__ T* src0ChunkUb = src0Ub + i * stride;
        __ubuf__ T* src1ChunkUb = src1Ub + i * stride;

        Lgamma(LgammaLocal, src0[stride * i], sliceCnt);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(tmpLocal, src0[stride * i], lanczos_g - 0.5f, sliceCnt);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(DivLocal, src1[stride * i], tmpLocal, sliceCnt);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Power(PowLocal, DivLocal, src0[stride * i], sliceCnt);
        AscendC::PipeBarrier<PIPE_V>();

        IGammaCInternal::IgammaCImplPreJudge<T, true>(dstChunkUb, src0ChunkUb, src1ChunkUb, workUb, sliceCnt);
		AscendC::PipeBarrier<PIPE_V>();
        IGammaCInternal::IgammaCImplAsymptoticMask<T>(src0ChunkUb, src1ChunkUb, workUb);
		AscendC::PipeBarrier<PIPE_V>();
        IGammaCInternal::IgammaCImplAsymptotic<T, true>(dstChunkUb, src0ChunkUb, src1ChunkUb, workUb);
		AscendC::PipeBarrier<PIPE_V>();
        IGammaInternal::IgammaImplXgt1p0AndGtA<T>(dstChunkUb, src0ChunkUb, src1ChunkUb, workUb);
		AscendC::PipeBarrier<PIPE_V>();
        IGammaInternal::IgammaImplXother<T>(dstChunkUb, src0ChunkUb, src1ChunkUb, workUb);
		AscendC::PipeBarrier<PIPE_V>();
    }
}
} // namespace AscendC

#endif // __ASCENDC_API_REGBASE_IGAMMA_H__