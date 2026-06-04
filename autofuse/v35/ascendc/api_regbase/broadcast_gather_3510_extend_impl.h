/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IMPL_PAD_BROADCAST_BROADCAST_GATHER_3510_EXTEND_IMPL_H
#define IMPL_PAD_BROADCAST_BROADCAST_GATHER_3510_EXTEND_IMPL_H

#include "kernel_basic_intf.h"
#include "kernel_tensor.h"

namespace AscendC {
namespace BroadcastInternal {
template <typename T>
__simd_vf__ inline void VfGenIndex(
    __ubuf__ T* indexUb, uint32_t sizeI0, uint32_t sizeI1, uint32_t sizeI2, uint32_t strideI0, uint32_t strideI1,
    uint32_t strideI2)
{
    Reg::RegTensor<T> v0;
    Reg::RegTensor<T> v1;
    Reg::RegTensor<T> v2;

    Reg::RegTensor<T> vr0;

    Reg::RegTensor<T> vd0;
    Reg::RegTensor<T> vd1;
    Reg::RegTensor<T> vd2;

    Reg::RegTensor<T> vi0;
    Reg::RegTensor<T> vi1;
    Reg::RegTensor<T> vi2;

    Reg::RegTensor<T> vs0;
    Reg::RegTensor<T> vs1;
    Reg::RegTensor<T> vs2;

    Reg::MaskReg p0 = Reg::CreateMask<T>();
    Reg::Arange(v0, 0);

    Reg::Duplicate(v1, (T)sizeI2, p0);
    Reg::Div(vd0, v0, v1, p0);
    Reg::Mul(v2, vd0, v1, p0);
    Reg::Sub(vi2, v0, v2, p0);

    Reg::Duplicate(v1, (T)sizeI1, p0);
    Reg::Div(vd1, vd0, v1, p0);
    Reg::Mul(v2, vd1, v1, p0);
    Reg::Sub(vi1, vd0, v2, p0);

    Reg::Duplicate(v1, (T)sizeI0, p0);
    Reg::Div(vd2, vd1, v1, p0);
    Reg::Mul(v2, vd2, v1, p0);
    Reg::Sub(vi0, vd1, v2, p0);

    Reg::Duplicate(vs0, (T)strideI0, p0);
    Reg::Duplicate(vs1, (T)strideI1, p0);
    Reg::Duplicate(vs2, (T)strideI2, p0);

    Reg::Mul(vr0, vs2, vi2, p0);
    Reg::MulAddDst(vr0, vs1, vi1, p0);
    Reg::MulAddDst(vr0, vs0, vi0, p0);

    Reg::StoreAlign(indexUb, vr0, p0);
}

__simd_vf__ inline void VfGenIndexB8(
    __ubuf__ int16_t* indexUb, uint32_t sizeI0, uint32_t sizeI1, uint32_t sizeI2, uint32_t strideI0,
    uint32_t strideI1, uint32_t strideI2)
{
    constexpr uint32_t VF_LEN_HALF = GetVecLen() / sizeof(int16_t);
    Reg::RegTensor<int16_t> v0, v1, v2, v3;
    Reg::RegTensor<int16_t> vr0, vr1;

    Reg::RegTensor<int16_t> vd0, vd1, vd2, vd3, vd4, vd5;
    Reg::RegTensor<int16_t> vi0, vi1, vi2, vi3, vi4, vi5;
    Reg::RegTensor<int16_t> vs0, vs1, vs2;

    Reg::MaskReg p0 = Reg::CreateMask<int16_t>();

    Reg::Arange(v0, (int16_t)0);
    Reg::Arange(v3, (int16_t)VF_LEN_HALF);

    Reg::Duplicate(v1, (int16_t)sizeI2, p0);
    Reg::Div(vd0, v0, v1, p0);
    Reg::Mul(v2, vd0, v1, p0);
    Reg::Sub(vi2, v0, v2, p0);
    Reg::Div(vd3, v3, v1, p0);
    Reg::Mul(v2, vd3, v1, p0);
    Reg::Sub(vi5, v3, v2, p0);

    Reg::Duplicate(v1, (int16_t)sizeI1, p0);
    Reg::Div(vd1, vd0, v1, p0);
    Reg::Mul(v2, vd1, v1, p0);
    Reg::Sub(vi1, vd0, v2, p0);
    Reg::Div(vd4, vd3, v1, p0);
    Reg::Mul(v2, vd4, v1, p0);
    Reg::Sub(vi4, vd3, v2, p0);

    Reg::Duplicate(v1, (int16_t)sizeI0, p0);
    Reg::Div(vd2, vd1, v1, p0);
    Reg::Mul(v2, vd2, v1, p0);
    Reg::Sub(vi0, vd1, v2, p0);
    Reg::Div(vd5, vd4, v1, p0);
    Reg::Mul(v2, vd5, v1, p0);
    Reg::Sub(vi3, vd4, v2, p0);

    Reg::Duplicate(vs0, (int16_t)strideI0, p0);
    Reg::Duplicate(vs1, (int16_t)strideI1, p0);
    Reg::Duplicate(vs2, (int16_t)strideI2, p0);

    Reg::Mul(vr0, vs2, vi2, p0);
    Reg::MulAddDst(vr0, vs1, vi1, p0);
    Reg::MulAddDst(vr0, vs0, vi0, p0);
    Reg::Mul(vr1, vs2, vi5, p0);
    Reg::MulAddDst(vr1, vs1, vi4, p0);
    Reg::MulAddDst(vr1, vs0, vi3, p0);

    Reg::StoreAlign(indexUb, vr0, p0);
    Reg::StoreAlign(indexUb + VF_LEN_HALF, vr1, p0);
}

template <typename T>
__simd_vf__ inline void VfGenIndexForFourDim(
    __ubuf__ T* indexUb, uint32_t sizeI0, uint32_t sizeI1, uint32_t sizeI2, uint32_t sizeI3, uint32_t strideI0,
    uint32_t strideI1, uint32_t strideI2, uint32_t strideI3)
{
    Reg::RegTensor<T> v0;
    Reg::RegTensor<T> v1;
    Reg::RegTensor<T> v2;

    Reg::RegTensor<T> vr0;

    Reg::RegTensor<T> vd0;
    Reg::RegTensor<T> vd1;
    Reg::RegTensor<T> vd2;
    Reg::RegTensor<T> vd3;

    Reg::RegTensor<T> vi0;
    Reg::RegTensor<T> vi1;
    Reg::RegTensor<T> vi2;
    Reg::RegTensor<T> vi3;

    Reg::RegTensor<T> vs0;
    Reg::RegTensor<T> vs1;
    Reg::RegTensor<T> vs2;
    Reg::RegTensor<T> vs3;

    Reg::MaskReg p0 = Reg::CreateMask<T>();
    Reg::Arange(v0, 0);

    Reg::Duplicate(v1, (T)sizeI3, p0);
    Reg::Div(vd0, v0, v1, p0);
    Reg::Mul(v2, vd0, v1, p0);
    Reg::Sub(vi3, v0, v2, p0);

    Reg::Duplicate(v1, (T)sizeI2, p0);
    Reg::Div(vd1, vd0, v1, p0);
    Reg::Mul(v2, vd1, v1, p0);
    Reg::Sub(vi2, vd0, v2, p0);

    Reg::Duplicate(v1, (T)sizeI1, p0);
    Reg::Div(vd2, vd1, v1, p0);
    Reg::Mul(v2, vd2, v1, p0);
    Reg::Sub(vi1, vd1, v2, p0);

    Reg::Duplicate(v1, (T)sizeI0, p0);
    Reg::Div(vd3, vd2, v1, p0);
    Reg::Mul(v2, vd3, v1, p0);
    Reg::Sub(vi0, vd2, v2, p0);

    Reg::Duplicate(vs0, (T)strideI0, p0);
    Reg::Duplicate(vs1, (T)strideI1, p0);
    Reg::Duplicate(vs2, (T)strideI2, p0);
    Reg::Duplicate(vs3, (T)strideI3, p0);

    Reg::Mul(vr0, vs3, vi3, p0);
    Reg::MulAddDst(vr0, vs2, vi2, p0);
    Reg::MulAddDst(vr0, vs1, vi1, p0);
    Reg::MulAddDst(vr0, vs0, vi0, p0);

    Reg::StoreAlign(indexUb, vr0, p0);
}

template <typename T>
__simd_vf__ inline void VfGatherBrc(
    __ubuf__ T* dstUb, __ubuf__ T* srcUb, __ubuf__ T* indexUb, uint16_t size1, uint16_t size2,
    uint16_t srcStride1, uint16_t srcStride2, uint32_t main, uint32_t tail)
{
    Reg::UnalignReg u0;
    Reg::RegTensor<T> vindex0;
    Reg::RegTensor<T> vindex;
    Reg::RegTensor<T> vstride1;
    Reg::RegTensor<T> vstride2;
    Reg::RegTensor<T> vbase1;
    Reg::RegTensor<T> voffset1;
    Reg::RegTensor<T> voffset2;

    Reg::RegTensor<T> vd0;
    Reg::RegTensor<T> vd1;

    Reg::MaskReg pa = Reg::CreateMask<T>();
    Reg::Duplicate(vstride1, (T)srcStride1, pa);
    Reg::Duplicate(vstride2, (T)srcStride2, pa);
    Reg::LoadAlign(vindex0, indexUb);
    for (uint16_t i1 = 0; i1 < size1; ++i1) {
        Reg::Muls(voffset1, vstride1, (T)i1, pa);
        Reg::Add(vbase1, vindex0, voffset1, pa);
        for (uint16_t i2 = 0; i2 < size2; ++i2) {
            Reg::Muls(voffset2, vstride2, (T)i2, pa);
            Reg::Add(vindex, vbase1, voffset2, pa);
            Reg::Gather(vd0, srcUb, vindex, pa);
            Reg::StoreUnAlign(dstUb, vd0, u0, main);
        }
        Reg::Muls(voffset2, vstride2, (T)size2, pa);
        Reg::Add(vindex, vbase1, voffset2, pa);
        Reg::Gather(vd1, srcUb, vindex, pa);
        Reg::StoreUnAlign(dstUb, vd1, u0, tail);
    }
    Reg::StoreUnAlignPost(dstUb, u0, 0);
}

template <typename T>
__simd_vf__ inline void VfGatherBrcB8(
    __ubuf__ T* dstUb, __ubuf__ T* srcUb, __ubuf__ uint16_t* indexUb, uint16_t size1, uint16_t size2,
    uint16_t srcStride1, uint16_t srcStride2, uint32_t main, uint32_t tail)
{
    constexpr uint32_t VF_LEN_HALF = GetVecLen() / sizeof(uint16_t);
    Reg::UnalignReg u0;
    Reg::RegTensor<uint16_t> vindexBase0;
    Reg::RegTensor<uint16_t> vindexBase1;
    Reg::RegTensor<uint16_t> vindex0;
    Reg::RegTensor<uint16_t> vindex1;
    Reg::RegTensor<uint16_t> vstride1;
    Reg::RegTensor<uint16_t> vstride2;
    Reg::RegTensor<uint16_t> vbase0;
    Reg::RegTensor<uint16_t> vbase1;
    Reg::RegTensor<uint16_t> voffset1;
    Reg::RegTensor<uint16_t> voffset2;

    Reg::RegTensor<uint16_t> vd0;
    Reg::RegTensor<uint16_t> vd1;
    Reg::RegTensor<T> vd;
    Reg::RegTensor<T> dummy;

    Reg::MaskReg pa = Reg::CreateMask<uint16_t>();
    Reg::Duplicate(vstride1, (uint16_t)srcStride1, pa);
    Reg::Duplicate(vstride2, (uint16_t)srcStride2, pa);
    Reg::LoadAlign(vindexBase0, indexUb);
    Reg::LoadAlign(vindexBase1, indexUb + VF_LEN_HALF);
    for (uint16_t i1 = 0; i1 < size1; ++i1) {
        Reg::Muls(voffset1, vstride1, (uint16_t)i1, pa);
        Reg::Add(vbase0, vindexBase0, voffset1, pa);
        Reg::Add(vbase1, vindexBase1, voffset1, pa);
        for (uint16_t i2 = 0; i2 < size2; ++i2) {
            Reg::Muls(voffset2, vstride2, (uint16_t)i2, pa);
            Reg::Add(vindex0, vbase0, voffset2, pa);
            Reg::Add(vindex1, vbase1, voffset2, pa);
            Reg::Gather(vd0, srcUb, vindex0, pa);
            Reg::Gather(vd1, srcUb, vindex1, pa);
            Reg::DeInterleave(vd, dummy, (Reg::RegTensor<T>&)vd0, (Reg::RegTensor<T>&)vd1);
            Reg::StoreUnAlign(dstUb, vd, u0, main);
        }
        Reg::Muls(voffset2, vstride2, (uint16_t)size2, pa);
        Reg::Add(vindex0, vbase0, voffset2, pa);
        Reg::Add(vindex1, vbase1, voffset2, pa);
        Reg::Gather(vd0, srcUb, vindex0, pa);
        Reg::Gather(vd1, srcUb, vindex1, pa);
        Reg::DeInterleave(vd, dummy, (Reg::RegTensor<T>&)vd0, (Reg::RegTensor<T>&)vd1);
        Reg::StoreUnAlign(dstUb, vd, u0, tail);
    }
    Reg::StoreUnAlignPost(dstUb, u0, 0);
}

template <typename T>
__simd_vf__ inline void VfGatherBrcForFourDim(
    __ubuf__ T* dstUb, __ubuf__ T* srcUb, __ubuf__ T* indexUb, uint16_t size1, uint16_t size2, uint16_t size3,
    uint16_t srcStride1, uint16_t srcStride2, uint16_t srcStride3, uint32_t main, uint32_t tail)
{
    Reg::UnalignReg u0;
    Reg::RegTensor<T> vindex0;
    Reg::RegTensor<T> vindex;
    Reg::RegTensor<T> vstride1;
    Reg::RegTensor<T> vstride2;
    Reg::RegTensor<T> vstride3;
    Reg::RegTensor<T> vbase1;
    Reg::RegTensor<T> vbase2;
    Reg::RegTensor<T> voffset1;
    Reg::RegTensor<T> voffset2;
    Reg::RegTensor<T> voffset3;

    Reg::RegTensor<T> vd0;
    Reg::RegTensor<T> vd1;

    Reg::MaskReg pa = Reg::CreateMask<T>();
    Reg::Duplicate(vstride1, (T)srcStride1, pa);
    Reg::Duplicate(vstride2, (T)srcStride2, pa);
    Reg::Duplicate(vstride3, (T)srcStride3, pa);
    Reg::LoadAlign(vindex0, indexUb);
    for (uint16_t i1 = 0; i1 < size1; ++i1) {
        Reg::Muls(voffset1, vstride1, (T)i1, pa);
        Reg::Add(vbase1, vindex0, voffset1, pa);
        for (uint16_t i2 = 0; i2 < size2; ++i2) {
            Reg::Muls(voffset2, vstride2, (T)i2, pa);
            Reg::Add(vbase2, vbase1, voffset2, pa);
            for (uint16_t i3 = 0; i3 < size3; ++i3) {
                Reg::Muls(voffset3, vstride3, (T)i3, pa);
                Reg::Add(vindex, vbase2, voffset3, pa);
                Reg::Gather(vd0, srcUb, vindex, pa);
                Reg::StoreUnAlign(dstUb, vd0, u0, main);
            }
            Reg::Muls(voffset3, vstride3, (T)size3, pa);
            Reg::Add(vindex, vbase2, voffset3, pa);
            Reg::Gather(vd1, srcUb, vindex, pa);
            Reg::StoreUnAlign(dstUb, vd1, u0, tail);
        }
    }
    Reg::StoreUnAlignPost(dstUb, u0, 0);
}

template <typename T>
__aicore__ inline void GenGatherIndex(__ubuf__ T* indexUb, const uint32_t* size, const uint32_t* srcStride)
{
    constexpr uint32_t VF_LEN = GetVecLen() / sizeof(T);
    uint32_t sizeI[3];

    if (size[2] * size[1] * size[0] < VF_LEN) {
        sizeI[0] = size[0];
        sizeI[1] = size[1];
        sizeI[2] = size[2];
    } else if (size[2] * size[1] < VF_LEN) {
        sizeI[0] = VF_LEN / (size[2] * size[1]);
        sizeI[1] = size[1];
        sizeI[2] = size[2];
    } else {
        sizeI[0] = 1;
        sizeI[1] = VF_LEN / size[2];
        sizeI[2] = size[2];
    }

    if constexpr (sizeof(T) == sizeof(uint8_t)) {
        VfGenIndexB8((__ubuf__ int16_t*)indexUb, sizeI[0], sizeI[1], sizeI[2], srcStride[0], srcStride[1],
            srcStride[2]);
    } else {
        VfGenIndex<T>(indexUb, sizeI[0], sizeI[1], sizeI[2], srcStride[0], srcStride[1], srcStride[2]);
    }
}

template <typename T>
__aicore__ inline void GenGatherIndexForFourDim(
    __ubuf__ T* indexUb, const uint32_t* size, const uint32_t* srcStride)
{
    constexpr uint32_t VF_LEN = GetVecLen() / sizeof(T);
    uint32_t sizeI[4];

    if (size[3] * size[2] * size[1] * size[0] < VF_LEN) {
        sizeI[0] = size[0];
        sizeI[1] = size[1];
        sizeI[2] = size[2];
        sizeI[3] = size[3];
    } else if (size[3] * size[2] * size[1] < VF_LEN) {
        sizeI[0] = VF_LEN / (size[3] * size[2] * size[1]);
        sizeI[1] = size[1];
        sizeI[2] = size[2];
        sizeI[3] = size[3];
    } else if (size[3] * size[2] < VF_LEN) {
        sizeI[0] = 1;
        sizeI[1] = VF_LEN / (size[3] * size[2]);
        sizeI[2] = size[2];
        sizeI[3] = size[3];
    } else {
        sizeI[0] = 1;
        sizeI[1] = 1;
        sizeI[2] = VF_LEN / size[3];
        sizeI[3] = size[3];
    }

    VfGenIndexForFourDim<T>(
        indexUb, sizeI[0], sizeI[1], sizeI[2], sizeI[3], srcStride[0], srcStride[1], srcStride[2], srcStride[3]);
}

template <typename T>
__aicore__ inline void GatherWrapper(
    __ubuf__ T* dstUb, __ubuf__ T* srcUb, const uint32_t* size, const uint32_t* srcStride)
{
    constexpr uint32_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t sizeI1, sizeI2;
    uint16_t srcStrideI1, srcStrideI2;
    uint32_t main, tail;
    uint32_t vlTile0, vlTile1, vlTile2;

    if (size[2] * size[1] < VF_LEN) {
        vlTile2 = size[2];
        vlTile1 = size[1];
        vlTile0 = VF_LEN / (vlTile2 * vlTile1);
        sizeI1 = 1;
        sizeI2 = static_cast<uint16_t>(size[0] / vlTile0);
        srcStrideI1 = 0;
        srcStrideI2 = static_cast<uint16_t>(srcStride[0] * vlTile0);
        main = vlTile2 * vlTile1 * vlTile0;
        tail = size[2] * size[1] * size[0] - sizeI2 * main;
    } else {
        vlTile2 = size[2];
        vlTile1 = VF_LEN / (vlTile2);
        sizeI1 = size[0];
        sizeI2 = size[1] / vlTile1;
        srcStrideI1 = static_cast<uint16_t>(srcStride[0]);
        srcStrideI2 = static_cast<uint16_t>(srcStride[1] * vlTile1);
        main = vlTile2 * vlTile1;
        tail = size[2] * size[1] - sizeI2 * main;
    }
    using BrcIndexType = typename ExtractIndexTypeBySize<sizeof(T)>::T;
    LocalTensor<BrcIndexType> indexUb;
    PopStackBuffer<BrcIndexType, TPosition::LCM>(indexUb);
    if constexpr (sizeof(T) == sizeof(uint8_t)) {
        GenGatherIndex((__ubuf__ int8_t*)indexUb.GetPhyAddr(), size, srcStride);
        VfGatherBrcB8<T>(
            dstUb, srcUb, (__ubuf__ uint16_t*)indexUb.GetPhyAddr(), sizeI1, sizeI2, srcStrideI1, srcStrideI2, main,
            tail);
    } else if constexpr (sizeof(T) == sizeof(uint32_t)) {
        GenGatherIndex((__ubuf__ int32_t*)indexUb.GetPhyAddr(), size, srcStride);
        VfGatherBrc<uint32_t>(
            (__ubuf__ uint32_t*)dstUb, (__ubuf__ uint32_t*)srcUb, (__ubuf__ uint32_t*)indexUb.GetPhyAddr(), sizeI1,
            sizeI2, srcStrideI1, srcStrideI2, main, tail);
    } else {
        GenGatherIndex((__ubuf__ int16_t*)indexUb.GetPhyAddr(), size, srcStride);
        VfGatherBrc<uint16_t>(
            (__ubuf__ uint16_t*)dstUb, (__ubuf__ uint16_t*)srcUb, (__ubuf__ uint16_t*)indexUb.GetPhyAddr(), sizeI1,
            sizeI2, srcStrideI1, srcStrideI2, main, tail);
    }
}

template <typename T>
__aicore__ inline void GatherWrapperForFourDim(
    __ubuf__ T* dstUb, __ubuf__ T* srcUb, const uint32_t* size, const uint32_t* srcStride)
{
    constexpr uint32_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t sizeI1, sizeI2, sizeI3;
    uint16_t srcStrideI1, srcStrideI2, srcStrideI3;
    uint32_t main, tail;
    uint32_t vlTile0, vlTile1, vlTile2, vlTile3;

    if (size[3] * size[2] * size[1] < VF_LEN) {
        vlTile3 = size[3];
        vlTile2 = size[2];
        vlTile1 = size[1];
        vlTile0 = VF_LEN / (vlTile3 * vlTile2 * vlTile1);
        sizeI1 = 1;
        sizeI2 = 1;
        sizeI3 = static_cast<uint16_t>(size[0] / vlTile0);
        srcStrideI1 = 0;
        srcStrideI2 = 0;
        srcStrideI3 = static_cast<uint16_t>(srcStride[0] * vlTile0);
        main = vlTile3 * vlTile2 * vlTile1 * vlTile0;
        tail = size[3] * size[2] * size[1] * size[0] - sizeI3 * main;
    } else if (size[3] * size[2] < VF_LEN) {
        vlTile3 = size[3];
        vlTile2 = size[2];
        vlTile1 = VF_LEN / (vlTile2 * vlTile3);
        sizeI1 = 1;
        sizeI2 = size[0];
        sizeI3 = static_cast<uint16_t>(size[1] / vlTile1);
        srcStrideI1 = 0;
        srcStrideI2 = static_cast<uint16_t>(srcStride[0]);
        srcStrideI3 = static_cast<uint16_t>(srcStride[1] * vlTile1);
        main = vlTile3 * vlTile2 * vlTile1;
        tail = size[3] * size[2] * size[1] - sizeI3 * main;
    } else {
        vlTile3 = size[3];
        vlTile2 = VF_LEN / vlTile3;
        sizeI1 = size[0];
        sizeI2 = size[1];
        sizeI3 = static_cast<uint16_t>(size[2] / vlTile2);
        srcStrideI1 = static_cast<uint16_t>(srcStride[0]);
        srcStrideI2 = static_cast<uint16_t>(srcStride[1]);
        srcStrideI3 = static_cast<uint16_t>(srcStride[2] * vlTile2);
        main = vlTile3 * vlTile2;
        tail = size[3] * size[2] - sizeI3 * main;
    }
    LocalTensor<T> indexUb;
    PopStackBuffer<T, TPosition::LCM>(indexUb);
    if constexpr (sizeof(T) == sizeof(uint32_t)) {
        GenGatherIndexForFourDim((__ubuf__ int32_t*)indexUb.GetPhyAddr(), size, srcStride);
        VfGatherBrcForFourDim<uint32_t>(
            (__ubuf__ uint32_t*)dstUb, (__ubuf__ uint32_t*)srcUb, (__ubuf__ uint32_t*)indexUb.GetPhyAddr(), sizeI1,
            sizeI2, sizeI3, srcStrideI1, srcStrideI2, srcStrideI3, main, tail);
    } else {
        GenGatherIndexForFourDim((__ubuf__ int16_t*)indexUb.GetPhyAddr(), size, srcStride);
        VfGatherBrcForFourDim<uint16_t>(
            (__ubuf__ uint16_t*)dstUb, (__ubuf__ uint16_t*)srcUb, (__ubuf__ uint16_t*)indexUb.GetPhyAddr(), sizeI1,
            sizeI2, sizeI3, srcStrideI1, srcStrideI2, srcStrideI3, main, tail);
    }
}
} // namespace BroadcastInternal
} // namespace AscendC
#endif // IMPL_PAD_BROADCAST_BROADCAST_GATHER_3510_EXTEND_IMPL_H
