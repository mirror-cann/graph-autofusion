/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_SIGNBIT_H__
#define __ASCENDC_API_SIGNBIT_H__

#include "utils.h"

using namespace AscendC;

/**
 * @brief Extract sign bit from input tensor and output as bool tensor.
 * 
 * For each element in src, extracts the sign bit and stores the result in dst:
 *   - dst[i] = true  if src[i] is negative (sign bit is 1)
 *   - dst[i] = false if src[i] is positive or zero (sign bit is 0)
 * 
 * Note: For floating-point values, this correctly handles -0.0 (returns true).
 * 
 * Implementation uses ShiftRights to extract sign bit directly:
 * 1. Load input as uint32_t (reinterpret from int32_t or float)
 * 2. Right shift by 31 bits to move sign bit to lowest position
 * 3. Store result as uint8_t using PACK4_B32 mode
 * 
 * @tparam T Input data type, must be int32_t or float.
 * @param dst Output tensor (bool type), stores the sign bit results.
 * @param src Input tensor (int32_t or float type).
 * @param sharedTmpBuffer Temporary buffer (reserved for future extension, not used).
 * @param calCount Number of elements to process.
 */
template <typename T>
__simd_vf__ inline void SignBitImplVF(__ubuf__ uint8_t* dstUb, __ubuf__ T* srcUb, uint32_t calcCnt)
{
    uint32_t oneRepeatSize = static_cast<uint32_t>(GetVecLen() / sizeof(T));
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calcCnt, oneRepeatSize));
    
    // Shift amount: 31 bits to extract sign bit (bit 31) to lowest position
    int16_t shift = 31;

    Reg::RegTensor<T> srcRegT;
    Reg::RegTensor<uint32_t> u32Reg;
    
    Reg::MaskReg curMaskReg, fullMask;
    fullMask = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        curMaskReg = Reg::UpdateMask<T>(calcCnt);
        Reg::LoadAlign(srcRegT, srcUb + i * oneRepeatSize);
        // Reinterpret as uint32_t to access bit representation
        u32Reg = (Reg::RegTensor<uint32_t>&)srcRegT;
        
        // Right shift by 31 bits: sign bit (bit 31) moves to lowest position
        Reg::ShiftRights(u32Reg, u32Reg, shift, fullMask);
        
        // Store as uint8_t using PACK4_B32 mode (packs 4 uint32_t to 4 uint8_t)
        Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(
            dstUb + i * oneRepeatSize, (Reg::RegTensor<uint8_t>&)u32Reg, curMaskReg);
    }
}

template <typename T>
__aicore__ inline void SignBitExtend(const LocalTensor<bool>& dst,
                                     const LocalTensor<T>& src,
                                     const LocalTensor<uint8_t>& sharedTmpBuffer,
                                     const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(std::is_same<T, int32_t>::value || std::is_same<T, float>::value,
                  "Unsupported data type for SignBitExtend");

    LocalTensor<uint8_t> dstUint = dst.template ReinterpretCast<uint8_t>();

    __ubuf__ uint8_t* dstUb = (__ubuf__ uint8_t*)dstUint.GetPhyAddr();
    __ubuf__ T* srcUb = (__ubuf__ T*)src.GetPhyAddr();
    
    SignBitImplVF<T>(dstUb, srcUb, calCount);
}

#endif // __ASCENDC_API_SIGNBIT_H__
