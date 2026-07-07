/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_NEXT_AFTER_H__
#define __ASCENDC_API_REGBASE_NEXT_AFTER_H__

constexpr uint32_t NEXTAFTER_FLOAT_NAN_U32 = AscendC::F32_NAN;
constexpr uint32_t NEXTAFTER_MIN_POS_SUBNORMAL_U32 = 0x00000001U;
constexpr uint32_t NEXTAFTER_MIN_NEG_SUBNORMAL_U32 = 0x80000001U;

template <typename T, bool IsScalarOther = false>
__simd_vf__ inline void NextAfterImplVF(__ubuf__ T *dstUb, __ubuf__ T *srcUb, __ubuf__ T *otherUb, T otherVal,
                                        uint32_t calcCnt) {
  uint32_t vlSize = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
  uint16_t repeatTimes = static_cast<uint16_t>(AscendC::CeilDivision(calcCnt, vlSize));

  AscendC::Reg::RegTensor<T> srcReg, otherReg, dstReg;
  AscendC::Reg::RegTensor<T> plusOneReg, minusOneReg, tempReg;
  AscendC::Reg::MaskReg mask, dirMask, signMask;
  if constexpr (IsScalarOther) {
    AscendC::Reg::Duplicate(otherReg, otherVal);
  }

  for (uint16_t i = 0U; i < repeatTimes; ++i) {
    mask = AscendC::Reg::UpdateMask<T>(calcCnt);
    AscendC::Reg::LoadAlign(srcReg, srcUb + i * vlSize);
    if constexpr (!IsScalarOther) {
      AscendC::Reg::LoadAlign(otherReg, otherUb + i * vlSize);
    }
    AscendC::Reg::RegTensor<uint32_t> &srcBits = (AscendC::Reg::RegTensor<uint32_t> &)srcReg;
    AscendC::Reg::RegTensor<uint32_t> &plusOneBits = (AscendC::Reg::RegTensor<uint32_t> &)plusOneReg;
    AscendC::Reg::RegTensor<uint32_t> &minusOneBits = (AscendC::Reg::RegTensor<uint32_t> &)minusOneReg;
    AscendC::Reg::Adds(plusOneBits, srcBits, (uint32_t)1, mask);
    AscendC::Reg::Adds(minusOneBits, srcBits, (uint32_t)0xFFFFFFFF, mask);
    AscendC::Reg::Compares<T, AscendC::CMPMODE::GT>(signMask, srcReg, (T)0.0, mask);
    if constexpr (IsScalarOther) {
      AscendC::Reg::Compares<T, AscendC::CMPMODE::LT>(dirMask, srcReg, otherVal, mask);
    } else {
      AscendC::Reg::Compare<T, AscendC::CMPMODE::LT>(dirMask, srcReg, otherReg, mask);
    }
    // x>0场景位+1,x<0场景位-1
    AscendC::Reg::Select(tempReg, plusOneReg, minusOneReg, signMask);
    // x>0,x<y场景位+1,x<0,x<y场景位-1,其余场景改回原值
    AscendC::Reg::Select(dstReg, tempReg, srcReg, dirMask);

    if constexpr (IsScalarOther) {
      AscendC::Reg::Compares<T, AscendC::CMPMODE::GT>(dirMask, srcReg, otherVal, mask);
    } else {
      AscendC::Reg::Compare<T, AscendC::CMPMODE::GT>(dirMask, srcReg, otherReg, mask);
    }
    // x>y场景，正数则位+1,负数位-1,x<=y场景不变
    AscendC::Reg::Select(tempReg, minusOneReg, plusOneReg, signMask);
    AscendC::Reg::Select(dstReg, tempReg, dstReg, dirMask);

    // 处理0值场景
    AscendC::Reg::Duplicate(plusOneBits, NEXTAFTER_MIN_POS_SUBNORMAL_U32);
    AscendC::Reg::Duplicate(minusOneBits, NEXTAFTER_MIN_NEG_SUBNORMAL_U32);
    AscendC::Reg::Compares<T, AscendC::CMPMODE::EQ>(signMask, srcReg, (T)0.0, mask);
    AscendC::Reg::Compares<T, AscendC::CMPMODE::GT>(dirMask, otherReg, (T)0.0, mask);
    AscendC::Reg::Select(tempReg, plusOneReg, dstReg, dirMask);
    AscendC::Reg::Compares<T, AscendC::CMPMODE::LT>(dirMask, otherReg, (T)0.0, mask);
    AscendC::Reg::Select(tempReg, minusOneReg, tempReg, dirMask);
    AscendC::Reg::Select(dstReg, tempReg, dstReg, signMask);

    // NaN场景覆盖
    AscendC::Reg::Duplicate(plusOneBits, NEXTAFTER_FLOAT_NAN_U32);
    AscendC::Reg::Compare<T, AscendC::CMPMODE::NE>(dirMask, srcReg, srcReg, mask);
    AscendC::Reg::Select(dstReg, plusOneReg, dstReg, dirMask);
    AscendC::Reg::Compare<T, AscendC::CMPMODE::NE>(dirMask, otherReg, otherReg, mask);
    AscendC::Reg::Select(dstReg, plusOneReg, dstReg, dirMask);

    AscendC::Reg::StoreAlign(dstUb + i * vlSize, dstReg, mask);
  }
}

/**
 * nextafter 算子的对外接口——向量版（other 为 LocalTensor）。
 *
 * @tparam T              浮点类型（当前仅支持 float）
 * @param dst             输出 LocalTensor，存放 nextafter(x, y) 结果
 * @param src             输入 LocalTensor，存放 x 值
 * @param other           输入 LocalTensor，存放 y 值
 * @param sharedTmpBuffer 共享临时 buffer
 * @param calCount        需要计算的元素总数
 */
template <typename T>
__aicore__ inline void NextAfterExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                       const AscendC::LocalTensor<T> &other,
                                       const AscendC::LocalTensor<uint8_t> &sharedTmpBuffer, const int32_t &calCount) {
  if ASCEND_IS_AIC {
    return;
  }
  static_assert(std::is_same<T, float>::value, "NextAfter currently only supports float");
  ASCENDC_ASSERT(calCount > 0, { KERNEL_LOG(KERNEL_ERROR, "calCount must be positive, got %d", calCount); });
  NextAfterImplVF<T, false>((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(),
                            (__ubuf__ T *)other.GetPhyAddr(), (T)0.0, static_cast<uint32_t>(calCount));
}

/**
 * nextafter 算子的对外接口——标量版（other 为标量 T）。
 *
 * @tparam T              浮点类型（当前仅支持 float）
 * @param dst             输出 LocalTensor，存放 nextafter(x, y) 结果
 * @param src             输入 LocalTensor，存放 x 值
 * @param other           标量 y 值
 * @param sharedTmpBuffer 共享临时 buffer
 * @param calCount        需要计算的元素总数
 */
template <typename T>
__aicore__ inline void NextAfterExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                       const T other, const AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
                                       const int32_t &calCount) {
  if ASCEND_IS_AIC {
    return;
  }
  static_assert(std::is_same<T, float>::value, "NextAfter currently only supports float");
  ASCENDC_ASSERT(calCount > 0, { KERNEL_LOG(KERNEL_ERROR, "calCount must be positive, got %d", calCount); });
  NextAfterImplVF<T, true>((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), nullptr, other,
                           static_cast<uint32_t>(calCount));
}

#endif  // __ASCENDC_API_REGBASE_NEXT_AFTER_H__
