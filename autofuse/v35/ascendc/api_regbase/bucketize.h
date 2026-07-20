/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_BUCKETIZE_H__
#define __ASCENDC_API_REGBASE_BUCKETIZE_H__

using namespace AscendC;

constexpr Reg::CastTrait castTraitNarrowToWide0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                   Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr Reg::CastTrait castTraitNarrowToWide1 = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                   Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

template <typename T>
struct PromotedTrait {
  using type = T;
};
template <>
struct PromotedTrait<int8_t> {
  using type = int16_t;
};
template <>
struct PromotedTrait<uint8_t> {
  using type = uint16_t;
};

template <typename T, typename PromoT>
__simd_callee__ inline void PromoteB8(__ubuf__ PromoT *dstUb, __ubuf__ T *srcUb, const uint32_t count) {
  constexpr uint32_t vlB8 = GetVecLen() / sizeof(T);
  uint16_t repeats = static_cast<uint16_t>(AscendC::CeilDivision(count, vlB8));
  uint32_t calCount = count;
  for (uint16_t i = 0; i < repeats; ++i) {
    Reg::MaskReg maskB8 = Reg::UpdateMask<T>(calCount);

    Reg::RegTensor<T> chunk;
    Reg::LoadAlign(chunk, srcUb + i * vlB8);
    // Widen the even / odd b8 lanes into two b16 registers ...
    Reg::RegTensor<PromoT> wide0;
    Reg::RegTensor<PromoT> wide1;
    Reg::Cast<PromoT, T, castTraitNarrowToWide0>(wide0, chunk, maskB8);
    Reg::Cast<PromoT, T, castTraitNarrowToWide1>(wide1, chunk, maskB8);
    Reg::StoreAlign<PromoT, Reg::StoreDist::DIST_INTLV_B16>((dstUb + i * vlB8), wide0, wide1, maskB8);
  }
}

template <typename U, typename EffT, typename IndexT>
__simd_callee__ inline void BucketizeStoreChunk(__ubuf__ U *dstUb, Reg::RegTensor<IndexT> &lowReg, uint16_t idx,
                                                Reg::MaskReg &mask) {
  constexpr uint32_t vlSize = GetVecLen() / sizeof(EffT);
  if constexpr (sizeof(U) == sizeof(IndexT)) {
    Reg::RegTensor<U> &outRef = (Reg::RegTensor<U> &)lowReg;
    Reg::StoreAlign(dstUb + idx * vlSize, outRef, mask);
  } else if constexpr (sizeof(U) < sizeof(IndexT)) {
    // b64 index -> int32: pack the low 32 bits of each valid b64 lane.
    Reg::StoreAlign<IndexT, Reg::StoreDist::DIST_PACK_B64>(reinterpret_cast<__ubuf__ IndexT *>(dstUb + idx * vlSize),
                                                           lowReg, mask);
  } else {
    // b16 index -> int32: widen the low/high halves and interleave.
    Reg::RegTensor<uint32_t> dstReg1, dstReg2;
    Reg::Cast<uint32_t, IndexT, castTraitNarrowToWide0>(dstReg1, lowReg, mask);
    Reg::Cast<uint32_t, IndexT, castTraitNarrowToWide1>(dstReg2, lowReg, mask);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
        reinterpret_cast<__ubuf__ uint32_t *>(dstUb + idx * vlSize), dstReg1, dstReg2, mask);
  }
}

// Clamp low to boundCount, handle NaN for float types, then store.
template <typename EffT, typename IndexT, typename U, bool IS_FLOAT>
__simd_callee__ inline void BucketizeClampAndStore(__ubuf__ U *dstUb, Reg::RegTensor<IndexT> &low,
                                                   Reg::RegTensor<EffT> &src, Reg::MaskReg &mask, IndexT idxBoundCount,
                                                   uint16_t laneIdx) {
  Reg::Mins(low, low, idxBoundCount, mask);
  if constexpr (IS_FLOAT) {
    Reg::RegTensor<IndexT> bound;
    Reg::Duplicate(bound, idxBoundCount, mask);
    Reg::MaskReg nan;
    Reg::Compare<EffT, CMPMODE::NE>(nan, src, src, mask);
    Reg::Select(low, bound, low, nan);
  }
  BucketizeStoreChunk<U, EffT, IndexT>(dstUb, low, laneIdx, mask);
}

// Processes one pair of lanes (i, i+1) for the main bucketize binary-search loop.
template <typename EffT, typename IndexT, typename U, bool RIGHT, bool IS_FLOAT>
__simd_callee__ inline void BucketizeProcessPair(__ubuf__ EffT *effSrcUb, __ubuf__ EffT *effBoundariesUb,
                                                 __ubuf__ U *dstUb, uint32_t vlSize, uint32_t boundCount,
                                                 uint16_t iters, uint32_t &count, uint16_t i) {
  const auto idxBoundCount = static_cast<IndexT>(boundCount);
  Reg::MaskReg mask0 = Reg::UpdateMask<EffT>(count);
  Reg::MaskReg mask1 = Reg::UpdateMask<EffT>(count);
  Reg::MaskReg cmp0, cmp1;
  Reg::RegTensor<EffT> src0, src1;
  Reg::LoadAlign(src0, effSrcUb + (i + 0) * vlSize);
  Reg::LoadAlign(src1, effSrcUb + (i + 1) * vlSize);
  Reg::RegTensor<IndexT> low0, low1, high0, high1;
  Reg::Duplicate(low0, static_cast<IndexT>(0), mask0);
  Reg::Duplicate(low1, static_cast<IndexT>(0), mask1);
  Reg::Duplicate(high0, idxBoundCount, mask0);
  Reg::Duplicate(high1, idxBoundCount, mask1);

  // Binary search — dual-lane interleaved for instruction-level parallelism.
  for (uint16_t iter = 0; iter < iters; ++iter) {
    Reg::RegTensor<IndexT> sum0, sum1, mid0, mid1, ms0, ms1, mp0, mp1;
    Reg::RegTensor<EffT> mv0, mv1;
    Reg::Add(sum0, low0, high0, mask0);
    Reg::Add(sum1, low1, high1, mask1);
    Reg::ShiftRights(mid0, sum0, static_cast<int16_t>(1), mask0);
    Reg::ShiftRights(mid1, sum1, static_cast<int16_t>(1), mask1);
    Reg::Mins(ms0, mid0, static_cast<IndexT>(boundCount - 1), mask0);
    Reg::Mins(ms1, mid1, static_cast<IndexT>(boundCount - 1), mask1);
    Reg::Gather(mv0, effBoundariesUb, ms0, mask0);
    Reg::Gather(mv1, effBoundariesUb, ms1, mask1);
    if constexpr (RIGHT) {
      Reg::Compare<EffT, CMPMODE::GT>(cmp0, mv0, src0, mask0);
      Reg::Compare<EffT, CMPMODE::GT>(cmp1, mv1, src1, mask1);
    } else {
      Reg::Compare<EffT, CMPMODE::GE>(cmp0, mv0, src0, mask0);
      Reg::Compare<EffT, CMPMODE::GE>(cmp1, mv1, src1, mask1);
    }
    Reg::Adds(mp0, mid0, static_cast<IndexT>(1), mask0);
    Reg::Adds(mp1, mid1, static_cast<IndexT>(1), mask1);
    Reg::Select(high0, mid0, high0, cmp0);
    Reg::Select(high1, mid1, high1, cmp1);
    Reg::Select(low0, low0, mp0, cmp0);
    Reg::Select(low1, low1, mp1, cmp1);
  }

  BucketizeClampAndStore<EffT, IndexT, U, IS_FLOAT>(dstUb, low0, src0, mask0, idxBoundCount, i + 0);
  BucketizeClampAndStore<EffT, IndexT, U, IS_FLOAT>(dstUb, low1, src1, mask1, idxBoundCount, i + 1);
}

// Processes a single leftover lane for the tail of the bucketize binary-search loop.
template <typename EffT, typename IndexT, typename U, bool RIGHT, bool IS_FLOAT>
__simd_callee__ inline void BucketizeProcessTail(__ubuf__ EffT *effSrcUb, __ubuf__ EffT *effBoundariesUb,
                                                 __ubuf__ U *dstUb, uint32_t vlSize, uint32_t boundCount,
                                                 uint16_t iters, uint32_t &count, uint16_t i) {
  const auto idxBoundCount = static_cast<IndexT>(boundCount);
  Reg::MaskReg mask0 = Reg::UpdateMask<EffT>(count);
  Reg::MaskReg cmpMask;
  Reg::RegTensor<EffT> srcReg;
  Reg::LoadAlign(srcReg, effSrcUb + i * vlSize);
  Reg::RegTensor<IndexT> lowReg, highReg;
  Reg::Duplicate(lowReg, static_cast<IndexT>(0), mask0);
  Reg::Duplicate(highReg, idxBoundCount, mask0);

  for (uint16_t iter = 0; iter < iters; ++iter) {
    Reg::RegTensor<IndexT> sumReg, midReg, midSafeReg, midP1Reg;
    Reg::RegTensor<EffT> midValReg;
    Reg::Add(sumReg, lowReg, highReg, mask0);
    Reg::ShiftRights(midReg, sumReg, static_cast<int16_t>(1), mask0);
    Reg::Mins(midSafeReg, midReg, static_cast<IndexT>(boundCount - 1), mask0);
    Reg::Gather(midValReg, effBoundariesUb, midSafeReg, mask0);
    if constexpr (RIGHT) {
      Reg::Compare<EffT, CMPMODE::GT>(cmpMask, midValReg, srcReg, mask0);
    } else {
      Reg::Compare<EffT, CMPMODE::GE>(cmpMask, midValReg, srcReg, mask0);
    }
    Reg::Adds(midP1Reg, midReg, static_cast<IndexT>(1), mask0);
    Reg::Select(highReg, midReg, highReg, cmpMask);
    Reg::Select(lowReg, lowReg, midP1Reg, cmpMask);
  }

  BucketizeClampAndStore<EffT, IndexT, U, IS_FLOAT>(dstUb, lowReg, srcReg, mask0, idxBoundCount, i);
}

template <typename T, typename U, bool RIGHT>
__simd_vf__ inline void BucketizeImplVF(__ubuf__ U *dstUb, __ubuf__ T *srcUb, __ubuf__ T *boundariesUb,
                                        __ubuf__ uint8_t *tmpUb, const uint32_t calCount, const uint32_t boundCount,
                                        const uint16_t iters) {
  // ---- Promote int8/uint8 to int16/uint16 in scratch memory ----
  using PromoT = typename PromotedTrait<T>::type;

  __ubuf__ PromoT *effSrcUb;
  __ubuf__ PromoT *effBoundariesUb;

  if constexpr (sizeof(T) == 1) {
    uint32_t srcRepeats = AscendC::CeilDivision(calCount, GetVecLen());
    uint32_t srcPromoBytes = srcRepeats * GetVecLen() * 2;
    effSrcUb = reinterpret_cast<__ubuf__ PromoT *>(tmpUb);
    effBoundariesUb = reinterpret_cast<__ubuf__ PromoT *>(tmpUb + srcPromoBytes);
    PromoteB8<T, PromoT>(effSrcUb, srcUb, calCount);
    PromoteB8<T, PromoT>(effBoundariesUb, boundariesUb, boundCount);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
  } else {
    effSrcUb = srcUb;
    effBoundariesUb = boundariesUb;
  }

  // ---- Use the effective (possibly promoted) type from here on ----
  using EffT = PromoT;
  using IndexT =
      typename std::conditional<sizeof(EffT) == 2, uint16_t,
                                typename std::conditional<sizeof(EffT) == 4, uint32_t, uint64_t>::type>::type;

  constexpr bool isFloat = SupportType<T, half, bfloat16_t, float>();
  constexpr uint32_t vlSize = GetVecLen() / sizeof(EffT);
  uint16_t repeatTimes = static_cast<uint16_t>(AscendC::CeilDivision(calCount, vlSize));
  uint32_t count = calCount;
  uint16_t mainCount = repeatTimes - (repeatTimes % 2);

  for (uint16_t i = 0; i < mainCount; i += 2) {
    BucketizeProcessPair<EffT, IndexT, U, RIGHT, isFloat>(effSrcUb, effBoundariesUb, dstUb, vlSize, boundCount, iters,
                                                          count, i);
  }

  for (uint16_t i = mainCount; i < repeatTimes; ++i) {
    BucketizeProcessTail<EffT, IndexT, U, RIGHT, isFloat>(effSrcUb, effBoundariesUb, dstUb, vlSize, boundCount, iters,
                                                          count, i);
  }
}

template <typename T, typename U>
__aicore__ inline void BucketizeExtend(const LocalTensor<U> &dst, const LocalTensor<T> &src,
                                       const LocalTensor<T> &boundaries, const LocalTensor<uint8_t> &sharedTmpBuffer,
                                       const uint32_t calCount, const uint32_t boundCount, bool right = false) {
  if ASCEND_IS_AIC {
    return;
  }

  static_assert(SupportType<T, uint8_t, int8_t, half, bfloat16_t, uint16_t, int16_t, float, uint32_t, int32_t, int64_t,
                            uint64_t>(),
                "Bucketize: input type must be uint8, int8, half, bfloat16, "
                "uint16, int16, float, uint32, int32, int64, or uint64");
  static_assert(SupportType<U, int32_t>(), "Bucketize: output type must be int32_t");
  // No boundaries → every element maps to bucket 0.
  if (boundCount == 0) {
    Duplicate(dst, static_cast<U>(0), calCount);
    return;
  }

  __ubuf__ U *dstUb = (__ubuf__ U *)dst.GetPhyAddr();
  __ubuf__ T *srcUb = (__ubuf__ T *)src.GetPhyAddr();
  __ubuf__ T *boundariesUb = (__ubuf__ T *)boundaries.GetPhyAddr();
  __ubuf__ uint8_t *tmpUb = (__ubuf__ uint8_t *)sharedTmpBuffer.GetPhyAddr();

  // Dispatch on `right` here (outside the VF) so the compare mode is a
  // compile-time template arg — keeps the search loop branch-free / Hardware Loop.
  uint16_t iters = static_cast<uint16_t>(32 - __builtin_clz(boundCount));
  if (right) {
    BucketizeImplVF<T, U, true>(dstUb, srcUb, boundariesUb, tmpUb, calCount, boundCount, iters);
  } else {
    BucketizeImplVF<T, U, false>(dstUb, srcUb, boundariesUb, tmpUb, calCount, boundCount, iters);
  }
}

#endif  // __ASCENDC_API_REGBASE_BUCKETIZE_H__
