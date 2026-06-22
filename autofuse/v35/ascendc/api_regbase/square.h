/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_SQUARE_H__
#define __ASCENDC_API_REGBASE_SQUARE_H__

/**
 * @brief SquareExtend - compute square of input (x * x)
 * Wrapper for Mul API to provide UnaryApiCall-compatible interface
 *
 * @tparam T data type
 * @param dst output tensor
 * @param src input tensor
 * @param size number of elements to compute
 */
template <typename T>
__aicore__ inline void SquareExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                    const uint32_t size) {
  AscendC::Mul(dst, src, src, size);
}

#endif  // __ASCENDC_API_REGBASE_SQUARE_H__
