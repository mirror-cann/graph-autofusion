/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#define GET_STRUCT_PTR(input_ptr, output_ptr) \
  do { \
    __asm__ volatile("MOV %0, %1" \
                     : "=l"(output_ptr) \
                     : "l"((int64_t)(input_ptr))); \
  } while (0)


#define __spk__ __attribute__((need_auto_sync)) __attribute__((aligned(32)))

#if defined(CONST_TILING)
#define GET_TILING_STR(TYPE, inTiling, tiling) const TYPE tiling = STATIC_TILING_VAR
#define GET_TILING_PTR(TYPE, inTiling, tiling) const TYPE tiling = &STATIC_TILING_VAR
#else
#define GET_TILING_STR(TYPE, inTiling, tiling) TYPE &tiling = inTiling
#define GET_TILING_PTR(TYPE, inTiling, tiling) TYPE &tiling = inTiling
#endif