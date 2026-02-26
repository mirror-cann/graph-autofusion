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

#include <stdint.h>

struct RMSNormTilingData {
    uint64_t num_row;
    uint64_t num_col;
    uint64_t num_col_align;
    uint64_t block_factor;
    uint32_t row_factor;
    uint32_t ub_factor;
    uint32_t reduce_mask;
    uint32_t left_num;
    uint32_t last_reduce_mask;
    uint32_t last_left_num;
    uint32_t rstd_size;
    uint32_t ub_loop;
    uint32_t col_buffer_length;
    uint32_t multi_n_num;
    uint32_t is_nddma;
    float epsilon;
    float avg_factor;
    uint8_t is_gemma;
    uint8_t RMSNormTilingDataPH[3];
};
