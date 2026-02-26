# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_library(gmm_kernel_v3 STATIC
    grouped_matmul.asc
)
target_compile_options(gmm_kernel_v3 PRIVATE   
   $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
   $<$<COMPILE_LANGUAGE:ASC>:--cce-aicore-input-parameter-size=2048>
   $<$<COMPILE_LANGUAGE:ASC>:-D__ASCENDC_ENABLE_SUPER_KERNEL__>
   -DDTYPE_X=int8_t -DORIG_DTYPE_X=DT_INT8 -DFORMAT_X=FORMAT_ND -DDTYPE_WEIGHT=int8_t -DORIG_DTYPE_WEIGHT=DT_INT8 -DFORMAT_WEIGHT=FORMAT_ND -DDTYPE_BIAS=int32_t -DORIG_DTYPE_BIAS=DT_INT32 -DFORMAT_BIAS=FORMAT_ND -DDTYPE_SCALE=bfloat16_t -DORIG_DTYPE_SCALE=DT_BF16 -DFORMAT_SCALE=FORMAT_ND -DDTYPE_OFFSET=float -DORIG_DTYPE_OFFSET=DT_FLOAT -DFORMAT_OFFSET=FORMAT_ND -DDTYPE_ANTIQUANT_SCALE=half -DORIG_DTYPE_ANTIQUANT_SCALE=DT_FLOAT16 -DFORMAT_ANTIQUANT_SCALE=FORMAT_ND -DDTYPE_ANTIQUANT_OFFSET=half -DORIG_DTYPE_ANTIQUANT_OFFSET=DT_FLOAT16 -DFORMAT_ANTIQUANT_OFFSET=FORMAT_ND -DDTYPE_GROUP_LIST=int64_t -DORIG_DTYPE_GROUP_LIST=DT_INT64 -DFORMAT_GROUP_LIST=FORMAT_ND -DDTYPE_PER_TOKEN_SCALE=float -DORIG_DTYPE_PER_TOKEN_SCALE=DT_FLOAT -DFORMAT_PER_TOKEN_SCALE=FORMAT_ND -DDTYPE_Y=bfloat16_t -DORIG_DTYPE_Y=DT_BF16 -DFORMAT_Y=FORMAT_ND
   -DCONST_TILING
)
target_include_directories(gmm_kernel_v3 PUBLIC . 
                            PRIVATE $<$<COMPILE_LANGUAGE:ASC>:${CMAKE_CURRENT_SOURCE_DIR}/../>)

add_subdirectory(tests)
