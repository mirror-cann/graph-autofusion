# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_library(dynamic_quant_kernel STATIC
    dynamic_quant.asc
)
target_compile_options(dynamic_quant_kernel PRIVATE
   $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
   $<$<COMPILE_LANGUAGE:ASC>:--cce-aicore-input-parameter-size=2048>
   $<$<COMPILE_LANGUAGE:ASC>:-D__ASCENDC_ENABLE_SUPER_KERNEL__>
   -DDTYPE_X=bfloat16_t -DORIG_DTYPE_X=DT_BF16 -DFORMAT_X=FORMAT_ND -DDTYPE_Y=int8_t -DORIG_DTYPE_Y=DT_INT8 -DFORMAT_Y=FORMAT_ND -DDTYPE_SCALE=float -DORIG_DTYPE_SCALE=DT_FLOAT -DFORMAT_SCALE=FORMAT_ND
   -DCONST_TILING

)
target_include_directories(dynamic_quant_kernel PUBLIC . 
                            PRIVATE $<$<COMPILE_LANGUAGE:ASC>:${CMAKE_CURRENT_SOURCE_DIR}/../>)

add_subdirectory(tests)
