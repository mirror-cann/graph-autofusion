# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_library(sk_kernel SHARED sk_entry.asc)
target_compile_options(sk_kernel PRIVATE
   $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
   $<$<COMPILE_LANGUAGE:ASC>:-D__ASCENDC_SUPERKERNEL_EARLY_START_V2>
)
target_include_directories(sk_kernel PUBLIC 
   ${CMAKE_CURRENT_SOURCE_DIR}/
   ${CMAKE_CURRENT_SOURCE_DIR}/../src/aot/)


ascendc_fatbin_library(sk_entry sk_entry.asc)
target_compile_options(sk_entry PRIVATE
   $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
   $<$<COMPILE_LANGUAGE:ASC>:-D__ASCENDC_SUPERKERNEL_EARLY_START_V2>
   $<$<COMPILE_LANGUAGE:ASC>:-DASCENDC_DEBUG>
)

target_include_directories(sk_entry PRIVATE
   $<$<COMPILE_LANGUAGE:ASC>:${CMAKE_CURRENT_SOURCE_DIR}/../src/aot/>
)