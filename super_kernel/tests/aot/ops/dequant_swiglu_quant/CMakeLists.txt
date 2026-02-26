# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

cmake_minimum_required(VERSION 3.16)
project(dswigluq)

add_library(dswigluq_kernel STATIC
    dequant_swiglu_quant.asc
)
target_compile_options(dswigluq_kernel PRIVATE
   $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
   $<$<COMPILE_LANGUAGE:ASC>:-D__ASCENDC_ENABLE_SUPER_KERNEL__>
   -DCONST_TILING
)
target_include_directories(dswigluq_kernel PUBLIC .
                           PRIVATE $<$<COMPILE_LANGUAGE:ASC>:${CMAKE_CURRENT_SOURCE_DIR}/../>
)

add_subdirectory(tests)
