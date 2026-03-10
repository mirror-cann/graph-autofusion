# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

cmake_minimum_required(VERSION 3.16)
project(ops_interface)

add_library(ops_interface SHARED
    DevContext.cpp
    OpsInterface.cpp
)

target_compile_options(ops_interface PRIVATE
   $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
   $<$<COMPILE_LANGUAGE:ASC>:-D__ASCENDC_ENABLE_SUPER_KERNEL__>
)
target_include_directories(ops_interface PUBLIC . $ENV{ASCEND_HOME_PATH}/include)
target_link_directories(ops_interface PUBLIC $<${RUN_MODEL}:$ENV{ASCEND_HOME_PATH}/tools/simulator/Ascend910B1/lib>
 $ENV{ASCEND_HOME_PATH}/lib64)

target_link_libraries(ops_interface PUBLIC
rms_norm_kernel 
tensor_list gmm_kernel 
wqbmm_kernel 
matmul_add_kernel 
dswigluq_kernel 
clear_ops_kernel 
gmm_kernel_v2 
gmm_kernel_v3 
dynamic_quant_kernel
ascendcl ascend_dump ascendc_runtime runtime error_manager c_sec mmpa profapi unified_dlog
)
