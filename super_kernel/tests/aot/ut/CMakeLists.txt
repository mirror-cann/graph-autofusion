# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16.0)
project(super_kernel_aot_ut)
set(CMAKE_CXX_STANDARD 17)
set(LOCAL_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(super_kernel_aot_utest
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sk_options_manager.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/sk_options_manager.cpp
)

target_include_directories(super_kernel_aot_utest PRIVATE
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/include/super_kernel/
    $ENV{ASCEND_HOME_PATH}/pkg_inc/base
    $ENV{ASCEND_HOME_PATH}/include
)

target_compile_definitions(super_kernel_aot_utest PRIVATE
    UT_TEST
)

target_compile_options(super_kernel_aot_utest PRIVATE
    -g
    -Werror
    -Wno-unused-function
)

target_link_libraries(super_kernel_aot_utest PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    unified_dlog
)

run_llt_test(
    TARGET super_kernel_aot_utest
    TASK_NUM 1
)