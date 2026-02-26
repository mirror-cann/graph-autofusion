# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

cmake_minimum_required(VERSION 3.16)
project(rt_hook)

add_library(rt_hook SHARED rt_hook.c rt_elf.c rt_sk_intf.cpp)
target_compile_options(rt_hook PRIVATE -g -O2)
target_include_directories(rt_hook PUBLIC .)
target_link_libraries(rt_hook PRIVATE c_sec)
if(DEFINED ENV{ASCEND_HOME_PATH})
    target_include_directories(rt_hook PUBLIC $ENV{ASCEND_HOME_PATH}/include)
    target_link_directories(rt_hook PUBLIC $ENV{ASCEND_HOME_PATH}/lib64)
endif()