# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_executable(test_gmm main.cpp)
target_include_directories(test_gmm PRIVATE $ENV{ASCEND_HOME_PATH}/include)
target_link_directories(test_gmm PRIVATE $<${RUN_MODEL}:$ENV{ASCEND_HOME_PATH}/tools/simulator/Ascend910B1/lib>
                                     $ENV{ASCEND_HOME_PATH}/lib64)
target_link_libraries(test_gmm PRIVATE ops_interface)

