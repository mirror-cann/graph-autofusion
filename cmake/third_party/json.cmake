# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

if(json_FOUND)
    return()
endif()

unset(json_FOUND CACHE)
unset(JSON_INCLUDE CACHE)

if(NOT CANN_3RD_LIB_PATH)
  set(CANN_3RD_LIB_PATH ${PROJECT_SOURCE_DIR}/third_party)
endif()

set(JSON_INSTALL_PATH ${CANN_3RD_LIB_PATH}/json)

find_path(JSON_INCLUDE
        NAMES nlohmann/json.hpp
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${JSON_INSTALL_PATH}/include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(json
        FOUND_VAR
        json_FOUND
        REQUIRED_VARS
        JSON_INCLUDE
        )
if(json_FOUND AND NOT FORCE_REBUILD_CANN_3RD)
    message(STATUS "[ThirdPartyLib][json] json found in ${JSON_INSTALL_PATH}, and not force rebuild cann third_party")
    set(JSON_INCLUDE_DIR ${JSON_INSTALL_PATH}/include)
    add_library(json INTERFACE)
    target_include_directories(json INTERFACE ${JSON_INCLUDE_DIR})
else()
    message(STATUS "[ThirdPartyLib][json] json not found in ${JSON_INSTALL_PATH}.")
    set(REQ_URL "${JSON_INSTALL_PATH}/json-3.11.3.tar.gz")
    set(JSON_EXTRA_ARGS "")
    if(EXISTS ${REQ_URL})
        message(STATUS "[json] ${REQ_URL} found.")
    else()
        message(STATUS "[json] ${REQ_URL} not found, need download.")
        set(REQ_URL "https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/json-3.11.3.tar.gz")
        list(APPEND JSON_EXTRA_ARGS
             DOWNLOAD_DIR ${JSON_INSTALL_PATH}
        )
    endif()
    include(ExternalProject)
    ExternalProject_Add(ascend_sk_json
            URL ${REQ_URL}
            TLS_VERIFY OFF
            ${JSON_EXTRA_ARGS}
            CONFIGURE_COMMAND ${CMAKE_COMMAND}
                            -DJSON_MultipleHeaders=ON
                            -DJSON_BuildTests=OFF
                            -DBUILD_SHARED_LIBS=OFF
                            -DCMAKE_INSTALL_PREFIX=${JSON_INSTALL_PATH}
                            <SOURCE_DIR>
            EXCLUDE_FROM_ALL TRUE
    )
    set(JSON_INCLUDE_DIR ${JSON_INSTALL_PATH}/include)
    add_library(json INTERFACE)
    target_include_directories(json INTERFACE ${JSON_INCLUDE_DIR})
    add_dependencies(json ascend_sk_json)
endif()