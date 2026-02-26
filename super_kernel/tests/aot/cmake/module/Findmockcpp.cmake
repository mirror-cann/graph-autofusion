# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
message(STATUS "in findmockcpp")
if (mockcpp_FOUND)
    message(STATUS "Package mockcpp has been found.")
    return()
endif()

set(MOCKCPP_DIR ${CMAKE_INSTALL_PREFIX}/mockcpp)

find_path(MOCKCPP_INCLUDE_ONE
        PATHS ${MOCKCPP_DIR}/include
        NO_DEFAULT_PATH
        CMAKE_FIND_ROOT_PATH_BOTH
        NAMES mockcpp/mockcpp.h)
mark_as_advanced(MOCKCPP_INCLUDE_ONE)


find_path(MOCKCPP_INCLUDE_TWO
        PATHS ${MOCKCPP_DIR}/include/3rdparty
        NO_DEFAULT_PATH
        CMAKE_FIND_ROOT_PATH_BOTH
        NAMES boost/config.hpp)
mark_as_advanced(MOCKCPP_INCLUDE_TWO)

message("MOCKCPP_INCLUDE_TWO is ${MOCKCPP_INCLUDE_TWO}")

find_library(MOCKCPP_STATIC_LIBRARY
        PATHS ${MOCKCPP_DIR}/lib
        NO_DEFAULT_PATH
        CMAKE_FIND_ROOT_PATH_BOTH
        NAMES libmockcpp.a)
mark_as_advanced(MOCKCPP_STATIC_LIBRARY)

message("MOCKCPP_STATIC_LIBRARY is ${MOCKCPP_STATIC_LIBRARY}")



include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(mockcpp
        REQUIRED_VARS MOCKCPP_INCLUDE_ONE MOCKCPP_INCLUDE_TWO MOCKCPP_STATIC_LIBRARY
    )

message("mockcpp_FOUND is ${mockcpp_FOUND}")

if(mockcpp_FOUND)
    set(MOCKCPP_INCLUDE_DIR ${MOCKCPP_INCLUDE_ONE} ${MOCKCPP_INCLUDE_TWO})
    get_filename_component(MOCKCPP_LIBRARY_DIR ${MOCKCPP_STATIC_LIBRARY} DIRECTORY)

    if(NOT TARGET mockcpp_static)
        add_library(mockcpp_static STATIC IMPORTED)
        set_target_properties(mockcpp_static PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${MOCKCPP_INCLUDE_DIR}"
            IMPORTED_LOCATION "${MOCKCPP_STATIC_LIBRARY}"
            )
    endif()
endif()
