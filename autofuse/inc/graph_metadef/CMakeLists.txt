# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (NOT DEFINED GE_METADEF_INC_DIR)
    # 开发环境编译兼容处理
    set(METADEF_DIR ${TOP_DIR}/metadef)
    set(GE_METADEF_DIR ${TOP_DIR}/air/graph_metadef)
    set(GE_METADEF_INC_DIR ${TOP_DIR}/air/inc/graph_metadef)
    set(AIR_CODE_DIR ${TOP_DIR}/air)
    set(RUNTIME_CODE_DIR ${TOP_DIR}/runtime)
elseif(NOT TARGET metadef_headers_af)
    add_library(metadef_headers_af INTERFACE)
endif ()

if (NOT ENABLE_OPEN_SRC)
    set(METADEF_DIR ${TOP_DIR}/metadef)
endif()

#set(AUTOFUSE_STUB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../stub)

# 影响sdk包导出cmake，增加时必须包含BUILD_INTERFACE
target_include_directories(metadef_headers_af INTERFACE
#    $<BUILD_INTERFACE:${AUTOFUSE_STUB_DIR}>
#    $<BUILD_INTERFACE:${AUTOFUSE_STUB_DIR}/base>
#    $<BUILD_INTERFACE:${mmpa_INCLUDE_DIR}>
#    $<BUILD_INTERFACE:${mmpa_INCLUDE_DIR}/mmpa>
#    $<BUILD_INTERFACE:${C_SEC_INCLUDE}>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/external>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/external/base>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/external/exe_graph>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/external/exe_graph/runtime>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/external/graph>
    $<BUILD_INTERFACE:${METADEF_DIR}/pkg_inc/base>
    $<BUILD_INTERFACE:${METADEF_DIR}/pkg_inc/base/asc>
    $<BUILD_INTERFACE:${METADEF_DIR}/pkg_inc/base/type>
    $<BUILD_INTERFACE:${METADEF_DIR}/pkg_inc/base/utils>
    $<BUILD_INTERFACE:${METADEF_DIR}/pkg_inc>
    $<BUILD_INTERFACE:${METADEF_DIR}/pkg_inc/graph/>
    $<BUILD_INTERFACE:${METADEF_DIR}/pkg_inc/common/>
    $<BUILD_INTERFACE:${METADEF_DIR}/pkg_inc/common/ge_common/>
    $<BUILD_INTERFACE:${METADEF_DIR}/pkg_inc/common/ge_common/debug/>
    $<BUILD_INTERFACE:${METADEF_DIR}/pkg_inc/exe_graph/runtime>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/external/graph/utils/>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/graph/utils/>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/graph/debug/>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/common/ge_common/>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/graph_metadef/graph/>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/graph_metadef/graph/symbolizer/>
    $<BUILD_INTERFACE:${GE_METADEF_INC_DIR}>
    $<BUILD_INTERFACE:${GE_METADEF_INC_DIR}/graph>
    $<BUILD_INTERFACE:${AIR_CODE_DIR}/inc>
    $<BUILD_INTERFACE:${GE_METADEF_INC_DIR}/common>
    $<BUILD_INTERFACE:${GE_METADEF_INC_DIR}/exe_graph>
    $<BUILD_INTERFACE:${GE_METADEF_INC_DIR}/op_common>
    $<BUILD_INTERFACE:${GE_METADEF_INC_DIR}/graph/utils>
    $<BUILD_INTERFACE:${GE_METADEF_INC_DIR}/register>
    $<BUILD_INTERFACE:${RUNTIME_CODE_DIR}/include/external>

    # --------------------start------------------------
    $<BUILD_INTERFACE:${GE_METADEF_DIR}>
    $<BUILD_INTERFACE:${GE_METADEF_DIR}/third_party/transformer>
    $<BUILD_INTERFACE:${GE_METADEF_DIR}/third_party/transformer/inc>
    $<BUILD_INTERFACE:${GE_METADEF_DIR}/third_party/transformer/src>
    $<BUILD_INTERFACE:${GE_METADEF_DIR}/exe_graph>
    $<BUILD_INTERFACE:${GE_METADEF_DIR}/graph>
    $<BUILD_INTERFACE:${GE_METADEF_DIR}/register>
    $<BUILD_INTERFACE:${GE_METADEF_DIR}/register/op_tiling>
    # ---------------------end-----------------------
)

message(STATUS "inc graph_metadef")
get_target_property(INCLUDE_PATHS metadef_headers_af INTERFACE_INCLUDE_DIRECTORIES)
foreach(line IN LISTS INCLUDE_PATHS)
    message(STATUS "metadef_headers_af ${line}")
endforeach()
