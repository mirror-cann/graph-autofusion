# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(autofuse_headers INTERFACE)
target_include_directories(autofuse_headers INTERFACE
    $<BUILD_INTERFACE:${AUTOFUSE_DIR}>
    $<BUILD_INTERFACE:${AUTOFUSE_DIR}/inc>
    $<BUILD_INTERFACE:${AUTOFUSE_DIR}/att>
    $<BUILD_INTERFACE:${AUTOFUSE_DIR}/common>
    $<BUILD_INTERFACE:${AUTOFUSE_DIR}/ascir/meta>
    $<BUILD_INTERFACE:${AUTOFUSE_DIR}/autofuse>
    $<BUILD_INTERFACE:${AUTOFUSE_DIR}/autofuse/utils>
    $<BUILD_INTERFACE:${AUTOFUSE_DIR}/inc/graph_metadef>
    $<BUILD_INTERFACE:${AUTOFUSE_DIR}/inc/graph_metadef/graph>
    $<BUILD_INTERFACE:${AUTOFUSE_DIR}/inc/graph_metadef/graph/ascendc_ir/ascendc_ir_core>
)