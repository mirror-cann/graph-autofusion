# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

ascendc_fatbin_library(sk_entry sk_entry.asc)
target_compile_options(sk_entry PRIVATE
   $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
   $<$<COMPILE_LANGUAGE:ASC>:-D__ASCENDC_SUPERKERNEL_EARLY_START_V2>
)

target_include_directories(sk_entry PRIVATE
   $<$<COMPILE_LANGUAGE:ASC>:${CMAKE_CURRENT_SOURCE_DIR}/../src/aot/>
)

set(SK_ENTRY_OBJECT ${CMAKE_CURRENT_BINARY_DIR}/sk_entry.o)

add_custom_command(
   OUTPUT ${GENERATED_SK_ENTRY_STUB_CPP}
   COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/gen_sk_entry_stub.py
   ${SK_ENTRY_OBJECT}
   ${GENERATED_SK_ENTRY_STUB_CPP}
   DEPENDS
      lld_copy_sk_entry
      ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/gen_sk_entry_stub.py
   COMMENT "Generating sk_entry stub cpp from sk_entry.o"
   VERBATIM
)

add_custom_target(generate_entry_stub
   DEPENDS ${GENERATED_SK_ENTRY_STUB_CPP}
)

ascendc_library(sk_scope STATIC
    sk_scope_kernel.asc
)

target_compile_options(sk_scope PRIVATE
   $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

target_include_directories(sk_scope PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/../src/aot/
)