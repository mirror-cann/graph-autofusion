# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(SK_KERNEL_ARCHS dav-2201 dav-3510)
set(SK_ENTRY_STUB_INPUTS)
set(SK_ENTRY_OBJECTS)
set(SK_ENTRY_TARGETS)

foreach(SK_KERNEL_ARCH IN LISTS SK_KERNEL_ARCHS)
   string(REPLACE "-" "_" SK_KERNEL_ARCH_SUFFIX ${SK_KERNEL_ARCH})
   if(SK_KERNEL_ARCH STREQUAL "dav-2201")
      set(SK_ENTRY_TARGET sk_entry)
   else()
      set(SK_ENTRY_TARGET sk_entry_${SK_KERNEL_ARCH_SUFFIX})
   endif()

   ascendc_fatbin_library(${SK_ENTRY_TARGET} sk_entry.asc)
   target_compile_options(${SK_ENTRY_TARGET} PRIVATE
      $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${SK_KERNEL_ARCH}>
      $<$<COMPILE_LANGUAGE:ASC>:-D__ASCENDC_SUPERKERNEL_EARLY_START_V2>
   )

   target_include_directories(${SK_ENTRY_TARGET} PRIVATE
      $<$<COMPILE_LANGUAGE:ASC>:${CMAKE_CURRENT_SOURCE_DIR}/../src/aot/>
   )

   set(SK_ENTRY_OBJECT ${CMAKE_CURRENT_BINARY_DIR}/${SK_ENTRY_TARGET}.o)
   list(APPEND SK_ENTRY_STUB_INPUTS ${SK_KERNEL_ARCH}=${SK_ENTRY_OBJECT})
   list(APPEND SK_ENTRY_OBJECTS ${SK_ENTRY_OBJECT})
   list(APPEND SK_ENTRY_TARGETS ${SK_ENTRY_TARGET})
endforeach()

add_custom_command(
   OUTPUT ${GENERATED_SK_ENTRY_STUB_CPP}
   COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/gen_sk_entry_stub.py
   ${GENERATED_SK_ENTRY_STUB_CPP}
   ${SK_ENTRY_STUB_INPUTS}
   DEPENDS
      ${SK_ENTRY_TARGETS}
      ${SK_ENTRY_OBJECTS}
      ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/gen_sk_entry_stub.py
   COMMENT "Generating sk_entry stub cpp from sk_entry objects"
   VERBATIM
)

add_custom_target(generate_entry_stub
   DEPENDS ${GENERATED_SK_ENTRY_STUB_CPP}
)

add_library(sk_scope STATIC
   sk_scope_kernel.asc
)

target_compile_options(sk_scope PRIVATE
   $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

target_include_directories(sk_scope PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/../src/aot/
)

add_library(sk_scope_dav_3510 STATIC
   sk_scope_kernel.asc
)

target_compile_options(sk_scope_dav_3510 PRIVATE
   $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>
)

target_compile_definitions(sk_scope_dav_3510 PRIVATE
   $<$<COMPILE_LANGUAGE:ASC>:SK_SCOPE_SYMBOL_SUFFIX=_dav_3510>
)

target_include_directories(sk_scope_dav_3510 PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/../src/aot/
)
