# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

file(GLOB_RECURSE SUPER_KERNEL_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

add_custom_target(superkernel_whl ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/superkernel-0.1.0-py3-none-any.whl)
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/superkernel-0.1.0-py3-none-any.whl
        COMMAND echo "Start to build superkernel wheel package"
        && cd ${CMAKE_CURRENT_SOURCE_DIR}
        && python3 -m build --wheel --no-isolation --outdir ${CMAKE_CURRENT_BINARY_DIR}
        && rm -rf build
        && echo "Build superkernel wheel package end"
        DEPENDS ${SUPER_KERNEL_SRC} pyproject.toml
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/superkernel-0.1.0-py3-none-any.whl OPTIONAL
        DESTINATION ${INSTALL_LIBRARY_DIR}
)

# find_package(ASC REQUIRED PATHS $ENV{ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
# enable_language(ASC)
# 
# set(GENERATED_SK_ENTRY_STUB_CPP "${CMAKE_CURRENT_BINARY_DIR}/kernel/sk_kernel_stub.cpp")
# set_source_files_properties(${GENERATED_SK_ENTRY_STUB_CPP} PROPERTIES GENERATED TRUE)
# 
# if (NOT TARGET sk_kernel)
#     add_subdirectory(kernel)
# endif()
# 
# add_subdirectory(tests/aot/rdv/runtime)
# 
# aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/aot SRC_FILES)
# 
# add_library(ascendsk SHARED
#     ${SRC_FILES}
#     ${GENERATED_SK_ENTRY_STUB_CPP}
# )
# add_dependencies(ascendsk generate_entry_stub)
# target_compile_options(ascendsk PRIVATE
#     -fPIC
#     -fvisibility=hidden
#     -Werror
#     -fno-common
#     -ftrapv
# )
# 
# target_compile_definitions(ascendsk PRIVATE
#     FUNC_VISIBILITY
# )
# 
# target_include_directories(ascendsk PUBLIC
#     ${CMAKE_CURRENT_SOURCE_DIR}
#     ${CMAKE_CURRENT_SOURCE_DIR}/include/super_kernel
#     ${CMAKE_CURRENT_SOURCE_DIR}/src/aot
#     PRIVATE
#     $ENV{ASCEND_HOME_PATH}/pkg_inc/
#     $ENV{ASCEND_HOME_PATH}/pkg_inc/base
#     $ENV{ASCEND_HOME_PATH}/include)
# target_link_libraries(ascendsk PRIVATE rt_hook error_manager c_sec unified_dlog sk_scope)
# 
# install(TARGETS ascendsk DESTINATION ascendsk/lib)
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/super_kernel/super_kernel.h
#     DESTINATION ascendsk/include
# )
# 
# 
# if(ENABLE_CPP_UTEST OR ENABLE_RDV_UTEST)
#     add_subdirectory(tests/aot)
# endif()
