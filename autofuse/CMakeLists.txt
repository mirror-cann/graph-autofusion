cmake_minimum_required(VERSION 3.16)
project(graph_autofusion_autofuse CXX C)
cmake_policy(SET CMP0079 NEW)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector-all -Wl,-z,now")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fstack-protector-all -Wl,-z,now")

if (ENABLE_ASAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=leak -fsanitize-recover=address,all -fno-stack-protector -fno-omit-frame-pointer -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fsanitize=leak -fsanitize-recover=address,all -fno-stack-protector -fno-omit-frame-pointer -g")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize=leak -fsanitize-recover=address")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address -fsanitize=leak -fsanitize-recover=address")
endif()

if(NOT MDC_COMPILE_RUNTIME)
    add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)
else()
    add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})
endif()

if (EXISTS "${CMAKE_SOURCE_DIR}/.dev_env")
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.dev_env")
        set(ENABLE_OPEN_SRC True)
        file(STRINGS "${CMAKE_SOURCE_DIR}/.dev_env" DEV_ENV_LINES)
        foreach (line IN LISTS DEV_ENV_LINES)
            string(STRIP "${line}" line)
            if (line MATCHES "^([^=]+)=(.*)$")
                set(var_name "${CMAKE_MATCH_1}")
                set(var_value "${CMAKE_MATCH_2}")
                string(STRIP "${var_name}" var_name)
                string(STRIP "${var_value}" var_value)
                set(${var_name} ${var_value})
            endif ()
        endforeach ()
        set(ASCEND_INSTALL_PATH ${ASCEND_INSTALL_PATH}/latest)
    endif ()
endif ()

set(CODE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(GE_METADEF_DIR ${CODE_ROOT_DIR}/graph_metadef)
set(GE_METADEF_INC_DIR ${CODE_ROOT_DIR}/inc/graph_metadef)
set(METADEF_PROTO_DIR ${GE_METADEF_DIR}/proto)

if(NOT DEFINED CANN_3RD_LIB_PATH AND DEFINED ENV{ASCEND_3RD_LIB_PATH})
    set(CANN_3RD_LIB_PATH "$ENV{ASCEND_3RD_LIB_PATH}" CACHE PATH "Ascend CANN third-party path" FORCE)
endif()
if (NOT DEFINED CANN_3RD_LIB_PATH)
    set(CANN_3RD_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/output/third_party)
endif ()

include_directories(${GE_METADEF_DIR}/third_party/transformer/inc)
include_directories(${GE_METADEF_INC_DIR})
include_directories(${GE_METADEF_INC_DIR}/common)
include_directories(${GE_METADEF_INC_DIR}/external)
include_directories(${GE_METADEF_INC_DIR}/external/graph)
include_directories(${GE_METADEF_INC_DIR}/graph)
include_directories(${GE_METADEF_INC_DIR}/graph/ascendc_ir)
include_directories(${GE_METADEF_INC_DIR}/graph/ascendc_ir/ascendc_ir_core)
include_directories(${GE_METADEF_INC_DIR}/graph/ascendc_ir/utils)
include_directories(${GE_METADEF_INC_DIR}/graph/utils)
include_directories(${GE_METADEF_DIR})
include_directories(${GE_METADEF_DIR}/graph)
include_directories(${CODE_ROOT_DIR}/autofuse)
include_directories(${CODE_ROOT_DIR}/common)
include_directories(${CODE_ROOT_DIR}/ascendc/api)
include_directories(${CODE_ROOT_DIR}/compiler/py_module)
include_directories(${CODE_ROOT_DIR}/ascir/meta)
include_directories(${CODE_ROOT_DIR}/att)
include_directories(${CODE_ROOT_DIR}/inc)
include_directories(${CODE_ROOT_DIR}/inc/stub)
if(DEFINED METADEF_INSTALL_PATH)
    if(EXISTS "${METADEF_INSTALL_PATH}/x86_64-linux")
        set(METADEF_PLATFORM_DIR ${METADEF_INSTALL_PATH}/x86_64-linux)
    elseif(EXISTS "${METADEF_INSTALL_PATH}/aarch64-linux")
        set(METADEF_PLATFORM_DIR ${METADEF_INSTALL_PATH}/aarch64-linux)
    else()
        set(METADEF_PLATFORM_DIR ${METADEF_INSTALL_PATH})
    endif()
    include_directories(${METADEF_PLATFORM_DIR}/include)
    include_directories(${METADEF_PLATFORM_DIR}/include/graph_metadef)
    include_directories(${METADEF_PLATFORM_DIR}/include/graph_metadef/graph)
    include_directories(${METADEF_PLATFORM_DIR}/pkg_inc)
    include_directories(${METADEF_PLATFORM_DIR}/pkg_inc/profiling)
endif()
if(NOT EXISTS "${ASCEND_INSTALL_PATH}/pkg_inc")
    if(DEFINED RUN_TEST AND RUN_TEST EQUAL 1)
        include_directories(${CMAKE_SOURCE_DIR}/tests/depends/securec)
    endif()
endif()

include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos_af)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
add_cann_third_party(json)
if(DEFINED RUN_TEST AND RUN_TEST EQUAL 1)
    add_cann_third_party(gtest)
endif()
add_cann_third_party(boost)
add_cann_third_party(symengine)
add_cann_third_party(protobuf)
find_cann_package(unified_dlog)
find_cann_package(mmpa)
find_cann_package(atrace)
find_cann_package(securec)
find_cann_package(metadef)

include(GoogleTest)
set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)

if (NOT TARGET intf_pub)
    add_library(intf_pub INTERFACE)
    target_compile_options(intf_pub INTERFACE -Wall -fPIC -fstack-protector-strong)
endif()
include(cmake/af_intf_pub.cmake)
if (NOT TARGET autofuse_shared_visibility)
    add_library(autofuse_shared_visibility INTERFACE)
    if (NOT TARGET_SYSTEM_NAME STREQUAL "Windows")
        target_compile_options(autofuse_shared_visibility INTERFACE
            -fvisibility=default
            -fno-visibility-inlines-hidden
        )
    endif ()
endif ()

function(autofuse_apply_shared_visibility target_name)
    target_link_libraries(${target_name} PRIVATE autofuse_shared_visibility)
endfunction()

if (DEFINED RUN_TEST)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
endif ()

add_library(aihac_codegen SHARED)

add_subdirectory(graph_metadef/proto)
add_subdirectory(inc/graph_metadef)
add_subdirectory(graph_metadef/graph)
add_subdirectory(ascendc)
add_subdirectory(ascir)
add_subdirectory(optimize)
add_subdirectory(att)
add_subdirectory(codegen)
add_subdirectory(common)
if (IS_DIRECTORY "${CODE_ROOT_DIR}/v35")
    add_subdirectory(v35)
endif()
if(NOT DEFINED RUN_TEST OR NOT RUN_TEST EQUAL 1)
    add_subdirectory(compiler)
endif ()

if (DEFINED RUN_TEST AND RUN_TEST EQUAL 1 AND IS_DIRECTORY "${CODE_ROOT_DIR}/tests")
    add_subdirectory(tests)
endif()

add_definitions(-DFORMULAS_PERF)

target_compile_definitions(aihac_codegen PRIVATE
    google=ascend_private
)

target_compile_options(aihac_codegen PRIVATE
    -O2
    -fPIC
    -fno-common
    -Wfloat-equal
    -Wextra
    -Wall
    -Werror
)

target_link_options(aihac_codegen PRIVATE
    -rdynamic
    -Wl,-Bsymbolic
    -Wl,--exclude-libs,All
    $<$<CONFIG:Release>:-s>
)

target_link_libraries(aihac_codegen PRIVATE runtime_headers metadef_headers)
target_link_libraries(aihac_codegen PUBLIC
        $<BUILD_INTERFACE:ascir_ops_headers>
        $<BUILD_INTERFACE:atrace_headers>
        graph_af
        graph_base_af
        aihac_ir
        aihac_ir_register
        aihac_symbolizer_af
        $<BUILD_INTERFACE:ge_log_utils>
        $<BUILD_INTERFACE:ascendc_api_extend>
        error_manager
        c_sec
        unified_dlog
        runtime
        atrace_share
        mmpa
        json
        ascend_protobuf
        symengine
        Boost::boost
        -lgcov
)


if(NOT DEFINED RUN_TEST OR NOT RUN_TEST EQUAL 1)
set(AUTOFUSE_INSTALL_COMPONENT graph-autofusion)
set(AUTOFUSE_INSTALL_LIB_DIR ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64)
set(AUTOFUSE_INSTALL_PYTHON_DIR python/site-packages)
set(AUTOFUSE_INSTALL_INCLUDE_DIR ${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/autofuse)

install(TARGETS aihac_codegen
    LIBRARY DESTINATION ${AUTOFUSE_INSTALL_LIB_DIR}
            COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
            OPTIONAL
    ARCHIVE DESTINATION ${AUTOFUSE_INSTALL_LIB_DIR}
            COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
            OPTIONAL
)

install(TARGETS aihac_ir aihac_ir_register
    LIBRARY DESTINATION ${AUTOFUSE_INSTALL_LIB_DIR}
            COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
            OPTIONAL
    ARCHIVE DESTINATION ${AUTOFUSE_INSTALL_LIB_DIR}
            COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
            OPTIONAL
)

install(TARGETS graph_af graph_base_af aihac_symbolizer_af
    LIBRARY DESTINATION ${AUTOFUSE_INSTALL_LIB_DIR} OPTIONAL
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

install(TARGETS pyautofuse
    LIBRARY DESTINATION ${AUTOFUSE_INSTALL_PYTHON_DIR}/autofuse OPTIONAL
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

install(FILES
        compiler/python/__init__.py
        compiler/python/asc_codegen_compile.py
        compiler/python/ascbc_kernel_compile.py
        compiler/python/compile_adapter.py
        compiler/python/ascendc_compile.py
        compiler/python/ascir_api.py
    DESTINATION ${AUTOFUSE_INSTALL_PYTHON_DIR}/autofuse
    PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
    OPTIONAL
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

install(DIRECTORY ${CODE_ROOT_DIR}/inc/autoschedule/
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/autoschedule
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY ${CODE_ROOT_DIR}/inc/backend/
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/backend
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY ${CODE_ROOT_DIR}/inc/fusion/
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/fusion
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY ${CODE_ROOT_DIR}/inc/autofuse/autofuse_frame/
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/autofuse_frame
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
    FILES_MATCHING PATTERN "*.h"
)

install(FILES
        ${CODE_ROOT_DIR}/inc/common/autofuse_base_type.h
        ${CODE_ROOT_DIR}/inc/common/platform_context.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/common
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

install(FILES
        ${CODE_ROOT_DIR}/autofuse/utils/auto_fuse_config.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/utils
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

install(FILES
        ${CODE_ROOT_DIR}/ascir/meta/ascend_graph_code_dumper.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/ascir/meta
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

install(FILES
        ${CODE_ROOT_DIR}/common/common_utils.h
        ${CODE_ROOT_DIR}/common/schedule_result.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/common
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

install(FILES
        ${CODE_ROOT_DIR}/att/base/base_types.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/base
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

install(FILES
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/ascendc_ir/ascendc_ir_check.h
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/ascendc_ir/ascir_register.h
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/ascendc_ir/ascir_registry.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/graph_metadef/graph/ascendc_ir
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

# ascendc_ir_def.h includes "attr_store.h" with a bare filename.
# GE adds ascendc_ir_core to its include path, so install alongside other ascendc_ir_core headers.
install(FILES ${CODE_ROOT_DIR}/inc/graph_metadef/graph/attr_store.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/graph_metadef/graph/ascendc_ir/ascendc_ir_core
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

# Bridge header for GE to add nodes into AscGraph without instantiating af-internal types.
install(FILES ${CODE_ROOT_DIR}/inc/graph_metadef/graph/ascendc_ir/ascendc_ir_core/asc_graph_ge_bridge.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/graph_metadef/graph/ascendc_ir/ascendc_ir_core
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

# attr_store.h includes "attribute_group/af_attr_group_base.h".
# GE adds graph_metadef/graph to its include path.
install(FILES ${CODE_ROOT_DIR}/inc/graph_metadef/graph/attribute_group/af_attr_group_base.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/graph_metadef/graph/attribute_group
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

# ascendc_ir_def.h includes "serialization/attr_serializer_registry.h" (internal impl dir).
# GE adds graph_metadef/graph to its include path.
install(FILES
        ${GE_METADEF_DIR}/graph/serialization/attr_serializer_registry.h
        ${GE_METADEF_DIR}/graph/serialization/attr_serializer.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/graph_metadef/graph/serialization
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

# ascendc_ir_def.h includes "graph/symbolizer/symbolic.h" and symbol_operator/symbol_checker.
# GE adds graph_metadef/graph to its include path.
install(FILES
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/symbolizer/symbolic.h
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/symbolizer/symbol_operator.h
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/symbolizer/symbol_checker.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/graph_metadef/graph/symbolizer
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

install(FILES ${CODE_ROOT_DIR}/inc/graph_metadef/graph/debug/ge_util.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/graph_metadef/graph/debug
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

# ascir_ops.h depends on af::ascir::cg types defined in cg_utils.h and its dependencies.
install(FILES
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/utils/cg_utils.h
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/utils/axis_utils.h
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/utils/node_utils_ex.h
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/utils/dtype_transform_utils.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/graph_metadef/graph/utils
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

# ascir/meta headers keep source-relative layout.
install(FILES
        ${CODE_ROOT_DIR}/ascir/meta/ascir.h
        ${CODE_ROOT_DIR}/ascir/meta/ascir_ops_utils.h
        ${CODE_ROOT_DIR}/ascir/meta/ascir_utils.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/ascir/meta
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

# generated ascir_ops.h (produced by ascir_builtin_ops_header at build time)
install(FILES ${CMAKE_BINARY_DIR}/ascir_builtin_ops/ascir_ops.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/ascir
    OPTIONAL
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

# Keep only the minimal relative-neighbor headers required by ascir_ops.h.
install(FILES ${CODE_ROOT_DIR}/inc/graph_metadef/graph/ascendc_ir/ascend_reg_ops.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/ascir/ascendc_ir
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

install(FILES
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/ascendc_ir/ascendc_ir_core/ascendc_ir_def.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/ascir/ascendc_ir/ascendc_ir_core
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

install(FILES
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/ascendc_ir/ascendc_ir_core/ascendc_ir_def.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/graph_metadef/graph/ascendc_ir/ascendc_ir_core
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

install(FILES
        ${CMAKE_BINARY_DIR}/proto/metadef_protos_af/proto/af_ir.pb.h
        ${CMAKE_BINARY_DIR}/proto/metadef_protos_af/proto/ascendc_ir.pb.h
        ${CMAKE_BINARY_DIR}/proto/metadef_protos_af/proto/af_attr_group_base.pb.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/proto
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

install(FILES
        ${CODE_ROOT_DIR}/inc/graph_metadef/external/graph/operator.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/graph_metadef/external/graph
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

install(FILES
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/anchor.h
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/node.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/graph_metadef/graph
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

install(FILES
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/ascendc_ir/utils/asc_graph_utils.h
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/ascendc_ir/utils/asc_tensor_utils.h
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/ascendc_ir/utils/ascendc_ir_dump_utils.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/graph_metadef/graph/ascendc_ir/utils
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

install(FILES ${CODE_ROOT_DIR}/inc/graph_metadef/graph/ascendc_ir/utils/cg_calc_tmp_buff_common_funcs.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/ascir/ascendc_ir/utils
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

install(FILES
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/utils/cg_utils.h
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/utils/axis_utils.h
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/utils/node_utils_ex.h
        ${CODE_ROOT_DIR}/inc/graph_metadef/graph/utils/dtype_transform_utils.h
    DESTINATION ${AUTOFUSE_INSTALL_INCLUDE_DIR}/ascir/utils
    COMPONENT ${AUTOFUSE_INSTALL_COMPONENT}
)

endif()
