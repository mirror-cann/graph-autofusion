set(GENERATOR_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# aihac_ir_register: ascendc_ir op 注册库
add_library(aihac_ir_register SHARED
    ${GENERATOR_DIR}/ascir_register.cc
    ${GENERATOR_DIR}/ascir_registry.cc
)

target_compile_options(aihac_ir_register PRIVATE
    -O2 -fPIC -fno-common
    -Wextra -Wfloat-equal -Wall
)

target_link_options(aihac_ir_register PRIVATE
    -rdynamic
    -Wl,-Bsymbolic
)

target_include_directories(aihac_ir_register PRIVATE
    ${GE_METADEF_INC_DIR}
    ${GE_METADEF_INC_DIR}/graph
    ${GE_METADEF_INC_DIR}/graph/ascendc_ir
    ${GE_METADEF_INC_DIR}/graph/ascendc_ir/ascendc_ir_core
    ${GE_METADEF_DIR}
    ${GE_METADEF_DIR}/graph
    ${CMAKE_BINARY_DIR}/proto/metadef_protos_af
    ${GENERATOR_DIR}
)

target_link_libraries(aihac_ir_register PRIVATE runtime_headers metadef_headers)
target_link_libraries(aihac_ir_register PRIVATE
    af_intf_pub
    -Wl,--no-as-needed
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
    graph_af
    graph_base_af
    aihac_symbolizer_af
    ascend_protobuf
    PUBLIC
    aihac_ir
    unified_dlog
    c_sec
)
autofuse_apply_shared_visibility(aihac_ir_register)

# ascir_generate: 代码生成库
add_library(ascir_generate SHARED
    ${GENERATOR_DIR}/generator.cc
)

target_compile_options(ascir_generate PRIVATE
    -O2 -fPIC -fno-common
    -Wextra -Wfloat-equal -Wall
)

target_link_options(ascir_generate PRIVATE
    -rdynamic
    -Wl,-Bsymbolic
)

target_include_directories(ascir_generate PRIVATE
    ${GE_METADEF_INC_DIR}
    ${GE_METADEF_INC_DIR}/graph
    ${GE_METADEF_INC_DIR}/graph/ascendc_ir
    ${GE_METADEF_INC_DIR}/graph/ascendc_ir/ascendc_ir_core
    ${GE_METADEF_DIR}
    ${GE_METADEF_DIR}/graph
    ${CMAKE_BINARY_DIR}/proto/metadef_protos_af
    ${GENERATOR_DIR}
)

target_link_libraries(ascir_generate PRIVATE
    af_intf_pub
    -Wl,--no-as-needed
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
    graph_base_af
    aihac_symbolizer_af
    aihac_ir_register
    aihac_ir
    ascend_protobuf
    metadef_headers
)
autofuse_apply_shared_visibility(ascir_generate)

# ascir_ops_header_generator: 可执行文件，用于生成 ascir_ops.h
add_executable(ascir_ops_header_generator
    ${GENERATOR_DIR}/ascir_ops_generator_main.cc
)

target_compile_options(ascir_ops_header_generator PRIVATE
    -O2 -fPIC -Wextra -Wfloat-equal -Wall
)

target_include_directories(ascir_ops_header_generator PRIVATE
    ${GE_METADEF_INC_DIR}
    ${GE_METADEF_INC_DIR}/graph
    ${GE_METADEF_INC_DIR}/graph/ascendc_ir
    ${GE_METADEF_INC_DIR}/graph/ascendc_ir/ascendc_ir_core
    ${GE_METADEF_DIR}
    ${GE_METADEF_DIR}/graph
    ${CMAKE_BINARY_DIR}/proto/metadef_protos_af
    ${GE_METADEF_DIR}/graph/ascendc_ir/generator
    ${CODE_ROOT_DIR}/common
)

target_link_libraries(ascir_ops_header_generator PRIVATE
    ascir_generate
    aihac_ir
    graph_af
    graph_base_af
    unified_dlog
    c_sec
    mmpa
)

include(generator.cmake)
