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
    ${GE_METADEF_INC_DIR}/external
    ${GE_METADEF_INC_DIR}/graph
    ${GE_METADEF_INC_DIR}/graph/ascendc_ir
    ${GE_METADEF_INC_DIR}/graph/ascendc_ir/ascendc_ir_core
    ${GE_METADEF_DIR}
    ${GE_METADEF_DIR}/graph
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
    ${GENERATOR_DIR}
)

target_link_libraries(aihac_ir_register PRIVATE
    intf_pub
    -Wl,--no-as-needed
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
    graph
    graph_base
    aihac_symbolizer
    ascend_protobuf
    PUBLIC
    aihac_ir
    unified_dlog
    c_sec
)
set_target_properties(aihac_ir_register PROPERTIES OUTPUT_NAME aihac_ir_register_af)
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
    ${GE_METADEF_INC_DIR}/external
    ${GE_METADEF_INC_DIR}/graph
    ${GE_METADEF_INC_DIR}/graph/ascendc_ir
    ${GE_METADEF_INC_DIR}/graph/ascendc_ir/ascendc_ir_core
    ${GE_METADEF_DIR}
    ${GE_METADEF_DIR}/graph
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
    ${GENERATOR_DIR}
)

target_link_libraries(ascir_generate PRIVATE
    intf_pub
    -Wl,--no-as-needed
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
    graph_base
    aihac_symbolizer
    aihac_ir_register
    aihac_ir
    ascend_protobuf
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
    ${GE_METADEF_INC_DIR}/external
    ${GE_METADEF_INC_DIR}/graph
    ${GE_METADEF_INC_DIR}/graph/ascendc_ir
    ${GE_METADEF_INC_DIR}/graph/ascendc_ir/ascendc_ir_core
    ${GE_METADEF_DIR}
    ${GE_METADEF_DIR}/graph
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
    ${GE_METADEF_DIR}/graph/ascendc_ir/generator
    ${CODE_ROOT_DIR}/common
)

target_link_libraries(ascir_ops_header_generator PRIVATE
    ascir_generate
    aihac_ir
    graph
    graph_base
    unified_dlog
    c_sec
    mmpa
)

include(generator.cmake)
