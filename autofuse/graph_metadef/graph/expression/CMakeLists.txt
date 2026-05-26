add_library(aihac_symbolizer_af SHARED
    expression.cc
    expression_impl.cc
    expr_print_manager.cc
    symbol_operator.cc
    symbolic_utils.cc
    symbol_checker.cc
    expr_parser.cc
    scanner.cc
    attr_group_shape_env_attr.cc
    attr_group_symbolic_desc_attr.cc
    guard_dfx_context.cc
)
target_compile_options(aihac_symbolizer_af PRIVATE -DNO_METADEF_ABI_COMPATIABLE)

target_compile_options(aihac_symbolizer_af PRIVATE
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>)

target_compile_definitions(aihac_symbolizer_af PRIVATE
    FMK_SUPPORT_DUMP
    google=ascend_private
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
)

target_include_directories(aihac_symbolizer_af PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${METADEF_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_protos_af
)

target_link_options(aihac_symbolizer_af PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(aihac_symbolizer_af
    PRIVATE
        af_intf_pub
        static_mmpa
        c_sec
        unified_dlog
        json
        platform
        symengine
        error_manager
        graph_base_af
        graph_af
        unified_dlog
        Boost::boost
        ascend_protobuf
        metadef_headers
    PUBLIC
        metadef_headers_af
)
set_target_properties(aihac_symbolizer_af PROPERTIES
        CXX_EXTENSIONS NO
        OUTPUT_NAME aihac_symbolizer_af)
autofuse_apply_shared_visibility(aihac_symbolizer_af)
