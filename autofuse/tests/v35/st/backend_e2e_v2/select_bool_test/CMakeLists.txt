backend_e2e_st_test(select_bool_test
    CODEGEN select_bool_backend_generate.cpp
    KERNEL_SRC
        select_bool_test_kernel.cpp
        select_bool_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_select_bool_kernel.cpp)

set_target_properties(select_bool_test_codegen_v2 PROPERTIES
    BUILD_RPATH
        "${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/ascendc_ir/generator;${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph;${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/expression;${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/ascendc_ir;${CMAKE_BINARY_DIR}/autofuse/ascir/meta")
target_link_options(select_bool_test_codegen_v2 PRIVATE -Wl,--disable-new-dtags)
