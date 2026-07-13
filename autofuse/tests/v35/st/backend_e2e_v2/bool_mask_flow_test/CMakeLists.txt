backend_e2e_st_test(bool_mask_flow_test
    CODEGEN bool_mask_flow_backend_generate.cpp
    KERNEL_SRC
        bool_mask_flow_test_kernel.cpp
        bool_mask_flow_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_bool_mask_flow_kernel.cpp)

set_target_properties(bool_mask_flow_test_codegen_v2 PROPERTIES
    BUILD_RPATH
        "${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/ascendc_ir/generator;${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph;${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/expression;${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/ascendc_ir;${CMAKE_BINARY_DIR}/autofuse/ascir/meta")
target_link_options(bool_mask_flow_test_codegen_v2 PRIVATE -Wl,--disable-new-dtags)
