backend_e2e_st_test(ndtri_store_test
    CODEGEN ndtri_store_test_backend_generator.cpp
    KERNEL_SRC
        ndtri_store_test_kernel.cpp
        ndtri_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_ndtri_store_test_kernel.cpp)
target_include_directories(ndtri_store_test_e2e_v2 PRIVATE ${CODE_ROOT_DIR}/v35/ascendc/api_regbase)
