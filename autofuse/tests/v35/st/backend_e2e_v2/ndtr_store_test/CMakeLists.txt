backend_e2e_st_test(ndtr_store_test
    CODEGEN ndtr_store_test_backend_generator.cpp
    KERNEL_SRC
        ndtr_store_test_kernel.cpp
        ndtr_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_ndtr_store_test_kernel.cpp)
target_include_directories(ndtr_store_test_e2e_v2 PRIVATE ${CODE_ROOT_DIR}/v35/ascendc/api_regbase)
