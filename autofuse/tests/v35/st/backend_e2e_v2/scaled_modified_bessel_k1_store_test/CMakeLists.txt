backend_e2e_st_test(scaled_modified_bessel_k1_store_test
    CODEGEN scaled_modified_bessel_k1_store_test_backend_generator.cpp
    KERNEL_SRC
        scaled_modified_bessel_k1_store_test_kernel.cpp
        scaled_modified_bessel_k1_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_scaled_modified_bessel_k1_store_test_kernel.cpp)
target_include_directories(scaled_modified_bessel_k1_store_test_e2e_v2 PRIVATE ${CODE_ROOT_DIR}/v35/ascendc/api_regbase)
