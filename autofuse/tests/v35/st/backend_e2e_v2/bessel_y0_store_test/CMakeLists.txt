backend_e2e_st_test(bessel_y0_store_test
    CODEGEN bessel_y0_store_test_backend_generator.cpp
    KERNEL_SRC
        bessel_y0_store_test_kernel.cpp
        bessel_y0_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_bessel_y0_store_test_kernel.cpp)
target_include_directories(bessel_y0_store_test_e2e_v2 PRIVATE ${CODE_ROOT_DIR}/v35/ascendc/api_regbase)
