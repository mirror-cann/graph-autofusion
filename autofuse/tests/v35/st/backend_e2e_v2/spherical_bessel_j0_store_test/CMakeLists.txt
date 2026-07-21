backend_e2e_st_test(spherical_bessel_j0_store_test
    CODEGEN spherical_bessel_j0_store_test_backend_generator.cpp
    KERNEL_SRC
        spherical_bessel_j0_store_test_kernel.cpp
        spherical_bessel_j0_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_spherical_bessel_j0_store_test_kernel.cpp)
target_include_directories(spherical_bessel_j0_store_test_e2e_v2 PRIVATE ${CODE_ROOT_DIR}/v35/ascendc/api_regbase)
