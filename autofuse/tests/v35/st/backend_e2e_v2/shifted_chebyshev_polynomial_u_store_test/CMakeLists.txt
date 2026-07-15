backend_e2e_st_test(shifted_chebyshev_polynomial_u_store_test
    CODEGEN shifted_chebyshev_polynomial_u_store_test_backend_generator.cpp
    KERNEL_SRC
        shifted_chebyshev_polynomial_u_store_test_kernel.cpp
        shifted_chebyshev_polynomial_u_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_shifted_chebyshev_polynomial_u_store_test_kernel.cpp)
target_include_directories(shifted_chebyshev_polynomial_u_store_test_e2e_v2 PRIVATE ${CODE_ROOT_DIR}/v35/ascendc/api_regbase)
