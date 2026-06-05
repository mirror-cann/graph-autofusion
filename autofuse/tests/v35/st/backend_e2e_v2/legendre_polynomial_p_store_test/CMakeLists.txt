backend_e2e_st_test(legendre_polynomial_p_store_test
    CODEGEN legendre_polynomial_p_store_backend_generator.cpp
    KERNEL_SRC
        legendre_polynomial_p_store_test_kernel.cpp
        legendre_polynomial_p_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_legendre_polynomial_p_store_kernel.cpp)
