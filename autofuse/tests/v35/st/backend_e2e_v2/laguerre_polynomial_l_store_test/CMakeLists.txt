backend_e2e_st_test(laguerre_polynomial_l_store_test
    CODEGEN laguerre_polynomial_l_store_backend_generator.cpp
    KERNEL_SRC
        laguerre_polynomial_l_store_test_kernel.cpp
        laguerre_polynomial_l_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_laguerre_polynomial_l_store_kernel.cpp)
