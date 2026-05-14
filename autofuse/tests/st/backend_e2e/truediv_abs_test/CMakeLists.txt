backend_e2e_st_test(truediv_abs_test
    CODEGEN truediv_abs_backend_generate.cpp
    KERNEL_SRC
        truediv_abs_test_kernel.cpp
        truediv_abs_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_truediv_abs_expect_kernel.cpp)
