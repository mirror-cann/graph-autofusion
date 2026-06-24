backend_e2e_st_test(isinf_maskedfill_test
    CODEGEN isinf_maskedfill_backend_generate.cpp
    KERNEL_SRC
        isinf_maskedfill_test_kernel.cpp
        isinf_maskedfill_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_isinf_maskedfill_expect_kernel.cpp)
