backend_e2e_st_test(remainder_int32_store_test
    CODEGEN remainder_int32_store_backend_generator.cpp
    KERNEL_SRC
        remainder_int32_store_test_kernel.cpp
        remainder_int32_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_remainder_int32_store_kernel.cpp)
