backend_e2e_st_test(mask_reg_chain_test
    CODEGEN mask_reg_chain_backend_generator.cpp
    KERNEL_SRC
        mask_reg_chain_test_kernel.cpp
        mask_reg_chain_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_mask_reg_chain_kernel.cpp)