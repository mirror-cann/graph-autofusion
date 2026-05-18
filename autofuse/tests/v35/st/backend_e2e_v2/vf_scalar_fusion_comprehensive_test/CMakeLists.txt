backend_e2e_st_test(vf_scalar_fusion_comprehensive_test
    CODEGEN vf_scalar_fusion_comprehensive_backend_generate.cpp
    KERNEL_SRC
        vf_scalar_fusion_comprehensive_test_kernel.cpp
        vf_scalar_fusion_comprehensive_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_vf_scalar_fusion_comprehensive_expect_kernel.cpp)
