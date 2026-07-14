backend_e2e_st_test(zeta_store_test
    CODEGEN zeta_store_test_backend_generator.cpp
    KERNEL_SRC
        zeta_store_test_kernel.cpp
        zeta_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_zeta_store_test_kernel.cpp)
target_include_directories(zeta_store_test_e2e_v2 PRIVATE ${CODE_ROOT_DIR}/v35/ascendc/api_regbase)
