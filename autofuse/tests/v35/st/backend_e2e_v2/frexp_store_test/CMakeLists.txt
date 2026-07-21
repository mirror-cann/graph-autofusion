backend_e2e_st_test(frexp_store_test
    CODEGEN frexp_store_backend_generator.cpp
    KERNEL_SRC
        frexp_store_test_kernel.cpp
        frexp_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_frexp_store_kernel.cpp)
target_include_directories(frexp_store_test_e2e_v2 PRIVATE ${CODE_ROOT_DIR}/v35/ascendc/api_regbase)
