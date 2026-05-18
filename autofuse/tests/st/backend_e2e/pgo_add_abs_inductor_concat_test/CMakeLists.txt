set(PGO_ADD_ABS_INDUCTOR_CONCAT_WORKDIR ${CMAKE_CURRENT_BINARY_DIR})
set(PGO_ADD_ABS_INDUCTOR_CONCAT_TILING_DEF ${PGO_ADD_ABS_INDUCTOR_CONCAT_WORKDIR}/autofuse_tiling_data.h)
set(PGO_ADD_ABS_INDUCTOR_CONCAT_HOST_CODE ${PGO_ADD_ABS_INDUCTOR_CONCAT_WORKDIR}/pgo_add_abs_inductor_concat_host.cpp)
set(PGO_ADD_ABS_INDUCTOR_CONCAT_DEVICE_CODE ${PGO_ADD_ABS_INDUCTOR_CONCAT_WORKDIR}/pgo_add_abs_inductor_concat_device.cpp)
set(PGO_ADD_ABS_INDUCTOR_CONCAT_OUTPUT_DIR ${PGO_ADD_ABS_INDUCTOR_CONCAT_WORKDIR}/split_compile_out)

set(PYAUTOFUSE_DIR ${CMAKE_BINARY_DIR}/tests/autofuse)
set(AUTOFUSE_PYTHON_DIR ${CMAKE_SOURCE_DIR}/compiler/graph/optimize/autofuse/compiler/python)

add_executable(pgo_add_abs_inductor_concat_test_codegen pgo_add_abs_inductor_concat_backend_generate.cpp)
target_link_options(pgo_add_abs_inductor_concat_test_codegen PRIVATE -Wl,--no-as-needed)
target_link_libraries(pgo_add_abs_inductor_concat_test_codegen
    atrace optimize att codegen common_stub share_graph ascgen_common
    aihac_symbolizer asc_slog_stub error_manager metadef json
    GTest::gtest GTest::gtest_main)
target_compile_definitions(pgo_add_abs_inductor_concat_test_codegen PRIVATE
    ATT_SO_NAME="./libatt.so"
    KERNEL_SRC_LIST="${PGO_ADD_ABS_INDUCTOR_CONCAT_TILING_DEF}:${PGO_ADD_ABS_INDUCTOR_CONCAT_HOST_CODE}:${PGO_ADD_ABS_INDUCTOR_CONCAT_DEVICE_CODE}"
)
add_test(NAME pgo_add_abs_inductor_concat_test_codegen
    COMMAND pgo_add_abs_inductor_concat_test_codegen --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/pgo_add_abs_inductor_concat_test_codegen.xml)
set_tests_properties(pgo_add_abs_inductor_concat_test_codegen PROPERTIES LABELS "st;build_backend_test1;pgo_add_abs_inductor_concat_test_codegen")

file(MAKE_DIRECTORY ${PGO_ADD_ABS_INDUCTOR_CONCAT_OUTPUT_DIR})
add_custom_command(
    OUTPUT ${PGO_ADD_ABS_INDUCTOR_CONCAT_TILING_DEF} ${PGO_ADD_ABS_INDUCTOR_CONCAT_HOST_CODE} ${PGO_ADD_ABS_INDUCTOR_CONCAT_DEVICE_CODE}
    COMMAND $<TARGET_FILE:pgo_add_abs_inductor_concat_test_codegen>
            --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/pgo_add_abs_inductor_concat_test_codegen.xml
    DEPENDS pgo_add_abs_inductor_concat_test_codegen
    VERBATIM)
add_custom_target(pgo_add_abs_inductor_concat_test_codegen_outputs
    DEPENDS ${PGO_ADD_ABS_INDUCTOR_CONCAT_TILING_DEF} ${PGO_ADD_ABS_INDUCTOR_CONCAT_HOST_CODE} ${PGO_ADD_ABS_INDUCTOR_CONCAT_DEVICE_CODE})
add_executable(pgo_add_abs_inductor_concat_test_e2e test_e2e_pgo_add_abs_inductor_concat_split_compile.cpp)
add_dependencies(pgo_add_abs_inductor_concat_test_e2e pgo_add_abs_inductor_concat_test_codegen_outputs)
target_link_libraries(pgo_add_abs_inductor_concat_test_e2e dl GTest::gtest GTest::gtest_main unified_dlog)
target_include_directories(pgo_add_abs_inductor_concat_test_e2e PRIVATE ${PGO_ADD_ABS_INDUCTOR_CONCAT_WORKDIR})
target_compile_definitions(pgo_add_abs_inductor_concat_test_e2e PRIVATE
    TILING_DEF_FILE="${PGO_ADD_ABS_INDUCTOR_CONCAT_TILING_DEF}"
    HOST_CODE_FILE="${PGO_ADD_ABS_INDUCTOR_CONCAT_HOST_CODE}"
    DEVICE_CODE_FILE="${PGO_ADD_ABS_INDUCTOR_CONCAT_DEVICE_CODE}"
    OUTPUT_DIR="${PGO_ADD_ABS_INDUCTOR_CONCAT_OUTPUT_DIR}"
    PYAUTOFUSE_DIR="${PYAUTOFUSE_DIR}"
    AUTOFUSE_PYTHON_DIR="${AUTOFUSE_PYTHON_DIR}"
    ASCEND_HOME_PATH="${ASCEND_INSTALL_PATH}"
)
add_test(NAME pgo_add_abs_inductor_concat_test_e2e
    COMMAND pgo_add_abs_inductor_concat_test_e2e --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/pgo_add_abs_inductor_concat_test_e2e.xml)
set_tests_properties(pgo_add_abs_inductor_concat_test_e2e PROPERTIES LABELS "st;build_backend_test2;pgo_add_abs_inductor_concat_test_e2e")
