set(TAIL_REDUCE_WORKDIR ${CMAKE_CURRENT_BINARY_DIR})
set(TAIL_REDUCE_TILING_DEF ${TAIL_REDUCE_WORKDIR}/autofuse_tiling_data.h)
set(TAIL_REDUCE_HOST_CODE ${TAIL_REDUCE_WORKDIR}/inductor_tail_brc_tail_reduce_host.cpp)
set(TAIL_REDUCE_DEVICE_CODE ${TAIL_REDUCE_WORKDIR}/inductor_tail_brc_tail_reduce_device.cpp)
set(TAIL_REDUCE_OUTPUT_DIR ${TAIL_REDUCE_WORKDIR}/split_compile_out)
file(TOUCH ${TAIL_REDUCE_TILING_DEF})

set(PYAUTOFUSE_DIR ${CMAKE_BINARY_DIR}/tests/autofuse)
set(AUTOFUSE_PYTHON_DIR ${CMAKE_SOURCE_DIR}/compiler/graph/optimize/autofuse/compiler/python)

add_executable(inductor_tail_brc_tail_reduce_test_codegen inductor_tail_brc_tail_reduce_backend_generate.cpp)
target_link_options(inductor_tail_brc_tail_reduce_test_codegen PRIVATE -Wl,--no-as-needed)
target_link_libraries(inductor_tail_brc_tail_reduce_test_codegen
    atrace optimize att codegen common_stub share_graph ascgen_common
    aihac_symbolizer asc_slog_stub error_manager metadef json
    GTest::gtest GTest::gtest_main)
target_compile_definitions(inductor_tail_brc_tail_reduce_test_codegen PRIVATE
    ATT_SO_NAME="./libatt.so"
    KERNEL_SRC_LIST="${TAIL_REDUCE_TILING_DEF}:${TAIL_REDUCE_HOST_CODE}:${TAIL_REDUCE_DEVICE_CODE}"
)
add_test(NAME inductor_tail_brc_tail_reduce_test_codegen
    COMMAND inductor_tail_brc_tail_reduce_test_codegen --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/inductor_tail_brc_tail_reduce_test_codegen.xml)
set_tests_properties(inductor_tail_brc_tail_reduce_test_codegen PROPERTIES LABELS "st;build_backend_test1;inductor_tail_brc_tail_reduce_test_codegen")

file(MAKE_DIRECTORY ${TAIL_REDUCE_OUTPUT_DIR})
add_executable(inductor_tail_brc_tail_reduce_test_e2e test_e2e_inductor_tail_brc_tail_reduce_split_compile.cpp)
target_link_libraries(inductor_tail_brc_tail_reduce_test_e2e dl GTest::gtest GTest::gtest_main unified_dlog)
target_include_directories(inductor_tail_brc_tail_reduce_test_e2e PRIVATE ${TAIL_REDUCE_WORKDIR})
target_compile_definitions(inductor_tail_brc_tail_reduce_test_e2e PRIVATE
    TILING_DEF_FILE="${TAIL_REDUCE_TILING_DEF}"
    HOST_CODE_FILE="${TAIL_REDUCE_HOST_CODE}"
    DEVICE_CODE_FILE="${TAIL_REDUCE_DEVICE_CODE}"
    OUTPUT_DIR="${TAIL_REDUCE_OUTPUT_DIR}"
    PYAUTOFUSE_DIR="${PYAUTOFUSE_DIR}"
    AUTOFUSE_PYTHON_DIR="${AUTOFUSE_PYTHON_DIR}"
    ASCEND_HOME_PATH="${ASCEND_INSTALL_PATH}"
)
add_test(NAME inductor_tail_brc_tail_reduce_test_e2e
    COMMAND inductor_tail_brc_tail_reduce_test_e2e --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/inductor_tail_brc_tail_reduce_test_e2e.xml)
set_tests_properties(inductor_tail_brc_tail_reduce_test_e2e PROPERTIES LABELS "st;build_backend_test2;inductor_tail_brc_tail_reduce_test_e2e")
