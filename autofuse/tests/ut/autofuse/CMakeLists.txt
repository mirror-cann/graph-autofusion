# autofuse/tests/ut/autofuse/CMakeLists.txt
# 仅测试 autofuse/utils/ 下的 AutoFuseConfig 和 SkipFuseConfig 接口
# Note: autofuse/autofuse/ is frontend code (not compiled in af repo).
# Only compile the specific source files needed by the tests.

set(AUTOFUSE_ROOT_DIR ${CODE_ROOT_DIR}/autofuse)

add_executable(autofuse_utils_ut
    auto_fuse_config_unittest.cpp
    skip_fuse_config_unittest.cpp
    ${AUTOFUSE_ROOT_DIR}/utils/auto_fuse_config.cpp
)

target_compile_options(autofuse_utils_ut PRIVATE
    -O0 -g --coverage -fprofile-arcs -ftest-coverage
    -Wall -Wfloat-equal -Wno-deprecated-declarations
    -fpermissive
)

target_compile_definitions(autofuse_utils_ut PRIVATE
    google=ascend_private
)

target_include_directories(autofuse_utils_ut PRIVATE
    ${AUTOFUSE_ROOT_DIR}
    ${CODE_ROOT_DIR}
    ${CODE_ROOT_DIR}/att
    ${CODE_ROOT_DIR}/inc
)

target_link_options(autofuse_utils_ut PRIVATE
    -Wl,--no-as-needed
)

target_link_libraries(autofuse_utils_ut PRIVATE
    -lrt -ldl -lgcov
    GTest::gtest
    GTest::gtest_main
    asc_slog_stub
    autofuse_runtime_stub
    error_manager
    metadef
    graph_af
    graph_base_af
    aihac_symbolizer_af
    json
    ascend_protobuf
)

gtest_discover_tests(autofuse_utils_ut)
