if(EXISTS "/home/xingzhixiong/TSPD_CODE/qiancang/ge/tests/autofuse/ut/e2e/load_broadcast_multi_axis_store_expect_code/test_load_broadcast_multi_axis_store_codegen")
  if(NOT EXISTS "/home/xingzhixiong/TSPD_CODE/qiancang/ge/tests/autofuse/ut/e2e/load_broadcast_multi_axis_store_expect_code/test_load_broadcast_multi_axis_store_codegen[1]_tests.cmake" OR
     NOT "/home/xingzhixiong/TSPD_CODE/qiancang/ge/tests/autofuse/ut/e2e/load_broadcast_multi_axis_store_expect_code/test_load_broadcast_multi_axis_store_codegen[1]_tests.cmake" IS_NEWER_THAN "/home/xingzhixiong/TSPD_CODE/qiancang/ge/tests/autofuse/ut/e2e/load_broadcast_multi_axis_store_expect_code/test_load_broadcast_multi_axis_store_codegen" OR
     NOT "/home/xingzhixiong/TSPD_CODE/qiancang/ge/tests/autofuse/ut/e2e/load_broadcast_multi_axis_store_expect_code/test_load_broadcast_multi_axis_store_codegen[1]_tests.cmake" IS_NEWER_THAN "${CMAKE_CURRENT_LIST_FILE}")
    include("/home/xingzhixiong/.local/lib/python3.8/site-packages/cmake/data/share/cmake-4.2/Modules/GoogleTestAddTests.cmake")
    gtest_discover_tests_impl(
      TEST_EXECUTABLE [==[/home/xingzhixiong/TSPD_CODE/qiancang/ge/tests/autofuse/ut/e2e/load_broadcast_multi_axis_store_expect_code/test_load_broadcast_multi_axis_store_codegen]==]
      TEST_EXECUTOR [==[]==]
      TEST_WORKING_DIR [==[/home/xingzhixiong/TSPD_CODE/qiancang/ge/tests/autofuse/ut/e2e/load_broadcast_multi_axis_store_expect_code]==]
      TEST_EXTRA_ARGS [==[]==]
      TEST_PROPERTIES [==[]==]
      TEST_PREFIX [==[]==]
      TEST_SUFFIX [==[]==]
      TEST_FILTER [==[]==]
      NO_PRETTY_TYPES [==[FALSE]==]
      NO_PRETTY_VALUES [==[FALSE]==]
      TEST_LIST [==[test_load_broadcast_multi_axis_store_codegen_TESTS]==]
      CTEST_FILE [==[/home/xingzhixiong/TSPD_CODE/qiancang/ge/tests/autofuse/ut/e2e/load_broadcast_multi_axis_store_expect_code/test_load_broadcast_multi_axis_store_codegen[1]_tests.cmake]==]
      TEST_DISCOVERY_TIMEOUT [==[5]==]
      TEST_DISCOVERY_EXTRA_ARGS [==[]==]
      TEST_XML_OUTPUT_DIR [==[]==]
    )
  endif()
  include("/home/xingzhixiong/TSPD_CODE/qiancang/ge/tests/autofuse/ut/e2e/load_broadcast_multi_axis_store_expect_code/test_load_broadcast_multi_axis_store_codegen[1]_tests.cmake")
else()
  add_test(test_load_broadcast_multi_axis_store_codegen_NOT_BUILT test_load_broadcast_multi_axis_store_codegen_NOT_BUILT)
endif()
