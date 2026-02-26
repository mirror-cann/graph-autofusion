
add_executable(test_wqbmm main.cpp)
target_include_directories(test_wqbmm PRIVATE $ENV{ASCEND_HOME_PATH}/include)
target_link_directories(test_wqbmm PRIVATE $<${RUN_MODEL}:$ENV{ASCEND_HOME_PATH}/tools/simulator/Ascend910B1/lib>
                                     $ENV{ASCEND_HOME_PATH}/lib64)
target_link_libraries(test_wqbmm PRIVATE ops_interface)

