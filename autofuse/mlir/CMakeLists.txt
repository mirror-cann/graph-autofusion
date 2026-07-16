cmake_minimum_required(VERSION 3.16)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AutofuseMlirDeps.cmake)

if(NOT ENABLE_AUTOFUSE_MLIR)
    return()
endif()

message(STATUS "Enable Autofuse MLIR migration infrastructure")

set(AF_PYASC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/pyasc" CACHE PATH "PyAsc upstream submodule path")
if(NOT EXISTS "${AF_PYASC_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "PyAsc submodule is missing. Run: bash autofuse/mlir/scripts/sync_pyasc_upstream.sh")
endif()

set(ASCIR_LLT_TEST OFF CACHE BOOL "Build PyAsc lit tests" FORCE)
if(NOT pybind11_DIR)
    if(NOT Python3_EXECUTABLE)
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
    endif()
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _af_pybind11_dir
        RESULT_VARIABLE _af_pybind11_result
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_af_pybind11_result EQUAL 0 AND EXISTS "${_af_pybind11_dir}/pybind11Config.cmake")
        set(pybind11_DIR "${_af_pybind11_dir}" CACHE PATH "pybind11 CMake config directory" FORCE)
    endif()
endif()
add_subdirectory(${AF_PYASC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/pyasc EXCLUDE_FROM_ALL)

add_subdirectory(dialect/AFIR)
add_subdirectory(conversion)
add_subdirectory(bridge)
add_subdirectory(tools/af-afir-gen)
add_subdirectory(demo)
add_subdirectory(tools/af-opt)
add_subdirectory(test)
