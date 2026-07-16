get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(AF_MLIR_OPT_LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}

    MLIRAffineAnalysis
    MLIRAnalysis
    MLIRCastInterfaces
    MLIRDialect
    MLIRMlirOptMain
    MLIROptLib
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRTransformUtils
    MLIRSupport
    MLIRIR

    MLIRAsc
    MLIRAscTransforms
    MLIRAscUtils
    MLIREmitAsc

    AFIRDialect
    AFIRToAscendC
    AFIRToAscendCQueue
)

add_executable(af-opt main.cpp)
target_include_directories(af-opt PRIVATE
    ${AF_PYASC_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/../../pyasc/include
)
target_link_libraries(af-opt PRIVATE
    ${AF_MLIR_OPT_LIBS}
)

if(COMMAND llvm_update_compile_flags)
    llvm_update_compile_flags(af-opt)
endif()
if(COMMAND mlir_check_all_link_libraries)
    mlir_check_all_link_libraries(af-opt)
endif()
