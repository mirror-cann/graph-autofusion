set(CODEGEN_DIR ${CODE_ROOT_DIR}/codegen)
file(GLOB_RECURSE CODEGEN_SRCS CONFIGURE_DEPENDS "${CODEGEN_DIR}/*.cpp")

target_sources(aihac_codegen PRIVATE
        ${CODEGEN_SRCS}
)

target_include_directories(aihac_codegen PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CODE_ROOT_DIR}
        ${CODE_ROOT_DIR}/common
        ${CODE_ROOT_DIR}/att
        ${TOP_DIR}/runtime/include/external
)

if(ENABLE_AUTOFUSE_MLIR)
        target_sources(aihac_codegen PRIVATE
                ${CODE_ROOT_DIR}/mlir/bridge/ScheduleToAfir/ScheduleToAfir.cpp
        )
        target_compile_definitions(aihac_codegen PRIVATE ENABLE_AUTOFUSE_MLIR=1)
        # In-process bridge: ScheduleToAfir needs AFIR dialect + AfirBuilder headers
        # plus the MLIR/LLVM headers. MLIR_INCLUDE_DIRS is not yet populated in this
        # scope (mlir/ subdir is processed after codegen/), so resolve the prebuilt
        # LLVM/MLIR include root directly from the deps root.
        if(DEFINED MLIR_INCLUDE_DIRS AND MLIR_INCLUDE_DIRS)
                target_include_directories(aihac_codegen SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
        else()
                set(_af_llvm_root "$ENV{AF_LLVM_ROOT}")
                if(NOT _af_llvm_root)
                        set(_af_llvm_root "$ENV{LLVM_BUILD_DIR}")
                endif()
                if(NOT _af_llvm_root)
                        set(_af_llvm_root "${AF_LLVM_ROOT}")
                endif()
                target_include_directories(aihac_codegen SYSTEM PRIVATE ${_af_llvm_root}/include)
        endif()
        target_include_directories(aihac_codegen PRIVATE
                ${CODE_ROOT_DIR}/mlir/bridge        # ScheduleToAfir/ , AfirBuilder/
                ${CODE_ROOT_DIR}/mlir/dialect      # AFIR/AFIR.h
                ${CMAKE_BINARY_DIR}/mlir/dialect/AFIR   # generated AFIR/*.inc
        )
        # MLIR/AFIR is heavy templated code; relax -Werror just for the bridge file.
        set_source_files_properties(
                ${CODE_ROOT_DIR}/mlir/bridge/ScheduleToAfir/ScheduleToAfir.cpp
                PROPERTIES COMPILE_OPTIONS "-Wno-error")
        target_link_libraries(aihac_codegen PRIVATE AfirBuilder AFIRDialect)
endif()
