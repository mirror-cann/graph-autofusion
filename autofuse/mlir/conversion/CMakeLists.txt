# AFIR -> ascendc lowering passes (autofuse MLIR migration).

# --- AFIR -> ascendc lowering (minimal, binary elementwise → L3; concept demo) ---
add_mlir_library(AFIRToAscendC
  AFIRToAscendC/AFIRToAscendC.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}

  LINK_LIBS PUBLIC
  AFIRDialect
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRTransformUtils
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRAsc
  MLIREmitAsc
)

# Headers resolve as "AFIRToAscendC/AFIRToAscendC.h"; pyasc ascendc headers as
# "ascir/Dialect/Asc/IR/Asc.h".
target_include_directories(AFIRToAscendC PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(AFIRToAscendC PRIVATE
  ${AF_PYASC_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/../pyasc/include
)

# --- AFIR -> ascendc queue-faithful lowering (tiling + multi-core + tail) ------
add_mlir_library(AFIRToAscendCQueue
  AFIRToAscendCQueue/AFIRToAscendCQueue.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}

  LINK_LIBS PUBLIC
  AFIRDialect
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRAsc
  MLIREmitAsc
)
target_include_directories(AFIRToAscendCQueue PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(AFIRToAscendCQueue PRIVATE
  ${AF_PYASC_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/../pyasc/include
)
