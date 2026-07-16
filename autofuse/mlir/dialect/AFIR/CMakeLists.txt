# AFIR dialect (autofuse MLIR migration demo).
# Subset ported from Ascend-MLIR-gsr, decoupled from PyAsc-fork traits so it
# builds against autofuse's official PyAsc submodule.

# --- TableGen: dialect / enums / attrs / ops -------------------------------
# tblgen outputs into the AFIR/ subdir; the Makefile generator (full autofuse
# build) does not auto-create it, unlike Ninja (standalone mlir/ build).
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/AFIR)

set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(AFIR/Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(AFIR/Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(AFIRDialectIncGen)

set(LLVM_TARGET_DEFINITIONS Core/Enums.td)
mlir_tablegen(AFIR/Enums.h.inc -gen-enum-decls)
mlir_tablegen(AFIR/Enums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(AFIREnumsIncGen)
add_dependencies(AFIREnumsIncGen AFIRDialectIncGen)

set(LLVM_TARGET_DEFINITIONS Core/Attrs.td)
mlir_tablegen(AFIR/Attrs.h.inc -gen-attrdef-decls)
mlir_tablegen(AFIR/Attrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(AFIRAttrsIncGen)
add_dependencies(AFIRAttrsIncGen AFIRDialectIncGen AFIREnumsIncGen)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(AFIR/Ops.h.inc -gen-op-decls)
mlir_tablegen(AFIR/Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(AFIROpsIncGen)
add_dependencies(AFIROpsIncGen AFIRDialectIncGen AFIRAttrsIncGen AFIREnumsIncGen)

# --- Library ---------------------------------------------------------------
add_mlir_dialect_library(AFIRDialect
  Dialect.cpp
  Ops/Basic/math.cpp
  Ops/Basic/data.cpp
  Ops/Basic/broadcast.cpp
  Ops/Basic/cast.cpp
  Ops/Basic/condition.cpp
  Ops/Adv/concat.cpp
  Ops/Adv/reduce.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}

  DEPENDS
  AFIRDialectIncGen
  AFIREnumsIncGen
  AFIRAttrsIncGen
  AFIROpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRInferTypeOpInterface
)

# Expose dirs so headers resolve as "AFIR/AFIR.h" (source) and
# "AFIR/*.h.inc" (generated, output names already carry the AFIR/ prefix).
target_include_directories(AFIRDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_BINARY_DIR}
)
