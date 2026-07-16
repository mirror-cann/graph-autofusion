# FusedScheduledResult -> AFIR bridge (autofuse MLIR migration).
#
# Two layers:
#   AfirBuilder     backend-neutral: plain structs -> AFIR module (built here)
#   ScheduleToAfir  in-process adapter: real FusedScheduledResult -> those structs.
#                   Compiled directly into the codegen lib (see codegen/CMakeLists.txt),
#                   not built as a standalone library here — it needs autofuse
#                   backend headers only available in the full build.

# --- AfirBuilder core: neutral schedule info -> AFIR module ----------------
add_mlir_library(AfirBuilder
  AfirBuilder/AfirBuilder.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}

  LINK_LIBS PUBLIC
  AFIRDialect
  MLIRIR
  MLIRFuncDialect
  MLIRSupport
)
target_include_directories(AfirBuilder PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
