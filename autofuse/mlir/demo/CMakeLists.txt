# Autofuse MLIR migration demo — reproduction material only.
#
# The business code moved out of this directory:
#   conversion/   AFIR -> ascendc lowering passes (AFIRToAscendC, AFIRToAscendCQueue)
#   bridge/       FusedScheduledResult -> AFIR (AfirBuilder, ScheduleToAfir)
#   tools/af-afir-gen/   standalone AFIR emitter
#
# What remains here is the end-to-end walkthrough for the 3D LoadAddStore use
# case: README.md (intro + pass design + reproduction), reproduce.sh (one-shot
# driver; AFIR generated from the real schedule at stage time), and
# loadaddstore_3d/ (versioned kernel wrapper / harness / launcher). Nothing to
# build — see README.md.
