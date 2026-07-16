#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
llvm_root="$(bash "${script_dir}/resolve_llvm_env.sh")"
mlir_config="${llvm_root}/lib/cmake/mlir/MLIRConfig.cmake"
llvm_config="${llvm_root}/lib/cmake/llvm/LLVMConfig.cmake"

if [ ! -f "${mlir_config}" ]; then
  echo "ERROR: MLIRConfig.cmake not found: ${mlir_config}" >&2
  exit 1
fi
if [ ! -f "${llvm_config}" ]; then
  echo "ERROR: LLVMConfig.cmake not found: ${llvm_config}" >&2
  exit 1
fi

out_dir="${repo_root}/autofuse/mlir/.artifacts/env"
mkdir -p "${out_dir}"
manifest="${out_dir}/llvm_manifest.json"

cat > "${manifest}" <<EOF
{
  "llvm_root": "${llvm_root}",
  "llvm_version": "$("${llvm_root}/bin/llvm-config" --version)",
  "llvm_bindir": "$("${llvm_root}/bin/llvm-config" --bindir)",
  "llvm_libdir": "$("${llvm_root}/bin/llvm-config" --libdir)",
  "llvm_config": "${llvm_config}",
  "mlir_config": "${mlir_config}",
  "arch": "$(uname -m)",
  "docker_image": "${AF_MLIR_DEV_IMAGE:-}",
  "docker_digest": "${AF_MLIR_DEV_IMAGE_DIGEST:-}"
}
EOF

echo "${manifest}"
