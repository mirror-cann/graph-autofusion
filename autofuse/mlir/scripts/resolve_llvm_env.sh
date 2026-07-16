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
mlir_root="$(cd "${script_dir}/.." && pwd)"

candidate="${LLVM_BUILD_DIR:-}"
if [ -z "${candidate}" ] && [ -n "${AF_LLVM_ROOT:-}" ]; then
  candidate="${AF_LLVM_ROOT}"
fi
if [ -z "${candidate}" ] && [ -n "${MLIR_DIR:-}" ]; then
  candidate="$(cd "${MLIR_DIR}/../../.." && pwd)"
fi
if [ -z "${candidate}" ] && [ -n "${LLVM_DIR:-}" ]; then
  candidate="$(cd "${LLVM_DIR}/../../.." && pwd)"
fi
if [ -z "${candidate}" ]; then
  candidate="${mlir_root}/.deps/llvm"
fi

if [ ! -x "${candidate}/bin/llvm-config" ]; then
  echo "ERROR: LLVM/MLIR build not found." >&2
  echo "Set LLVM_BUILD_DIR or AF_LLVM_ROOT to a directory containing bin/llvm-config." >&2
  echo "Checked: ${candidate}" >&2
  exit 1
fi

echo "${candidate}"
