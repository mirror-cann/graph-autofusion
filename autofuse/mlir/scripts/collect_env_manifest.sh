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

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
out_dir="${repo_root}/autofuse/mlir/.artifacts/env"
mkdir -p "${out_dir}"
manifest="${out_dir}/dev_env_manifest.json"

json_string() {
    python3 -c 'import json, sys; print(json.dumps(sys.stdin.read().rstrip("\n")))' <<< "${1:-}"
}

first_line_or_empty() {
    if ! command -v "$1" >/dev/null 2>&1; then
        return 0
    fi
    "$@" 2>&1 | head -n 1 || true
}

cann_root="${CANN_ROOT:-${ASCEND_HOME_PATH:-}}"
cann_version=""
version_files=(
    "${ASCEND_HOME_PATH:-}/version.info"
    "${ASCEND_HOME_PATH:-}/share/info/runtime/version.info"
    "${ASCEND_HOME_PATH:-}/share/info/asc-devkit/version.info"
    "${cann_root}/latest/share/info/runtime/version.info"
    "${cann_root}/latest/share/info/asc-devkit/version.info"
    "${cann_root}/version.info"
)
if [ -n "${cann_root}" ]; then
    for version_file in \
        "${cann_root}"/*/runtime/version.info \
        "${cann_root}"/*/*-linux/ascend_toolkit_install.info \
        "${cann_root}/latest/var/manager/version.info"; do
        if [ -f "${version_file}" ]; then
            version_files+=("${version_file}")
        fi
    done
fi
for version_file in "${version_files[@]}"; do
    if [ -n "${version_file}" ] && [ -f "${version_file}" ]; then
        cann_version="$(sed -n '/^Version=/p' "${version_file}" | head -n 1 || true)"
        if [ -z "${cann_version}" ]; then
            cann_version="$(head -n 1 "${version_file}" || true)"
        fi
        break
    fi
done
if [ -z "${cann_version}" ] && [ -n "${cann_root}" ]; then
    fallback_version_file="$(find "${cann_root}" -maxdepth 3 -name version.info -print -quit 2>/dev/null || true)"
    if [ -n "${fallback_version_file}" ]; then
        cann_version="$(sed -n '/^Version=/p' "${fallback_version_file}" | head -n 1 || true)"
        if [ -z "${cann_version}" ]; then
            cann_version="$(head -n 1 "${fallback_version_file}" || true)"
        fi
    fi
fi

os_release=""
if [ -f /etc/os-release ]; then
    os_release="$(head -n 1 /etc/os-release || true)"
fi

llvm_root="${AF_LLVM_ROOT:-${LLVM_BUILD_DIR:-}}"
if [ -z "${llvm_root}" ] && [ -d /opt/llvm ]; then
    llvm_root=/opt/llvm
fi
llvm_config=""
mlir_opt=""
mlir_tblgen=""
llvm_lit=""
llvm_cxxflags=""
llvm_has_rtti=""
if [ -n "${llvm_root}" ]; then
    if [ -x "${llvm_root}/bin/llvm-config" ]; then
        llvm_config="$("${llvm_root}/bin/llvm-config" --version 2>&1 | head -n 1 || true)"
        llvm_cxxflags="$("${llvm_root}/bin/llvm-config" --cxxflags 2>&1 || true)"
        llvm_has_rtti="$("${llvm_root}/bin/llvm-config" --has-rtti 2>/dev/null || true)"
    fi
    if [ -x "${llvm_root}/bin/mlir-opt" ]; then
        mlir_opt="$("${llvm_root}/bin/mlir-opt" --version 2>&1 | head -n 1 || true)"
    fi
    if [ -x "${llvm_root}/bin/mlir-tblgen" ]; then
        mlir_tblgen="$("${llvm_root}/bin/mlir-tblgen" --version 2>&1 | head -n 1 || true)"
    fi
    if [ -x "${llvm_root}/bin/llvm-lit" ]; then
        llvm_lit="$("${llvm_root}/bin/llvm-lit" --version 2>&1 | head -n 1 || true)"
    fi
fi
third_party_root="${AF_CANN_3RD_LIB_PATH:-}"

cat > "${manifest}" <<EOF
{
  "arch": $(json_string "$(uname -m)"),
  "os": $(json_string "$(uname -s)"),
  "os_release": $(json_string "${os_release}"),
  "gcc": $(json_string "$(first_line_or_empty gcc --version)"),
  "gxx": $(json_string "$(first_line_or_empty g++ --version)"),
  "cmake": $(json_string "$(first_line_or_empty cmake --version)"),
  "ninja": $(json_string "$(first_line_or_empty ninja --version)"),
  "python": $(json_string "$(first_line_or_empty python3 --version)"),
  "cann_root": $(json_string "${cann_root}"),
  "ascend_home_path": $(json_string "${ASCEND_HOME_PATH:-}"),
  "cann_version": $(json_string "${cann_version}"),
  "cann_3rd_lib_path": $(json_string "${third_party_root}"),
  "llvm_root": $(json_string "${llvm_root}"),
  "llvm_config": $(json_string "${llvm_config}"),
  "mlir_opt": $(json_string "${mlir_opt}"),
  "mlir_tblgen": $(json_string "${mlir_tblgen}"),
  "llvm_lit": $(json_string "${llvm_lit}"),
  "llvm_dir": $(json_string "${LLVM_DIR:-}"),
  "mlir_dir": $(json_string "${MLIR_DIR:-}"),
  "llvm_cxxflags": $(json_string "${llvm_cxxflags}"),
  "llvm_has_rtti": $(json_string "${llvm_has_rtti}"),
  "docker_image": $(json_string "${AF_MLIR_DEV_IMAGE:-}"),
  "docker_digest": $(json_string "${AF_MLIR_DEV_IMAGE_DIGEST:-}")
}
EOF

echo "${manifest}"
