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
build_sh="${repo_root}/build.sh"
run_autofuse_test_sh="${repo_root}/scripts/test/run_autofuse_test.sh"
autofuse_cmake="${repo_root}/autofuse/CMakeLists.txt"
mlir_cmake="${repo_root}/autofuse/mlir/CMakeLists.txt"
dockerfile_llvm="${repo_root}/autofuse/mlir/docker/Dockerfile.llvm"
validate_llvm_image_sh="${repo_root}/autofuse/mlir/docker/validate_llvm_image.sh"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

assert_file_contains() {
    local file="$1"
    local pattern="$2"
    if ! grep -Fq -- "${pattern}" "${file}"; then
        echo "--- ${file} ---" >&2
        sed -n '1,220p' "${file}" >&2
        fail "expected '${file}' to contain '${pattern}'"
    fi
}

assert_file_matches() {
    local file="$1"
    local pattern="$2"
    if ! grep -Eq -- "${pattern}" "${file}"; then
        fail "expected '${file}' to match '${pattern}'"
    fi
}

test -f "${build_sh}" || fail "missing ${build_sh}"
test -f "${run_autofuse_test_sh}" || fail "missing ${run_autofuse_test_sh}"
test -f "${autofuse_cmake}" || fail "missing ${autofuse_cmake}"
test -f "${mlir_cmake}" || fail "missing ${mlir_cmake}"
test -f "${dockerfile_llvm}" || fail "missing ${dockerfile_llvm}"
test -f "${validate_llvm_image_sh}" || fail "missing ${validate_llvm_image_sh}"

assert_file_contains "${build_sh}" '--enable-mlir'
assert_file_contains "${build_sh}" 'ENABLE_AUTOFUSE_MLIR="off"'
assert_file_contains "${build_sh}" 'enable-mlir'
assert_file_contains "${build_sh}" '--enable-mlir)'
assert_file_contains "${build_sh}" 'ENABLE_AUTOFUSE_MLIR="on"'
assert_file_contains "${build_sh}" '--no-autofuse)'
assert_file_contains "${build_sh}" 'ENABLE_AUTOFUSE="off"'
assert_file_contains "${build_sh}" '-DENABLE_AUTOFUSE_MLIR=ON'
assert_file_contains "${build_sh}" '-DENABLE_AUTOFUSE_MLIR=OFF'
assert_file_contains "${build_sh}" 'test_args+=("--enable-mlir")'

assert_file_contains "${run_autofuse_test_sh}" '--enable-mlir'
assert_file_contains "${run_autofuse_test_sh}" 'ENABLE_AUTOFUSE_MLIR="off"'
assert_file_contains "${run_autofuse_test_sh}" 'ENABLE_AUTOFUSE_MLIR="on"'
assert_file_contains "${run_autofuse_test_sh}" '-D ENABLE_AUTOFUSE_MLIR=ON'
assert_file_contains "${run_autofuse_test_sh}" '-D ENABLE_AUTOFUSE_MLIR=OFF'
assert_file_contains "${run_autofuse_test_sh}" 'require_coverage_tools'

assert_file_matches "${autofuse_cmake}" 'option\(ENABLE_AUTOFUSE_MLIR "Enable Autofuse MLIR migration infrastructure" OFF\)'
assert_file_matches "${autofuse_cmake}" 'if\(ENABLE_AUTOFUSE_MLIR\)'
assert_file_matches "${autofuse_cmake}" 'add_subdirectory\(mlir\)'
assert_file_contains "${mlir_cmake}" 'include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AutofuseMlirDeps.cmake)'
assert_file_contains "${mlir_cmake}" 'if(NOT ENABLE_AUTOFUSE_MLIR)'
assert_file_contains "${mlir_cmake}" 'return()'

assert_file_contains "${dockerfile_llvm}" 'lcov'
assert_file_contains "${validate_llvm_image_sh}" 'lcov genhtml gcov'

echo "MLIR build switch check passed."
