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
mlir_root="${repo_root}/autofuse/mlir"
tmp_dir="$(mktemp -d "${mlir_root}/.artifacts/check-llvm-deps.XXXXXX")"
created_default_llvm=0
locked_default_llvm=0
lock_dir="${mlir_root}/.artifacts/check-llvm-deps.lock"

cleanup() {
  if [ "${created_default_llvm}" -eq 1 ]; then
    rm -f "${mlir_root}/.deps/llvm"
  fi
  if [ "${locked_default_llvm}" -eq 1 ]; then
    rmdir "${lock_dir}" 2>/dev/null || true
  fi
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

acquire_default_llvm_lock() {
  while ! mkdir "${lock_dir}" 2>/dev/null; do
    sleep 0.1
  done
  locked_default_llvm=1
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

assert_equal() {
  local expected="$1"
  local actual="$2"
  local context="$3"
  if [ "${expected}" != "${actual}" ]; then
    fail "${context}: expected '${expected}', got '${actual}'"
  fi
}

assert_file_contains() {
  local file="$1"
  local pattern="$2"
  if ! grep -Fq -- "${pattern}" "${file}"; then
    echo "--- ${file} ---" >&2
    cat "${file}" >&2
    fail "expected '${file}' to contain '${pattern}'"
  fi
}

fake_llvm="${tmp_dir}/fake-llvm"
mkdir -p "${fake_llvm}/bin" \
         "${fake_llvm}/include" \
         "${fake_llvm}/lib/cmake/llvm" \
         "${fake_llvm}/lib/cmake/mlir"

cat > "${fake_llvm}/bin/llvm-config" <<'EOF'
#!/bin/bash
set -euo pipefail
case "${1:-}" in
  --version)
    echo "21.0.0git"
    ;;
  --bindir)
    cd "$(dirname "$0")"
    pwd
    ;;
  --libdir)
    cd "$(dirname "$0")/../lib"
    pwd
    ;;
  *)
    echo "unsupported llvm-config argument: ${1:-}" >&2
    exit 2
    ;;
esac
EOF
chmod +x "${fake_llvm}/bin/llvm-config"

cat > "${fake_llvm}/lib/cmake/mlir/MLIRConfig.cmake" <<'EOF'
set(MLIR_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(MLIR_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/../../../include")
set(LLVM_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/../../../include")
set(LLVM_DEFINITIONS "-DAF_FAKE_LLVM")
EOF
touch "${fake_llvm}/lib/cmake/llvm/LLVMConfig.cmake"
touch "${fake_llvm}/lib/cmake/mlir/TableGen.cmake"
touch "${fake_llvm}/lib/cmake/mlir/AddLLVM.cmake"
touch "${fake_llvm}/lib/cmake/mlir/AddMLIR.cmake"
touch "${fake_llvm}/lib/cmake/mlir/HandleLLVMOptions.cmake"

resolved="$(LLVM_BUILD_DIR="${fake_llvm}" AF_LLVM_ROOT="${tmp_dir}/unused" \
  bash "${script_dir}/resolve_llvm_env.sh")"
assert_equal "${fake_llvm}" "${resolved}" "LLVM_BUILD_DIR should take precedence"

resolved="$(env -u LLVM_BUILD_DIR AF_LLVM_ROOT="${fake_llvm}" \
  bash "${script_dir}/resolve_llvm_env.sh")"
assert_equal "${fake_llvm}" "${resolved}" "AF_LLVM_ROOT should be used when LLVM_BUILD_DIR is unset"

resolved="$(env -u LLVM_BUILD_DIR -u AF_LLVM_ROOT MLIR_DIR="${fake_llvm}/lib/cmake/mlir" \
  bash "${script_dir}/resolve_llvm_env.sh")"
assert_equal "${fake_llvm}" "${resolved}" "MLIR_DIR should resolve to the LLVM root"

acquire_default_llvm_lock
if [ ! -e "${mlir_root}/.deps/llvm" ] && [ ! -L "${mlir_root}/.deps/llvm" ]; then
  mkdir -p "${mlir_root}/.deps"
  ln -s "${fake_llvm}" "${mlir_root}/.deps/llvm"
  created_default_llvm=1
fi
resolved="$(env -u LLVM_BUILD_DIR -u AF_LLVM_ROOT -u MLIR_DIR -u LLVM_DIR \
  bash "${script_dir}/resolve_llvm_env.sh")"
if [ "${created_default_llvm}" -eq 1 ]; then
  assert_equal "${mlir_root}/.deps/llvm" "${resolved}" "default .deps/llvm should be used"
fi

missing_log="${tmp_dir}/missing-llvm.log"
if LLVM_BUILD_DIR="${tmp_dir}/missing-llvm" bash "${script_dir}/resolve_llvm_env.sh" >"${missing_log}" 2>&1; then
  fail "resolve_llvm_env.sh should fail when llvm-config is missing"
fi
assert_file_contains "${missing_log}" "Set LLVM_BUILD_DIR or AF_LLVM_ROOT"

manifest_path="$(LLVM_BUILD_DIR="${fake_llvm}" \
  AF_MLIR_DEV_IMAGE="fake:image" \
  AF_MLIR_DEV_IMAGE_DIGEST="sha256:fake" \
  bash "${script_dir}/prepare_mlir_deps.sh")"
[ -f "${manifest_path}" ] || fail "manifest was not created: ${manifest_path}"
assert_file_contains "${manifest_path}" "\"llvm_root\": \"${fake_llvm}\""
assert_file_contains "${manifest_path}" "\"llvm_version\": \"21.0.0git\""
assert_file_contains "${manifest_path}" "\"mlir_config\": \"${fake_llvm}/lib/cmake/mlir/MLIRConfig.cmake\""
assert_file_contains "${manifest_path}" "\"docker_image\": \"fake:image\""

cmake_off_src="${tmp_dir}/cmake-off"
mkdir -p "${cmake_off_src}"
cat > "${cmake_off_src}/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.16)
project(check_mlir_deps_off NONE)
set(ENABLE_AUTOFUSE_MLIR OFF)
include("${mlir_root}/cmake/AutofuseMlirDeps.cmake")
EOF
cmake -S "${cmake_off_src}" -B "${tmp_dir}/cmake-off-build" >/dev/null

cmake_missing_src="${tmp_dir}/cmake-missing"
mkdir -p "${cmake_missing_src}"
cat > "${cmake_missing_src}/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.16)
project(check_mlir_deps_missing NONE)
set(ENABLE_AUTOFUSE_MLIR ON)
include("${mlir_root}/cmake/AutofuseMlirDeps.cmake")
EOF
cmake_missing_log="${tmp_dir}/cmake-missing.log"
if cmake -S "${cmake_missing_src}" -B "${tmp_dir}/cmake-missing-build" \
    -DLLVM_BUILD_DIR= \
    -DAF_LLVM_ROOT="${tmp_dir}/missing-llvm" \
    -DMLIR_DIR= \
    -DLLVM_DIR= >"${cmake_missing_log}" 2>&1; then
  fail "CMake should fail when ENABLE_AUTOFUSE_MLIR=ON and MLIRConfig.cmake is missing"
fi
assert_file_contains "${cmake_missing_log}" "ENABLE_AUTOFUSE_MLIR=ON requires MLIRConfig.cmake"

cmake_valid_src="${tmp_dir}/cmake-valid"
mkdir -p "${cmake_valid_src}"
cat > "${cmake_valid_src}/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.16)
project(check_mlir_deps_valid CXX)
set(ENABLE_AUTOFUSE_MLIR ON)
include("${mlir_root}/cmake/AutofuseMlirDeps.cmake")
EOF
cmake -S "${cmake_valid_src}" -B "${tmp_dir}/cmake-valid-build" \
  -DLLVM_BUILD_DIR= \
  -DAF_LLVM_ROOT="${fake_llvm}" \
  -DMLIR_DIR= \
  -DLLVM_DIR= >/dev/null

echo "LLVM dependency resolution check passed."
