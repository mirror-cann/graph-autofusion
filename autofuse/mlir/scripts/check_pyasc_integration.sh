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
tmp_dir="$(mktemp -d "${mlir_root}/.artifacts/check-pyasc.XXXXXX")"

cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

fail() {
  echo "ERROR: $*" >&2
  exit 1
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

assert_executable() {
  local file="$1"
  [ -x "${file}" ] || fail "expected executable file: ${file}"
}

assert_file_contains "${repo_root}/.gitmodules" "[submodule \"autofuse/mlir/externals/pyasc\"]"
assert_file_contains "${repo_root}/.gitmodules" "url = https://gitcode.com/cann/pyasc.git"
assert_file_contains "${mlir_root}/patches/pyasc/README.md" \
  "PyAsc submodule follows upstream \`https://gitcode.com/cann/pyasc.git\`."
assert_file_contains "${mlir_root}/patches/pyasc/README.md" \
  "Export only selected Autofuse-required commits or ranges"
assert_file_contains "${mlir_root}/README.md" \
  "bash mlir/scripts/sync_pyasc_upstream.sh --unprotect-gitlink"
assert_executable "${script_dir}/sync_pyasc_upstream.sh"
assert_executable "${script_dir}/export_pyasc_patch.sh"
bash -n "${script_dir}/sync_pyasc_upstream.sh" "${script_dir}/export_pyasc_patch.sh"
assert_file_contains "${script_dir}/sync_pyasc_upstream.sh" "--no-protect-gitlink"
assert_file_contains "${script_dir}/sync_pyasc_upstream.sh" "--unprotect-gitlink"
assert_file_contains "${script_dir}/sync_pyasc_upstream.sh" "update-index --skip-worktree"
assert_file_contains "${script_dir}/export_pyasc_patch.sh" "--pyasc-dir"

if compgen -G "${mlir_root}/patches/pyasc/*.patch" >/dev/null; then
  pinned_pyasc_commit="$(git -C "${repo_root}" ls-files --stage autofuse/mlir/externals/pyasc | awk '{print $2}')"
  [ -n "${pinned_pyasc_commit}" ] || fail "failed to resolve pinned PyAsc gitlink commit"
  patch_check_dir="${tmp_dir}/pyasc-patch-check"
  git -C "${mlir_root}/externals/pyasc" worktree add --detach "${patch_check_dir}" "${pinned_pyasc_commit}" >/dev/null
  git -C "${patch_check_dir}" am "${mlir_root}"/patches/pyasc/*.patch >/dev/null
fi

export_log="${tmp_dir}/export-default.log"
if "${script_dir}/export_pyasc_patch.sh" --output-dir "${tmp_dir}/patches-default" >"${export_log}" 2>&1; then
  fail "export_pyasc_patch.sh should require explicit --commit or --range"
fi
assert_file_contains "${export_log}" "choose PyAsc commits explicitly"

mkdir -p "${tmp_dir}/patches-selected"
"${script_dir}/export_pyasc_patch.sh" --output-dir "${tmp_dir}/patches-selected" --commit HEAD >/dev/null
find "${tmp_dir}/patches-selected" -name '*.patch' -print -quit | grep -q . || \
  fail "expected explicit --commit export to create one patch in a temporary directory"

mkdir -p "${tmp_dir}/patches-external"
"${script_dir}/export_pyasc_patch.sh" \
  --pyasc-dir "${mlir_root}/externals/pyasc" \
  --output-dir "${tmp_dir}/patches-external" \
  --commit HEAD >/dev/null
find "${tmp_dir}/patches-external" -name '*.patch' -print -quit | grep -q . || \
  fail "expected export from an explicit PyAsc checkout to create one patch"

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
cat > "${fake_llvm}/lib/cmake/mlir/AddLLVM.cmake" <<'EOF'
function(llvm_update_compile_flags target_name)
endfunction()

function(configure_lit_site_cfg input_file output_file)
  configure_file("${input_file}" "${output_file}" @ONLY)
endfunction()

function(add_lit_testsuite target_name comment)
  add_custom_target("${target_name}")
endfunction()
EOF
cat > "${fake_llvm}/lib/cmake/mlir/AddMLIR.cmake" <<'EOF'
function(mlir_check_all_link_libraries target_name)
endfunction()
EOF
touch "${fake_llvm}/lib/cmake/mlir/HandleLLVMOptions.cmake"

make_wrapper_project() {
  local src_dir="$1"
  local enable_mlir="$2"
  mkdir -p "${src_dir}"
  cat > "${src_dir}/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.16)
project(check_pyasc_integration CXX)
set(ENABLE_AUTOFUSE_MLIR ${enable_mlir})
add_subdirectory("${mlir_root}" "\${CMAKE_BINARY_DIR}/autofuse-mlir")
EOF
}

fake_pyasc="${tmp_dir}/fake-pyasc"
mkdir -p "${fake_pyasc}/include" "${fake_pyasc}/lib/TableGen" "${fake_pyasc}/lib" "${fake_pyasc}/bin"
cat > "${fake_pyasc}/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.16)
project(fake_pyasc CXX)
EOF
for subdir in include lib/TableGen lib bin; do
  cat > "${fake_pyasc}/${subdir}/CMakeLists.txt" <<EOF
add_custom_target(fake_pyasc_${subdir//\//_})
EOF
done

off_src="${tmp_dir}/off-src"
make_wrapper_project "${off_src}" OFF
cmake -S "${off_src}" -B "${tmp_dir}/off-build" >/dev/null

missing_src="${tmp_dir}/missing-src"
make_wrapper_project "${missing_src}" ON
missing_log="${tmp_dir}/missing-pyasc.log"
if cmake -S "${missing_src}" -B "${tmp_dir}/missing-build" \
    -DLLVM_BUILD_DIR= \
    -DAF_LLVM_ROOT="${fake_llvm}" \
    -DMLIR_DIR= \
    -DLLVM_DIR= \
    -DAF_PYASC_DIR="${tmp_dir}/missing-pyasc" >"${missing_log}" 2>&1; then
  fail "CMake should fail when ENABLE_AUTOFUSE_MLIR=ON and PyAsc is missing"
fi
assert_file_contains "${missing_log}" "PyAsc submodule is missing"

valid_src="${tmp_dir}/valid-src"
make_wrapper_project "${valid_src}" ON
cmake -S "${valid_src}" -B "${tmp_dir}/valid-build" \
  -DLLVM_BUILD_DIR= \
  -DAF_LLVM_ROOT="${fake_llvm}" \
  -DMLIR_DIR= \
  -DLLVM_DIR= \
  -DAF_PYASC_DIR="${fake_pyasc}" >/dev/null

echo "PyAsc integration check passed."
