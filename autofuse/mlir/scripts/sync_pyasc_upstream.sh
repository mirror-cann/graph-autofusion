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
pyasc_dir="${repo_root}/autofuse/mlir/externals/pyasc"
pyasc_path="autofuse/mlir/externals/pyasc"
patch_dir="${repo_root}/autofuse/mlir/patches/pyasc"
base_ref="${AF_PYASC_BASE_REF:-}"
expected_url="https://gitcode.com/cann/pyasc.git"
apply_patches=1
protect_gitlink=1
unprotect_gitlink=0

usage() {
  cat <<'EOF'
Usage:
  sync_pyasc_upstream.sh [--ref <ref>] [--no-apply-patches] [--no-protect-gitlink]
  sync_pyasc_upstream.sh --unprotect-gitlink

Options:
  --ref <ref>             Checkout a specific PyAsc ref.
  --no-apply-patches      Stop after checking out the pinned or selected upstream ref.
  --no-protect-gitlink    Do not protect the PyAsc gitlink after applying patches.
  --unprotect-gitlink     Remove local gitlink protection and exit.
  -h, --help              Show this help.

By default this script uses the PyAsc commit pinned by the Autofuse gitlink.
It does not update to the latest upstream commit unless --ref or
AF_PYASC_BASE_REF is set explicitly.

When patches are applied, the script protects the local PyAsc gitlink from
git status noise and accidental git add -A staging. Use --unprotect-gitlink
before intentionally updating and staging the pinned upstream baseline.
EOF
}

protect_pyasc_gitlink() {
  git -C "${repo_root}" config "submodule.${pyasc_path}.ignore" all
  git -C "${repo_root}" update-index --skip-worktree "${pyasc_path}"
}

unprotect_pyasc_gitlink() {
  git -C "${repo_root}" update-index --no-skip-worktree "${pyasc_path}" 2>/dev/null || true
  git -C "${repo_root}" config --unset "submodule.${pyasc_path}.ignore" 2>/dev/null || true
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --ref)
      [ "$#" -ge 2 ] || { echo "ERROR: --ref requires a value" >&2; exit 2; }
      base_ref="$2"
      shift 2
      ;;
    --no-apply-patches)
      apply_patches=0
      shift
      ;;
    --no-protect-gitlink)
      protect_gitlink=0
      shift
      ;;
    --unprotect-gitlink)
      unprotect_gitlink=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ "${unprotect_gitlink}" -eq 1 ]; then
  unprotect_pyasc_gitlink
  exit 0
fi

actual_url="$(git -C "${repo_root}" config --file .gitmodules --get submodule.autofuse/mlir/externals/pyasc.url)"
if [ "${actual_url}" != "${expected_url}" ]; then
  echo "ERROR: PyAsc submodule must use official upstream: ${expected_url}" >&2
  echo "Actual: ${actual_url}" >&2
  exit 2
fi

if [ -z "${base_ref}" ]; then
  git -C "${repo_root}" submodule update --init --recursive autofuse/mlir/externals/pyasc
else
  git -C "${repo_root}" submodule update --init --recursive autofuse/mlir/externals/pyasc
  git -C "${pyasc_dir}" fetch origin
  git -C "${pyasc_dir}" checkout "${base_ref}"
fi

if [ "${apply_patches}" -eq 1 ] && compgen -G "${patch_dir}/*.patch" >/dev/null; then
  git -C "${pyasc_dir}" am "${patch_dir}"/*.patch
  if [ "${protect_gitlink}" -eq 1 ]; then
    protect_pyasc_gitlink
  fi
fi
