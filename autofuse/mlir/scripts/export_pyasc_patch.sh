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
patch_dir="${repo_root}/autofuse/mlir/patches/pyasc"

usage() {
  cat <<'EOF'
Usage:
  export_pyasc_patch.sh [--pyasc-dir <dir>] --commit <commit> [--commit <commit> ...]
  export_pyasc_patch.sh [--pyasc-dir <dir>] --range <rev-range> [--range <rev-range> ...]

Options:
  --commit <commit>     Export exactly one selected PyAsc commit with format-patch -1.
  --range <rev-range>   Export an explicitly selected PyAsc revision range.
  --pyasc-dir <dir>     Export from an explicit PyAsc checkout.
  --output-dir <dir>    Override patch output directory.
  -h, --help            Show this help.

Examples:
  bash autofuse/mlir/scripts/export_pyasc_patch.sh --commit HEAD
  bash autofuse/mlir/scripts/export_pyasc_patch.sh --range topic-base..topic-head
  bash autofuse/mlir/scripts/export_pyasc_patch.sh --pyasc-dir /path/to/pyasc --commit HEAD

The script intentionally refuses to export origin/master..HEAD implicitly. Keep
Autofuse PyAsc patches minimal and reviewable; do not vendor unrelated PyAsc
upstream history into this patchset.
EOF
}

commits=""
ranges=""
commit_count=0
range_count=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --commit)
      [ "$#" -ge 2 ] || { echo "ERROR: --commit requires a value" >&2; exit 2; }
      commits="${commits}${2}"$'\n'
      commit_count=$((commit_count + 1))
      shift 2
      ;;
    --range)
      [ "$#" -ge 2 ] || { echo "ERROR: --range requires a value" >&2; exit 2; }
      ranges="${ranges}${2}"$'\n'
      range_count=$((range_count + 1))
      shift 2
      ;;
    --pyasc-dir)
      [ "$#" -ge 2 ] || { echo "ERROR: --pyasc-dir requires a value" >&2; exit 2; }
      pyasc_dir="$2"
      shift 2
      ;;
    --output-dir)
      [ "$#" -ge 2 ] || { echo "ERROR: --output-dir requires a value" >&2; exit 2; }
      patch_dir="$2"
      shift 2
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

if [ ! -d "${pyasc_dir}/.git" ] && ! git -C "${pyasc_dir}" rev-parse --git-dir >/dev/null 2>&1; then
  echo "ERROR: PyAsc submodule is missing. Run: bash autofuse/mlir/scripts/sync_pyasc_upstream.sh" >&2
  exit 2
fi

if [ "${commit_count}" -eq 0 ] && [ "${range_count}" -eq 0 ]; then
  echo "ERROR: choose PyAsc commits explicitly with --commit or --range." >&2
  echo "Refusing to export the full upstream-to-HEAD range implicitly." >&2
  usage >&2
  exit 2
fi

mkdir -p "${patch_dir}"
while IFS= read -r commit; do
  [ -n "${commit}" ] || continue
  git -C "${pyasc_dir}" format-patch -1 --output-directory "${patch_dir}" "${commit}"
done <<EOF
${commits}
EOF
while IFS= read -r range; do
  [ -n "${range}" ] || continue
  git -C "${pyasc_dir}" format-patch --output-directory "${patch_dir}" "${range}"
done <<EOF
${ranges}
EOF
