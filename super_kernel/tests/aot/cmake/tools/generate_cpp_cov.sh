#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

CUR_DIR="$(cd "$(dirname "$0")" && pwd)"

source ${CUR_DIR}/util.sh

set -e

LCOV_VERSION=""
LCOV_COMPAT_MODE=""
LCOV_MAJOR_VERSION=0
LCOV_CAPTURE_OPTS=()
LCOV_FILTER_OPTS=()

init_lcov_compat_options() {
  LCOV_VERSION="$(lcov --version 2>/dev/null | awk '{print $NF}')"
  LCOV_MAJOR_VERSION="$(printf '%s\n' "${LCOV_VERSION}" | awk -F. '{print $1}')"
  if [[ -z "${LCOV_MAJOR_VERSION}" || ! "${LCOV_MAJOR_VERSION}" =~ ^[0-9]+$ ]]; then
    LCOV_MAJOR_VERSION=0
  fi

  LCOV_CAPTURE_OPTS=()
  LCOV_FILTER_OPTS=()
  if (( LCOV_MAJOR_VERSION >= 2 )); then
    LCOV_COMPAT_MODE="modern"
    LCOV_CAPTURE_OPTS=(
      --ignore-errors mismatch,mismatch,negative,gcov
      --rc geninfo_unexecuted_blocks=1
    )
    LCOV_FILTER_OPTS=(
      --ignore-errors mismatch,mismatch,negative,gcov,empty,unused
    )
  else
    LCOV_COMPAT_MODE="legacy"
    LCOV_CAPTURE_OPTS=(
      --ignore-errors gcov
    )
  fi

  logging "Detected lcov version: ${LCOV_VERSION:-unknown}, compatibility mode: ${LCOV_COMPAT_MODE}"
}

# using lcov to generate coverage for cpp files
generate_coverage() {
  local _source_dir="$1"
  local _coverage_file="$2"
  local _cann_pkg_path="$3"

  if [[ -z "${_source_dir}" ]]; then
    logging "directory required to find the .da files"
    exit 1
  fi

  if [[ ! -d "${_source_dir}" ]]; then
    logging "directory is not exist, please check ${_source_dir}"
    exit 1
  fi

  if [[ -z "${_coverage_file}" ]]; then
    _coverage_file="coverage.info"
    logging "using default file name to generate coverage"
  fi

  \which lcov >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    logging "lcov is required to generate coverage data, please install"
    exit 1
  fi
  init_lcov_compat_options

  local _path_to_gen="$(dirname ${_coverage_file})"
  if [[ ! -d "${_path_to_gen}" ]]; then
    mk_dir "${_path_to_gen}"
  fi
  lcov -c -d "${_source_dir}" "${LCOV_CAPTURE_OPTS[@]}" -o "${_coverage_file}"
  lcov -r "${_coverage_file}" "${_cann_pkg_path}/*" "/home/jenkins/opensource/*" "${_src}/build/*" "${_src}/build_out/*" "${_src}/output/*" "${_src}/super_kernel/tests/*" "${LCOV_FILTER_OPTS[@]}" -o "${_coverage_file}"
  logging "generated coverage file ${_coverage_file} ${_src}"
}

# filter out some unused directories or files
filter_coverage() {
  local _coverage_file="$1"
  local _filtered_file="$2"

  if [[ ! -f "${_coverage_file}" ]]; then
    logging "coverage data file required"
    exit 1
  fi

  \which lcov >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    logging "lcov is required to generate coverage data, please install"
    exit 1
  fi
  lcov --remove "${_coverage_file}" '/usr/include/*' '/usr/local/include/*' \
         "${LCOV_FILTER_OPTS[@]}" \
         -o "${_filtered_file}"
}

# generate html report
generate_html() {
  local _filtered_file="$1"
  local _out_path="$2"

  \which genhtml >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    logging "genhtml is required to generate coverage html report, please install"
    exit 1
  fi

  local _path_to_gen="$(dirname ${_out_path})"
  if [[ ! -d "${_out_path}" ]]; then
    mk_dir "${_out_path}"
  fi
  genhtml "${_filtered_file}" -o "${_out_path}"
}


if [[ $# -ne 4 ]]; then
  logging "Usage: $0 DIR COV_FILE OUT_PATH CANN_PATH"
  exit 0
fi

_src="$1"
_cov_file="$2"
_out="$3"
_cann_path="$4"

generate_coverage "${_src}" "${_cov_file}" "${_cann_path}"
filter_coverage   "${_cov_file}" "${_cov_file}_filtered"
generate_html     "${_cov_file}_filtered" "${_out}"
