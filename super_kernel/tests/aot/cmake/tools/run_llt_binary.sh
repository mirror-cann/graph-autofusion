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

set -e

ENV_FILE="$1"
shift
LD_LIBRARY_PATH_VALUE="$1"
shift
LD_PRELOAD_VALUE="$1"
shift
ASAN_OPTIONS_VALUE="$1"
shift
BINARY="$1"
shift

if [[ "${ENV_FILE}" == "__EMPTY__" ]]; then
  ENV_FILE=""
fi

if [[ "${LD_PRELOAD_VALUE}" == "__EMPTY__" ]]; then
  LD_PRELOAD_VALUE=""
fi

if [[ "${ASAN_OPTIONS_VALUE}" == "__EMPTY__" ]]; then
  ASAN_OPTIONS_VALUE=""
fi

if [[ -n "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH_VALUE}"

if [[ -n "${LD_PRELOAD_VALUE}" ]]; then
  export LD_PRELOAD="${LD_PRELOAD_VALUE}"
fi

if [[ -n "${ASAN_OPTIONS_VALUE}" ]]; then
  export ASAN_OPTIONS="${ASAN_OPTIONS_VALUE}"
  ulimit -s 32768
fi

HAS_GTEST_FILTER=0
LIST_LOG_FILE=""
LOG_FILE=""

cleanup() {
  rm -f "${LIST_LOG_FILE:-}" "${LOG_FILE:-}"
}

trap cleanup EXIT

for arg in "$@"; do
  case "${arg}" in
    --gtest_filter)
      HAS_GTEST_FILTER=1
      break
      ;;
    --gtest_filter=*)
      if [[ -n "${arg#--gtest_filter=}" ]]; then
        HAS_GTEST_FILTER=1
      fi
      break
      ;;
  esac
done

if [[ ${HAS_GTEST_FILTER} -eq 1 ]]; then
  LIST_LOG_FILE="$(mktemp)"

  set +e
  "${BINARY}" "$@" --gtest_list_tests >"${LIST_LOG_FILE}" 2>&1
  LIST_STATUS=$?
  set -e

  cat "${LIST_LOG_FILE}"

  if [[ ${LIST_STATUS} -ne 0 ]]; then
    exit "${LIST_STATUS}"
  fi

  if ! grep -Eq '^[[:space:]]{2,}[^[:space:]].*$' "${LIST_LOG_FILE}"; then
    echo "[ERROR] No AOT C++ tests were selected. Please check --test_case / GTEST_FILTER." >&2
    exit 1
  fi
fi

LOG_FILE="$(mktemp)"

set +e
"${BINARY}" "$@" 2>&1 | tee "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}
set -e

if [[ ${STATUS} -ne 0 ]]; then
  exit "${STATUS}"
fi

if grep -Eq '^\[[[:space:]]*SKIPPED[[:space:]]*\][[:space:]]+[1-9][0-9]*[[:space:]]+tests?' "${LOG_FILE}"; then
  echo "[ERROR] AOT C++ tests contain skipped cases. Please fix or explicitly remove the skipped tests." >&2
  grep -E '^\[[[:space:]]*SKIPPED[[:space:]]*\]' "${LOG_FILE}" >&2 || true
  exit 1
fi
