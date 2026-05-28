#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

get_lcov_major_version() {
    local major_version
    if ! major_version=$(set -o pipefail; lcov --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)*' | head -1 | cut -d. -f1); then
        echo "Error: Failed to parse LCOV major version number, please check 'lcov --version'." >&2
        exit 1
    fi
    echo "$major_version"
}

get_lcov_minor_version() {
    local minor_version
    if ! minor_version=$(set -o pipefail; lcov --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)*' | head -1 | cut -d. -f2); then
        echo "Error: Failed to parse LCOV minor version number, please check 'lcov --version'." >&2
        exit 1
    fi
    echo "$minor_version"
}

check_lcov_version_ge() {
    local expected_major="$1"
    local expected_minor="$2"
    local current_major=$(get_lcov_major_version)
    local current_minor=$(get_lcov_minor_version)
    
    if [ "$current_major" -gt "$expected_major" ] 2>/dev/null; then
        return 0
    elif [ "$current_major" -eq "$expected_major" ] 2>/dev/null; then
        if [ "$current_minor" -ge "$expected_minor" ] 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

add_lcov_ops_by_major_version() {
    local expected_major_version="$1"
    local ops_to_be_added="$2"
    if [ "$(get_lcov_major_version)" -ge $expected_major_version ] 2>/dev/null; then
        echo "$ops_to_be_added"
    fi
}

get_lcov_parallel_params() {
    local thread_count="$1"
    if [ -z "$thread_count" ]; then
        thread_count=4
    fi
    
    if check_lcov_version_ge 2 3; then
        local minor=$(get_lcov_minor_version)
        if [ "$minor" -ge 4 ] 2>/dev/null; then
            echo "--parallel --thread-count ${thread_count}"
        else
            echo "--parallel ${thread_count}"
        fi
    fi
}

detect_lcov_ignore_errors() {
    local lcov_version
    if lcov_version=$(lcov --version 2>/dev/null); then
        LCOV_VERSION="$(echo "$lcov_version" | awk '{print $NF}')"
        LCOV_MAJOR_VERSION="$(printf '%s\n' "${LCOV_VERSION}" | awk -F. '{print $1}')"
        if [[ -z "${LCOV_MAJOR_VERSION}" || ! "${LCOV_MAJOR_VERSION}" =~ ^[0-9]+$ ]]; then
            LCOV_MAJOR_VERSION=0
        fi
        if (( LCOV_MAJOR_VERSION >= 2 )); then
            LCOV_IGNORE_ERRORS="mismatch,unused,negative"
        else
            LCOV_IGNORE_ERRORS=""
        fi
    else
    echo "Warning: Unable to detect lcov version, assuming version < 2.0"
  fi
}

get_lcov_parallel_ignore_errors() {
    local major=$(get_lcov_major_version)
    local minor=$(get_lcov_minor_version)
    
    if [ "$major" -ge 2 ] 2>/dev/null; then
        if [ "$minor" -ge 3 ] 2>/dev/null; then
            echo "--ignore-errors child,inconsistent,negative,mismatch,empty"
        else
            echo "--ignore-errors inconsistent,negative,mismatch,empty"
        fi
    fi
}

get_lcov_base_ignore_errors() {
    local major=$(get_lcov_major_version)
    
    if [ "$major" -ge 2 ] 2>/dev/null; then
        echo "--ignore-errors inconsistent,negative,mismatch,empty"
    fi
}

get_lcov_unexecuted_blocks_param() {
    local major=$(get_lcov_major_version)
    
    if [ "$major" -ge 2 ] 2>/dev/null; then
        echo "--rc geninfo_unexecuted_blocks=1"
    fi
}

get_genhtml_ignore_errors() {
    local major=$(get_lcov_major_version)
    
    if [ "$major" -ge 2 ] 2>/dev/null; then
        echo "--ignore-errors inconsistent,corrupt"
    fi
}