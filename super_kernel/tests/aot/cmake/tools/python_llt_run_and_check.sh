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

CUR_DIR=$(dirname $(readlink -f "$0"))
# TOP_DIR=$(readlink -f "$CUR_DIR/../../..")

source ${CUR_DIR}/comm_log.sh

# echo command line
log_info "$0 $@"

OUTDIR="$1"
LOCAL_MODULE="$2"
SRC_PATH="$3"
TEST_PATH="$4"
RUN_ENV_FILE="$5"
RUN_ENV_PARAMS="$6"
EXPORT_PYTHONPATH="$7"
COVERAGERC_PATH="$8"
PYTHON_VERSION="$9"
LLT_RUN_MOD="${10}"
TASK_NUM="${11}"
LLT_KILL_TIME=1200
if [[ "$TASK_NUM" -gt 8 ]];then
    TASK_NUM=8
fi
log_info "begin run python llt target:${LOCAL_MODULE}"
log_info "TASK_NUM is ${TASK_NUM}"
# 去掉前后空格
function trim()
{
    sed 's/^ *//g; s/ *$//g'
}

OUTDIR=$(echo "${OUTDIR}" | trim)
LOCAL_MODULE=$(echo "${LOCAL_MODULE}" | trim)
SRC_PATH=$(echo "${SRC_PATH}" | trim)
TEST_PATH=$(echo "${TEST_PATH}" | trim)
RUN_ENV_FILE=$(echo "${RUN_ENV_FILE}" | trim)
RUN_ENV_PARAMS=$(echo "${RUN_ENV_PARAMS}" | trim)
EXPORT_PYTHONPATH=$(echo "${EXPORT_PYTHONPATH}" | trim)
COVERAGERC_PATH=$(echo "${COVERAGERC_PATH}" | trim)
PYTHON_VERSION=$(echo "${PYTHON_VERSION}" | trim)
if [ "$LLT_RUN_MOD"x = "x" ]; then
    LLT_RUN_MOD="single"
fi

PYTEST_CACHE="${OUTDIR}"/.pytest_cache/"${LOCAL_MODULE}"
COVERAGE_RESULT_PATH="${OUTDIR}"/coverage_result
COVERAGE_FILE="${OUTDIR}"/coverage_result/.coverage."${LOCAL_MODULE}"
SRC_PATH_FILE="${OUTDIR}"/coverage_result/.src_path."${LOCAL_MODULE}"

log_info ${PYTEST_CACHE}
log_info ${COVERAGE_RESULT_PATH}
log_info ${COVERAGE_FILE}
log_info ${SRC_PATH_FILE}

if [ "${RUN_ENV_FILE}" != "" ]; then
    if [ -f "${RUN_ENV_FILE}" ]; then
        log_info "source ${RUN_ENV_FILE} ${RUN_ENV_PARAMS}"
        # ${RUN_ENV_PARAMS} 不可以加引号，因为它可能有多个参数
        source "${RUN_ENV_FILE}" ${RUN_ENV_PARAMS}
    else
        log_info "WARNING: RUN_ENV_FILE ${RUN_ENV_FILE} does not exist, skip source."
    fi
fi

#python_path还需要配上系统自带的库路径
export PYTHONPATH="${EXPORT_PYTHONPATH}":"${PYTHONPATH}"

#--------------------------------------开始执行测试用例---------------------------------------

mkdir -p "$COVERAGE_RESULT_PATH"

#执行用例，并生成覆盖率文件
if [ "$LLT_RUN_MOD" = "single" ]; then
    if [ "${COVERAGERC_PATH}" != "" ]; then
        log_info "${PYTHON_VERSION} -m coverage run --source=${SRC_PATH} --rcfile=${COVERAGERC_PATH} -m pytest ${TEST_PATH} -o cache_dir=${PYTEST_CACHE}"
        COVERAGE_FILE="$COVERAGE_FILE" timeout -s SIGKILL "${LLT_KILL_TIME}"s "${PYTHON_VERSION}" -m coverage run --source="${SRC_PATH}" --rcfile="${COVERAGERC_PATH}" -m pytest -s "${TEST_PATH}" -o cache_dir="${PYTEST_CACHE}" --durations=20
    else
        log_info "${PYTHON_VERSION} -m coverage run --source=${SRC_PATH} -m pytest ${TEST_PATH} -o cache_dir=${PYTEST_CACHE}"
        COVERAGE_FILE="$COVERAGE_FILE" timeout -s SIGKILL "${LLT_KILL_TIME}"s "${PYTHON_VERSION}" -m coverage run --source="${SRC_PATH}" -m pytest -s  "${TEST_PATH}" -o cache_dir="${PYTEST_CACHE}" --durations=20
    fi
else
    pytest_param="-o cache_dir=${PYTEST_CACHE} --cov=${SRC_PATH} --durations=20 -n ${TASK_NUM} --dist loadfile"
    if [ "${COVERAGERC_PATH}" != "" ]; then
        pytest_param="$pytest_param --cov-config=${COVERAGERC_PATH}"
    fi
    log_info "COVERAGE_FILE=$COVERAGE_FILE timeout -s SIGKILL ${LLT_KILL_TIME}s ${PYTHON_VERSION} -m pytest  ${TEST_PATH}  ${pytest_param}"
    COVERAGE_FILE="$COVERAGE_FILE" timeout -s SIGKILL "${LLT_KILL_TIME}"s  "${PYTHON_VERSION}" -m pytest -s "${TEST_PATH}"  ${pytest_param}
fi



res=$?
if [ "$res" != 0 ];then
    log_error "ERROR: exec pytest failed"
    exit 1
else
    log_info "INFO: exec pytest success"
fi
log_info "llt run time consuming (module=${LOCAL_MODULE}):$SECONDS seconds"
echo "${SRC_PATH}" > "${SRC_PATH_FILE}"

#----------------------------------------end------------------------------------------------
