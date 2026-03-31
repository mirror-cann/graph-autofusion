#!/bin/bash
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------

set -e

BASEPATH=$(cd "$(dirname $0)"; pwd)
BUILD_PATH="${BASEPATH}/build"
OUTPUT_PATH="${BASEPATH}/build_out"
CPU_NUM=$(($(cat /proc/cpuinfo | grep "^processor" | wc -l)))
THREAD_NUM=${CPU_NUM}
ASCEND_INSTALL_PATH="${ASCEND_HOME_PATH}"
BUILD_TYPE="Release"
CUSTOM_OPTION="-DCMAKE_INSTALL_PREFIX=${BUILD_PATH} -DASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH}"
# Detect Python command to use
if [ -n "$VIRTUAL_ENV" ]; then
  PYTHON_CMD="python3"
elif [ -f "${BASEPATH}/venv/bin/python" ]; then
  PYTHON_CMD="${BASEPATH}/venv/bin/python"
else
  PYTHON_CMD="python3"
fi

# print usage message
usage() {
  echo "Usage:"
  echo "  sh build.sh [-h|--help] [--pkg] [--cpp_utest] [-u|--ut] [-s|--st] [-c|--coverage] [-j]"
  echo "              [--output_path=<PATH>] [--build-type=<TYPE>]"
  echo ""
  echo "Options:"
  echo "    -h, --help            Print usage"
  echo "    --pkg                 Build run package"
  echo "    -j                    Compile thread nums, default is 16, eg: -j 8"
  echo "    --cpp_utest           Run cpp unit test"
  echo "        --test_case=NAME  Run specific test case (e.g. --test_case=SkScopeSplitTest.*)"
  echo "    -u, --ut              Run all unit test"
  echo "        =superkernel      Run superkernel unit test"
  echo "    -s, --st              Run all system test"
  echo "        =superkernel      Run superkernel system test"
  echo "    -c, --coverage        Run tests with coverage report generation"
  echo "    --output_path=<PATH>"
  echo "                          Set output path, where the run package will be generated, default ./build_out"
  echo "    --run_example         Run all examples"
  echo "        =superkernel      Run superkernel examples"
  echo "    --build-type=<TYPE>   Set build type: Debug, Release(default: Release)"
  echo ""
}

check_param_j() {
  local thread_num=$1
  if [[ -z "$thread_num" ]]; then
    echo "ERROR: -j must specify a positive integer (e.g. -j8, -j=16)"
    usage
    exit 1
  fi
  if [[ ! "$thread_num" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: -j only support positive integers (0/negative/non-number are not allowed)"
    usage
    exit 1
  fi

  if [[ "$thread_num" -gt "$CPU_NUM" ]]; then
    thread_num=$CPU_NUM
  fi
  echo "$thread_num"
}

# parse and set options
checkopts() {
  if [ $# -eq 0 ]; then
    echo "ERROR: 'build.sh' has no options available, please select at least one option!"
    usage
    exit 1
  fi

  ENABLE_BUILD_PACKAGE="off"
  ENABLE_UT="off"
  ENABLE_ST="off"
  ENABLE_COVERAGE="off"
  ENABLE_SUPERKERNEL_UT="off"
  ENABLE_SUPERKERNEL_ST="off"
  ENABLE_RUN_EXAMPLE="off"
  ENABLE_SUPERKERNEL_RUN_EXAMPLE="off"

  # Process the options - 添加了 build-type 选项
  parsed_args=$(getopt -a -o j:hu::s::c -l help,pkg,cpp_utest,test_case:,run_example::,ut::,st::,coverage,output_path:,build-type: -- "$@") || {
    usage
    exit 1
  }

  eval set -- "$parsed_args"

  while true; do
    case "$1" in
      -h | --help)
        usage
        exit 0
        ;;
      --pkg)
        ENABLE_BUILD_PACKAGE="on"
        shift
        ;;
      -j)
        local raw_thread_num="$2"
        raw_thread_num="${raw_thread_num#=}"
        THREAD_NUM=$(check_param_j "$raw_thread_num")
        shift 2
        ;;
      -u | --ut)
        ENABLE_UT="on"
        case "$2" in
          "")
            ENABLE_SUPERKERNEL_UT="on"
            shift 2
            ;;
          "superkernel")
            ENABLE_SUPERKERNEL_UT="on"
            shift 2
            ;;
          *)
            usage
            exit 1
        esac
        ;;
      -s | --st)
        ENABLE_ST="on"
        case "$2" in
          "")
            ENABLE_SUPERKERNEL_ST="on"
            shift 2
            ;;
          "superkernel")
            ENABLE_SUPERKERNEL_ST="on"
            shift 2
            ;;
          *)
            usage
            exit 1
        esac
        ;;
      -c | --coverage)
        ENABLE_COVERAGE="on"
        shift
        ;;
      --cpp_utest)
        ENABLE_CPP_UTEST="on"
        shift
        ;;
      --test_case)
        CPP_UTEST_FILTER="$2"
        shift 2
        ;;
      --run_example)
        ENABLE_RUN_EXAMPLE="on"
        case "$2" in
          "")
            ENABLE_SUPERKERNEL_RUN_EXAMPLE="on"
            shift 2
            ;;
          "superkernel")
            ENABLE_SUPERKERNEL_RUN_EXAMPLE="on"
            shift 2
            ;;
          *)
            usage
            exit 1
        esac
        ;;
      --output_path)
        OUTPUT_PATH="$(realpath $2)"
        shift 2
        ;;
      --build-type)  # 新增的 build-type 选项
        BUILD_TYPE="$2"
        # 验证 BUILD_TYPE 是否有效
        if [[ ! "$BUILD_TYPE" =~ ^(Debug|Release)$ ]]; then
          echo "ERROR: Invalid build type: $BUILD_TYPE"
          echo "       Valid types: Debug, Release"
          exit 1
        fi
        shift 2
        ;;
      --)
        shift
        break
        ;;
      *)
        echo "Undefined option: $1"
        usage
        exit 1
        ;;
    esac
  done
  
}

function cmake_config()
{
  local extra_option="$1"
  log "Info: cmake config ${CUSTOM_OPTION} ${extra_option} ."
  cmake .. ${CUSTOM_OPTION} ${extra_option}
}

function build()
{
  local target="$1"
  cmake --build . --target ${target} -j ${THREAD_NUM}
}

function build_test() {
  cmake_config
  build all
}

function build_package_inner(){
  cmake_config
  build package
}

function build_test_cpp_utest() {
  echo "---------------- Start run cpp utest ----------------"
  
  CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_CPP_UTEST=ON -DENABLE_GCOV=ON"
  
  # Pass gtest filter to cmake if specified
  if [ -n "${CPP_UTEST_FILTER}" ]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DGTEST_FILTER=--gtest_filter=${CPP_UTEST_FILTER}"
  fi
  
  mkdir -pv ${BUILD_PATH} &&
  cd ${BUILD_PATH} &&
  build_test
  echo "Build run cpp utest success!"
}

build_package() {
  echo "---------------- Start build run package ----------------"
  mkdir -pv ${BUILD_PATH} &&
  cd ${BUILD_PATH} &&
  build_package_inner &&
  mkdir -pv ${OUTPUT_PATH} &&
  cp _CPack_Packages/makeself_staging/cann-graph-autofusion*.run ${OUTPUT_PATH} &&
  output_run_path=`ls -1 ${OUTPUT_PATH}/cann-graph-autofusion*.run 2>/dev/null` &&
  echo "Buid run package success!" &&
  echo "package: ${output_run_path}"
}

superkernel_run_example() {
  echo "---------------- Start running examples ----------------"
  ${PYTHON_CMD} ${BASEPATH}/super_kernel/examples/super_kernel_base/superkernel_scope.py
  ${PYTHON_CMD} ${BASEPATH}/super_kernel/examples/super_kernel_profiling/superkernel_compare.py
  ${PYTHON_CMD} ${BASEPATH}/super_kernel/examples/super_kernel_runtime_ascendc_only/superkernel_runtime_ascendc_basic.py
  echo "Run all examples success"
}

superkernel_ut() {
  echo "---------------- Start UT ----------------"
  cd ${BASEPATH}/super_kernel &&
  ${PYTHON_CMD} -m pip install -e . --force-reinstall --no-deps -q &&
  if [ "X$ENABLE_COVERAGE" == "Xon" ]; then
    ${PYTHON_CMD} -m pytest tests/ut -m ut \
                                     --cov-config=scripts/sk_ut_cfg.toml \
                                     --cov=superkernel \
                                     --cov-report=term-missing \
                                     --cov-report=html \
                                     --cov-report=xml
  else
    ${PYTHON_CMD} -m pytest tests/ut -m ut
  fi
}

superkernel_st() {
  echo "---------------- Start ST ----------------"
  cd ${BASEPATH}/super_kernel &&
  ${PYTHON_CMD} -m pip install -e . --force-reinstall --no-deps -q &&
  if [ "X$ENABLE_COVERAGE" == "Xon" ]; then
    ${PYTHON_CMD} -m pytest tests/st -m st --cov-config=scripts/sk_st_cfg.toml \
                                     --cov=superkernel \
                                     --cov-report=term-missing \
                                     --cov-report=html \
                                     --cov-report=xml
  else
    ${PYTHON_CMD} -m pytest tests/st -m st
  fi
}

main() {
  checkopts "$@"

  # Clean coverage data if coverage is enabled
  if [ "X$ENABLE_COVERAGE" == "Xon" ]; then
    # All UT and ST is enabled when only '-c' without '-u' or '-s'
    if [ "X$ENABLE_UT" != "Xon" ] && [ "X$ENABLE_ST" != "Xon" ]; then
      ENABLE_UT="on"
      ENABLE_ST="on"
      ENABLE_SUPERKERNEL_UT="on"
      ENABLE_SUPERKERNEL_ST="on"
    fi
    echo "---------------- Clean Coverage Data ----------------"
    cd ${BASEPATH}/super_kernel && ${PYTHON_CMD} -m coverage erase
  fi

  if [ "X$ENABLE_BUILD_PACKAGE" == "Xon" ]; then
    build_package || { echo "Build run package failed."; exit 1; }
  fi

  if [ "X$ENABLE_CPP_UTEST" == "Xon" ]; then
    build_test_cpp_utest || { echo "Build and run cpp part of unit tests failed."; exit 1; }
  fi

  if [ "X$ENABLE_UT" == "Xon" ]; then
    if [ "X$ENABLE_SUPERKERNEL_UT" == "Xon" ]; then
      superkernel_ut || { echo "Run superkernel UT failed."; exit 1; }
    fi
  fi

  if [ "X$ENABLE_ST" == "Xon" ]; then
    if [ "X$ENABLE_SUPERKERNEL_ST" == "Xon" ]; then
      superkernel_st || { echo "Run superkernel ST failed."; exit 1; }
    fi
  fi

  if [ "X$ENABLE_RUN_EXAMPLE" == "Xon" ]; then
    if [ "X$ENABLE_SUPERKERNEL_RUN_EXAMPLE" == "Xon" ]; then
      superkernel_run_example || { echo "Run examples failed."; exit 1; }
    fi
  fi
}

main "$@"
