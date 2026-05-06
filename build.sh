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

SUPPORTED_MODULES=("superkernel" "autofuse_framework" "autofuse_ascendc_api" "autofuse_e2e")
declare -A MODULE_ACTION_HANDLERS=(
  ["superkernel:py_ut"]="superkernel_py_ut"
  ["superkernel:cpp_ut"]="superkernel_cpp_ut"
  ["superkernel:py_st"]="superkernel_py_st"
  ["superkernel:py_run_example"]="superkernel_py_run_example"
  # Reserved module-level test entrypoints. Fill the value with a handler name when implemented.
  ["autofuse_framework:ut"]="reserved_module_test_suite"
  ["autofuse_framework:st"]="reserved_module_test_suite"
  ["autofuse_ascendc_api:ut"]="reserved_module_test_suite"
  ["autofuse_ascendc_api:st"]="reserved_module_test_suite"
  ["autofuse_e2e:st"]="reserved_module_test_suite"
)

# print usage message
function log()
{
  local commoninfo="$1"
  local timestr
  timestr=$(date '+%Y-%m-%d %H:%M:%S.%N ' | cut -b 1-23)
  echo "[${timestr}]${commoninfo}"
}

usage() {
  echo "Usage:"
  echo "  sh build.sh [-h|--help] [--pkg] [-u|--ut] [-s|--st] [--impl=<py|cpp|all>]"
  echo "              [--module=<name>] [-c|--coverage] [-j]"
  echo "              [--output_path=<PATH>] [--cann_3rd_lib_path=<PATH>] [--build-type=<TYPE>] [--no-autofuse]"
  echo ""
  echo "Options:"
  echo "    -h, --help            Print usage"
  echo "    --pkg                 Build run package"
  echo "    --no-autofuse         Skip autofuse backend build/package artifacts"
  echo "    -j                    Compile thread nums, default is 16, eg: -j 8"
  echo "    -u, --ut              Run unit tests for supported implementations"
  echo "    -s, --st              Run system tests for supported implementations"
  echo "    --impl=<py|cpp|all>   Select implementation under -u/-s (default: all supported)"
  echo "    --module=<name>       Select module, default: all supported modules"
  echo "    -c, --coverage        Run tests with coverage report generation"
  echo "                          Without explicit test selection, run supported tests for the selected module"
  echo "    --output_path=<PATH>"
  echo "                          Set output path, where the run package will be generated, default ./build_out"
  echo "    --run_example         Run examples for the selected module"
  echo "    --cann_3rd_lib_path=<PATH>"
  echo "                          Set third_party package install path, default ./output/third_party"
  echo "                          (Third_party package will cost a little time during the first compilation,"
  echo "                          it will skip compilation to save time during subsequent builds)"
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

is_supported_module() {
  local requested_module="$1"
  local module
  for module in "${SUPPORTED_MODULES[@]}"; do
    if [[ "${module}" == "${requested_module}" ]]; then
      return 0
    fi
  done
  return 1
}

has_module_action_entry() {
  local action="$1"
  local module="$2"
  local target_module

  if [[ "${module}" == "all" ]]; then
    for target_module in "${SUPPORTED_MODULES[@]}"; do
      if [[ -v MODULE_ACTION_HANDLERS["${target_module}:${action}"] ]]; then
        return 0
      fi
    done
    return 1
  fi

  [[ -v MODULE_ACTION_HANDLERS["${module}:${action}"] ]]
}

normalize_test_selection() {
  EXEC_ACTIONS=()

  if [[ -z "${TARGET_MODULE}" ]]; then
    TARGET_MODULE="all"
  fi

  if [[ "${ENABLE_COVERAGE}" == "on" && "${ENABLE_UT}" != "on" && "${ENABLE_ST}" != "on" ]]; then
    ENABLE_UT="on"
    ENABLE_ST="on"
  fi

  local selected_modules=()
  if [[ "${TARGET_MODULE}" == "all" ]]; then
    selected_modules=("${SUPPORTED_MODULES[@]}")
  else
    selected_modules=("${TARGET_MODULE}")
  fi

  for suite in ut st; do
    local enabled_flag="ENABLE_${suite^^}"
    if [[ "${!enabled_flag}" != "on" ]]; then
      continue
    fi

    local impls=()
    case "${TEST_IMPL_MODE}" in
      py) impls=(py) ;;
      cpp) impls=(cpp) ;;
      all) impls=(py cpp) ;;
    esac

    for module in "${selected_modules[@]}"; do
      if has_module_action_entry "${suite}" "${module}"; then
        EXEC_ACTIONS+=("${module}:${suite}")
        continue
      fi

      for impl in "${impls[@]}"; do
        local action="${impl}_${suite}"
        if ! has_module_action_entry "${action}" "${module}"; then
          continue
        fi

        EXEC_ACTIONS+=("${module}:${action}")
      done
    done
  done
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
  ENABLE_RUN_EXAMPLE="off"
  TEST_IMPL_MODE="all"
  TARGET_MODULE="all"
  CANN_3RD_LIB_PATH="$BASEPATH/output/third_party"
  ENABLE_AUTOFUSE="on"

  # Process the options - 添加了 build-type 选项
  parsed_args=$(getopt -a -o j:husc -l help,pkg,autofuse,no-autofuse,impl:,module:,test_case:,run_example,ut,st,coverage,output_path:,cann_3rd_lib_path:,build-type: -- "$@") || {
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
      --autofuse)
        ENABLE_AUTOFUSE="on"
        shift
        ;;
      --no-autofuse)
        ENABLE_AUTOFUSE="off"
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
        shift
        ;;
      -s | --st)
        ENABLE_ST="on"
        shift
        ;;
      -c | --coverage)
        ENABLE_COVERAGE="on"
        shift
        ;;
      --impl)
        if [[ ! "$2" =~ ^(py|cpp|all)$ ]]; then
          echo "ERROR: Invalid implementation type: $2"
          echo "       Valid values: py, cpp, all"
          exit 1
        fi
        TEST_IMPL_MODE="$2"
        shift 2
        ;;
      --module)
        if [[ "$2" != "all" ]] && ! is_supported_module "$2"; then
          echo "ERROR: Unsupported module '$2'. Currently supported values: all, ${SUPPORTED_MODULES[*]}."
          exit 1
        fi
        if [[ "${TARGET_MODULE}" != "all" && "${TARGET_MODULE}" != "$2" ]]; then
          echo "ERROR: Conflicting module selection: '${TARGET_MODULE}' vs '$2'."
          exit 1
        fi
        TARGET_MODULE="$2"
        shift 2
        ;;
      --test_case)
        CPP_UTEST_FILTER="$2"
        shift 2
        ;;
      --run_example)
        ENABLE_RUN_EXAMPLE="on"
        shift
        ;;
      --output_path)
        OUTPUT_PATH="$(realpath $2)"
        shift 2
        ;;
      --cann_3rd_lib_path)
        CANN_3RD_LIB_PATH="$(realpath $2)"
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

  normalize_test_selection

  if [[ "${ENABLE_RUN_EXAMPLE}" == "on" ]]; then
    local selected_modules=()
    if [[ "${TARGET_MODULE}" == "all" ]]; then
      selected_modules=("${SUPPORTED_MODULES[@]}")
    else
      selected_modules=("${TARGET_MODULE}")
    fi

    local module
    for module in "${selected_modules[@]}"; do
      if [[ -z "${MODULE_ACTION_HANDLERS["${module}:py_run_example"]}" ]]; then
        continue
      fi
      EXEC_ACTIONS+=("${module}:py_run_example")
    done
  fi

  if [ "X$ENABLE_BUILD_PACKAGE" != "Xon" ] && [ ${#EXEC_ACTIONS[@]} -eq 0 ]; then
    echo "ERROR: No supported actions for the requested selection."
    echo "       module=${TARGET_MODULE}, impl=${TEST_IMPL_MODE}, ut=${ENABLE_UT}, st=${ENABLE_ST}, coverage=${ENABLE_COVERAGE}, run_example=${ENABLE_RUN_EXAMPLE}"
    exit 1
  fi
}

function cmake_config()
{
  local extra_option="$1"
  local cmake_option="${CUSTOM_OPTION} -DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
  if [ "X$ENABLE_AUTOFUSE" == "Xon" ]; then
    extra_option="${extra_option} -DBUILD_AUTOFUSE=ON"
    if [ -n "${CANN_3RD_LIB_PATH}" ]; then
      extra_option="${extra_option} -DASCEND_CANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}"
    fi
  fi
  if [ "X$ENABLE_AUTOFUSE" == "Xoff" ]; then
    extra_option="${extra_option} -DBUILD_AUTOFUSE=OFF"
  fi
  echo "Info: cmake config ${cmake_option} ${extra_option} ."
  cmake .. ${cmake_option} ${extra_option}
}

function build()
{
  local target="$1"
  cmake --build . --target ${target} -j ${THREAD_NUM}
}

function prepare_autofuse_third_party() {
  local third_party_dir="${CANN_3RD_LIB_PATH}"
  local script_path="${BASEPATH}/autofuse/build_third_party.sh"
  if [ "X$ENABLE_AUTOFUSE" != "Xon" ]; then
    return
  fi
  if [ -z "${third_party_dir}" ]; then
    third_party_dir="${BASEPATH}/autofuse/output/third_party"
  fi
  if [ ! -f "${script_path}" ]; then
    echo "Build autofuse failed: missing ${script_path}"
    exit 1
  fi

  echo "---------------- Prepare autofuse third_party in ${third_party_dir} ----------------"
  bash "${script_path}" "${third_party_dir}" "${THREAD_NUM}" "" "" "0" || { echo "Prepare autofuse third_party failed."; exit 1; }
}

clean_coverage_artifacts() {
  if [ "X$ENABLE_COVERAGE" != "Xon" ]; then
    return
  fi

  local action_entry
  local has_python_tests="off"
  local has_cpp_tests="off"
  for action_entry in "${EXEC_ACTIONS[@]}"; do
    case "${action_entry}" in
      *:py_ut|*:py_st)
        has_python_tests="on"
        ;;
      *:cpp_ut|*:cpp_st)
        has_cpp_tests="on"
        ;;
    esac
  done

  if [ "X$has_python_tests" == "Xon" ]; then
    echo "---------------- Clean Python Coverage Data ----------------"
    cd ${BASEPATH}/super_kernel && ${PYTHON_CMD} -m coverage erase
  fi

  if [ "X$has_cpp_tests" == "Xon" ]; then
    echo "---------------- Clean AOT C++ Coverage Artifacts ----------------"
    rm -rf ${BASEPATH}/super_kernel/coverage/cpp_ut
    find ${BUILD_PATH} -name "*.gcda" -delete 2>/dev/null || true
  fi
}

function build_package_inner(){
  cmake_config "-DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}"
  build "all package"
}

build_package() {
  echo "---------------- Start build run package ----------------"
  mkdir -pv ${BUILD_PATH} &&
  cd ${BUILD_PATH} &&
  build_package_inner &&
  mkdir -pv ${OUTPUT_PATH} &&
  cp _CPack_Packages/makeself_staging/cann-graph-autofusion*.run ${OUTPUT_PATH} &&
  output_run_path=`ls -1 ${OUTPUT_PATH}/cann-graph-autofusion*.run 2>/dev/null` &&
  echo "Build run package success!" &&
  echo "package: ${output_run_path}"
}

superkernel_py_run_example() {
  echo "---------------- Start running examples ----------------"
  ${PYTHON_CMD} ${BASEPATH}/super_kernel/examples/super_kernel_base/superkernel_scope.py &&
  ${PYTHON_CMD} ${BASEPATH}/super_kernel/examples/super_kernel_profiling/superkernel_compare.py &&
  ${PYTHON_CMD} ${BASEPATH}/super_kernel/examples/super_kernel_runtime_ascendc_only/superkernel_runtime_ascendc_basic.py &&
  echo "Run all examples success"
}

superkernel_py_ut() {
  echo "---------------- Start UT ----------------"
  cd ${BASEPATH}/super_kernel &&
  ${PYTHON_CMD} -m pip install --upgrade pip -q &&
  ${PYTHON_CMD} -m pip install -e .[dev] --force-reinstall -q &&
  if [ "X$ENABLE_COVERAGE" == "Xon" ]; then
    ${PYTHON_CMD} -m pytest tests/ut -m ut -n auto \
                                     --cov-config=scripts/sk_ut_cfg.toml \
                                     --cov=superkernel \
                                     --cov-report=term-missing \
                                     --cov-report=html \
                                     --cov-report=xml
  else
    ${PYTHON_CMD} -m pytest tests/ut -m ut -n auto
  fi
}

superkernel_py_st() {
  echo "---------------- Start ST ----------------"
  cd ${BASEPATH}/super_kernel &&
  ${PYTHON_CMD} -m pip install --upgrade pip -q &&
  ${PYTHON_CMD} -m pip install -e .[dev] --force-reinstall -q &&
  if [ "X$ENABLE_COVERAGE" == "Xon" ]; then
    ${PYTHON_CMD} -m pytest tests/st -m st -n auto --cov-config=scripts/sk_st_cfg.toml \
                                     --cov=superkernel \
                                     --cov-report=term-missing \
                                     --cov-report=html \
                                     --cov-report=xml
  else
    ${PYTHON_CMD} -m pytest tests/st -m st -n auto
  fi
}

function superkernel_cpp_ut() {
  echo "---------------- Start run cpp utest ----------------"

  CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_CPP_UTEST=ON"

  if [ "X$ENABLE_COVERAGE" == "Xon" ]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_GCOV=ON"
  fi

  # Pass gtest filter to cmake if specified
  if [ -n "${CPP_UTEST_FILTER}" ]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DGTEST_FILTER=--gtest_filter=${CPP_UTEST_FILTER}"
  else
    CUSTOM_OPTION="${CUSTOM_OPTION} -DGTEST_FILTER="
  fi

  mkdir -pv ${BUILD_PATH} &&
  cd ${BUILD_PATH} &&
  cmake_config &&
  if [ "X$ENABLE_COVERAGE" == "Xon" ]; then
    build clean &&
    build collect_coverage_data
  else
    build run_super_kernel_aot_utest
  fi &&
  echo "Build run cpp utest success!"
}

reserved_module_test_suite() {
  local module="$1"
  local suite="$2"
  echo "更多功能支持中，已识别预留测试入口，暂未接入执行: module=${module}, suite=${suite}."
  if [[ "${TEST_IMPL_MODE}" != "all" ]]; then
    echo "该模块暂不区分 py/cpp，忽略 --impl=${TEST_IMPL_MODE}."
  fi
}

main() {
  checkopts "$@"

  clean_coverage_artifacts

  if [ "X$ENABLE_AUTOFUSE" == "Xon" ]; then
    echo "---------------- Start build autofuse ----------------"
    prepare_autofuse_third_party
    mkdir -pv ${BUILD_PATH} &&
    cd ${BUILD_PATH} &&
    cmake_config "-DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}" &&
    build all || { echo "Build autofuse failed."; exit 1; }
    echo "Build autofuse success!"
  fi

  if [ "X$ENABLE_BUILD_PACKAGE" == "Xon" ]; then
    build_package || { echo "Build run package failed."; exit 1; }
  fi

  local action_entry
  local module
  local action
  local handler
  for action_entry in "${EXEC_ACTIONS[@]}"; do
    module="${action_entry%%:*}"
    action="${action_entry#*:}"
    handler="${MODULE_ACTION_HANDLERS["${module}:${action}"]}"

    ${handler} "${module}" "${action}" || {
      echo "Run ${module} ${action} failed."
      exit 1
    }
  done
}

main "$@"
