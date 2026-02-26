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

CURRENT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))

export HI_PYTHON=python3.11

WORK_DIR=$(cd ${CURRENT_DIR}/../..; pwd)
CCACHE_LOG_DIR=${WORK_DIR}/output/ccache

[ ! -d "${CCACHE_LOG_DIR}" ] && mkdir -p "${CCACHE_LOG_DIR}"
export CCACHE_LOGFILE="${CCACHE_LOG_DIR}/ccache_$$.log"
export CCACHE_STATSLOG="${CCACHE_LOG_DIR}/stat_ccache.log"
export CCACHE_COMPILERCHECK=content
export CCACHE_SLOPPINESS=include_file_ctime,include_file_mtime,time_macros
export CCACHE_BASEDIR=${WORK_DIR}
export CCACHE_MAXSIZE=200GB
# 云龙构建场景设置远端缓存
[ -n "${CLOUD_BUILD_RECORD_ID}" ] && export CCACHE_SECONDARY_STORAGE=redis://cloud-cache.turing.huawei.com:6381
export CCACHE_NOINODECACHE=true
export CCACHE_NOHASHDIR=true
export CCACHE_UMASK=002
export LANG=en_US.UTF-8
export LC_ALL=C

if [ "z$enable_ccache_debug" == "ztrue" ] ;then
    export CCACHE_DEBUG=true
    export CCACHE_DEBUGDIR="${CCACHE_LOG_DIR}/debug"
fi

# 设置set_ccache_env时只设置上述环境变量，不执行实际编译
if [ -z "${set_ccache_env}" ]; then

    # -Orecurse 选项防止在并发编译时日志打印串行，但是在centos系统上的make不支持该选项
    os_type=$(lsb_release -i | sed 's@Distributor ID:\t@@g')
    if [ "${os_type}" != "CentOS" ]; then
        extra_options="-- -Orecurse"
    fi

    if [ "$1" == "default" ]; then
        cmake --build . ${extra_options}
    else
        if [ "${enable_tbuild}" == "true" ]; then
            # 嵌套tbuildlite场景，superbuild调用tbuildlite执行，ExternalProject也调用tbuildlite执行，
            # 不同层级的构建数据会生成在同一个buildId上，不利于分析定位。
            # 清空CLOUD_BUILD_RECORD_ID环境变量，使ExternalProject生成一个新的buildId。
            unset CLOUD_BUILD_RECORD_ID
            tbuildlite make $1
        else
            if [ "${enable_custom}" == "false" ] || [ -z "${TARGETS}" ]; then
                cmake --build . --target $1 ${extra_options}
            else
                cmake --build . --target ${TARGETS}
            fi
        fi
    fi
fi
