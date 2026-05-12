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

username="$(id -un)"
usergroup="$(id -gn)"
is_quiet=n
pylocal=n
in_install_for_all=n
docker_root=""
install_autofuse="n"
curpath=$(dirname $(readlink -f "$0"))
common_func_path="${curpath}/common_func.inc"
pkg_version_path="${curpath}/../version.info"

. "${common_func_path}"

if [ $1 ]; then
    input_install_dir="$2"
    is_quiet="$4"
    in_install_for_all="$8"
    common_parse_type="$3"
    pylocal="$5"
    docker_root="$7"
    pkg_version_dir="$9"
    install_autofuse="${10}"
fi

if [ "x${docker_root}" != "x" ]; then
    common_parse_dir="${docker_root}${input_install_dir}"
else
    common_parse_dir="${input_install_dir}"
fi

get_version "pkg_version" "$pkg_version_path"
is_multi_version_pkg "pkg_is_multi_version" "$pkg_version_path"
if [ "$pkg_is_multi_version" = "true" ]; then
    common_parse_dir="$common_parse_dir/$pkg_version_dir"
fi

if [ $(id -u) -ne 0 ]; then
    log_dir="${HOME}/var/log/ascend_seclog"
else
    log_dir="/var/log/ascend_seclog"
fi
logfile="${log_dir}/ascend_install.log"

install_info="${common_parse_dir}/share/info/graph_autofusion/ascend_install.info"

# 写日志
log() {
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    local log_type="$1"
    shift
    if [ "$log_type" = "INFO" -o "$log_type" = "WARNING" -o "$log_type" = "ERROR" ]; then
        echo -e "[graph_autofusion] [$cur_date] [$log_type]: $*"
    else
        echo "[graph_autofusion] [$cur_date] [$log_type]: $*" 1> /dev/null
    fi
    echo "[graph_autofusion] [$cur_date] [$log_type]: $*" >> "$logfile"
}

# 静默模式日志打印
new_echo() {
    local log_type="$1"
    local log_msg="$2"
    if [ "${is_quiet}" = "n" ]; then
        echo "${log_type}" "${log_msg}" 1> /dev/null
    fi
}

echo_progress() {
    new_echo "INFO" "upgrade upgradePercentage:$1%"
    log "INFO" "upgrade upgradePercentage:$1%"
}

create_filtered_filelist() {
    local src_filelist="$1"
    local dst_filelist="$2"
    if [ "$install_autofuse" = "y" ]; then
        cp "$src_filelist" "$dst_filelist"
    else
        python3 - "$src_filelist" "$dst_filelist" <<'PY'
import sys
from pathlib import Path
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
lines = src.read_text().splitlines()
autofuse_shared_libs = (
    'libaihac_codegen.so',
    'libaihac_ir.so',
    'libaihac_ir_register.so',
    'libgraph_af.so',
    'libgraph_base_af.so',
    'libaihac_symbolizer_af.so',
)
filtered = [
    line for line in lines
    if 'python/site-packages/autofuse' not in line
    and not any(lib in line for lib in autofuse_shared_libs)
]
dst.write_text("\n".join(filtered) + "\n")
PY
    fi
}

update_installed_filelist() {
    local src_filelist="$1"
    local installed_filelist="${common_parse_dir}/share/info/graph_autofusion/script/filelist.csv"
    if [ ! -f "$src_filelist" ]; then
        log "ERROR" "ERR_NO:0x0085;ERR_DES:source filelist $src_filelist is not exist."
        return 1
    fi
    if [ ! -d "$(dirname "$installed_filelist")" ]; then
        log "ERROR" "ERR_NO:0x0085;ERR_DES:installed script directory is not exist."
        return 1
    fi

    chmod u+w "$installed_filelist" 2> /dev/null
    cp "$src_filelist" "$installed_filelist"
    if [ $? -ne 0 ]; then
        log "ERROR" "ERR_NO:0x0085;ERR_DES:failed to update installed filelist."
        return 1
    fi

    return 0
}

create_latest_linux_softlink() {
    if [ "$pkg_is_multi_version" = "true" ] && [ "$hetero_arch" = "y" ]; then
        local linux_path="$(realpath $common_parse_dir/..)"
        local arch_path="$(basename $linux_path)"
        local latest_path="$(realpath $linux_path/../..)/cann"
        if [ -d "$latest_path" ]; then
            if [ ! -e "$latest_path/$arch_path" ] || [ -L "$latest_path/$arch_path" ]; then
                ln -srfn "$linux_path" "$latest_path"
            fi
        fi
    fi
}

##########################################################################
log "INFO" "step into run_graph_autofusion_upgrade.sh ......"
log "INFO" "upgrade target dir $common_parse_dir, type $common_parse_type."

if [ ! -d "$common_parse_dir" ]; then
    log "ERROR" "ERR_NO:0x0001;ERR_DES:path $common_parse_dir is not exist."
    exit 1
fi

new_upgrade() {
    echo_progress 10

    local filtered_filelist
    filtered_filelist="$(mktemp /tmp/graph_autofusion_filelist.upgrade.XXXXXX.csv)"
    if [ $? -ne 0 ]; then
        log "ERROR" "ERR_NO:0x0085;ERR_DES:failed to create temporary filelist."
        return 1
    fi
    create_filtered_filelist "$curpath/filelist.csv" "$filtered_filelist"
    if [ $? -ne 0 ]; then
        rm -f "$filtered_filelist"
        log "ERROR" "ERR_NO:0x0085;ERR_DES:failed to prepare install filelist."
        return 1
    fi

    # 执行安装
    custom_options="--custom-options=--common-parse-dir=$common_parse_dir,--logfile=$logfile,--stage=upgrade,--quiet=$is_quiet,--pylocal=$pylocal,--autofuse=$install_autofuse"
    sh "$curpath/install_common_parser.sh" --package="graph_autofusion" --install --username="$username" --usergroup="$usergroup" --set-cann-uninstall --upgrade \
        --version=$pkg_version --version-dir=$pkg_version_dir --use-share-info \
        $in_install_for_all --docker-root="$docker_root" \
        $custom_options "$common_parse_type" "$input_install_dir" "$filtered_filelist"
    local ret=$?
    if [ $ret -ne 0 ]; then
        rm -f "$filtered_filelist"
        log "ERROR" "ERR_NO:0x0085;ERR_DES:failed to install package."
        return 1
    fi
    update_installed_filelist "$filtered_filelist"
    if [ $? -ne 0 ]; then
        rm -f "$filtered_filelist"
        return 1
    fi
    rm -f "$filtered_filelist"

    create_latest_linux_softlink
    return 0
}

new_upgrade
if [ $? -ne 0 ]; then
    exit 1
fi

echo_progress 100
exit 0
