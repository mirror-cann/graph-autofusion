#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ctypes
import os
import re
import sys
import shutil
import tempfile
import argparse
import subprocess
import platform
from typing import List
PYF_PATH = os.path.dirname(os.path.realpath(__file__))
ASCEND_PATH = os.path.join(PYF_PATH, "..", "..", "..")
machine = platform.machine()
if not os.path.exists(ASCEND_PATH):
    ASCEND_PATH = os.getenv("ASCEND_HOME_PATH", ASCEND_PATH)


class CompileError(Exception):
    """Compile failed exception."""


def get_soc_type(args):
    """根据 soc_version 返回对应的类型"""
    if args.soc_version.startswith("Ascend910B"):
        return "dav-2201"
    elif args.soc_version.startswith("Ascend910_93"):
        return "dav-2201"
    elif args.soc_version.startswith("Ascend950"):
        return "dav-3510"
    else:
        raise ValueError(f"Unsupported soc_version: {args.soc_version}")


def run_compile_command(cmd: List[str], stage_name):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error_msg = f"{stage_name} compile failed with code {result.returncode}"
        if result.stderr:
            error_msg += f"\nstderr: {result.stderr}"
        raise CompileError(error_msg)
    if result.stdout:
        print(f"[{stage_name}] {result.stdout}")


def link_shared(target_file, obj_files):
    link_command = [f"{ASCEND_PATH}/tools/bisheng_compiler/bin/bisheng"]
    link_command.extend(obj_files)
    link_command.extend(["-fPIC", "--shared", "-o", target_file])
    run_compile_command(link_command, "LinkObj")
    return target_file


def compile_host_obj(args: argparse.Namespace, temp_dir):
    base_host_file = os.path.basename(args.host_files)
    soc_version = get_soc_type(args)
    host_compile_cmd = [
        f"{ASCEND_PATH}/tools/bisheng_compiler/bin/bisheng",
        "-D", "kernel_EXPORTS",
        "-I", f"{temp_dir}/host",
        "-I", f"{ASCEND_PATH}/include",
        "-I", f"{ASCEND_PATH}/pkg_inc/base",
        "-I", f"{ASCEND_PATH}/include/base",
        "-I", f"{ASCEND_PATH}/include/experiment",
        "-I", f"{ASCEND_PATH}/{machine}-linux/pkg_inc/base",
        "-I", f"{ASCEND_PATH}/{machine}-linux/include",
        "-I", f"{ASCEND_PATH}/{machine}-linux/ascendc/include/highlevel_api/tiling/platform",
        "-fPIC", f"--npu-arch={soc_version}", "-O2", "-fno-common", "-Wextra", "-Wfloat-equal", "-fvisibility=default",
        *args.compile_options.split(),
        "-D", "LOG_CPP", "-o",
        f"{temp_dir}/host/{base_host_file}.o", "-c", "-x", "asc",
        f"{temp_dir}/host/{base_host_file}"]
    run_compile_command(host_compile_cmd, "Host")
    return f"{temp_dir}/host/{base_host_file}.o"


def compile_device_obj(args: argparse.Namespace, temp_dir):
    base_device_file = os.path.basename(args.device_files)
    soc_version = get_soc_type(args)
    device_compile_cmd = [
        f"{ASCEND_PATH}/tools/bisheng_compiler/bin/bisheng",
        "-I", f"{temp_dir}/device",
        "-fPIC", "-D", "HAVE_TILING", "-D", "AUTO_FUSE_DEVICE=1", f"--npu-arch={soc_version}",
        "-o", f"{temp_dir}/device/{base_device_file}.o",
        "-c", "-x", "asc", f"{temp_dir}/device/{base_device_file}"]
    run_compile_command(device_compile_cmd, "Device")
    return f"{temp_dir}/device/{base_device_file}.o"


def build_device_so(args: argparse.Namespace, host_obj_path, temp_dir):
    device_obj_path = compile_device_obj(args, temp_dir)
    target_file = os.path.join(temp_dir, os.path.basename(args.output_file))
    obj_files = [device_obj_path]
    if host_obj_path is not None:
        obj_files.insert(0, host_obj_path)
    return link_shared(target_file, obj_files)


def clean_before_modify(temp_dir):
    src_directory = os.getcwd()
    keep_dirs = {'host', 'device'}
    for entry in os.listdir(temp_dir):
        entry_path = os.path.join(temp_dir, entry)
        if os.path.isfile(entry_path):
            os.remove(entry_path)
            print(f"delete file: {entry_path}")
        elif entry not in keep_dirs:
            shutil.rmtree(entry_path)
            print(f"delete dir: {entry_path}")
    os.chdir(src_directory)


def static_shape_kernel_proc(args: argparse.Namespace, temp_dir, tiling_repr=None):
    clean_before_modify(temp_dir)
    base_device_files = os.path.basename(args.device_files)
    kernel_file = os.path.join(temp_dir, "device", base_device_files)
    pattern = re.compile(r'^extern\s+"C"\s+__global__\s+__aicore__\s+void\s+(\w+)\s*\(([^)]*)\)\s*{')

    with open(kernel_file, 'r') as f:
        lines = f.readlines()

    result = []
    for line in lines:
        match = pattern.match(line)
        if not match:
            result.append(line)
            continue

        func_name = match.group(1)
        params_str = match.group(2).strip()
        if not params_str:
            result.append(line)
            continue

        params = [p.strip() for p in params_str.split(',')]
        if not params or params[-1] != 'AutofuseTilingData t':
            result.append(line)
            continue

        # 匹配到 AutofuseTilingData t 参数的 kernel 函数，进行静态 shape 优化：
        # 1. 将参数名 t 改为 param，避免与后续声明的 t 变量冲突
        # 2. 在函数体内插入 t 变量声明，用于存储静态 tiling 数据
        params[-1] = 'AutofuseTilingData param'
        new_params = ', '.join(params)
        new_line = f'extern "C" __global__ __aicore__ void {func_name}({new_params}) {{\n'
        result.append(new_line)
        if tiling_repr is None:
            result.append('  const AutofuseTilingData t;\n')
        else:
            result.append(f'  constexpr AutofuseTilingData t = {tiling_repr};\n')

    with open(kernel_file, 'w') as f:
        f.writelines(result)


def try_static_shape_compile(args: argparse.Namespace, temp_dir, so_path):
    if args.force_unknown:
        return False
    lib = ctypes.CDLL(so_path)
    lib.AutofuseIsStaticShape.argtypes = []
    lib.AutofuseIsStaticShape.restype = ctypes.c_bool
    if not bool(lib.AutofuseIsStaticShape()):
        return False
    print("static shape detected, recompile kernel with const tiling data")
    static_shape_kernel_proc(args, temp_dir)
    lib.GenConstTilingData.argtypes = [ctypes.c_char_p]
    lib.GenConstTilingData.restype = ctypes.c_char_p
    config_file = ctypes.c_char_p(args.config_file.encode('utf-8'))
    result = lib.GenConstTilingData(config_file)
    const_tiling_data = result.decode('utf-8')
    tiling_data = os.path.join(temp_dir, "device", "autofuse_tiling_data.h")
    tiling_data_bak = os.path.join(temp_dir, "device", "autofuse_tiling_data_bak.h")
    shutil.copy(tiling_data, tiling_data_bak)
    with open(tiling_data, "w") as file:
        file.write(const_tiling_data)
    return True


def link_host_target(args, temp_dir):
    # 处理 host 编译阶段
    host_obj_path = compile_host_obj(args, temp_dir)
    so_file = args.host_files.replace('.cpp', '.so')
    link_shared(so_file, [host_obj_path])
    return so_file


def link_kernel_target(args, host_obj_path, temp_dir):
    if args.stage == 'device' and args.tiling_repr is not None:
        print("process static shape kernel with tiling_repr")
        static_shape_kernel_proc(args, temp_dir, args.tiling_repr)

    # 首次编译
    so_file = build_device_so(args, host_obj_path, temp_dir)

    # kernel_compile场景一次性生成so，链接device.o
    if args.stage == 'device':
        return so_file

    # jit_compile场景，检测是否为静态shape
    re_compile = try_static_shape_compile(args, temp_dir, so_file)
    if not re_compile:
        return so_file

    # 重编译，最终产物链接host.o+device.o
    return build_device_so(args, host_obj_path, temp_dir)


def copy_so_to_output(so_file, args, src_directory):
    dst_file = os.path.realpath(args.output_file)
    dst_dir_path = os.path.dirname(dst_file)
    if not os.path.exists(dst_dir_path):
        os.makedirs(dst_dir_path)

    shutil.copy(so_file, dst_file)
    print(f'copy file {so_file} to {dst_file}')
    os.chdir(src_directory)


def main(args):
    print("compile args:", args)
    src_directory = os.getcwd()
    os.chdir(args.temp_dir)
    print("change work dir:", os.getcwd())

    if args.stage == 'host':
        so_file = link_host_target(args, args.temp_dir)
    elif args.stage == 'device':
        so_file = link_kernel_target(args, None, args.temp_dir)
    else:  # all
        host_obj_path = compile_host_obj(args, args.temp_dir)
        so_file = link_kernel_target(args, host_obj_path, args.temp_dir)

    copy_so_to_output(so_file, args, src_directory)

def main_with_except(argv: List[str]):
    """Main process with except exceptions."""
    try:
        print("Enter main func")
        return main(argv)
    except argparse.ArgumentError as ex:
        print(f'error: check arguments error, {ex}')
        return False

if __name__ == "__main__":
    if not main_with_except(sys.argv[1:]):
        sys.exit(1)
