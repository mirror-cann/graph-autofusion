# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os

import pytest

from compile_test_utils import PYTHON_DIR, load_compile_module

MODULE_NAME = "autofuse.compiler.python.ascendc_compile"
MODULE_PATH = os.path.join(PYTHON_DIR, "ascendc_compile.py")

_LAUNCH_FUNC = (
    'extern "C" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, void* input0, void* output0, '
    'void* workspace, AutofuseTilingData* tiling_data)\n'
    '{\n'
    '  static_kernel<<<blockDim, nullptr, stream>>>(\n'
    '      (uint8_t*)input0, (uint8_t*)output0,\n'
    '      (uint8_t*)workspace, *tiling_data);\n'
    '  return 0;\n'
    '}\n'
)


@pytest.fixture()
def ascendc_compile_module():
    with load_compile_module(MODULE_NAME, MODULE_PATH) as loaded_module:
        yield loaded_module


def _noop_run_compile_command(cmd, stage_name):
    return None


def test_link_shared_adds_requested_libraries(ascendc_compile_module):
    captured = {}

    def fake_run_compile_command(cmd, stage_name):
        captured["cmd"] = cmd
        captured["stage_name"] = stage_name

    ascendc_compile_module.module.run_compile_command = fake_run_compile_command
    ascendc_compile_module.module.ASCEND_PATH = "/usr/local/Ascend/cann"
    ascendc_compile_module.module.machine = "x86_64"

    result = ascendc_compile_module.link_shared("kernel.so", ["host.o"], link_libraries=["graph_base", "register"])

    assert result == "kernel.so"
    assert captured["stage_name"] == "LinkObj"
    assert "-L" in captured["cmd"]
    assert "/usr/local/Ascend/cann/x86_64-linux/lib64" in captured["cmd"]
    assert "-lgraph_base" in captured["cmd"]
    assert "-lregister" in captured["cmd"]
    assert captured["cmd"].index("-lgraph_base") < captured["cmd"].index("-lregister")


def test_link_shared_skips_libraries_by_default(ascendc_compile_module):
    captured = {}

    def fake_run_compile_command(cmd, stage_name):
        captured["cmd"] = cmd

    ascendc_compile_module.module.run_compile_command = fake_run_compile_command

    ascendc_compile_module.link_shared("kernel.so", ["device.o"])

    assert "-lgraph_base" not in captured["cmd"]


def test_host_target_records_compile_and_link_stage(ascendc_compile_module, tmpdir, capsys):
    host_dir = tmpdir.mkdir("host")
    host_file = host_dir.join("graph_tiling_func.cpp")
    host_file.write("host")
    args = type("Args", (), {
        "host_files": str(host_file),
        "compile_options": "",
        "soc_version": "Ascend910B",
        "stage": "host",
        "graph_name": "graph",
    })()

    ascendc_compile_module.module.run_compile_command = _noop_run_compile_command
    ascendc_compile_module.link_host_target(args, str(tmpdir))

    labels = [item[0] for item in ascendc_compile_module.duration_records]
    assert ["InductorCompile", "host", "CompileHostObj", "graph"] in labels
    assert ["InductorCompile", "host", "LinkHostSo", "graph"] in labels
    assert capsys.readouterr().out == ""


def test_kernel_target_records_device_compile_and_link_stage(ascendc_compile_module, tmpdir):
    device_dir = tmpdir.mkdir("device")
    device_file = device_dir.join("graph_op_kernel.cpp")
    device_file.write(
        'extern "C" __global__ __aicore__ void graph_kernel(GM_ADDR input0) {}\n'
    )
    output_file = tmpdir.join("kernel.so")
    args = type("Args", (), {
        "device_files": str(device_file),
        "output_file": str(output_file),
        "soc_version": "Ascend910B",
        "stage": "device",
        "tiling_repr": None,
        "force_unknown": True,
        "graph_name": "graph",
    })()

    ascendc_compile_module.module.run_compile_command = _noop_run_compile_command
    ascendc_compile_module.link_kernel_target(args, None, str(tmpdir))

    labels = [item[0] for item in ascendc_compile_module.duration_records]
    assert ["InductorCompile", "device", "CompileDeviceObj", "graph"] in labels
    assert ["InductorCompile", "device", "LinkDeviceSo", "graph"] in labels


def test_try_static_shape_compile_records_stage_when_force_unknown(ascendc_compile_module, tmpdir):
    args = type("Args", (), {"force_unknown": True, "stage": "all", "graph_name": "graph"})()

    assert ascendc_compile_module.try_static_shape_compile(args, str(tmpdir), "kernel.so") is False

    labels = [item[0] for item in ascendc_compile_module.duration_records]
    assert ["InductorCompile", "all", "TryStaticShapeCompile", "graph"] in labels


def test_copy_so_to_output_records_stage(ascendc_compile_module, tmpdir):
    src_file = tmpdir.join("source.so")
    dst_file = tmpdir.mkdir("out").join("target.so")
    src_file.write("binary")
    src_directory = os.getcwd()
    args = type("Args", (), {
        "output_file": str(dst_file),
        "stage": "host",
        "graph_name": "graph",
    })()

    ascendc_compile_module.copy_so_to_output(str(src_file), args, src_directory)

    labels = [item[0] for item in ascendc_compile_module.duration_records]
    assert ["InductorCompile", "host", "CopyOutput", "graph"] in labels
    assert dst_file.read() == "binary"


def test_static_shape_kernel_proc_removes_tiling_data_from_launch(ascendc_compile_module, tmpdir):
    device_dir = tmpdir.mkdir("device")
    kernel_file = device_dir.join("static_kernel.cpp")
    kernel_file.write(
        'extern "C" __global__ __aicore__ void static_kernel(GM_ADDR input0, GM_ADDR output0, '
        'GM_ADDR workspace, AutofuseTilingData t) {\n'
        '  use(t);\n'
        '}\n'
        'void init_static_kernel(void) {}\n'
        'extern "C" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, void* input0, void* output0, '
        'void* workspace, AutofuseTilingData* tiling_data)\n'
        '{\n'
        '  static_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, (uint8_t*)output0, '
        '(uint8_t*)workspace, *tiling_data);\n'
        '  return 0;\n'
        '}\n'
    )
    args = type("Args", (), {"device_files": str(kernel_file)})()

    ascendc_compile_module.static_shape_kernel_proc(args, str(tmpdir), "AutofuseTilingData{.block_dim = 8}")

    content = kernel_file.read()
    assert (
        'extern "C" __global__ __aicore__ void static_kernel(GM_ADDR input0, GM_ADDR output0, GM_ADDR workspace) {'
        in content
    )
    assert "constexpr AutofuseTilingData t = AutofuseTilingData{.block_dim = 8};" in content
    assert ("static_kernel<<<blockDim, nullptr, stream>>>("
            "(uint8_t*)input0, (uint8_t*)output0, (uint8_t*)workspace);"
            in content)
    assert "*tiling_data" not in content


def test_static_shape_kernel_proc_removes_multiline_tiling_data_from_launch(ascendc_compile_module, tmpdir):
    device_dir = tmpdir.mkdir("device")
    kernel_file = device_dir.join("static_kernel.cpp")
    kernel_file.write(
        'extern "C" __global__ __aicore__ void static_kernel(GM_ADDR input0, GM_ADDR output0, '
        'GM_ADDR workspace, AutofuseTilingData t) {\n'
        '  use(t);\n'
        '}\n'
        + _LAUNCH_FUNC
    )
    args = type("Args", (), {"device_files": str(kernel_file)})()

    ascendc_compile_module.static_shape_kernel_proc(args, str(tmpdir), "AutofuseTilingData{.block_dim = 8}")

    content = kernel_file.read()
    assert "*tiling_data" not in content
    assert "(uint8_t*)workspace);" in content


def test_static_shape_kernel_proc_removes_multiline_kernel_param(ascendc_compile_module, tmpdir):
    device_dir = tmpdir.mkdir("device")
    kernel_file = device_dir.join("static_kernel.cpp")
    kernel_file.write(
        'extern "C" __global__ __aicore__ void static_kernel(\n'
        '    GM_ADDR input0,\n'
        '    GM_ADDR output0,\n'
        '    GM_ADDR workspace,\n'
        '    AutofuseTilingData t) {\n'
        '  use(t);\n'
        '}\n'
        + _LAUNCH_FUNC
    )
    args = type("Args", (), {"device_files": str(kernel_file)})()

    ascendc_compile_module.static_shape_kernel_proc(args, str(tmpdir), "AutofuseTilingData{.block_dim = 8}")

    content = kernel_file.read()
    assert "GM_ADDR workspace) {" in content
    assert "AutofuseTilingData t) {" not in content
    assert "constexpr AutofuseTilingData t = AutofuseTilingData{.block_dim = 8};" in content
    assert "(uint8_t*)workspace);" in content


def test_static_shape_kernel_proc_removes_indented_kernel_param(ascendc_compile_module, tmpdir):
    device_dir = tmpdir.mkdir("device")
    kernel_file = device_dir.join("static_kernel.cpp")
    kernel_file.write(
        '  extern "C" __global__ __aicore__ void static_kernel(GM_ADDR input0, GM_ADDR output0, '
        'GM_ADDR workspace, AutofuseTilingData t) {\n'
        '  use(t);\n'
        '}\n'
        + _LAUNCH_FUNC
    )
    args = type("Args", (), {"device_files": str(kernel_file)})()

    ascendc_compile_module.static_shape_kernel_proc(args, str(tmpdir), "AutofuseTilingData{.block_dim = 8}")

    content = kernel_file.read()
    assert "AutofuseTilingData t) {" not in content
    assert "constexpr AutofuseTilingData t = AutofuseTilingData{.block_dim = 8};" in content
    assert "(uint8_t*)workspace);" in content


def test_static_shape_kernel_proc_keeps_launch_when_kernel_definition_not_rewritten(
        ascendc_compile_module, tmpdir):
    device_dir = tmpdir.mkdir("device")
    kernel_file = device_dir.join("static_kernel.cpp")
    kernel_file.write(
        '__global__ __aicore__ __launch_bounds__(1) void static_kernel('
        'GM_ADDR input0, GM_ADDR workspace, AutofuseTilingData t) {\n'
        '  use(t);\n'
        '}\n'
        'extern "C" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, void* input0, '
        'void* workspace, AutofuseTilingData* tiling_data)\n'
        '{\n'
        '  static_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, '
        '(uint8_t*)workspace, *tiling_data);\n'
        '  return 0;\n'
        '}\n'
    )
    args = type("Args", (), {"device_files": str(kernel_file)})()

    ascendc_compile_module.static_shape_kernel_proc(args, str(tmpdir), "AutofuseTilingData{.block_dim = 8}")

    content = kernel_file.read()
    assert "AutofuseTilingData t) {" in content
    assert "constexpr AutofuseTilingData t =" not in content
    assert (
        "static_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, (uint8_t*)workspace, *tiling_data);"
        in content
    )


def test_static_shape_kernel_proc_only_rewrites_launch_for_rewritten_kernel(ascendc_compile_module, tmpdir):
    device_dir = tmpdir.mkdir("device")
    kernel_file = device_dir.join("static_kernel.cpp")
    kernel_file.write(
        'extern "C" __global__ __aicore__ void static_kernel(GM_ADDR input0, GM_ADDR workspace, '
        'AutofuseTilingData t) {\n'
        '  use(t);\n'
        '}\n'
        'extern "C" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, void* input0, '
        'void* workspace, AutofuseTilingData* tiling_data)\n'
        '{\n'
        '  static_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, (uint8_t*)workspace, *tiling_data);\n'
        '  other_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, (uint8_t*)workspace, *tiling_data);\n'
        '  return 0;\n'
        '}\n'
    )
    args = type("Args", (), {"device_files": str(kernel_file)})()

    ascendc_compile_module.static_shape_kernel_proc(args, str(tmpdir), "AutofuseTilingData{.block_dim = 8}")

    content = kernel_file.read()
    assert "static_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, (uint8_t*)workspace);" in content
    assert (
        "other_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, (uint8_t*)workspace, *tiling_data);"
        in content
    )


def test_static_shape_kernel_proc_keeps_dynamic_inductor_tiling_when_tiling_repr_is_none(
        ascendc_compile_module, tmpdir):
    device_dir = tmpdir.mkdir("device")
    kernel_file = device_dir.join("static_kernel.cpp")
    kernel_file.write(
        'extern "C" __global__ __aicore__ void static_kernel(GM_ADDR input0, GM_ADDR workspace, '
        'AutofuseTilingData t) {\n'
        '#ifdef INDUCTOR_CONST_TILING_DATA\n'
        '  const AutofuseTilingData t;\n'
        '#else\n'
        '  use(t);\n'
        '#endif\n'
        '}\n'
        'extern "C" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, void* input0, '
        'void* workspace, AutofuseTilingData* tiling_data)\n'
        '{\n'
        '  static_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, '
        '(uint8_t*)workspace, *tiling_data);\n'
        '  return 0;\n'
        '}\n'
    )
    args = type("Args", (), {"device_files": str(kernel_file), "stage": "device"})()

    ascendc_compile_module.static_shape_kernel_proc(args, str(tmpdir), None)

    content = kernel_file.read()
    assert "INDUCTOR_CONST_TILING_DATA" not in content
    assert "AutofuseTilingData t) {" in content
    assert "const AutofuseTilingData t;" not in content
    assert (
        "static_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, "
        "(uint8_t*)workspace, *tiling_data);"
        in content
    )
