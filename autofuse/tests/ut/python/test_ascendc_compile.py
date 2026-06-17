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

import importlib.util
import os

import pytest


_BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.realpath(__file__)))))
)
_PYTHON_DIR = os.path.join(_BASE_DIR, "autofuse/compiler/python")
MODULE_NAME = "autofuse.compiler.python.ascendc_compile"
MODULE_PATH = os.path.join(_PYTHON_DIR, "ascendc_compile.py")

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
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec is not None and spec.loader is not None:
        spec.loader.exec_module(module)
    return module


def test_link_shared_adds_requested_libraries(ascendc_compile_module):
    captured = {}

    def fake_run_compile_command(cmd, stage_name):
        captured["cmd"] = cmd
        captured["stage_name"] = stage_name

    ascendc_compile_module.run_compile_command = fake_run_compile_command
    ascendc_compile_module.ASCEND_PATH = "/usr/local/Ascend/cann"
    ascendc_compile_module.machine = "x86_64"

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

    ascendc_compile_module.run_compile_command = fake_run_compile_command

    ascendc_compile_module.link_shared("kernel.so", ["device.o"])

    assert "-lgraph_base" not in captured["cmd"]


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
