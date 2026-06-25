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
import types

import pytest

from compile_test_utils import PYTHON_DIR, load_compile_module

MODULE_NAME = "autofuse.compile_adapter"
MODULE_PATH = os.path.join(PYTHON_DIR, "compile_adapter.py")


@pytest.fixture()
def compile_adapter_module():
    ascendc_compile_module = types.ModuleType("autofuse.ascendc_compile")

    def _noop_main(args):
        return None

    def _record_duration(stage, step, graph_name, start, duration):
        from autofuse.pyautofuse import ascir
        labels = ["InductorCompile", stage, step, graph_name]
        ascir.utils.duration_record(labels, int(start), int(duration))

    ascendc_compile_module.main = _noop_main
    ascendc_compile_module.record_inductor_compile_duration = _record_duration
    with load_compile_module(
            MODULE_NAME,
            MODULE_PATH,
            extra_autofuse_attrs={"ascendc_compile": ascendc_compile_module},
            extra_modules={"autofuse.ascendc_compile": ascendc_compile_module}) as loaded_module:
        yield loaded_module


def test_host_compile_defaults_to_abi_1(compile_adapter_module):
    argv = ["--output_file=host.so", "--compile_options=-Werror"]

    args, temp_dir_ctx, auto_cleanup = compile_adapter_module.prepare_compile_context(argv, "host", None)

    assert auto_cleanup is True
    assert temp_dir_ctx is not None
    assert "-Werror" in args.compile_options
    assert "-D_GLIBCXX_USE_CXX11_ABI=1" in args.compile_options
    temp_dir_ctx.cleanup()


@pytest.mark.parametrize("abi", ["0", "1"])
def test_host_compile_keeps_explicit_abi(compile_adapter_module, abi):
    option = f"-D_GLIBCXX_USE_CXX11_ABI={abi}"
    argv = ["--output_file=host.so", f"--compile_options=-Werror {option}"]

    args, temp_dir_ctx, _ = compile_adapter_module.prepare_compile_context(argv, "host", None)

    assert args.compile_options == f"-Werror {option}"
    temp_dir_ctx.cleanup()


def test_device_compile_does_not_add_host_default_abi(compile_adapter_module):
    argv = ["--output_file=device.so", "--compile_options=-Werror"]

    args, temp_dir_ctx, _ = compile_adapter_module.prepare_compile_context(argv, "device", None)

    assert args.compile_options == "-Werror"
    temp_dir_ctx.cleanup()


def test_jit_compile_records_atrace_and_reports(compile_adapter_module, tmpdir, capsys):
    output_file = tmpdir.join("jit.so")
    argv = [f"--output_file={output_file}", f"--output_path={tmpdir}"]

    compile_adapter_module.jit_compile("tiling", "host", "kernel", argv)

    labels = [item[0] for item in compile_adapter_module.duration_records]
    assert ["InductorCompile", "jit_compile", "GenerateHostSource", "autofuse"] in labels
    assert ["InductorCompile", "jit_compile", "GenerateDeviceSource", "autofuse"] in labels
    assert ["InductorCompile", "jit_compile", "AscendCCompile", "autofuse"] in labels
    assert ["InductorCompile", "jit_compile", "Total", "autofuse"] in labels
    assert compile_adapter_module.duration_reports == [True]
    assert capsys.readouterr().out == ""


def test_host_compile_records_duration_without_stdout(compile_adapter_module, tmpdir, capsys):
    output_file = tmpdir.join("host.so")
    argv = [f"--output_file={output_file}", f"--output_path={tmpdir}"]

    compile_adapter_module.host_compile("tiling", "host", argv)

    labels = [item[0] for item in compile_adapter_module.duration_records]
    assert ["InductorCompile", "host_compile", "GenerateHostSource", "autofuse"] in labels
    assert ["InductorCompile", "host_compile", "AscendCCompile", "autofuse"] in labels
    assert ["InductorCompile", "host_compile", "Total", "autofuse"] in labels
    assert capsys.readouterr().out == ""


def test_kernel_compile_records_device_stage(compile_adapter_module, tmpdir):
    output_file = tmpdir.join("kernel.so")
    argv = [f"--output_file={output_file}", f"--output_path={tmpdir}"]

    compile_adapter_module.kernel_compile("tiling", "kernel", argv, tiling_repr="AutofuseTilingData{}")

    labels = [item[0] for item in compile_adapter_module.duration_records]
    assert ["InductorCompile", "kernel_compile", "GenerateDeviceSource", "autofuse"] in labels
    assert ["InductorCompile", "kernel_compile", "AscendCCompile", "autofuse"] in labels
    assert ["InductorCompile", "kernel_compile", "Total", "autofuse"] in labels
