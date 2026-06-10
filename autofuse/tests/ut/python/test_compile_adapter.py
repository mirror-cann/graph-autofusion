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
import sys
import types

import pytest


_BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.realpath(__file__)))))
)
_PYTHON_DIR = os.path.join(_BASE_DIR, "autofuse/compiler/python")
MODULE_NAME = "autofuse.compile_adapter"
MODULE_PATH = os.path.join(_PYTHON_DIR, "compile_adapter.py")


@pytest.fixture()
def compile_adapter_module():
    original_autofuse = sys.modules.get("autofuse")
    original_ascendc_compile = sys.modules.get("autofuse.ascendc_compile")
    autofuse_module = types.ModuleType("autofuse")
    ascendc_compile_module = types.ModuleType("autofuse.ascendc_compile")

    def _noop_main(args):
        return None

    ascendc_compile_module.main = _noop_main
    autofuse_module.ascendc_compile = ascendc_compile_module
    sys.modules["autofuse"] = autofuse_module
    sys.modules["autofuse.ascendc_compile"] = ascendc_compile_module

    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec is not None and spec.loader is not None:
        spec.loader.exec_module(module)
    yield module

    if original_autofuse is None:
        sys.modules.pop("autofuse", None)
    else:
        sys.modules["autofuse"] = original_autofuse
    if original_ascendc_compile is None:
        sys.modules.pop("autofuse.ascendc_compile", None)
    else:
        sys.modules["autofuse.ascendc_compile"] = original_ascendc_compile


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
