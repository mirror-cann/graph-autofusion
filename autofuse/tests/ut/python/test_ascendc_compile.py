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
