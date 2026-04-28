#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------

"""Unit tests for super_kernel_option_parse module."""

import logging
from unittest import mock
import pytest
from asc_op_compile_base.asc_op_compiler.super_kernel_utility import CommonUtility
from superkernel.super_kernel_option_parse import parse_super_kernel_options
from superkernel.super_kernel_constants import SuperKernelEarlyStartMode


class TestParseSuperKernelOptions:
    @staticmethod
    def setup_method():
        logging.info("---------------SetUp---------------")

    @staticmethod
    def teardown_method():
        logging.info("--------------TearDown-------------")

    @staticmethod
    def test_convert_underscore_to_hyphen_true():
        with mock.patch.object(CommonUtility, 'ascendc_raise_python_err'):
            res = parse_super_kernel_options(
                "early_start=1", convert_underscore_to_hyphen=True)
            assert 'early-start' in res
            assert res['early-start'] == SuperKernelEarlyStartMode.EarlyStartEnableV2

    @staticmethod
    def test_convert_underscore_to_hyphen_false():
        with mock.patch.object(CommonUtility, 'ascendc_raise_python_err',
                               side_effect=Exception) as mock_raise:
            with pytest.raises(Exception):
                parse_super_kernel_options(
                    "early_start=1", convert_underscore_to_hyphen=False)
            mock_raise.assert_called()

    @staticmethod
    def test_strip_quotes():
        with mock.patch.object(CommonUtility, 'ascendc_raise_python_err'):
            res = parse_super_kernel_options('"early-start=1"')
            assert 'early-start' in res
            assert res['early-start'] == SuperKernelEarlyStartMode.EarlyStartEnableV2
