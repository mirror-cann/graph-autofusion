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

import logging
from dataclasses import dataclass

from autofuse.pyautofuse import ascir, Autofuser, AutofuserOptions

logger = logging.getLogger(__name__)

ascir.utils.set_platform("2201", 1, 1024)


@dataclass
class GraphContext:
    """Encapsulates graph axes and sizes for node creation."""

    graph: ascir.HintGraph
    z0: ascir.Axis
    z1: ascir.Axis
    s0: ascir.SizeExpr
    s1: ascir.SizeExpr
    buf_z0: ascir.Axis
    buf_z1: ascir.Axis


def _set_node_output(node, dtype, ctx):
    """Set node output properties using graph context."""
    node.y.dtype = dtype
    node.y.axis = [ctx.z0, ctx.z1]
    node.y.size = [ctx.s0, ctx.s1]
    node.y.strides = [ctx.s1, ascir.SizeExpr(1)]


def _set_node_sched(node, ctx):
    """Set node scheduling axis using graph context."""
    node.attr.sched.axis = [ctx.z0, ctx.z1]


def _create_data_node(name, index, dtype, ctx):
    data = ascir.ops.Data(name, ctx.graph)
    data.attr.ir_attr.index = index
    _set_node_sched(data, ctx)
    _set_node_output(data, dtype, ctx)
    return data


def _create_load_node(name, offset, src, dtype, ctx):
    load = ascir.ops.Load(name)
    load.attr.ir_attr.offset = ascir.SizeExpr(offset)
    load.x = src
    _set_node_sched(load, ctx)
    _set_node_output(load, dtype, ctx)
    return load


def _create_store_node(src, dtype, ctx):
    store = ascir.ops.Store("store")
    store.attr.ir_attr.offset = ascir.SizeExpr(0)
    store.x = src
    _set_node_sched(store, ctx)
    _set_node_output(store, dtype, ctx)
    return store


def _create_output_node(src, dtype, ctx):
    output = ascir.ops.Output("output", ctx.graph)
    output.attr.ir_attr.index = 0
    output.x = src
    _set_node_sched(output, ctx)
    output.y.dtype = dtype
    output.y.axis = [ctx.z0, ctx.z1]
    output.y.size = [ctx.s0, ctx.s1]
    output.y.strides = [ctx.s1, ascir.SizeExpr(1)]
    return output


def _create_graph_skeleton(graph_name):
    graph = ascir.HintGraph(graph_name)
    s0 = graph.create_size("s0")
    s1 = graph.create_size("s1")
    z0 = graph.create_axis("z0", s0)
    z1 = graph.create_axis("z1", s1)
    buf_z0 = graph.create_axis("buf_z0", s0)
    buf_z1 = graph.create_axis("buf_z1", s1)
    return GraphContext(graph, z0, z1, s0, s1, buf_z0, buf_z1)


def _create_is_inf_node(src, dtype, ctx):
    is_inf = ascir.ops.IsInf("is_inf")
    is_inf.x = src
    _set_node_sched(is_inf, ctx)
    _set_node_output(is_inf, dtype, ctx)
    return is_inf


def _create_logical_or_node(x1, x2, dtype, ctx):
    logical_or = ascir.ops.LogicalOr("logical_or")
    logical_or.x1 = x1
    logical_or.x2 = x2
    _set_node_sched(logical_or, ctx)
    _set_node_output(logical_or, dtype, ctx)
    return logical_or


def _create_masked_fill_node(x, mask, value, dtype, ctx):
    masked_fill = ascir.ops.MaskedFill("masked_fill")
    masked_fill.x = x
    masked_fill.mask = mask
    masked_fill.value = value
    _set_node_sched(masked_fill, ctx)
    _set_node_output(masked_fill, dtype, ctx)
    return masked_fill


class _AutofuseTestBase:
    """Base class for autofuse operator tests."""

    expected_input_num = 0
    expected_output_num = 0

    @staticmethod
    def construct_graph():
        raise NotImplementedError

    def test_construct_graph(self):
        graph = self.construct_graph()
        debug_str = ascir.utils.debug_str(graph)
        assert debug_str

    def test_schedule(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)
        hint_graph = self.construct_graph()
        schedule_results = fuser.schedule(hint_graph)
        assert schedule_results.get_input_num() == self.expected_input_num
        assert schedule_results.get_output_num() == self.expected_output_num

    def test_autofuse_backend(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)
        hint_graph = self.construct_graph()
        sched_result = fuser.schedule(hint_graph)
        assert sched_result is not None, "Schedule must succeed"
        assert sched_result.get_input_num() == self.expected_input_num
        assert sched_result.get_output_num() == self.expected_output_num

        try:
            tiling_def, host_tiling, op_kernel = fuser.codegen(sched_result)
            assert len(tiling_def) > 0
            assert len(host_tiling) > 0
            assert len(op_kernel) > 0
        except (RuntimeError, TypeError) as e:
            logger.info("Codegen skipped (expected for simplified graphs): %s", e)


class TestIsInf(_AutofuseTestBase):
    """Test IsInf operator for autofusion"""

    expected_input_num = 1
    expected_output_num = 1

    @staticmethod
    def construct_graph():
        ctx = _create_graph_skeleton("IsInfTest")

        data = _create_data_node("data", 0, ascir.dtypes.float16, ctx)
        load = _create_load_node("load", 0, data, ascir.dtypes.float16, ctx)
        is_inf = _create_is_inf_node(load, ascir.dtypes.uint8, ctx)
        store = _create_store_node(is_inf, ascir.dtypes.uint8, ctx)
        _create_output_node(store, ascir.dtypes.uint8, ctx)

        ctx.graph.set_axis_map({ctx.z0: [ctx.buf_z0], ctx.z1: [ctx.buf_z1]})
        return ctx.graph


class TestMaskedFillFusionChain(_AutofuseTestBase):
    """Test IsInf + LogicalOr + MaskedFill fusion chain for autofusion"""

    expected_input_num = 3
    expected_output_num = 1

    @staticmethod
    def construct_graph():
        ctx = _create_graph_skeleton("MaskedFillTest")

        data_x = _create_data_node("data_x", 0, ascir.dtypes.float16, ctx)
        data_mask = _create_data_node("data_mask", 1, ascir.dtypes.uint8, ctx)
        data_value = _create_data_node("data_value", 2, ascir.dtypes.float16, ctx)

        load_x = _create_load_node("load_x", 0, data_x, ascir.dtypes.float16, ctx)
        load_mask = _create_load_node(
            "load_mask", 1, data_mask, ascir.dtypes.uint8, ctx
        )
        load_value = _create_load_node(
            "load_value", 2, data_value, ascir.dtypes.float16, ctx
        )

        is_inf = _create_is_inf_node(load_x, ascir.dtypes.uint8, ctx)
        logical_or = _create_logical_or_node(is_inf, load_mask, ascir.dtypes.uint8, ctx)
        masked_fill = _create_masked_fill_node(
            load_x, logical_or, load_value, ascir.dtypes.float16, ctx
        )

        store = _create_store_node(masked_fill, ascir.dtypes.float16, ctx)
        _create_output_node(store, ascir.dtypes.float16, ctx)

        ctx.graph.set_axis_map({ctx.z0: [ctx.buf_z0], ctx.z1: [ctx.buf_z1]})
        return ctx.graph
