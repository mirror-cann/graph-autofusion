#!/usr/bin/env python3
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

from autofuse.pyautofuse import ascir
from autofuse.pyautofuse import Autofuser, AutofuserOptions

NpuKernel0Graph = ascir.HintGraph("isinf_maskedfill_fusion")
A0 = ascir.SizeExpr(128)
A1 = ascir.SizeExpr(128)
buf_a0 = NpuKernel0Graph.create_axis("buf_z0", A0)
buf_a1 = NpuKernel0Graph.create_axis("buf_z1", A1)

data_x = ascir.ops.Data("data_x", NpuKernel0Graph)
data_x.attr.ir_attr.index = 0
data_x.y.dtype = ascir.dtypes.float16

data_mask = ascir.ops.Data("data_mask", NpuKernel0Graph)
data_mask.attr.ir_attr.index = 1
data_mask.y.dtype = ascir.dtypes.uint8

data_value = ascir.ops.Data("data_value", NpuKernel0Graph)
data_value.attr.ir_attr.index = 2
data_value.y.dtype = ascir.dtypes.float16

load_x = ascir.ops.Load("load_x", NpuKernel0Graph)
load_x.attr.ir_attr.offset = ascir.SizeExpr(0)
load_x.attr.sched.axis = [buf_a0, buf_a1]
load_x.x = data_x.y
load_x.y.axis = [buf_a0, buf_a1]
load_x.y.strides = [A1, ascir.SizeExpr(1)]
load_x.y.size = [A0, A1]
load_x.y.dtype = ascir.dtypes.float16

load_mask = ascir.ops.Load("load_mask", NpuKernel0Graph)
load_mask.attr.ir_attr.offset = ascir.SizeExpr(0)
load_mask.attr.sched.axis = [buf_a0, buf_a1]
load_mask.x = data_mask.y
load_mask.y.axis = [buf_a0, buf_a1]
load_mask.y.strides = [A1, ascir.SizeExpr(1)]
load_mask.y.size = [A0, A1]
load_mask.y.dtype = ascir.dtypes.uint8

load_value = ascir.ops.Load("load_value", NpuKernel0Graph)
load_value.attr.ir_attr.offset = ascir.SizeExpr(0)
load_value.attr.sched.axis = [buf_a0, buf_a1]
load_value.x = data_value.y
load_value.y.axis = [buf_a0, buf_a1]
load_value.y.strides = [A1, ascir.SizeExpr(1)]
load_value.y.size = [A0, A1]
load_value.y.dtype = ascir.dtypes.float16

is_inf = ascir.ops.IsInf("is_inf", NpuKernel0Graph)
is_inf.attr.sched.axis = [buf_a0, buf_a1]
is_inf.x = load_x.y
is_inf.y.axis = [buf_a0, buf_a1]
is_inf.y.strides = [A1, ascir.SizeExpr(1)]
is_inf.y.size = [A0, A1]
is_inf.y.dtype = ascir.dtypes.uint8

logical_or = ascir.ops.LogicalOr("logical_or", NpuKernel0Graph)
logical_or.attr.sched.axis = [buf_a0, buf_a1]
logical_or.x1 = is_inf.y
logical_or.x2 = load_mask.y
logical_or.y.axis = [buf_a0, buf_a1]
logical_or.y.strides = [A1, ascir.SizeExpr(1)]
logical_or.y.size = [A0, A1]
logical_or.y.dtype = ascir.dtypes.uint8

masked_fill = ascir.ops.MaskedFill("masked_fill", NpuKernel0Graph)
masked_fill.attr.sched.axis = [buf_a0, buf_a1]
masked_fill.x = load_x.y
masked_fill.mask = logical_or.y
masked_fill.value = load_value.y
masked_fill.y.axis = [buf_a0, buf_a1]
masked_fill.y.strides = [A1, ascir.SizeExpr(1)]
masked_fill.y.size = [A0, A1]
masked_fill.y.dtype = ascir.dtypes.float16

store = ascir.ops.Store("store", NpuKernel0Graph)
store.attr.ir_attr.offset = ascir.SizeExpr(0)
store.attr.sched.axis = [buf_a0, buf_a1]
store.x = masked_fill.y
store.y.axis = [buf_a0, buf_a1]
store.y.strides = [A1, ascir.SizeExpr(1)]
store.y.size = [A0, A1]
store.y.dtype = ascir.dtypes.float16

output = ascir.ops.Output("output", NpuKernel0Graph)
output.attr.ir_attr.index = 0
output.x = store.y
output.y.dtype = ascir.dtypes.float16

fuser = Autofuser(AutofuserOptions())
fused_NpuKernel0Graph = fuser.schedule(NpuKernel0Graph)
tiling_def, host_impl, device_impl = fuser.codegen(fused_NpuKernel0Graph)
print("=================================")
print(device_impl)
