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

import numpy as np

shape = (128, 128)
rng = np.random.default_rng(20260613)

input_x = rng.normal(0.0, 1.0, shape).astype(np.float16)
input_x[0, 0] = np.inf
input_x[3, 7] = -np.inf
input_x[64, 32] = np.inf

input_mask = np.zeros(shape, dtype=np.uint8)
input_mask[1::17, 2::19] = 1
input_mask[5, 5] = 1

input_value = np.full(shape, np.float16(-10000.0), dtype=np.float16)

input_x.tofile("Data_0.bin")
input_mask.tofile("Data_1.bin")
input_value.tofile("Data_2.bin")

golden = np.where(np.isinf(input_x) | (input_mask != 0), input_value, input_x).astype(np.float16)
golden.tofile("../out/golden.bin")
