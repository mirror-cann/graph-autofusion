# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Reference implementation for the add_custom elementwise-add operator.

Extension point: drop a reference_impls/<entry_name>.py exposing REFERENCE_IMPL
metadata + a compute(inputs) function. The registry indexes by REFERENCE_IMPL
["entry_name"]; the auto-build-correctness-oracle command calls compute() to
derive expected outputs from the synthesised input values.

`inputs` maps each kernel parameter name to {"shape", "dtype", "value"} where
value is the nested JSON list / scalar from the input-values spec. compute()
returns the expected primary output value (the same JSON-list shape the runner
emits), used directly as the oracle's expected_output.value.
"""

import numpy as np

REFERENCE_IMPL = {
    "entry_name": "add_custom",
    "description": "Elementwise add: z = x + y.",
    "comparator": "exact",
    "tolerance": {"rtol": 0, "atol": 0},
}


def compute(inputs: dict) -> object:
    x = np.array(inputs["x"]["value"], dtype=np.float64)
    y = np.array(inputs["y"]["value"], dtype=np.float64)
    z = x + y
    return z.tolist()
