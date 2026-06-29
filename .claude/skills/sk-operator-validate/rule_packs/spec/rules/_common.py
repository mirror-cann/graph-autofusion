# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Shared regex helpers for spec-check rules.

Loaded only as a helper (filename starts with `_` so the rule loader skips it).
Rule modules import it relative to their own directory via the loader's sys.path
fallback; to keep rules self-contained we instead duplicate the few tiny regex
helpers each rule needs. This module exists for documentation of the shared
patterns and is not imported by the bundled rules.
"""
import re

GLOBAL_QUALIFIER_RE = re.compile(r'extern\s+"C"\s+__global__\s+(?P<kt>__vector__|__cube__|__aicore__|__mix__\s*\([^)]*\))')
SK_FN_RE = re.compile(r'__sk__\s+(?P<kt>__vector__|__cube__|__aicore__|__mix__\s*\([^)]*\))\s+void\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)')
SK_BIND_RE = re.compile(r"SK_BIND\s*\(\s*(?P<orig>\w+)\s*,\s*(?P<mask>\d+)\s*,(?P<splits>[^)]*)\)")
