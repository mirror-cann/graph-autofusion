# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Rule: SK system-args API must use current names.

kernel-launch-adapt.md Q6: the current CANN headers renamed the SkSystemArgs
members. Legacy `skBlockNum` / `SkGetBlockNum()` fail to compile and must be
`skNumBlocks` / `SkGetNumBlocks()`.
"""

import re

RULE = {
    "id": "sk.sys-args-api-current",
    "severity": "blocker",
    "category": "spec",
    "description": "Use current SkSystemArgs API names (skNumBlocks / SkGetNumBlocks); legacy names fail to compile.",
}

_LEGACY_NAMES = (("skBlockNum", "skNumBlocks"), ("SkGetBlockNum", "SkGetNumBlocks"))


def check(units):
    findings = []
    for unit in units:
        for legacy, modern in _LEGACY_NAMES:
            if re.search(rf"\b{legacy}\b", unit["text"]):
                findings.append(
                    {
                        "rule_id": RULE["id"],
                        "severity": RULE["severity"],
                        "category": RULE["category"],
                        "actionable_by": ["codegen.apply-remediation"],
                        "remediation_hint": {
                            "kind": "rename-symbol",
                            "target_file": unit["rel"],
                            "old_value": legacy,
                            "new_value": modern,
                        },
                        "message": f"legacy SkSystemArgs API {legacy!r}; rename to {modern!r} (legacy name no longer exists in current CANN headers).",
                        "target_file": unit["rel"],
                        "evidence_signature": legacy,
                    }
                )
    return findings
