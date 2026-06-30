# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Rule: the original __global__ function must be preserved alongside SK adaptation.

kernel-launch-adapt.md step 1 + Q7: the original __global__ kernel must stay for
non-SuperKernel deployment. If a file has SK adaptation (__sk__ / SK_BIND) but no
matching __global__ entry, the original was lost. Requires human review (cannot be
auto-recreated).
"""

import re

RULE = {
    "id": "sk.global-preserved",
    "severity": "blocker",
    "category": "spec",
    "description": "Original __global__ kernel must be preserved alongside the __sk__ adaptation.",
}


def check(units):
    findings = []
    for unit in units:
        text = unit["text"]
        has_sk = bool(re.search(r"__sk__\b", text)) or bool(
            re.search(r"\bSK_BIND\s*\(", text)
        )
        has_global = bool(re.search(r"__global__\b", text))
        if has_sk and not has_global:
            findings.append(
                {
                    "rule_id": RULE["id"],
                    "severity": RULE["severity"],
                    "category": RULE["category"],
                    "actionable_by": ["human"],
                    "remediation_hint": {"kind": "human-decision"},
                    "message": (
                        "file has SK adaptation (__sk__/SK_BIND) but no original "
                        "__global__ kernel; the original kernel must be preserved."
                    ),
                    "target_file": unit["rel"],
                    "evidence_signature": "global-missing",
                }
            )
    return findings
