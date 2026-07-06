# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Rule: kernel source must include kernel_operator.h.

Any .asc/.cpp that defines a kernel entry (__global__ or __sk__) must include
"kernel_operator.h" for the AscendC primitives. Auto-remediable by adding the
include line.
"""

import re

RULE = {
    "id": "includes.kernel-operator-h",
    "severity": "warning",
    "category": "spec",
    "description": 'Kernel source defining a kernel entry must #include "kernel_operator.h".',
}

_ENTRY_RE = re.compile(r"__global__\b|__sk__\b")


def check(units):
    findings = []
    for unit in units:
        if not unit["rel"].endswith((".asc", ".cpp", ".cc", ".cxx")):
            continue
        text = unit["text"]
        if not _ENTRY_RE.search(text):
            continue
        if '"kernel_operator.h"' in text:
            continue
        findings.append(
            {
                "rule_id": RULE["id"],
                "severity": RULE["severity"],
                "category": RULE["category"],
                "actionable_by": ["codegen.apply-remediation"],
                "remediation_hint": {
                    "kind": "add-include",
                    "target_file": unit["rel"],
                    "new_value": '#include "kernel_operator.h"',
                },
                "message": 'kernel source defines a kernel entry but does not include "kernel_operator.h".',
                "target_file": unit["rel"],
                "evidence_signature": "missing-kernel-operator-h",
            }
        )
    return findings
