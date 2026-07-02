# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Rule: Args struct fields smaller than 4 bytes must use alignas(4).

kernel-launch-adapt.md s3.1 rule 3 + s6.1: small integer fields (int8/uint8/
int16/uint16/bool) in the SK Args struct must be declared `alignas(4)` for ABI
correctness.
"""

import re

RULE = {
    "id": "sk.args-struct-alignas",
    "severity": "blocker",
    "category": "spec",
    "description": "Args struct fields < 4 bytes must use alignas(4) (ABI requirement).",
}

_SMALL_TYPES = ("int8_t", "uint8_t", "int16_t", "uint16_t", "bool")
_ARGS_STRUCT_RE = re.compile(
    r"struct\s+(?P<name>\w*Args)\s*\{(?P<body>[^}]*)\}", re.DOTALL
)


def check(units):
    findings = []
    for unit in units:
        for m in _ARGS_STRUCT_RE.finditer(unit["text"]):
            struct_name = m.group("name")
            for line in m.group("body").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("//"):
                    continue
                # A field line with a small type that is not preceded by alignas(.
                if (
                    any(re.search(rf"\b{t}\b", stripped) for t in _SMALL_TYPES)
                    and "alignas" not in stripped
                ):
                    findings.append(
                        {
                            "rule_id": RULE["id"],
                            "severity": RULE["severity"],
                            "category": RULE["category"],
                            "actionable_by": ["human"],
                            "remediation_hint": {"kind": "human-decision"},
                            "message": (
                                f"struct {struct_name} field {stripped!r} is a sub-4-byte type and must use alignas(4)."
                            ),
                            "target_file": unit["rel"],
                            "evidence_signature": f"alignas:{struct_name}:{stripped}",
                        }
                    )
    return findings
