# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Rule: __sk__ sub-function kernel type must match the original __global__'s.

kernel-launch-adapt.md s3.1 rule 2: SK sub-functions must carry the same kernel
type qualifier as the original __global__ function -- with the special cases
__mix__(1,0) -> __cube__ and __mix__(0,1) -> __vector__.

This rule operates per file: it pairs each SK sub-function (stripped of its
`_sk<...>` suffix) to a __global__ entry by name prefix and flags an obvious
qualifier mismatch. It is conservative -- if it cannot confidently pair them it
emits nothing.
"""

import re

RULE = {
    "id": "sk.kernel-type-match",
    "severity": "warning",
    "category": "spec",
    "description": (
        "SK sub-function kernel type must match the original __global__ kernel "
        "type (with __mix__(1,0)->__cube__ and __mix__(0,1)->__vector__ "
        "special cases)."
    ),
}

_MIX_RE = re.compile(r"__mix__\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)")
_GLOBAL_RE = re.compile(
    r"__global__\s+(?P<kt>__vector__|__cube__|__aicore__|__mix__\s*\([^)]*\))\s+void\s+(?P<name>\w+)"
)
_SK_RE = re.compile(r"__sk__\s+(?P<kt>__vector__|__cube__|__aicore__|__mix__\s*\([^)]*\))\s+void\s+(?P<name>\w+)")


def _expected_sk_type(global_kt: str) -> str:
    m = _MIX_RE.search(global_kt)
    if m:
        c, v = int(m.group(1)), int(m.group(2))
        if c >= 1 and v == 0:
            return "__cube__"
        if c == 0 and v >= 1:
            return "__vector__"
        return f"__mix__({c},{v})"
    return global_kt.replace(" ", "")


def _canonical(kt: str) -> str:
    return kt.replace(" ", "")


def check(units):
    findings = []
    for unit in units:
        globals_by_base = {m.group("name"): _expected_sk_type(m.group("kt")) for m in _GLOBAL_RE.finditer(unit["text"])}
        if not globals_by_base:
            continue
        for sm in _SK_RE.finditer(unit["text"]):
            sk_name = sm.group("name")
            base = sk_name[:-3] if sk_name.endswith("_sk") else sk_name
            # Trim a trailing _sk that the adapter appends; pair by base name.
            expected = globals_by_base.get(base)
            if expected is None:
                continue
            actual = _canonical(sm.group("kt"))
            if _canonical(expected) != actual:
                findings.append(
                    {
                        "rule_id": RULE["id"],
                        "severity": RULE["severity"],
                        "category": RULE["category"],
                        "actionable_by": ["human"],
                        "remediation_hint": {"kind": "human-decision"},
                        "message": (
                            f"SK function {sk_name!r} uses kernel type {actual!r} "
                            f"but the matching __global__ {base!r} implies "
                            f"{_canonical(expected)!r}."
                        ),
                        "target_file": unit["rel"],
                        "evidence_signature": f"kt_mismatch:{sk_name}",
                    }
                )
    return findings
