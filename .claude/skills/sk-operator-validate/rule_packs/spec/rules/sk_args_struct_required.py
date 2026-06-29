# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Rule: an __sk__ sub-function's first parameter must be a const Args struct pointer.

kernel-launch-adapt.md s3.1 rule 1 + rule 3: SK sub-functions take parameters via a
struct pointer `const <ArgsStruct> *args` (optionally followed by `sk::SkSystemArgs
*sysArgs`), never as positional kernel arguments.
"""
import re

RULE = {
    "id": "sk.args-struct-required",
    "severity": "blocker",
    "category": "spec",
    "description": "An __sk__ sub-function's first parameter must be `const <ArgsStruct> *args`.",
}

_SK_FN_RE = re.compile(
    r"__sk__\s+(?:__vector__|__cube__|__aicore__|__mix__\s*\([^)]*\))\s+void\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)"
)
_FIRST_PARAM_OK_RE = re.compile(r"^\s*const\s+\w+(?:\s*<.+>)?\s*\*\s*\w+\s*$")


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth_angle = 0
    depth_paren = 0
    depth_bracket = 0
    depth_brace = 0
    for index, char in enumerate(text):
        if char == "<":
            depth_angle += 1
        elif char == ">" and depth_angle:
            depth_angle -= 1
        elif char == "(":
            depth_paren += 1
        elif char == ")" and depth_paren:
            depth_paren -= 1
        elif char == "[":
            depth_bracket += 1
        elif char == "]" and depth_bracket:
            depth_bracket -= 1
        elif char == "{":
            depth_brace += 1
        elif char == "}" and depth_brace:
            depth_brace -= 1
        elif char == "," and not any((depth_angle, depth_paren, depth_bracket, depth_brace)):
            parts.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def check(units):
    findings = []
    for unit in units:
        for m in _SK_FN_RE.finditer(unit["text"]):
            params = m.group("params").strip()
            # No-arg SK functions (e.g. side-effect kernels like clear) are valid.
            if not params:
                continue
            split_params = _split_top_level_commas(params)
            first = split_params[0] if split_params else ""
            if not _FIRST_PARAM_OK_RE.match(first):
                findings.append(
                    {
                        "rule_id": RULE["id"],
                        "severity": RULE["severity"],
                        "category": RULE["category"],
                        "actionable_by": ["human"],
                        "remediation_hint": {"kind": "human-decision"},
                        "message": f"__sk__ function {m.group('name')!r} first parameter is {first!r}; expected `const <ArgsStruct> *args`.",
                        "target_file": unit["rel"],
                        "evidence_signature": f"sk_args_first_param:{m.group('name')}",
                    }
                )
    return findings
