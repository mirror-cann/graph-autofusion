# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Rule: SK_BIND may bind at most 4 SK sub-functions.

kernel-launch-adapt.md s6.3: SK_BIND supports binding at most 4 sub-functions.
"""
import re

RULE = {
    "id": "sk.bind-max-splits",
    "severity": "blocker",
    "category": "spec",
    "description": "SK_BIND must bind at most 4 SK sub-functions.",
}

_SK_BIND_RE = re.compile(r"\bSK_BIND\s*\(")


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


def _iter_sk_bind_args(text: str):
    for match in _SK_BIND_RE.finditer(text):
        body_start = match.end()
        depth = 1
        index = body_start
        while index < len(text):
            char = text[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    yield _split_top_level_commas(text[body_start:index])
                    break
            index += 1


def check(units):
    findings = []
    for unit in units:
        for args in _iter_sk_bind_args(unit["text"]):
            if len(args) < 3:
                continue
            orig = args[0]
            splits = [s.strip() for s in args[2:] if s.strip()]
            if len(splits) > 4:
                findings.append(
                    {
                        "rule_id": RULE["id"],
                        "severity": RULE["severity"],
                        "category": RULE["category"],
                        "actionable_by": ["human"],
                        "remediation_hint": {"kind": "human-decision"},
                        "message": f"SK_BIND({orig}, ...) binds {len(splits)} sub-functions; the maximum is 4.",
                        "target_file": unit["rel"],
                        "evidence_signature": f"sk_bind_too_many:{orig}",
                    }
                )
    return findings
