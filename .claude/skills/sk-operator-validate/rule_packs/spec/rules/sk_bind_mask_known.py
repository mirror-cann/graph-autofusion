# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Rule: SK_BIND mask must be a known value.

kernel-launch-adapt.md s6.4: the mask (second SK_BIND argument) is a bit mask
over known bits: 0 (no capability bits), 4 (DCCI, recommended default), 2
(early start set flag), 1 (early start wait flag), or a bitwise OR of those
bits (0..7). Anything outside that range is suspicious and warrants review.
"""

import re

RULE = {
    "id": "sk.bind-mask-known",
    "severity": "warning",
    "category": "spec",
    "description": (
        "SK_BIND mask should be 0 or a combination of the known bits "
        "(DCCI=4, early-start-set=2, early-start-wait=1)."
    ),
}

_KNOWN_MASKS = frozenset(range(0, 8))  # any OR of bits 1/2/4, including 0
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
        elif char == "," and not any(
            (depth_angle, depth_paren, depth_bracket, depth_brace)
        ):
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
            if len(args) < 2 or not args[1].isdigit():
                continue
            orig = args[0]
            mask = int(args[1])
            if mask not in _KNOWN_MASKS:
                findings.append(
                    {
                        "rule_id": RULE["id"],
                        "severity": RULE["severity"],
                        "category": RULE["category"],
                        "actionable_by": ["human"],
                        "remediation_hint": {"kind": "human-decision"},
                        "message": (
                            f"SK_BIND({orig}, {mask}, ...) uses an unrecognised "
                            "mask; expected 0 or a combination of DCCI=4 / "
                            "early-start-set=2 / early-start-wait=1."
                        ),
                        "target_file": unit["rel"],
                        "evidence_signature": f"sk_bind_mask:{orig}:{mask}",
                    }
                )
    return findings
