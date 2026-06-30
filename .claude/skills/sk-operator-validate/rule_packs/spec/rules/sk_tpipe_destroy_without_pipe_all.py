# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Rule: SK functions using TPipe must call DestroyWithoutPipeAll before exit.

superkernel advanced features guidance requires explicitly destroying TPipe in
SK functions to avoid the default destructor's PipeBarrierAll behavior between
fused SK sub-functions.
"""

import re

RULE = {
    "id": "sk.tpipe-destroy-without-pipe-all",
    "severity": "blocker",
    "category": "spec",
    "description": "SK functions that declare TPipe must call DestroyWithoutPipeAll() on it.",
}

_SK_FN_RE = re.compile(
    r"__sk__\s+(?:__vector__|__cube__|__aicore__|__mix__\s*\([^)]*\))\s+void\s+(?P<name>\w+)\s*\([^)]*\)\s*\{"
)


def _find_matching_brace(text: str, open_pos: int) -> int | None:
    depth = 0
    index = open_pos
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _tpipe_declarations(body: str) -> list[str]:
    names: list[str] = []
    for line in body.splitlines():
        if line.strip().startswith("//"):
            continue
        for match in re.finditer(
            r"\b(?:AscendC::)?TPipe\s+([A-Za-z_]\w*)\s*(?:;|=)", line
        ):
            names.append(match.group(1))
    return list(dict.fromkeys(names))


def check(units):
    findings = []
    for unit in units:
        for match in _SK_FN_RE.finditer(unit["text"]):
            open_brace = match.end() - 1
            close_brace = _find_matching_brace(unit["text"], open_brace)
            if close_brace is None:
                continue
            body = unit["text"][open_brace + 1 : close_brace]
            for name in _tpipe_declarations(body):
                if re.search(
                    rf"\b{re.escape(name)}\s*\.\s*DestroyWithoutPipeAll\s*\(", body
                ):
                    continue
                findings.append(
                    {
                        "rule_id": RULE["id"],
                        "severity": RULE["severity"],
                        "category": RULE["category"],
                        "actionable_by": ["human"],
                        "remediation_hint": {"kind": "human-decision"},
                        "message": f"__sk__ function {match.group('name')!r} declares TPipe {name!r} but does not call {name}.DestroyWithoutPipeAll().",
                        "target_file": unit["rel"],
                        "evidence_signature": f"sk_tpipe_destroy:{match.group('name')}:{name}",
                    }
                )
    return findings
