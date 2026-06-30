# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import re
from dataclasses import dataclass


CANONICAL_OWNER = "sk-operator-asset-adapter"
SCHEMA_VERSION = 1
_MIX_QUALIFIER_RE = r"__mix__(?:\s*\([^)]*\))?"


@dataclass(frozen=True)
class ParsedKernelEntry:
    name: str


_GLOBAL_FN_RE = re.compile(
    r'(?P<extern>extern\s+"C"\s+)?'
    rf"(?P<qualifiers>(?:__global__|__aicore__|__vector__|__cube__|{_MIX_QUALIFIER_RE}|__inline__|inline|\s)+?)\s+"
    r"(?P<rettype>void)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*"
    r"\((?P<params>[^()]*)\)\s*(?:\\\s*)?\{",
    re.MULTILINE,
)


def parse_global_entries(source_text: str) -> list[ParsedKernelEntry]:
    return [
        ParsedKernelEntry(name=match.group("name"))
        for match in _GLOBAL_FN_RE.finditer(source_text or "")
        if "__global__" in " ".join(match.group("qualifiers").split())
    ]
