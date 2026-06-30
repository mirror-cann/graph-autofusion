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

import json
import re
from pathlib import Path
from typing import Any


CHIP_ARCH_SOURCES = {
    "torch_npu_soc_versions": "https://gitcode.com/Ascend/pytorch/blob/master/torch_npu/csrc/core/npu/NpuVariables.h",
    "torch_npu_soc_strings": "https://gitcode.com/Ascend/pytorch/blob/master/torch_npu/csrc/core/npu/NpuVariables.cpp",
    "asc_devkit_arch_examples": "https://gitcode.com/cann/asc-devkit/tree/master/examples",
}

SUPPORTED_CHIP_ARCH_RULES = (
    {
        "canonical_chip": "ascend910b",
        "arch": "dav-2201",
        "string_prefixes": ("ascend910b",),
        "enum_values": (104, 220, 221, 222, 223, 224, 225),
        "source": [
            CHIP_ARCH_SOURCES["torch_npu_soc_versions"],
            CHIP_ARCH_SOURCES["torch_npu_soc_strings"],
            CHIP_ARCH_SOURCES["asc_devkit_arch_examples"],
        ],
    },
    {
        "canonical_chip": "ascend950",
        "arch": "dav-3510",
        "string_prefixes": ("ascend950",),
        "enum_values": (260,),
        "source": [
            CHIP_ARCH_SOURCES["torch_npu_soc_versions"],
            CHIP_ARCH_SOURCES["torch_npu_soc_strings"],
            CHIP_ARCH_SOURCES["asc_devkit_arch_examples"],
        ],
    },
)

ADD_CONFIG_RE = re.compile(r"\bAddConfig\s*\(\s*['\"]([^'\"]+)['\"]")
REGISTER_OP_AICORE_CONFIG_RE = re.compile(
    r"\bREGISTER_OP_AICORE_CONFIG\s*\(\s*[A-Za-z_]\w*\s*,\s*([A-Za-z0-9_]+)\s*,"
)
ASCEND_COMPUTE_UNIT_RE = re.compile(r"\bASCEND_COMPUTE_UNIT\b")
SOC_VALUE_RE = re.compile(
    r"\b(?:ascend|Ascend)"
    r"(?:"
    r"910[A-Za-z0-9_\\-]*|"
    r"310[A-Za-z0-9_\\-]*|"
    r"950[A-Za-z0-9_\\-]*"
    r")\b"
)


def normalize_chip(chip: object) -> str:
    return str(chip or "").strip().lower().replace("_", "").replace("-", "")


def normalize_arch(arch: object) -> str:
    return str(arch or "").strip().lower().replace("_", "-")


def split_target_chips(raw_chips: object) -> list[str]:
    chips: list[str] = []
    seen: set[str] = set()
    for item in re.split(r"[,;]", str(raw_chips or "")):
        chip = item.strip()
        normalized = normalize_chip(chip)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        chips.append(chip)
    return chips


def split_arches(raw_arches: object) -> list[str]:
    arches: list[str] = []
    seen: set[str] = set()
    for item in re.split(r"[,;]", str(raw_arches or "")):
        arch = normalize_arch(item)
        if not arch or arch in seen:
            continue
        seen.add(arch)
        arches.append(arch)
    return arches


def resolve_chip_to_arch(chip: object) -> dict[str, Any] | None:
    raw = str(chip or "").strip()
    if not raw:
        return None
    normalized = normalize_chip(raw)
    enum_value = None
    try:
        enum_value = int(raw)
    except (TypeError, ValueError):
        enum_value = None
    for rule in SUPPORTED_CHIP_ARCH_RULES:
        if any(
            normalized.startswith(normalize_chip(prefix))
            for prefix in rule["string_prefixes"]
        ):
            return {
                "chip": raw,
                "canonical_chip": rule["canonical_chip"],
                "arch": rule["arch"],
                "source": list(rule["source"]),
                "confidence": "source-backed",
            }
        if enum_value is not None and enum_value in set(rule["enum_values"]):
            return {
                "chip": raw,
                "canonical_chip": rule["canonical_chip"],
                "arch": rule["arch"],
                "source": list(rule["source"]),
                "confidence": "source-backed",
            }
    return None


def resolve_target_chips(
    raw_chips: object,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resolved: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    for chip in split_target_chips(raw_chips):
        item = resolve_chip_to_arch(chip)
        if item is None:
            unsupported.append(
                {
                    "chip": chip,
                    "reason": "unsupported-chip-arch-mapping",
                    "supported_chips": [
                        rule["canonical_chip"] for rule in SUPPORTED_CHIP_ARCH_RULES
                    ],
                }
            )
        else:
            resolved.append(item)
    return resolved, unsupported


def arch_to_module_suffix(arch: object) -> str:
    return normalize_arch(arch).replace("-", "_")


def extract_supported_soc_versions_from_text(text: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for match in ADD_CONFIG_RE.finditer(text or ""):
        value = match.group(1).strip()
        normalized = normalize_chip(value)
        if value and normalized not in seen:
            seen.add(normalized)
            values.append(value)
    for match in REGISTER_OP_AICORE_CONFIG_RE.finditer(text or ""):
        value = match.group(1).strip()
        normalized = normalize_chip(value)
        if value and normalized not in seen:
            seen.add(normalized)
            values.append(value)
    return values


def _walk_json_values(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_json_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_json_values(item)
    else:
        yield value


def extract_supported_soc_versions_from_cmake_presets(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    values: list[str] = []
    seen: set[str] = set()
    for item in _walk_json_values(payload):
        text = str(item or "")
        if not ASCEND_COMPUTE_UNIT_RE.search(
            json.dumps(item, ensure_ascii=False)
            if isinstance(item, (dict, list))
            else text
        ):
            # Also accept direct values because cacheVariables nesting separates key and value.
            pass
        for match in SOC_VALUE_RE.findall(text):
            normalized = normalize_chip(match)
            if normalized and normalized not in seen:
                seen.add(normalized)
                values.append(match)
    return values


def extract_supported_soc_versions_from_misc_text(text: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for match in SOC_VALUE_RE.findall(text or ""):
        value = match.strip()
        normalized = normalize_chip(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            values.append(value)
    return values


def supported_arches_for_soc_versions(soc_versions: list[str]) -> list[str]:
    arches: list[str] = []
    seen: set[str] = set()
    for chip in soc_versions:
        resolved = resolve_chip_to_arch(chip)
        if resolved is None:
            continue
        arch = normalize_arch(resolved["arch"])
        if arch not in seen:
            seen.add(arch)
            arches.append(arch)
    return arches


def build_target_resolution(
    supported_soc_versions: list[str],
    target_chips: object = "",
) -> dict[str, Any]:
    requested, unsupported = resolve_target_chips(target_chips)
    supported_norm = {normalize_chip(chip) for chip in supported_soc_versions}
    supported_arch_norm = {
        normalize_arch(resolved["arch"])
        for chip in supported_soc_versions
        for resolved in [resolve_chip_to_arch(chip)]
        if resolved is not None
    }
    has_declared_support = bool(supported_norm)
    resolutions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    candidate_chips = (
        requested
        if requested
        else [
            resolved
            for chip in supported_soc_versions
            for resolved in [resolve_chip_to_arch(chip)]
            if resolved is not None
        ]
    )
    for item in candidate_chips:
        canonical_norm = normalize_chip(item["canonical_chip"])
        raw_norm = normalize_chip(item["chip"])
        arch_norm = normalize_arch(item["arch"])
        if (
            has_declared_support
            and canonical_norm not in supported_norm
            and raw_norm not in supported_norm
            and arch_norm not in supported_arch_norm
        ):
            skipped.append(
                {
                    "chip": item["chip"],
                    "arch": item["arch"],
                    "reason": "not-declared-by-operator",
                }
            )
            continue
        resolutions.append(item)
    if not has_declared_support and requested:
        resolutions = [{**item, "confidence": "compat-inferred"} for item in requested]
    arches = []
    seen_arches: set[str] = set()
    for item in resolutions:
        arch = normalize_arch(item["arch"])
        if arch not in seen_arches:
            seen_arches.add(arch)
            arches.append(arch)
    return {
        "requested_chips": [item["chip"] for item in requested],
        "unsupported_requested_chips": unsupported,
        "resolutions": resolutions,
        "skipped": skipped,
        "arches": arches,
        "support_source": "declared" if has_declared_support else "compat-inferred",
    }
