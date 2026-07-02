# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Spec rule-pack runner for sk-operator-validate."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any, Callable

SOURCE_SUFFIXES = (".asc", ".cpp", ".cc", ".cxx", ".h", ".hpp")
RULES_DIR = Path(__file__).resolve().parent / "spec" / "rules"

SCHEMA_VERSION = 1


class CliUsageError(Exception):
    """Raised for user-facing usage errors."""


def _stable_finding_id(
    rule_id: str,
    target_file: str,
    target_location: dict | None,
    evidence_signature: str,
) -> str:
    location_token = ""
    if target_location:
        location_items = []
        for key in sorted(target_location):
            location_items.append(f"{key}={target_location[key]}")
        location_token = "|".join(location_items)
    base = "|".join(
        [rule_id, target_file or "", location_token, evidence_signature or ""]
    )
    return f"{rule_id}:{hashlib.sha1(base.encode('utf-8')).hexdigest()[:12]}"


def _normalize_finding(raw: dict[str, Any]) -> dict[str, Any]:
    rule_id = raw["rule_id"]
    target_file = raw.get("target_file", "")
    target_location = raw.get("target_location")
    evidence_signature = raw.get("evidence_signature", "")
    return {
        "finding_id": _stable_finding_id(
            rule_id, target_file, target_location, evidence_signature
        ),
        "rule_id": rule_id,
        "severity": raw["severity"],
        "category": raw.get("category", "spec"),
        "actionable_by": list(raw.get("actionable_by", ["human"])),
        "remediation_hint": dict(
            raw.get("remediation_hint", {"kind": "human-decision"})
        ),
        "evidence": list(raw.get("evidence", [])),
        "message": raw["message"],
    }


def _build_envelope(
    stage: str, iteration_index: int, findings: list[dict]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "skill_source": "operator-validate",
        "stage": stage,
        "iteration_index": iteration_index,
        "findings": findings,
    }


def load_rule_modules(rules_dir: Path = RULES_DIR) -> list[tuple[dict, Callable]]:
    modules: list[tuple[dict, Callable]] = []
    if not rules_dir.exists():
        return modules
    for path in sorted(rules_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"sk_spec_rule_{path.stem}", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rule_meta = getattr(module, "RULE", None)
        check_fn = getattr(module, "check", None)
        if isinstance(rule_meta, dict) and callable(check_fn):
            modules.append((rule_meta, check_fn))
    return modules


def _collect_source_units(asset_path: Path) -> list[dict[str, Any]]:
    if asset_path.is_file():
        files = [asset_path]
        base = asset_path.parent
    else:
        files = sorted(
            p
            for p in asset_path.rglob("*")
            if p.is_file() and p.suffix in SOURCE_SUFFIXES
        )
        base = asset_path
    units: list[dict[str, Any]] = []
    for path in files:
        rel = (
            str(path.relative_to(base))
            if base in path.parents or path.parent == base
            else path.name
        )
        units.append(
            {"rel": rel, "text": path.read_text(encoding="utf-8", errors="replace")}
        )
    return units


def run_spec_rules(
    asset_path: Path, *, stage: str, iteration_index: int
) -> dict[str, Any]:
    asset_path = asset_path.resolve()
    if not asset_path.exists():
        raise CliUsageError(f"asset path not found: {asset_path}")

    units = _collect_source_units(asset_path)
    if not units:
        raise CliUsageError(f"no source files found under {asset_path}")

    rules = load_rule_modules(RULES_DIR)
    raw_findings: list[dict] = []
    for _meta, check_fn in rules:
        produced = check_fn(units) or []
        raw_findings.extend(produced)

    findings = [_normalize_finding(raw) for raw in raw_findings]
    return _build_envelope(
        stage=stage, iteration_index=iteration_index, findings=findings
    )


def list_spec_rules() -> dict[str, Any]:
    rules = load_rule_modules(RULES_DIR)
    return {
        "rules": [
            {
                "id": meta.get("id"),
                "severity": meta.get("severity"),
                "description": meta.get("description"),
            }
            for meta, _ in rules
        ]
    }
