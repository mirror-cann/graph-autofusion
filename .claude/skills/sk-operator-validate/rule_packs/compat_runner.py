# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Compatibility rule-pack runner for sk-operator-validate."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlparse

SOURCE_SUFFIXES = (".asc", ".cpp", ".cc", ".cxx", ".h", ".hpp")
MATRICES_DIR = Path(__file__).resolve().parent / "compat" / "matrices"
SCHEMA_VERSION = 1

_ACL_API_RE = re.compile(r"\bacl[A-Za-z][A-Za-z0-9]+\b")
_MIX_RE = re.compile(r"__mix__\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)")
_ARCH_MARKER_RE = re.compile(r"//\s*sk-arch:\s*(?P<alias>[\w-]+)")
_VERSION_NUMBER_RE = re.compile(r"\d+")
_OFFICIAL_SOURCE_HOSTS = frozenset({"www.hiascend.com", "hiascend.com", "support.huawei.com"})


class CliUsageError(Exception):
    pass


# ---------- embedded finding builder ----------


class FindingInput(NamedTuple):
    rule_id: str
    severity: str
    actionable_by: list[str]
    remediation_hint: dict[str, Any]
    message: str
    target_file: str = ""
    evidence_signature: str = ""
    evidence: list[Any] | None = None


def _stable_finding_id(rule_id: str, target_file: str, evidence_signature: str) -> str:
    base = "|".join([rule_id, target_file or "", evidence_signature or ""])
    return f"{rule_id}:{hashlib.sha1(base.encode('utf-8')).hexdigest()[:12]}"


def _finding(finding_input: FindingInput):
    return {
        "finding_id": _stable_finding_id(
            finding_input.rule_id,
            finding_input.target_file,
            finding_input.evidence_signature,
        ),
        "rule_id": finding_input.rule_id,
        "severity": finding_input.severity,
        "category": "compat",
        "actionable_by": list(finding_input.actionable_by),
        "remediation_hint": dict(finding_input.remediation_hint),
        "evidence": list(finding_input.evidence or []),
        "message": finding_input.message,
    }


def _envelope(stage, iteration_index, findings):
    return {
        "schema_version": SCHEMA_VERSION,
        "skill_source": "operator-validate",
        "stage": stage,
        "iteration_index": iteration_index,
        "findings": findings,
    }


def _load_matrix() -> dict[str, dict]:
    import yaml  # pyyaml is available in the toolchain env

    chips: dict[str, dict] = {}
    matrices_dir = Path(os.getenv("SK_COMPAT_MATRIX_DIR", str(MATRICES_DIR)))
    if not matrices_dir.exists():
        return chips
    for path in sorted(matrices_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for entry in data.get("chip_families", []):
            _validate_official_sources(entry, path)
            chips[entry["id"]] = entry
    return chips


def _is_official_source_url(value: str) -> bool:
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and host in _OFFICIAL_SOURCE_HOSTS


def _validate_official_sources(chip: dict, matrix_path: Path) -> None:
    sources = chip.get("official_sources", {})
    if not sources:
        return
    if not isinstance(sources, dict):
        raise CliUsageError(
            f"{matrix_path}: official_sources for {chip.get('id')!r} must be a mapping of field -> official HTTPS URL"
        )
    for field, source in sources.items():
        if not isinstance(source, str) or not _is_official_source_url(source):
            raise CliUsageError(
                f"{matrix_path}: official_sources.{field} for {chip.get('id')!r} must be an official HTTPS URL "
                "(for example https://www.hiascend.com/document/detail/...). Local paths are not allowed."
            )


def _field_source(chip: dict, field: str) -> str:
    sources = chip.get("official_sources", {})
    if not isinstance(sources, dict):
        return ""
    source = sources.get(field, "")
    return source if isinstance(source, str) else ""


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _version_key(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _VERSION_NUMBER_RE.findall(value))


def _compare_version_keys(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    width = max(len(left), len(right))
    padded_left = left + (0,) * (width - len(left))
    padded_right = right + (0,) * (width - len(right))
    if padded_left < padded_right:
        return -1
    if padded_left > padded_right:
        return 1
    return 0


def _version_in_range(version: str, scope: dict) -> bool:
    version_key = _version_key(version)
    if not version_key:
        return False
    min_version = scope.get("min") or scope.get("from")
    max_version = scope.get("max") or scope.get("to")
    if min_version and _compare_version_keys(version_key, _version_key(str(min_version))) < 0:
        return False
    if max_version and _compare_version_keys(version_key, _version_key(str(max_version))) > 0:
        return False
    return bool(min_version or max_version)


def _cann_scope_applies(scope: Any, target_cann: str) -> bool:
    if scope in ("all", "*"):
        return True
    if isinstance(scope, dict):
        if scope.get("all") is True:
            return True
        if "versions" in scope and target_cann in _string_list(scope.get("versions")):
            return True
        return _version_in_range(target_cann, scope)
    return target_cann in _string_list(scope)


def _field_cann_scope(chip: dict, field: str) -> Any:
    scopes = chip.get("fact_scopes", {})
    field_scope = scopes.get(field, {}) if isinstance(scopes, dict) else {}
    if isinstance(field_scope, dict) and "cann" in field_scope:
        return field_scope["cann"]
    if isinstance(field_scope, dict) and "cann_scope" in field_scope:
        return field_scope["cann_scope"]
    if isinstance(field_scope, (str, list, tuple, set)):
        return field_scope
    # Legacy/conservative default: a source-backed fact without its own CANN
    # scope applies only to recorded contexts, never to future CANN releases.
    return chip.get("cann_versions", [])


def _field_applies_to_target_cann(chip: dict, field: str, target_cann: str) -> bool:
    scope = _field_cann_scope(chip, field)
    if not target_cann:
        return scope in ("all", "*") or (isinstance(scope, dict) and scope.get("all") is True)
    return _cann_scope_applies(scope, target_cann)


def _rule_id_for_field(field: str) -> str:
    return {
        "arch_alias": "compat.arch-alias-mismatch",
        "forbidden_apis": "compat.forbidden-acl-api",
        "supports_mix_aic_aiv": "compat.mix-aic-aiv-unsupported",
    }.get(field, f"compat.{field}")


def _collect_sources(asset_path: Path) -> list[tuple[str, str]]:
    if asset_path.is_file():
        return [(asset_path.name, asset_path.read_text(encoding="utf-8", errors="replace"))]
    base = asset_path
    units = []
    for path in sorted(p for p in asset_path.rglob("*") if p.is_file() and p.suffix in SOURCE_SUFFIXES):
        units.append(
            (
                str(path.relative_to(base)),
                path.read_text(encoding="utf-8", errors="replace"),
            )
        )
    return units


def run_compat_rules(
    asset_path: Path,
    *,
    target_chip: str,
    target_cann: str,
    stage: str,
    iteration_index: int,
) -> dict[str, Any]:
    asset_path = asset_path.resolve()
    if not asset_path.exists():
        raise CliUsageError(f"asset path not found: {asset_path}")

    chips = _load_matrix()
    chip = chips.get(target_chip, {})

    units = _collect_sources(asset_path)
    if not units:
        raise CliUsageError(f"no source files found under {asset_path}")

    recorded_cann_versions = chip.get("cann_versions", [])
    target_chip_known = bool(target_chip and chip)
    source_backed_fields = {
        field for field in ("arch_alias", "forbidden_apis", "supports_mix_aic_aiv") if _field_source(chip, field)
    }
    applicable_source_backed_fields = {
        field for field in source_backed_fields if _field_applies_to_target_cann(chip, field, target_cann)
    }
    arch_alias = chip.get("arch_alias", "") if "arch_alias" in applicable_source_backed_fields else ""
    has_applicable_source_backed_checks = bool(applicable_source_backed_fields)

    findings: list[dict] = []
    for rel, text in units:
        if "forbidden_apis" in applicable_source_backed_fields:
            forbidden = set(chip.get("forbidden_apis", []))
            for api in sorted(set(_ACL_API_RE.findall(text))):
                if api in forbidden:
                    findings.append(
                        _finding(
                            FindingInput(
                                "compat.forbidden-acl-api",
                                "blocker",
                                ["human"],
                                {"kind": "human-decision"},
                                (
                                    f"ACL API {api!r} is unsupported on {target_chip} "
                                    "according to the recorded official source."
                                ),
                                target_file=rel,
                                evidence_signature=f"{target_chip}:{api}",
                                evidence=[api],
                            ),
                        )
                    )
        if "supports_mix_aic_aiv" in applicable_source_backed_fields and chip.get("supports_mix_aic_aiv") is False:
            for m in _MIX_RE.finditer(text):
                c, v = int(m.group(1)), int(m.group(2))
                if c > 0 and v > 0:
                    findings.append(
                        _finding(
                            FindingInput(
                                "compat.mix-aic-aiv-unsupported",
                                "blocker",
                                ["human"],
                                {"kind": "human-decision"},
                                (
                                    f"kernel uses __mix__({c},{v}) (mixed AIC/AIV), "
                                    f"which is unsupported on {target_chip} according "
                                    "to the recorded official source."
                                ),
                                target_file=rel,
                                evidence_signature=f"{target_chip}:mix:{c}:{v}",
                            ),
                        )
                    )
        # Arch marker mismatch is actionable only when arch_alias has an
        # official source in the matrix. Otherwise preserve the marker.
        marker = _ARCH_MARKER_RE.search(text)
        if arch_alias and marker and marker.group("alias") != arch_alias:
            findings.append(
                _finding(
                    FindingInput(
                        "compat.arch-alias-mismatch",
                        "warning",
                        ["codegen.apply-remediation"],
                        {
                            "kind": "replace-pattern",
                            "target_file": rel,
                            "old_value": f"sk-arch: {marker.group('alias')}",
                            "new_value": f"sk-arch: {arch_alias}",
                        },
                        (
                            f"declared arch {marker.group('alias')!r} disagrees "
                            f"with target chip arch_alias {arch_alias!r}."
                        ),
                        target_file=rel,
                        evidence_signature=f"arch:{marker.group('alias')}->{arch_alias}",
                    ),
                )
            )

    envelope = _envelope(stage, iteration_index, findings)
    # Derive an overall verdict from severities for human consumption.
    if any(f["severity"] == "blocker" for f in findings):
        overall = "incompatible"
    elif findings:
        overall = "requires-adaptation"
    else:
        overall = "not-verified"
    if not chip:
        verification_note = f"target chip {target_chip!r} is not recorded in the compatibility declarations"
    elif not source_backed_fields:
        verification_note = "target chip is recorded, but no official-source-backed capability checks were declared"
    elif not target_cann and not has_applicable_source_backed_checks:
        verification_note = (
            "target CANN was not provided, so CANN-scoped official-source-backed compatibility checks were not applied"
        )
    elif not has_applicable_source_backed_checks:
        verification_note = (
            f"no official-source-backed checks were applicable to target CANN {target_cann!r}; "
            f"recorded CANN versions are informational only: {recorded_cann_versions}"
        )
    elif not findings:
        verification_note = (
            "official-source-backed checks applicable to the target context found no findings; "
            "full compatibility is not asserted by this skill"
        )
    else:
        verification_note = "official-source-backed checks produced findings for the target context"
    envelope["metadata"] = {
        "rule_pack": "compat",
        "overall_verdict": overall,
        "target_chip": target_chip,
        "target_cann": target_cann,
        "target_chip_known": target_chip_known,
        "recorded_cann_versions": list(recorded_cann_versions),
        "source_backed_fields": sorted(source_backed_fields),
        "applicable_source_backed_fields": sorted(applicable_source_backed_fields),
        "checked_rules": sorted(_rule_id_for_field(field) for field in applicable_source_backed_fields),
        "verification": verification_note,
    }
    return envelope


def list_compat_targets() -> dict[str, Any]:
    chips = _load_matrix()
    return {"chips": [{"id": cid, "cann_versions": c.get("cann_versions", [])} for cid, c in sorted(chips.items())]}
