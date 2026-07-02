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

from sk_codegen_lib import parse_global_entries


SCHEMA_VERSION = 1
SUPPORTED_BUILD_KINDS = {
    "sk-native-wheel",
    "sk-standalone",
    "cmake-project",
    "python-wheel",
    "external-command",
}
SUPPORTED_VERIFY_KINDS = {
    "command",
    "bind-target-on-wheel",
    "reference-impls-numpy",
    "standalone",
}
STATUS_VALUES = {"ready", "needs-human", "not-configured"}
TOOLCHAIN_INCLUDE_NAMES = {"kernel_operator.h", "tikicpulib.h"}
TOOLCHAIN_INCLUDE_PREFIXES = ("acl/", "register/", "tiling/", "kernel_tiling/")


class ContractError(ValueError):
    """Raised when an operator contract is not safe for core skill execution."""


def safe_slug(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value or "")).strip("._-")
    return slug or "item"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return payload


def _require_schema(payload: dict[str, Any], contract_name: str) -> None:
    if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ContractError(f"{contract_name} schema_version must be {SCHEMA_VERSION}")


def _asset_root(payload: dict[str, Any], base_dir: Path | None = None) -> Path:
    raw = str(payload.get("asset_root", ""))
    if not raw:
        raise ContractError("asset_root is required")
    root = Path(raw)
    if not root.is_absolute() and base_dir is not None:
        root = base_dir / root
    root = root.resolve()
    if not root.exists():
        raise ContractError(f"asset_root not found: {root}")
    return root


def _rel_path(value: object, *, field: str) -> Path:
    raw = str(value or "")
    if not raw:
        raise ContractError(f"{field} is required")
    path = Path(raw)
    if path.is_absolute():
        raise ContractError(f"{field} must be relative: {raw}")
    if ".." in path.parts:
        raise ContractError(f"{field} must not contain '..': {raw}")
    return path


def _check_existing_file(
    root: Path, value: object, *, field: str, required: bool = True
) -> None:
    if not value:
        if required:
            raise ContractError(f"{field} is required")
        return
    rel = _rel_path(value, field=field)
    if not (root / rel).is_file():
        raise ContractError(f"{field} not found: {rel.as_posix()}")


def _global_entry_names(root: Path, value: object, *, field: str) -> list[str]:
    rel = _rel_path(value, field=field)
    path = root / rel
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ContractError(f"{field} cannot be read: {rel.as_posix()}") from exc
    return [entry.name for entry in parse_global_entries(text)]


def _check_status(payload: dict[str, Any], *, contract_name: str) -> None:
    status = str(payload.get("status", ""))
    if status not in STATUS_VALUES:
        raise ContractError(
            f"{contract_name} status must be one of {sorted(STATUS_VALUES)}"
        )


def validate_asset_layout_contract(
    layout: dict[str, Any], base_dir: Path | None = None
) -> None:
    _require_schema(layout, "operator-asset-layout")
    _check_status(layout, contract_name="operator-asset-layout")
    root = _asset_root(layout, base_dir)
    root = root if root.is_dir() else root.parent
    seen: set[str] = set()
    for index, unit in enumerate(layout.get("operator_units", [])):
        if not isinstance(unit, dict):
            raise ContractError(f"operator_units[{index}] must be an object")
        unit_id = str(unit.get("unit_id", ""))
        if not unit_id:
            raise ContractError(f"operator_units[{index}].unit_id is required")
        if unit_id in seen:
            raise ContractError(f"duplicate operator unit_id: {unit_id}")
        seen.add(unit_id)
        entry_name = str(unit.get("entry_name") or "")
        if not entry_name:
            raise ContractError(f"operator unit {unit_id} entry_name is required")
        _check_existing_file(
            root,
            unit.get("kernel_source"),
            field=f"operator unit {unit_id} kernel_source",
        )
        entry_names = _global_entry_names(
            root,
            unit.get("kernel_source"),
            field=f"operator unit {unit_id} kernel_source",
        )
        if entry_name not in entry_names:
            raise ContractError(
                f"operator unit {unit_id} entry_name {entry_name} is not defined in kernel_source"
            )
        _check_existing_file(
            root,
            unit.get("host_source"),
            field=f"operator unit {unit_id} host_source",
            required=False,
        )
        _check_existing_file(
            root,
            unit.get("json_spec"),
            field=f"operator unit {unit_id} json_spec",
            required=False,
        )
        for key in ("tiling_headers", "support_files"):
            values = unit.get(key, []) or []
            if not isinstance(values, list):
                raise ContractError(f"operator unit {unit_id} {key} must be a list")
            for value in values:
                _check_existing_file(
                    root, value, field=f"operator unit {unit_id} {key}"
                )


def _list_target_values(raw: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [item.strip() for item in re.split(r"[,;]", raw) if item.strip()]
    return [str(item).strip() for item in raw if str(item).strip()]


def _unique_sorted(values: list[str]) -> list[str]:
    return sorted(dict.fromkeys(values))


def _inventory_unresolved_includes(
    inventory: dict[str, Any] | None,
) -> list[dict[str, str]]:
    unresolved: list[dict[str, str]] = []
    if not inventory:
        return unresolved
    for item in inventory.get("files", []):
        for include in item.get("unresolved_includes", []) or []:
            unresolved.append(
                {"source": str(item.get("path", "")), "include": str(include)}
            )
    return unresolved


def _is_toolchain_include(include: str) -> bool:
    return include in TOOLCHAIN_INCLUDE_NAMES or include.startswith(
        TOOLCHAIN_INCLUDE_PREFIXES
    )


def build_default_build_context(
    layout: dict[str, Any],
    inventory: dict[str, Any] | None = None,
    *,
    target_chips: str | list[str] | None = None,
    target_arches: str | list[str] | None = None,
) -> dict[str, Any]:
    include_dirs: set[str] = set()
    unit_ids: list[str] = []
    for unit in layout.get("operator_units", []) or []:
        unit_ids.append(str(unit.get("unit_id", "")))
        for key in ("kernel_source", "host_source", "json_spec"):
            value = unit.get(key)
            if value:
                parent = Path(str(value)).parent.as_posix()
                if parent != ".":
                    include_dirs.add(parent)
        for key in ("tiling_headers", "support_files"):
            for value in unit.get(key, []) or []:
                parent = Path(str(value)).parent.as_posix()
                if parent != ".":
                    include_dirs.add(parent)
    all_unresolved_includes = _inventory_unresolved_includes(inventory)
    unresolved_includes = [
        item
        for item in all_unresolved_includes
        if not _is_toolchain_include(item["include"])
    ]
    toolchain_includes = [
        item
        for item in all_unresolved_includes
        if _is_toolchain_include(item["include"])
    ]
    questions: list[dict[str, Any]] = []
    if layout.get("status") != "ready":
        questions.append(
            {
                "id": "layout-required",
                "message": "operator asset layout must be ready before build context can be ready",
            }
        )
    if unresolved_includes:
        questions.append(
            {
                "id": "unresolved-include",
                "message": "some quoted includes could not be resolved inside the asset root",
            }
        )
    status = (
        "ready"
        if layout.get("status") == "ready" and not unresolved_includes and unit_ids
        else "needs-human"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "context_source": "default-asset-adapter",
        "asset_root": str(Path(str(layout["asset_root"])).resolve()),
        "target": {
            "chips": _list_target_values(target_chips),
            "arches": _list_target_values(target_arches),
        },
        "include_dirs": _unique_sorted(list(include_dirs)),
        "system_include_dirs": [],
        "compile_definitions": [],
        "compile_options": [],
        "link_dirs": [],
        "link_libraries": [],
        "env_requirements": [{"name": "ASCEND_HOME_PATH", "required": False}],
        "external_dependencies": [
            {
                "kind": "ascend-cann-toolchain",
                "include_count": len(toolchain_includes),
                "env": ["ASCEND_HOME_PATH"],
            }
        ]
        if toolchain_includes
        else [],
        "unresolved_includes": unresolved_includes,
        "build_flows": [
            {
                "name": "sk-native-wheel",
                "kind": "sk-native-wheel",
                "unit_ids": unit_ids,
                "target_arches": _list_target_values(target_arches),
                "expected_outputs": ["dist/*.whl"],
                "jobs": "auto",
            }
        ]
        if unit_ids
        else [],
        "baseline_flows": [],
        "human_questions": questions,
    }


def build_default_verify_context(layout: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "not-configured",
        "context_source": "default-asset-adapter",
        "asset_root": str(Path(str(layout["asset_root"])).resolve()),
        "verify_flows": [],
        "human_questions": [
            {
                "id": "verify-context-required",
                "message": "verification is not configured; core pipeline must not report equivalence as passed",
            }
        ],
    }


def validate_build_context_contract(
    build_context: dict[str, Any], base_dir: Path | None = None
) -> None:
    _require_schema(build_context, "operator-build-context")
    _check_status(build_context, contract_name="operator-build-context")
    _asset_root(build_context, base_dir)
    for flow in build_context.get("build_flows", []) or []:
        if not isinstance(flow, dict):
            raise ContractError("build_flows entries must be objects")
        kind = str(flow.get("kind", ""))
        name = str(flow.get("name", ""))
        if not name:
            raise ContractError("build flow name is required")
        if kind not in SUPPORTED_BUILD_KINDS:
            raise ContractError(f"unsupported build flow kind: {kind}")
        expected_outputs = flow.get("expected_outputs", [])
        if not isinstance(expected_outputs, list) or not expected_outputs:
            raise ContractError(
                f"build flow {name} expected_outputs must be a non-empty list"
            )
        if kind == "external-command":
            command = flow.get("command", [])
            if not isinstance(command, list) or not command:
                raise ContractError(
                    f"external-command build flow {name} requires command list"
                )


def validate_verify_context_contract(
    verify_context: dict[str, Any], base_dir: Path | None = None
) -> None:
    _require_schema(verify_context, "operator-verify-context")
    _check_status(verify_context, contract_name="operator-verify-context")
    _asset_root(verify_context, base_dir)
    for flow in verify_context.get("verify_flows", []) or []:
        if not isinstance(flow, dict):
            raise ContractError("verify_flows entries must be objects")
        kind = str(flow.get("kind", ""))
        name = str(flow.get("name", ""))
        if not name:
            raise ContractError("verify flow name is required")
        if kind not in SUPPORTED_VERIFY_KINDS:
            raise ContractError(f"unsupported verify flow kind: {kind}")


def build_adapter_report(
    *,
    asset_root: Path,
    inventory: dict[str, Any],
    layout: dict[str, Any],
    build_context: dict[str, Any],
    verify_context: dict[str, Any],
) -> dict[str, Any]:
    layout_ready = layout.get("status") == "ready"
    build_ready = build_context.get("status") == "ready"
    verify_ready = verify_context.get("status") == "ready"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready" if layout_ready else "needs-human",
        "adapter_source": "default-asset-adapter",
        "asset_root": str(asset_root.resolve()),
        "readiness": {
            "layout": layout.get("status"),
            "build": build_context.get("status"),
            "verify": verify_context.get("status"),
            "can_adapt": layout_ready,
            "can_build": layout_ready and build_ready,
            "can_claim_equivalence": layout_ready and build_ready and verify_ready,
        },
        "counts": {
            "files": len(inventory.get("files", [])),
            "operator_units": len(layout.get("operator_units", [])),
            "skipped_items": len(layout.get("skipped_items", [])),
            "unresolved_includes": len(build_context.get("unresolved_includes", [])),
        },
        "outputs": {
            "inventory": "operator-asset-inventory.json",
            "layout": "operator-asset-layout.json",
            "build_context": "operator-build-context.json",
            "verify_context": "operator-verify-context.json",
        },
        "human_questions": [
            *list(layout.get("human_questions", []) or []),
            *list(build_context.get("human_questions", []) or []),
            *list(verify_context.get("human_questions", []) or []),
        ],
    }


def validate_adapter_report_contract(
    report: dict[str, Any], base_dir: Path | None = None
) -> None:
    _require_schema(report, "adapter-report")
    _check_status(report, contract_name="adapter-report")
    _asset_root(report, base_dir)
    readiness = report.get("readiness", {})
    if not isinstance(readiness, dict):
        raise ContractError("adapter-report readiness must be an object")


def contract_finding(
    rule_id: str, message: str, *, severity: str = "blocker"
) -> dict[str, Any]:
    return {
        "finding_id": f"{rule_id}:{safe_slug(message)[:48]}",
        "rule_id": rule_id,
        "severity": severity,
        "category": "contract",
        "actionable_by": ["adapter"],
        "remediation_hint": {"kind": "contract-edit"},
        "evidence": [message],
        "message": message,
    }
