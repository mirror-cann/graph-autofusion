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
from pathlib import Path
from typing import Any


def _kind_from_c_type(c_type: str) -> str:
    if "GM_ADDR" in c_type or "*" in c_type or "__gm__" in c_type:
        return "device_buffer"
    return "scalar"


def _fixture_payload(fixture_status: dict[str, Any]) -> dict[str, Any] | None:
    raw_path = fixture_status.get("path")
    if not raw_path:
        return None
    path = Path(raw_path) / "operator-sk-runtime-fixture.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _parameters_from_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    parameters: list[dict[str, Any]] = []
    for index, param in enumerate(entry.get("parameters", [])):
        c_type = str(param.get("c_type", ""))
        parameters.append(
            {
                "index": index,
                "name": param.get("name", f"arg{index}"),
                "kind": _kind_from_c_type(c_type),
                "source_type": c_type,
                "compare": False,
                "source": "entry-signature",
            }
        )
    return parameters


def _parameters_from_fixture(
    entry: dict[str, Any], fixture: dict[str, Any]
) -> list[dict[str, Any]]:
    by_name = {
        str(item.get("name")): item
        for item in fixture.get("parameters", [])
        if isinstance(item, dict) and item.get("name")
    }
    parameters: list[dict[str, Any]] = []
    for param in _parameters_from_entry(entry):
        fixture_param = by_name.get(str(param["name"]), {})
        merged = {
            **param,
            "source": "runtime-fixture" if fixture_param else param["source"],
        }
        for key in ("kind", "dtype", "shape", "bytes", "fill", "value", "compare"):
            if key in fixture_param:
                merged[key] = fixture_param[key]
        parameters.append(merged)
    return parameters


def build_runtime_contract(
    output_dir: Path,
    *,
    entry: dict[str, Any],
    fixture_status: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = _fixture_payload(fixture_status)
    parameters = (
        _parameters_from_fixture(entry, fixture)
        if fixture
        else _parameters_from_entry(entry)
    )
    compare_outputs = [param["name"] for param in parameters if param.get("compare")]
    if fixture_status.get("device_runnable") is True and compare_outputs:
        status = "available"
        reason = "runtime fixture declares runnable inputs and comparable outputs"
    elif fixture_status.get("device_runnable") is True:
        status = "needs-user-confirmation"
        reason = "runtime fixture is runnable but comparable outputs are not declared"
    else:
        status = "needs-runtime-fixture"
        reason = fixture_status.get("reason", "runtime fixture not available")
    contract = {
        "schema_version": 1,
        "status": status,
        "reason": reason,
        "entry_name": entry["entry_name"],
        "source_file": entry.get("source_file", ""),
        "parameters": parameters,
        "comparison": {
            "mode": "bytewise",
            "outputs": compare_outputs,
            "tolerance": None,
            "status": "available" if compare_outputs else "needs-user-confirmation",
        },
        "fixture": {
            "path": fixture_status.get("path", ""),
            "status": fixture_status.get("status", "unknown"),
            "device_runnable": fixture_status.get("device_runnable"),
        },
    }
    (output_dir / "operator-runtime-contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    return contract
