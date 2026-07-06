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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OperatorUnit:
    unit_id: str
    entry_name: str
    display_name: str
    kernel_source: str | None
    host_source: str | None
    json_spec: str | None
    tiling_headers: list[str]
    support_files: list[str]
    features: dict[str, Any]
    supported_soc_versions: list[str]
    supported_arches: list[str]
    target_resolution: dict[str, Any]
    support_source: str
    build_backends: list[str]
    runtime_contract_status: str
    human_questions: list[dict[str, Any]]
    normalized_asset: str | None = None
    source_asset: str | None = None


@dataclass(frozen=True)
class AssetUnderstanding:
    schema_version: int
    status: str
    asset_root: str
    asset_kind: str
    operator_units: list[OperatorUnit]
    unsupported_items: list[dict[str, Any]]
    warnings: list[dict[str, Any]]


def understanding_to_dict(manifest: AssetUnderstanding) -> dict[str, Any]:
    return asdict(manifest)


def write_understanding(path: Path, manifest: AssetUnderstanding) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(understanding_to_dict(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_understanding(path: Path) -> AssetUnderstanding:
    payload = json.loads(path.read_text(encoding="utf-8"))
    units = [
        OperatorUnit(
            **{
                "supported_soc_versions": [],
                "supported_arches": [],
                "target_resolution": {},
                "support_source": "unknown",
                **item,
            }
        )
        for item in payload.get("operator_units", [])
    ]
    return AssetUnderstanding(
        schema_version=int(payload["schema_version"]),
        status=str(payload["status"]),
        asset_root=str(payload["asset_root"]),
        asset_kind=str(payload["asset_kind"]),
        operator_units=units,
        unsupported_items=list(payload.get("unsupported_items", [])),
        warnings=list(payload.get("warnings", [])),
    )
