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

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def safe_public_name(value: object, fallback: str = "op") -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", str(value or "").strip()).strip("_").lower()
    if not name:
        name = fallback
    if name[0].isdigit():
        name = f"op_{name}"
    return name


def _short_hash(*parts: object) -> str:
    text = "\n".join(str(part or "") for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _asset_namespace(root: Path | None, asset_path: Path) -> str:
    try:
        raw = "__".join(asset_path.resolve().relative_to(root.resolve()).parts) if root is not None else asset_path.name
    except ValueError:
        raw = asset_path.name
    if asset_path.is_file():
        raw = Path(raw).stem
    return safe_public_name(raw, "asset")


def _asset_key(root: Path | None, asset_path: Path) -> str:
    try:
        return asset_path.resolve().relative_to(root.resolve()).as_posix() if root is not None else asset_path.name
    except ValueError:
        return asset_path.name


def build_name_resolution(
    *,
    assets: list[dict[str, Any]],
    root: Path | None = None,
    policy: str = "reject",
) -> dict[str, Any]:
    if policy not in {"reject", "namespace"}:
        raise ValueError(f"unsupported duplicate entry policy: {policy}")
    entries: list[dict[str, Any]] = []
    for asset in assets:
        asset_path = Path(asset["path"]).resolve()
        namespace = safe_public_name(asset.get("namespace") or _asset_namespace(root, asset_path), "asset")
        for entry in asset.get("entries", []):
            source_entry = str(entry)
            entries.append(
                {
                    "asset_path": str(asset_path),
                    "asset_key": _asset_key(root, asset_path),
                    "asset_namespace": namespace,
                    "source_entry_name": source_entry,
                }
            )
    by_entry: dict[str, list[dict[str, Any]]] = {}
    for item in entries:
        by_entry.setdefault(item["source_entry_name"], []).append(item)
    duplicate_groups = {entry: items for entry, items in by_entry.items() if len(items) > 1}
    used_public_names: dict[str, dict[str, Any]] = {}
    resolutions: list[dict[str, Any]] = []
    for item in entries:
        source_entry = item["source_entry_name"]
        duplicate = source_entry in duplicate_groups
        if duplicate and policy == "namespace":
            base_name = safe_public_name(f"{item['asset_namespace']}__{source_entry}", "op")
        else:
            base_name = source_entry
        public_name = base_name
        previous = used_public_names.get(public_name)
        public_name_collision = False
        previous_is_different_asset = previous is not None and (
            previous["asset_path"] != item["asset_path"] or previous["source_entry_name"] != source_entry
        )
        if policy == "namespace" and previous_is_different_asset:
            public_name_collision = True
            public_name = f"{base_name}__{_short_hash(item['asset_namespace'], source_entry, item['asset_key'])}"
        used_public_names[public_name] = item
        renamed = public_name != source_entry
        rename_reason = ""
        if duplicate and renamed:
            rename_reason = "duplicate-entry"
        elif public_name_collision and renamed:
            rename_reason = "public-name-collision"
        resolutions.append(
            {
                "asset_path": item["asset_path"],
                "asset_key": item["asset_key"],
                "asset_namespace": item["asset_namespace"],
                "source_entry_name": source_entry,
                "public_entry_name": public_name,
                "internal_symbol_name": safe_public_name(public_name, "op"),
                "bind_target": source_entry,
                "renamed": renamed,
                "rename_reason": rename_reason,
                "collision_group": source_entry if duplicate else "",
            }
        )
    return {
        "schema_version": 1,
        "policy": policy,
        "duplicate_entry_groups": {
            entry: [
                {
                    "asset_path": item["asset_path"],
                    "asset_key": item["asset_key"],
                    "asset_namespace": item["asset_namespace"],
                    "source_entry_name": item["source_entry_name"],
                }
                for item in items
            ]
            for entry, items in sorted(duplicate_groups.items())
        },
        "renamed_entry_count": sum(1 for item in resolutions if item["renamed"]),
        "resolutions": resolutions,
    }


def write_name_resolution(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_name_resolution(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def public_name_resolution_payload(payload: dict[str, Any]) -> dict[str, Any]:
    public_payload = json.loads(json.dumps(payload))
    for items in public_payload.get("duplicate_entry_groups", {}).values():
        for item in items:
            if item.get("asset_key"):
                item["asset_path"] = item["asset_key"]
    for item in public_payload.get("resolutions", []):
        if item.get("asset_key"):
            item["asset_path"] = item["asset_key"]
    return public_payload


def resolution_by_asset_and_entry(
    payload: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for item in payload.get("resolutions", []):
        index[(str(Path(item["asset_path"]).resolve()), str(item["source_entry_name"]))] = dict(item)
    return index
