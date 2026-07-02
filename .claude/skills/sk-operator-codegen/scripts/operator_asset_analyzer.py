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
import shutil
from pathlib import Path
from typing import Any

from operator_asset_contract import (
    AssetUnderstanding,
    OperatorUnit,
    write_understanding,
)
from operator_asset_layout import (
    build_inventory,
    build_layout_from_inventory,
    read_layout,
    validate_layout,
)
from operator_target_arch import (
    build_target_resolution,
    extract_supported_soc_versions_from_cmake_presets,
    extract_supported_soc_versions_from_misc_text,
    extract_supported_soc_versions_from_text,
    supported_arches_for_soc_versions,
)

SOURCE_SUFFIXES = {".asc", ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
KERNEL_SOURCE_SUFFIXES = {".asc", ".cpp", ".cc", ".cxx"}
SKIP_DIRS = {
    ".git",
    "__pycache__",
    "build",
    "cmake-build-debug",
    "dist",
    "out",
    "output",
    "outputs",
    "run",
    "runs",
    "prof",
    "prof_total",
    "profiling",
    "cache",
    "kernel_meta",
    "kernel_meta_temp",
    "sk_meta",
    "static_kernel_compile_outputs",
    "tmp",
}
GLOBAL_ENTRY_RE = re.compile(
    r"(?:template\s*<[^;{}]*>\s*)?"
    r"(?:extern\s+\"C\"\s+)?"
    r"(?P<qualifiers>(?:__global__|__aicore__|__vector__|__cube__|__mix__(?:\s*\([^)]*\))?|inline|__inline__|\s)+?)"
    r"\s+void\s+(?P<name>[A-Za-z_]\w*)\s*\(",
    re.MULTILINE | re.DOTALL,
)
ANY_KERNEL_ENTRY_RE = re.compile(
    r"(?:extern\s+\"C\"\s+)?(?:(?:__global__|__spk__|__sk__)[\w\s_()*,:&<>]*\s+void\s+)([A-Za-z_]\w*)\s*\(",
    re.MULTILINE,
)
OP_ADD_RE = re.compile(r"\bOP_ADD\s*\(\s*([A-Za-z_]\w*)\s*\)")
QUOTED_INCLUDE_RE = re.compile(r"^\s*#\s*include\s+\"([^\"]+)\"", re.MULTILINE)


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _runtime_wrapper_support_files(root: Path, entry_name: str) -> list[str]:
    contract_path = root / "operator-io-contract.json"
    if not contract_path.is_file():
        return []
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return []
    entry_payload = entries.get(entry_name)
    if not isinstance(entry_payload, dict):
        return []
    runtime_wrapper = entry_payload.get("runtime_wrapper")
    if not isinstance(runtime_wrapper, dict):
        return []
    source = str(runtime_wrapper.get("source") or "").strip()
    if not source:
        return []
    source_path = Path(source)
    if source_path.is_absolute() or ".." in source_path.parts:
        return []
    resolved = (root / source_path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return []
    return [_rel(resolved, root)] if resolved.is_file() else []


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip("._-")
    return slug or "op"


def _iter_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS for part in rel_parts[:-1]):
            continue
        files.append(path)
    return files


def _source_files(root: Path) -> list[Path]:
    return [path for path in _iter_files(root) if path.suffix in SOURCE_SUFFIXES]


def _global_entries(path: Path) -> list[str]:
    text = _safe_read(path)
    entries: list[str] = []
    seen: set[str] = set()
    for match in GLOBAL_ENTRY_RE.finditer(text):
        qualifiers = " ".join(match.group("qualifiers").split())
        if "__global__" not in qualifiers:
            continue
        name = match.group("name")
        if name in seen:
            continue
        seen.add(name)
        entries.append(name)
    return entries


def _any_kernel_entries(path: Path) -> list[str]:
    text = _safe_read(path)
    entries: list[str] = []
    seen: set[str] = set()
    for match in ANY_KERNEL_ENTRY_RE.finditer(text):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            entries.append(name)
    return entries


def _display_name_from_host(path: Path | None, fallback: str) -> str:
    if path is not None and path.is_file():
        match = OP_ADD_RE.search(_safe_read(path))
        if match:
            return match.group(1)
    return (
        "".join(part.capitalize() for part in re.split(r"[_\-.]+", fallback) if part)
        or fallback
    )


def _json_for_stem(root: Path, stem: str) -> str | None:
    for candidate in (root / f"{stem}.json", root / "op_dev" / f"{stem}.json"):
        if candidate.is_file():
            return _rel(candidate, root)
    matches = sorted(root.glob(f"**/{stem}.json"))
    return _rel(matches[0], root) if matches else None


def _tiling_headers_for_stem(root: Path, stem: str, host_dir: Path | None) -> list[str]:
    headers: list[Path] = []
    search_roots = [host_dir] if host_dir is not None and host_dir.is_dir() else []
    search_roots.append(root)
    patterns = [
        f"{stem}_tiling.h",
        f"{stem}_tiling.hpp",
        f"tiling_key_{stem}.h",
        f"tiling_key_{stem}.hpp",
    ]
    for search_root in search_roots:
        for pattern in patterns:
            headers.extend(sorted(search_root.glob(pattern)))
            headers.extend(sorted(search_root.glob(f"**/{pattern}")))
    seen: set[str] = set()
    result: list[str] = []
    for header in headers:
        if not header.is_file():
            continue
        rel = _rel(header, root)
        if rel not in seen:
            seen.add(rel)
            result.append(rel)
    return result


def _features_for_sources(kernel_text: str, host_text: str) -> dict[str, Any]:
    joined = kernel_text + "\n" + host_text
    return {
        "has_global": "__global__" in kernel_text,
        "has_host_registration": "OP_ADD" in host_text,
        "has_tiling_func": "TilingFunc" in host_text or "SetTiling" in host_text,
        "has_template_global": bool(
            re.search(
                r"template\s*<[^;{}]*>\s*(?:extern\s+\"C\"\s+)?__global__",
                kernel_text,
                re.DOTALL,
            )
        ),
        "has_tiling_key": "TILING_KEY" in joined or "tiling_key_" in joined,
        "has_native_dtype_template": "native_dtype" in joined.lower(),
        "has_native_format_template": "native_format" in joined.lower(),
        "uses_tpipe": "TPipe" in joined,
        "uses_system_args": "systemArgs" in joined or "SkSystemArgs" in joined,
        "uses_workspace": "workspace" in joined,
    }


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.strip().lower().replace("_", "").replace("-", "")
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _soc_versions_for_sources(
    root: Path, kernel_source: Path | None, host_source: Path | None
) -> tuple[list[str], str]:
    kernel_text = _safe_read(kernel_source) if kernel_source is not None else ""
    host_text = _safe_read(host_source) if host_source is not None else ""
    soc_versions = _dedupe(
        extract_supported_soc_versions_from_text(host_text)
        + extract_supported_soc_versions_from_text(kernel_text)
    )
    if soc_versions:
        return soc_versions, "source-add-config"
    preset_values: list[str] = []
    search_root = root if root.is_dir() else root.parent
    for preset in sorted(search_root.glob("**/CMakePresets.json")):
        skip_preset = False
        for part in preset.relative_to(search_root).parts[:-1]:
            if part in SKIP_DIRS:
                skip_preset = True
                break
        if skip_preset:
            continue
        preset_values.extend(extract_supported_soc_versions_from_cmake_presets(preset))
    soc_versions = _dedupe(preset_values)
    if soc_versions:
        return soc_versions, "cmake-presets"
    misc_values: list[str] = []
    for name in ("run.sh", "*.yaml", "*.yml"):
        for path in sorted(search_root.glob(f"**/{name}")):
            skip_path = False
            for part in path.relative_to(search_root).parts[:-1]:
                if part in SKIP_DIRS:
                    skip_path = True
                    break
            if skip_path:
                continue
            misc_values.extend(
                extract_supported_soc_versions_from_misc_text(_safe_read(path))
            )
    soc_versions = _dedupe(misc_values)
    if soc_versions:
        return soc_versions, "runtime-scripts"
    return [], "compat-inferred"


def _support_files_for_source(
    root: Path, source: Path | None, host_source: Path | None = None
) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    search_roots = [path.parent for path in (source, host_source) if path is not None]
    queue = [path for path in (source, host_source) if path is not None]
    scanned: set[Path] = set()
    root_resolved = root.resolve()
    while queue:
        current = queue.pop(0)
        try:
            current_resolved = current.resolve()
        except OSError:
            continue
        if current_resolved in scanned:
            continue
        scanned.add(current_resolved)
        include_roots = [current_resolved.parent, *search_roots]
        for include in QUOTED_INCLUDE_RE.findall(_safe_read(current_resolved)):
            include_path = Path(include)
            candidates: list[Path] = []
            if not include_path.is_absolute():
                candidates.extend(base / include_path for base in include_roots)
                candidates.append(root / include_path)
            for candidate in candidates:
                try:
                    resolved = candidate.resolve()
                    resolved.relative_to(root_resolved)
                except (OSError, ValueError):
                    continue
                if not resolved.is_file():
                    continue
                rel = _rel(resolved, root)
                if rel not in seen:
                    seen.add(rel)
                    files.append(rel)
                    queue.append(resolved)
                break
    return files


def _unit(
    *,
    root: Path,
    unit_id: str,
    entry_name: str,
    kernel_source: Path | None,
    host_source: Path | None = None,
    json_spec: str | None = None,
    tiling_headers: list[str] | None = None,
    asset_kind: str,
) -> OperatorUnit:
    kernel_text = _safe_read(kernel_source) if kernel_source is not None else ""
    host_text = _safe_read(host_source) if host_source is not None else ""
    supported_soc_versions, support_source = _soc_versions_for_sources(
        root, kernel_source, host_source
    )
    supported_arches = supported_arches_for_soc_versions(supported_soc_versions)
    build_backends = ["sk_aclgraph_wheel"]
    if asset_kind in {
        "op_dev_pool",
        "op_dev_source_pool",
        "custom_op_package",
        "pure_global_kernel",
        "source_asset",
        "direct_kernel_invocation",
    }:
        build_backends.insert(0, "baseline_direct_asc")
    return OperatorUnit(
        unit_id=unit_id,
        entry_name=entry_name,
        display_name=_display_name_from_host(host_source, entry_name),
        kernel_source=_rel(kernel_source, root) if kernel_source is not None else None,
        host_source=_rel(host_source, root) if host_source is not None else None,
        json_spec=json_spec,
        tiling_headers=tiling_headers or [],
        support_files=sorted(
            set(_support_files_for_source(root, kernel_source, host_source))
            | set(_runtime_wrapper_support_files(root, entry_name))
            | {
                item
                for wrapper_rel in _runtime_wrapper_support_files(root, entry_name)
                for item in _support_files_for_source(root, root / wrapper_rel)
            }
        ),
        features=_features_for_sources(kernel_text, host_text),
        supported_soc_versions=supported_soc_versions,
        supported_arches=supported_arches,
        target_resolution=build_target_resolution(supported_soc_versions),
        support_source=support_source,
        build_backends=build_backends,
        runtime_contract_status="needs-input-plan",
        human_questions=[],
        source_asset=str(root),
    )


def _unit_from_layout(
    root: Path, item: dict[str, Any], fallback_asset_kind: str
) -> OperatorUnit:
    kernel_source = root / str(item["kernel_source"])
    host_value = item.get("host_source")
    host_source = root / str(host_value) if host_value else None
    asset_kind = str(item.get("asset_kind") or fallback_asset_kind or "source_asset")
    return _unit(
        root=root,
        unit_id=_safe_slug(str(item.get("unit_id") or item["entry_name"])),
        entry_name=str(item["entry_name"]),
        kernel_source=kernel_source,
        host_source=host_source,
        json_spec=item.get("json_spec"),
        tiling_headers=list(item.get("tiling_headers") or []),
        asset_kind=asset_kind,
    )


def analyze_asset_layout(layout: dict[str, Any]) -> AssetUnderstanding:
    validate_layout(layout)
    asset_root = Path(str(layout["asset_root"])).resolve()
    root = asset_root if asset_root.is_dir() else asset_root.parent
    layout_units = list(layout.get("operator_units", []))
    units = [
        _unit_from_layout(root, item, str(item.get("asset_kind") or "source_asset"))
        for item in layout_units
    ]
    kinds = sorted(
        {str(item.get("asset_kind") or "source_asset") for item in layout_units}
    )
    asset_kind = kinds[0] if len(kinds) == 1 else "layout_operator_asset"
    human_questions = list(layout.get("human_questions", []))
    if human_questions and units:
        units = [
            OperatorUnit(
                **{
                    **unit.__dict__,
                    "human_questions": human_questions,
                }
            )
            for unit in units
        ]
    warnings = list(layout.get("warnings", []))
    warnings.extend(
        {
            "kind": "layout-human-question",
            "id": str(question.get("id", "")),
            "message": str(question.get("message", "")),
        }
        for question in human_questions
    )
    return AssetUnderstanding(
        schema_version=1,
        status="analyzed"
        if units and layout.get("status") == "ready"
        else "needs-human",
        asset_root=str(asset_root),
        asset_kind=asset_kind,
        operator_units=units,
        unsupported_items=list(layout.get("skipped_items", [])),
        warnings=warnings,
    )


def _analyze_op_dev(root: Path) -> AssetUnderstanding:
    op_kernel = root / "op_kernel"
    op_host = root / "op_host"
    units: list[OperatorUnit] = []
    warnings: list[dict[str, Any]] = []
    for kernel in sorted(path for path in op_kernel.rglob("*.cpp") if path.is_file()):
        entries = _global_entries(kernel)
        if not entries:
            warnings.append(
                {"kind": "kernel-without-global", "path": _rel(kernel, root)}
            )
            continue
        stem = kernel.stem
        host = op_host / f"{stem}.cpp"
        host_path = host if host.is_file() else None
        for entry in entries:
            units.append(
                _unit(
                    root=root,
                    unit_id=_safe_slug(entry),
                    entry_name=entry,
                    kernel_source=kernel,
                    host_source=host_path,
                    json_spec=_json_for_stem(root, stem),
                    tiling_headers=_tiling_headers_for_stem(root, stem, op_host),
                    asset_kind="op_dev_source_pool",
                )
            )
    return AssetUnderstanding(
        schema_version=1,
        status="analyzed" if units else "needs-human",
        asset_root=str(root),
        asset_kind="op_dev_source_pool",
        operator_units=units,
        unsupported_items=[]
        if units
        else [{"kind": "no-kernel-entry-detected", "path": str(root)}],
        warnings=warnings,
    )


def _analyze_generic(root: Path, asset_kind: str) -> AssetUnderstanding:
    units: list[OperatorUnit] = []
    warnings: list[dict[str, Any]] = []
    candidates: list[tuple[Path, list[str]]] = []
    source_files = []
    for path in _source_files(root):
        if path.suffix in KERNEL_SOURCE_SUFFIXES:
            source_files.append(path)
    if root.is_dir():
        op_kernel_sources = []
        for path in source_files:
            if "op_kernel" in path.relative_to(root).parts:
                op_kernel_sources.append(path)
        if op_kernel_sources:
            source_files = op_kernel_sources
    for source in sorted(source_files):
        entries = _any_kernel_entries(source)
        if not entries:
            continue
        candidates.append((source, entries))
    use_single_preferred_unit = (
        bool(candidates)
        and asset_kind in {"source_asset", "direct_kernel_invocation"}
        and not (
            root.is_dir()
            and any(
                "op_kernel" in source.relative_to(root).parts
                for source, _ in candidates
            )
        )
    )
    if use_single_preferred_unit:
        base_root = root if root.is_dir() else root.parent
        preferred_stem = root.stem if root.is_file() else root.name
        selected_source, selected_entries = candidates[0]
        for source, entries in candidates:
            if source.suffix == ".asc" and source.stem == preferred_stem:
                selected_source, selected_entries = source, entries
                break
        entry = selected_entries[0]
        units.append(
            _unit(
                root=base_root,
                unit_id=_safe_slug(entry),
                entry_name=entry,
                kernel_source=selected_source,
                asset_kind=asset_kind,
            )
        )
    else:
        seen_entries: set[str] = set()
        for source, entries in candidates:
            for entry in entries:
                if entry in seen_entries:
                    continue
                seen_entries.add(entry)
                units.append(
                    _unit(
                        root=root if root.is_dir() else root.parent,
                        unit_id=_safe_slug(entry),
                        entry_name=entry,
                        kernel_source=source,
                        asset_kind=asset_kind,
                    )
                )
    if not units:
        warnings.append({"kind": "no-kernel-entry-detected", "path": str(root)})
    return AssetUnderstanding(
        schema_version=1,
        status="analyzed" if units else "needs-human",
        asset_root=str(root),
        asset_kind=asset_kind,
        operator_units=units,
        unsupported_items=[]
        if units
        else [{"kind": "no-kernel-entry-detected", "path": str(root)}],
        warnings=warnings,
    )


def analyze_asset(asset: Path) -> AssetUnderstanding:
    root = asset.resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    inventory = build_inventory(root)
    layout = build_layout_from_inventory(inventory)
    has_layout_signal = bool(
        layout.get("status") == "ready" or layout.get("human_questions")
    )
    has_kernel_signal = bool(
        inventory.get("kernel_source_count")
        or inventory.get("kernel_candidate_count")
        or inventory.get("rtc_host_program_count")
    )
    if has_layout_signal or has_kernel_signal:
        return analyze_asset_layout(layout)
    base = root if root.is_dir() else root.parent
    if root.is_dir() and (root / "op_kernel").is_dir() and (root / "op_host").is_dir():
        return _analyze_op_dev(root)
    if root.is_dir() and (root / "custom_op").is_dir():
        custom = root / "custom_op"
        if (custom / "op_kernel").is_dir() and (custom / "op_host").is_dir():
            manifest = _analyze_op_dev(custom)
            operator_units = []
            for unit in manifest.operator_units:
                tiling_headers = []
                for item in unit.tiling_headers:
                    tiling_headers.append(f"custom_op/{item}")
                operator_units.append(
                    OperatorUnit(
                        **{
                            **unit.__dict__,
                            "kernel_source": f"custom_op/{unit.kernel_source}"
                            if unit.kernel_source
                            else None,
                            "host_source": f"custom_op/{unit.host_source}"
                            if unit.host_source
                            else None,
                            "json_spec": f"custom_op/{unit.json_spec}"
                            if unit.json_spec
                            else None,
                            "tiling_headers": tiling_headers,
                            "source_asset": str(root),
                        }
                    )
                )
            return AssetUnderstanding(
                schema_version=1,
                status=manifest.status,
                asset_root=str(root),
                asset_kind="custom_op_package",
                operator_units=operator_units,
                unsupported_items=manifest.unsupported_items,
                warnings=manifest.warnings,
            )
    if root.is_dir() and (root / "CMakeLists.txt").is_file():
        return _analyze_generic(root, "direct_kernel_invocation")
    if root.is_file():
        return _analyze_generic(root, "source_asset")
    return _analyze_generic(base, "source_asset")


def analyze_asset_from_layout_file(layout_path: Path) -> AssetUnderstanding:
    return analyze_asset_layout(read_layout(layout_path.resolve()))


def _copy_rel(root: Path, rel: str | None, dest_root: Path) -> None:
    if not rel:
        return
    source = root / rel
    if not source.is_file():
        return
    dest = dest_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def _copy_tree_filtered(source_root: Path, dest_root: Path) -> None:
    if source_root.is_file():
        dest = dest_root / source_root.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root, dest)
        return
    for source in sorted(source_root.rglob("*")):
        rel = source.relative_to(source_root)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        dest = dest_root / rel
        if source.is_symlink():
            try:
                real_source = source.resolve(strict=True)
            except OSError:
                continue
            if real_source.is_dir():
                shutil.copytree(real_source, dest, dirs_exist_ok=True, symlinks=False)
            elif real_source.is_file():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(real_source, dest)
            continue
        if source.is_dir():
            if source.name not in SKIP_DIRS:
                dest.mkdir(parents=True, exist_ok=True)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def _materialize_sk_asset(
    root: Path, unit: OperatorUnit, unit_dir: Path, asset_kind: str
) -> str:
    sk_dir = unit_dir / "sk_source"
    sk_dir.mkdir(parents=True, exist_ok=True)
    for rel in [
        unit.kernel_source,
        unit.json_spec,
        *unit.tiling_headers,
        *unit.support_files,
    ]:
        _copy_rel(root, rel, sk_dir)
    return str(sk_dir)


def materialize_operator_units(
    asset: Path, manifest: AssetUnderstanding, output_dir: Path
) -> dict[str, str]:
    source_root = asset.resolve()
    root = source_root.parent if source_root.is_file() else source_root
    full_copy_root = source_root if source_root.is_file() else root
    units_root = output_dir / "operator-units"
    units_root.mkdir(parents=True, exist_ok=True)
    materialized: dict[str, str] = {}
    sk_assets: dict[str, str] = {}
    preserve_full_asset = (
        manifest.asset_kind in {"source_asset", "direct_kernel_invocation"}
        and len(manifest.operator_units) == 1
    )
    for unit in manifest.operator_units:
        unit_dir = units_root / unit.unit_id
        if unit_dir.exists():
            shutil.rmtree(unit_dir)
        unit_dir.mkdir(parents=True)
        if preserve_full_asset:
            _copy_tree_filtered(full_copy_root, unit_dir)
        else:
            for rel in [
                unit.kernel_source,
                unit.host_source,
                unit.json_spec,
                *unit.tiling_headers,
                *unit.support_files,
            ]:
                _copy_rel(root, rel, unit_dir)
        sk_assets[unit.unit_id] = _materialize_sk_asset(
            root, unit, unit_dir, manifest.asset_kind
        )
        normalized_unit = OperatorUnit(
            **{
                **unit.__dict__,
                "normalized_asset": str(unit_dir),
            }
        )
        unit_manifest = AssetUnderstanding(
            schema_version=manifest.schema_version,
            status=manifest.status,
            asset_root=str(unit_dir),
            asset_kind=manifest.asset_kind,
            operator_units=[normalized_unit],
            unsupported_items=[],
            warnings=[],
        )
        write_understanding(
            unit_dir / "operator-asset-understanding.json", unit_manifest
        )
        materialized[unit.unit_id] = str(unit_dir)
    index = {
        "schema_version": 1,
        "asset_root": str(root),
        "asset_kind": manifest.asset_kind,
        "operator_units": [
            {
                "unit_id": unit.unit_id,
                "entry_name": unit.entry_name,
                "asset": materialized[unit.unit_id],
                "sk_asset": sk_assets[unit.unit_id],
                "source_asset": str(root),
                "asset_kind": manifest.asset_kind,
                "kernel_source": unit.kernel_source,
                "host_source": unit.host_source,
                "json_spec": unit.json_spec,
                "tiling_headers": unit.tiling_headers,
                "support_files": unit.support_files,
                "supported_soc_versions": unit.supported_soc_versions,
                "supported_arches": unit.supported_arches,
                "target_resolution": unit.target_resolution,
                "support_source": unit.support_source,
                "build_backends": unit.build_backends,
            }
            for unit in manifest.operator_units
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "operator-units.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return materialized
