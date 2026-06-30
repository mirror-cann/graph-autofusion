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

from operator_asset_contracts import validate_asset_layout_contract
from operator_asset_parser import parse_global_entries


SOURCE_SUFFIXES = {".asc", ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
KERNEL_SOURCE_SUFFIXES = {".asc", ".c", ".cc", ".cpp", ".cxx"}
INCLUDE_SUFFIXES = {".h", ".hh", ".hpp", ".hxx"}
SKIP_DIRS = {
    ".git",
    "__pycache__",
    "build",
    "build_out",
    "cmake-build-debug",
    "dist",
    "input",
    "kernel_meta",
    "kernel_meta_temp",
    "kernel_template",
    "out",
    "output",
    "outputs",
    "prof",
    "prof_total",
    "profiling",
    "run",
    "runs",
    "sk_meta",
    "static_kernel_compile_outputs",
    "tmp",
}
TEST_DIR_NAMES = {"test", "tests", "op_verify", "op_testcase", "op_testcases"}
BUILD_FILENAMES = {
    "CMakeLists.txt",
    "setup.py",
    "pyproject.toml",
    "compile.sh",
    "build.sh",
}
OP_ADD_RE = re.compile(r"\bOP_ADD\s*\(\s*([A-Za-z_]\w*)\s*\)")
QUOTED_INCLUDE_RE = re.compile(r"^\s*#\s*include\s+\"([^\"]+)\"", re.MULTILINE)
NPU_OP_KERNEL_SOURCES_RE = re.compile(
    r"npu_op_kernel_sources\s*\([^)]*?OP_TYPE\s+([A-Za-z_]\w*)[^)]*?KERNEL_FILE\s+([^\s)]+)",
    re.DOTALL,
)
RAW_RTC_RE = re.compile(
    r"\baclrtc(?:CreateProg|CompileProg|AddNameExpr|GetLoweredName)\b|R\"{2,}\("
)


def safe_slug(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value or "")).strip("._-")
    return slug or "op"


def safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def iter_files(root: Path) -> list[Path]:
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


def _is_test_path(path: Path, root: Path) -> bool:
    return any(part in TEST_DIR_NAMES for part in path.relative_to(root).parts[:-1])


def _file_role(path: Path, root: Path, text: str, entries: list[str]) -> str:
    rel_parts = path.relative_to(root).parts
    if _is_test_path(path, root):
        return "test"
    if path.name in BUILD_FILENAMES:
        return "build"
    if path.suffix in INCLUDE_SUFFIXES:
        return "include"
    if RAW_RTC_RE.search(text):
        return "rtc_host_program"
    if entries:
        return "kernel_source"
    if "int main" in text or "aclrt" in text or "aclCreate" in text:
        return "host_runtime"
    if OP_ADD_RE.search(text) or "TilingFunc" in text or "npu_op" in text:
        return "host_registration"
    if path.suffix in KERNEL_SOURCE_SUFFIXES and (
        path.suffix == ".asc" or "op_kernel" in rel_parts
    ):
        return "kernel_candidate"
    if path.suffix == ".json":
        return "json_spec"
    return "support"


def _source_records(
    scan_root: Path, rel_root: Path | None = None
) -> list[dict[str, Any]]:
    rel_root = rel_root or scan_root
    records: list[dict[str, Any]] = []
    for path in iter_files(scan_root):
        text = safe_read(path)
        entries = (
            [entry.name for entry in parse_global_entries(text)]
            if path.suffix in SOURCE_SUFFIXES
            else []
        )
        quoted_includes = (
            QUOTED_INCLUDE_RE.findall(text) if path.suffix in SOURCE_SUFFIXES else []
        )
        unresolved_includes = []
        for include in quoted_includes:
            if not _resolve_include(rel_root, rel(path, rel_root), include):
                unresolved_includes.append(include)
        records.append(
            {
                "path": rel(path, rel_root),
                "suffix": path.suffix,
                "role": _file_role(path, rel_root, text, entries),
                "kernel_entries": entries,
                "op_add_entries": OP_ADD_RE.findall(text)
                if path.suffix in SOURCE_SUFFIXES
                else [],
                "quoted_includes": quoted_includes,
                "unresolved_includes": unresolved_includes,
                "has_raw_rtc": bool(RAW_RTC_RE.search(text)),
                "is_test_path": _is_test_path(path, rel_root),
            }
        )
    return records


def _find_json(root: Path, stem: str) -> str | None:
    candidates = sorted(path for path in root.rglob(f"{stem}.json") if path.is_file())
    return rel(candidates[0], root) if candidates else None


def _nearest_asset_root(root: Path, source_rel: str) -> Path:
    source_path = root / source_rel
    for parent in [source_path.parent, *source_path.parents]:
        if parent == root.parent:
            break
        if parent.name == "op_kernel":
            continue
        if (parent / "op_kernel").is_dir() or (parent / "CMakeLists.txt").is_file():
            return parent
    return source_path.parent


def _find_tiling_headers(root: Path, stem: str, source_rel: str) -> list[str]:
    patterns = [
        f"{stem}_tiling.h",
        f"{stem}_tiling.hpp",
        f"tiling_key_{stem}.h",
        f"tiling_key_{stem}.hpp",
    ]
    results: list[str] = []
    seen: set[str] = set()
    search_root = _nearest_asset_root(root, source_rel)
    for pattern in patterns:
        for path in sorted(search_root.rglob(pattern)):
            if not path.is_file():
                continue
            item = rel(path, root)
            if item not in seen:
                seen.add(item)
                results.append(item)
    return results


def _resolve_include(root: Path, source_rel: str, include: str) -> str | None:
    include_path = Path(include)
    if include_path.is_absolute() or not include_path.parts:
        return None
    source_path = root / source_rel
    search_roots = [source_path.parent, root]
    for search_root in search_roots:
        candidate = (search_root / include_path).resolve()
        try:
            candidate.relative_to(root.resolve())
        except (OSError, ValueError):
            continue
        if candidate.is_file():
            return rel(candidate, root)
    return None


def _support_files_for_record(root: Path, record: dict[str, Any]) -> list[str]:
    support: list[str] = []
    seen: set[str] = set()
    for include in record.get("quoted_includes", []):
        resolved = _resolve_include(root, str(record["path"]), str(include))
        if resolved and resolved not in seen:
            seen.add(resolved)
            support.append(resolved)
    return support


def build_inventory(asset: Path) -> dict[str, Any]:
    root = asset.resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    base = root.parent if root.is_file() else root
    files = _source_records(root, base)
    kernel_records = [item for item in files if item["role"] == "kernel_source"]
    kernel_candidate_records = [
        item for item in files if item["role"] == "kernel_candidate"
    ]
    rtc_records = [item for item in files if item["role"] == "rtc_host_program"]
    build_records = [item for item in files if item["role"] == "build"]
    cmake_kernel_sources: list[dict[str, str]] = []
    for item in build_records:
        if Path(str(item["path"])).name != "CMakeLists.txt":
            continue
        text = safe_read((root if root.is_dir() else base) / str(item["path"]))
        for op_type, kernel_file in NPU_OP_KERNEL_SOURCES_RE.findall(text):
            cmake_kernel_sources.append(
                {"op_type": op_type, "kernel_file": kernel_file}
            )
    return {
        "schema_version": 1,
        "asset_root": str(root),
        "base_dir": str(base),
        "status": "scanned",
        "files": files,
        "kernel_source_count": len(kernel_records),
        "kernel_candidate_count": len(kernel_candidate_records),
        "rtc_host_program_count": len(rtc_records),
        "build_files": [item["path"] for item in build_records],
        "cmake_kernel_sources": cmake_kernel_sources,
    }


def _host_candidates_by_stem(
    root: Path, files: list[dict[str, Any]], stem: str, entry: str
) -> list[str]:
    candidates = [
        item
        for item in files
        if item["role"] == "host_registration"
        and (
            Path(str(item["path"])).stem == stem
            or entry in item.get("op_add_entries", [])
        )
    ]
    return [
        str(item["path"])
        for item in sorted(candidates, key=lambda item: str(item["path"]))
    ]


def _asset_kind_for_unit(
    root: Path, kernel_rel: str, host_rel: str | None, files: list[dict[str, Any]]
) -> str:
    parts = Path(kernel_rel).parts
    if host_rel and "op_kernel" in parts:
        return "op_dev_pool"
    if any(Path(str(item["path"])).name == "CMakeLists.txt" for item in files):
        return "direct_kernel_invocation"
    if root.is_file():
        return "source_asset"
    return "source_asset"


def build_layout_from_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    root = Path(str(inventory["asset_root"])).resolve()
    files = list(inventory.get("files", []))
    units: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    if inventory.get("rtc_host_program_count", 0):
        for item in files:
            if item["role"] == "rtc_host_program":
                skipped.append(
                    {
                        "path": item["path"],
                        "reason": "rtc-host-program-not-auto-converted",
                    }
                )
        questions.append(
            {
                "id": "rtc-host-program",
                "message": "RTC host programs embed kernel source in strings and need an explicit extraction policy before conversion.",
            }
        )
    for item in files:
        if item["role"] == "kernel_candidate":
            skipped.append(
                {"path": item["path"], "reason": "kernel-source-without-global-entry"}
            )
            continue
        if item["role"] != "kernel_source":
            continue
        kernel_rel = str(item["path"])
        stem = Path(kernel_rel).stem
        support_files = _support_files_for_record(
            root if root.is_dir() else root.parent, item
        )
        for entry in item.get("kernel_entries", []):
            host_candidates = _host_candidates_by_stem(
                root if root.is_dir() else root.parent, files, stem, str(entry)
            )
            host_rel = host_candidates[0] if len(host_candidates) == 1 else None
            if len(host_candidates) > 1:
                questions.append(
                    {
                        "id": "ambiguous-host-registration",
                        "entry_name": str(entry),
                        "kernel_source": kernel_rel,
                        "candidates": host_candidates,
                        "message": "Multiple host registration files match this kernel entry; provide an explicit asset layout.",
                    }
                )
            tiling_headers = _find_tiling_headers(
                root if root.is_dir() else root.parent, stem, kernel_rel
            )
            support = sorted({*support_files, *tiling_headers})
            unit = {
                "unit_id": safe_slug(entry),
                "entry_name": entry,
                "asset_kind": _asset_kind_for_unit(root, kernel_rel, host_rel, files),
                "kernel_source": kernel_rel,
                "host_source": host_rel,
                "host_source_candidates": host_candidates,
                "json_spec": _find_json(root if root.is_dir() else root.parent, stem),
                "tiling_headers": tiling_headers,
                "support_files": support,
                "source_asset": str(root),
            }
            units.append(unit)
    grouped_units: dict[str, list[dict[str, Any]]] = {}
    for unit in units:
        grouped_units.setdefault(str(unit["unit_id"]), []).append(unit)
    deduped_units: list[dict[str, Any]] = []
    duplicate_ids: list[str] = []
    for unit_id, group in sorted(grouped_units.items()):
        if len(group) == 1:
            deduped_units.append(group[0])
            continue
        duplicate_ids.append(unit_id)
        candidate_sources = sorted(str(unit["kernel_source"]) for unit in group)
        for unit in group:
            skipped.append(
                {
                    "path": unit["kernel_source"],
                    "reason": "duplicate-unit-id-requires-explicit-layout",
                    "entry_name": unit["entry_name"],
                    "unit_id": unit_id,
                    "candidate_kernel_sources": candidate_sources,
                }
            )
    units = sorted(
        deduped_units,
        key=lambda unit: (str(unit["kernel_source"]), str(unit["entry_name"])),
    )
    if duplicate_ids:
        questions.append(
            {
                "id": "duplicate-unit-id",
                "duplicate_unit_ids": duplicate_ids,
                "message": f"Duplicate unit ids require an explicit asset layout or namespace policy: {', '.join(duplicate_ids)}",
            }
        )
    has_ambiguous_host = any(
        question.get("id") == "ambiguous-host-registration" for question in questions
    )
    status = (
        "ready"
        if units
        and not duplicate_ids
        and not has_ambiguous_host
        and not inventory.get("rtc_host_program_count", 0)
        else "needs-human"
    )
    return {
        "schema_version": 1,
        "status": status,
        "asset_root": str(root),
        "layout_source": "script-inventory",
        "operator_units": units,
        "skipped_items": skipped,
        "human_questions": questions,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_layout(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_layout(payload, path.parent)
    return payload


def validate_layout(layout: dict[str, Any], base_dir: Path | None = None) -> None:
    validate_asset_layout_contract(layout, base_dir)
