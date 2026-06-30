# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""aclgraph-canonical pybind package handoff for sk-operator-build-package.

Codegen now owns the native package source tree:

    operator-sk-adapted/
      csrc/<op>.asc
      csrc/pybind11.asc
      csrc/pybind11_<op>.asc
      <python_package>/__init__.py
      <python_package>/_arch_selector.py
      <python_package>/_torch_library.py
      setup.py

This module validates that tree and returns the binding manifest consumed by
`build-native-wheel`. It intentionally does not generate per-entry wrapper
sources or an aggregate C++ module.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _collect_entries(adapted_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries_by_name: dict[str, dict[str, Any]] = {}
    for per_file in adapted_manifest.get("per_file", []):
        for entry in per_file.get("entries", []):
            entries_by_name.setdefault(
                entry["entry_name"], {**entry, "source_file": per_file["file"]}
            )
    return list(entries_by_name.values())


def _normalize_arches(values: Any) -> list[str]:
    arches: list[str] = []
    seen: set[str] = set()
    iterable = values or []
    if isinstance(iterable, str):
        iterable = iterable.replace(";", ",").split(",")
    for item in iterable:
        arch = str(item or "").strip().lower().replace("_", "-")
        if not arch or arch in seen:
            continue
        seen.add(arch)
        arches.append(arch)
    return arches


def _safe_entry_module_base(entry_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", entry_name).strip("_")
    if not safe:
        safe = "entry"
    if safe[0].isdigit():
        safe = f"_{safe}"
    return safe


def _entry_pybind_file(entry_name: str) -> str:
    return f"pybind11_{_safe_entry_module_base(entry_name)}.asc"


def _required_tree_files(adapted_dir: Path, package_dir_name: str) -> list[Path]:
    return [
        adapted_dir / "csrc" / "pybind11.asc",
        adapted_dir / "setup.py",
        adapted_dir / package_dir_name / "__init__.py",
        adapted_dir / package_dir_name / "_arch_selector.py",
        adapted_dir / package_dir_name / "_torch_library.py",
    ]


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def generate_pybind_artifacts(
    adapted_dir: Path,
    adapted_manifest: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Validate and publish the in-place aclgraph-canonical binding tree."""
    entries = _collect_entries(adapted_manifest)
    if not entries:
        raise ValueError("no kernel entries in adapted manifest; nothing to bind")
    if adapted_manifest.get("pybind_layout") != "aclgraph-canonical":
        raise ValueError(
            "adapted manifest is not aclgraph-canonical; rerun codegen.adapt-sk-from-global"
        )

    package_name = adapted_manifest.get("package_name") or "op_extension"
    package_dir_name = (
        adapted_manifest.get("python_package")
        or re.sub(r"[^A-Za-z0-9_]", "_", package_name).strip("_")
        or "op_extension"
    )
    missing = [
        path
        for path in _required_tree_files(adapted_dir, package_dir_name)
        if not path.is_file()
    ]
    entry_pybind_paths = [
        adapted_dir / "csrc" / _entry_pybind_file(entry["entry_name"])
        for entry in entries
    ]
    missing.extend(path for path in entry_pybind_paths if not path.is_file())
    pybind_names = {"pybind11.asc", *(path.name for path in entry_pybind_paths)}
    csrc_sources = sorted(
        path
        for path in (adapted_dir / "csrc").glob("*.asc")
        if path.name not in pybind_names
    )
    csrc_support = sorted(
        path
        for path in (adapted_dir / "csrc").rglob("*")
        if path.is_file() and path not in csrc_sources and path.name not in pybind_names
    )
    if not csrc_sources:
        missing.append(adapted_dir / "csrc" / "<op>.asc")
    if missing:
        formatted = ", ".join(_relative(path, adapted_dir) for path in missing)
        raise ValueError(f"aclgraph-canonical binding tree incomplete: {formatted}")

    module_name = adapted_manifest.get("pybind_module")
    if not module_name:
        raise ValueError("adapted manifest missing pybind_module")

    written_files = sorted(
        _relative(path, adapted_dir)
        for path in [
            *csrc_sources,
            *entry_pybind_paths,
            *csrc_support,
            *_required_tree_files(adapted_dir, package_dir_name),
        ]
    )

    extension_modules_by_entry = {
        entry[
            "entry_name"
        ]: f"{package_dir_name}.{_safe_entry_module_base(entry['entry_name'])}_<arch_suffix>"
        for entry in entries
    }
    supported_arches_by_entry = {
        entry["entry_name"]: _normalize_arches(entry.get("supported_arches"))
        for entry in entries
    }
    return {
        "status": "generated",
        "package_name": package_name,
        "python_package": package_dir_name,
        "package_version": adapted_manifest.get("package_version", ""),
        "package_root": str(adapted_dir),
        "pybind_layout": "aclgraph-canonical",
        "extension_module": f"{package_dir_name}.{module_name}",
        "extension_module_base": f"{package_dir_name}.{module_name}",
        "extension_module_pattern": f"{package_dir_name}.<entry_module_base>_<arch_suffix>",
        "extension_modules_by_entry": extension_modules_by_entry,
        "supported_arches_by_entry": supported_arches_by_entry,
        "kernel_entries": [
            {
                "entry_name": entry["entry_name"],
                "source_file": entry["source_file"],
                "param_count": entry["param_count"],
                "supported_arches": supported_arches_by_entry.get(
                    entry["entry_name"], []
                ),
                "supported_soc_versions": list(entry.get("supported_soc_versions", [])),
                "support_source": entry.get("support_source", ""),
            }
            for entry in entries
        ],
        "written_files": written_files,
        "output_root": str(output_root),
    }
