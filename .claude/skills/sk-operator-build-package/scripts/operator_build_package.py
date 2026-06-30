# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""SK Operator Build + Package CLI -- compile SK source, export validated versions, and build pybind wheels."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, NamedTuple

SOURCE_SUFFIXES = {".asc", ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
INCLUDE_SUFFIXES = {".h", ".hh", ".hpp", ".hxx"}
COMPILABLE_SOURCE_SUFFIXES = SOURCE_SUFFIXES - INCLUDE_SUFFIXES
ASCEND_COMPILABLE_SOURCE_SUFFIXES = {".asc"}
BUILD_FILENAMES = {
    "CMakeLists.txt",
    "setup.py",
    "pyproject.toml",
    "compile.sh",
    "build.sh",
}
TEST_FILENAMES = {"run.sh", "main.cpp", "test.py"}
PACKAGE_FILENAMES = {"setup.py", "pyproject.toml"}
DOC_FILENAMES = {"README", "README.md", "readme.md"}
TEST_PACKAGE_DIR_NAMES = {"test", "tests"}
SCAFFOLD_CONTRACT_ORDER = (
    "operator_entry_contract",
    "build_contract",
    "test_contract",
    "package_contract",
    "delivery_docs_contract",
    "sk_binding_contract",
    "operator_semantics_contract",
)
SCAFFOLD_SPECS = {
    "operator_entry_contract": {
        "kind": "entry",
        "priority": "high",
        "reason": "missing_operator_entry_contract",
        "suggested_path": "",
        "template": "operator-entry-contract",
        "description": (
            "Define the operator entry contract before build or test scaffolds "
            "can target a stable callable unit."
        ),
    },
    "build_contract": {
        "kind": "build",
        "priority": "high",
        "reason": "missing_build_contract",
        "suggested_path": "CMakeLists.txt",
        "template": "minimal-cmake-ascendc",
        "description": "Add a minimal build contract that can compile the detected operator sources.",
    },
    "test_contract": {
        "kind": "test",
        "priority": "high",
        "reason": "missing_test_contract",
        "suggested_path": "tests/test_operator.py",
        "template": "minimal-python-smoke-test",
        "description": "Add a smoke test entrypoint that can exercise the operator delivery surface.",
    },
    "package_contract": {
        "kind": "package",
        "priority": "medium",
        "reason": "missing_package_contract",
        "suggested_path": "pyproject.toml",
        "template": "minimal-python-package",
        "description": "Add package metadata so the operator asset has a distributable boundary.",
    },
    "delivery_docs_contract": {
        "kind": "docs",
        "priority": "medium",
        "reason": "missing_delivery_docs_contract",
        "suggested_path": "README.md",
        "template": "operator-delivery-readme",
        "description": "Document build, test, and handoff commands for the operator asset.",
    },
    "sk_binding_contract": {
        "kind": "sk_binding",
        "priority": "high",
        "reason": "missing_sk_binding_contract",
        "suggested_path": "",
        "template": "sk-binding-contract-notes",
        "description": "Record how the operator binds to SK runtime expectations.",
    },
    "operator_semantics_contract": {
        "kind": "semantics",
        "priority": "high",
        "reason": "missing_operator_semantics_contract",
        "suggested_path": "operator-semantics.md",
        "template": "operator-semantics-contract",
        "description": "Capture operator inputs, outputs, constraints, and correctness expectations.",
    },
    "build_entry_without_command_contract": {
        "kind": "build",
        "priority": "medium",
        "reason": "build_entry_without_command_contract",
        "suggested_path": "CMakeLists.txt",
        "template": "cmake-command-contract",
        "description": "Promote the build entry into an explicit command contract.",
    },
    "smoke_test_without_correctness_or_graph_sk": {
        "kind": "validation",
        "priority": "medium",
        "reason": "smoke_test_without_correctness_or_graph_sk",
        "suggested_path": "tests/test_operator_correctness.py",
        "template": "python-correctness-assertion-test",
        "description": "Strengthen smoke coverage with correctness or graph-SK validation evidence.",
    },
}
GRAPH_SK_TEST_PATTERNS = (
    ("torch.ops", re.compile(r"\btorch\.ops\b")),
    ("ascendc_sk", re.compile(r"\bascendc_sk\b")),
    ("aclgraph", re.compile(r"\baclgraph\b", re.IGNORECASE)),
    ("aclgraph_sk", re.compile(r"\baclgraph_sk\b", re.IGNORECASE)),
)
CORRECTNESS_TEST_PATTERNS = (
    ("torch.testing", re.compile(r"\btorch\.testing\b")),
    ("np.testing", re.compile(r"\bnp\.testing\b")),
    ("assert_close", re.compile(r"\bassert_close\b")),
    ("assert_allclose", re.compile(r"\bassert_allclose\b")),
    ("EXPECT_", re.compile(r"\bEXPECT_[A-Za-z_0-9]*\b")),
    ("ASSERT_", re.compile(r"\bASSERT_[A-Za-z_0-9]*\b")),
    ("self.assert", re.compile(r"\bself\.assert[A-Za-z_0-9]*\b")),
    ("pytest.raises", re.compile(r"\bpytest\.raises\b")),
)
TOOLCHAIN_BUILD_PATTERNS = (
    ("npu_op_package", re.compile(r"(?<![A-Za-z0-9_])npu_op_package\s*\(")),
    ("torch.utils.cpp_extension", re.compile(r"\btorch\.utils\.cpp_extension\b")),
    ("CppExtension", re.compile(r"(?<![A-Za-z0-9_])CppExtension\s*\(")),
    ("CUDAExtension", re.compile(r"(?<![A-Za-z0-9_])CUDAExtension\s*\(")),
    ("bisheng", re.compile(r"(?<![A-Za-z0-9_])bisheng\b")),
    ("ascendc", re.compile(r"(?<![A-Za-z0-9_])ascendc\b")),
    ("msopgen", re.compile(r"(?<![A-Za-z0-9_])msopgen\b")),
    ("opc", re.compile(r"(?<![A-Za-z0-9_])opc\b")),
    ("atc", re.compile(r"(?<![A-Za-z0-9_])atc\b")),
)
COMMAND_BUILD_PATTERNS = (
    ("add_library", re.compile(r"(?<![A-Za-z0-9_])add_library\s*\(")),
    ("add_executable", re.compile(r"(?<![A-Za-z0-9_])add_executable\s*\(")),
    ("project", re.compile(r"(?<![A-Za-z0-9_])project\s*\(")),
    (
        "cmake_minimum_required",
        re.compile(r"(?<![A-Za-z0-9_])cmake_minimum_required\s*\("),
    ),
    ("setup", re.compile(r"(?<![A-Za-z0-9_])setup\s*\(")),
    ("build-system", re.compile(r"(?m)^\s*\[build-system\]\s*$")),
    ("python -m build", re.compile(r"(?m)(?:^|[;&|]\s*)python\s+-m\s+build\b")),
    ("pip install", re.compile(r"(?m)(?:^|[;&|]\s*)pip\s+install\b")),
    ("cmake", re.compile(r"(?m)(?:^|[;&|]\s*)cmake\b")),
    ("make", re.compile(r"(?m)(?:^|[;&|]\s*)make\b")),
)
UNCOMPILED_SOURCE_WARNING = "linker input file unused because linking not done"
MISSING_INCLUDE_RE = re.compile(r"fatal error:\s+([^:]+): No such file or directory")
ASCEND_COMPILE_UNITS_MISSING = "ascend_compile_units_missing"
TOOLCHAIN_INCLUDE_NAMES = {"kernel_operator.h"}
TOOLCHAIN_INCLUDE_PREFIXES = ("kernel_tiling/",)
ASCEND_ARCH_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
KERNEL_ENTRY_RE = re.compile(
    r"(?:extern\s+\"C\"\s+)?(?:(?:__global__|__spk__|__sk__)[\w\s_()*,:&<>]*\s+void\s+)([A-Za-z_]\w*)\s*\(",
    re.MULTILINE,
)
SK_CONVERSION_BOUNDARY = [
    "sk source not generated",
    "build not executed",
    "runtime not executed",
    "operator correctness not validated",
    "customer handoff not completed",
]
SK_SOURCE_SCAFFOLD_BOUNDARY = [
    "original asset not modified",
    "build not executed",
    "runtime not executed",
    "operator correctness not validated",
    "customer handoff not completed",
]
SK_SOURCE_SCAFFOLD_DIR = "operator-sk-source-scaffold"
SK_SOURCE_VERSION_DIR = "operator-sk-source-version"
SK_SOURCE_VERSION_BUILD_DIR = "operator-sk-source-version-build"
SK_SOURCE_VERSION_VALIDATION_PATH = "operator-sk-source-version-validation.json"
SK_SOURCE_VERSION_PIPELINE_PATH = "operator-sk-source-version-pipeline.json"
SK_SOURCE_VERSION_GENERATED_BOUNDARY = [
    "source version prepared from exact-current generated SK source scaffold",
    "build validation passed before source version preparation",
    "runtime not executed",
    "operator correctness not validated",
    "customer handoff not completed",
]
SK_SOURCE_VERSION_BLOCKED_BOUNDARY = [
    "source version not prepared because sk build validation did not pass",
    "runtime not executed",
    "operator correctness not validated",
    "customer handoff not completed",
]
SK_SOURCE_VERSION_VALIDATION_PASSED_BOUNDARY = [
    "source version build validated",
    "runtime not executed",
    "operator correctness not validated",
    "customer handoff not completed",
]
SK_SOURCE_VERSION_VALIDATION_BLOCKED_BOUNDARY = [
    "source version build validation blocked",
    "runtime not executed",
    "operator correctness not validated",
    "customer handoff not completed",
]
SK_SOURCE_VERSION_VALIDATION_FAILED_BOUNDARY = [
    "source version build validation failed",
    "runtime not executed",
    "operator correctness not validated",
    "customer handoff not completed",
]
SK_SOURCE_VERSION_PIPELINE_PASSED_BOUNDARY = [
    "source version build validated",
    "runtime not executed",
    "operator correctness not validated",
    "customer handoff not completed",
]
SK_SOURCE_VERSION_PIPELINE_BLOCKED_BOUNDARY = [
    "source version pipeline blocked",
    "runtime not executed",
    "operator correctness not validated",
    "customer handoff not completed",
]
SK_SOURCE_VERSION_PIPELINE_FAILED_BOUNDARY = [
    "source version pipeline failed",
    "runtime not executed",
    "operator correctness not validated",
    "customer handoff not completed",
]
SK_BUILD_VALIDATION_CHECK_NAMES = (
    "sk_source_scaffold_generated",
    "copied_sources_current",
    "cmake_command_available",
    "cmake_contract_current",
    "build_dir_available",
    "cmake_configure",
    "cmake_build",
)
SK_SOURCE_VERSION_VALIDATION_CHECK_NAMES = (
    "sk_source_version_current",
    "cmake_command_available",
    "build_dir_available",
    "cmake_configure",
    "cmake_build",
)
SK_BUILD_VALIDATION_BOUNDARY = [
    "runtime not executed",
    "operator correctness not validated",
    "customer handoff not completed",
]
CMAKE_EXECUTION_TIMEOUT_SECONDS = 120
SUPPORT_DIR_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

SCAFFOLD_MANIFEST_KEYS = {
    "status",
    "asset_level",
    "source_files",
    "kernel_entries",
    "copied_source_files",
    "generated_files",
    "skipped_items",
    "blocked_reasons",
    "execution_boundary",
}
VALIDATION_CHECK_NAMES = (
    "cmake_contract_static",
    "copied_sources_present",
    "generated_files_present",
    "kernel_entries_visible",
    "pytest_contract_tests",
    "scaffold_manifest_present",
    "scaffold_project_present",
)
VALIDATION_BOUNDARY = [
    "ascend build not executed",
    "ascend runtime not executed",
    "package not produced",
    "operator correctness not validated",
]
PREFLIGHT_CHECK_NAMES = (
    "ascend_toolchain_detected",
    "cmake_contract_present",
    "copied_sources_current",
    "package_contract_present",
    "pytest_contract_present",
    "python_available",
    "scaffold_project_ready",
    "validation_manifest_accepted",
)
PREFLIGHT_BOUNDARY = [
    "build not executed",
    "pytest not executed",
    "package not produced",
    "ascend runtime not executed",
    "operator correctness not validated",
]
PREFLIGHT_TEST_CONTRACTS = (
    "tests/test_operator.py",
    "tests/test_operator_correctness.py",
)
SCAFFOLD_TEST_CHECK_NAMES = (
    "preflight_ready",
    "pytest_contract_files_present",
    "pytest_contract_content_current",
    "pytest_import_context_clean",
    "pytest_contract_execution",
)
SCAFFOLD_TEST_BOUNDARY = [
    "build not executed",
    "package not produced",
    "ascend runtime not executed",
    "operator correctness not validated",
]
SCAFFOLD_BUILD_CHECK_NAMES = (
    "preflight_ready",
    "pytest_contract_passed",
    "copied_sources_current",
    "cmake_command_available",
    "cmake_contract_current",
    "build_dir_available",
    "cmake_configure",
    "cmake_build",
)
SCAFFOLD_BUILD_BOUNDARY = [
    "package not produced",
    "ascend runtime not executed",
    "operator correctness not validated",
]
SCAFFOLD_READINESS_CHECK_NAMES = (
    "scaffold_manifest_accepted",
    "validation_result_accepted",
    "preflight_result_accepted",
    "pytest_contract_result_accepted",
    "cmake_build_result_accepted",
    "package_boundary_open",
    "runtime_boundary_open",
    "correctness_boundary_open",
    "handoff_boundary_open",
)
SCAFFOLD_READINESS_BOUNDARY = [
    "package not produced",
    "ascend runtime not executed",
    "operator correctness not validated",
    "customer handoff not completed",
]
PACKAGE_CONTRACT_CHECK_NAMES = (
    "readiness_local_build_passed",
    "package_files_declared",
    "pyproject_contract_current",
    "package_marker_current",
)
PACKAGE_CONTRACT_FILES = [
    "operator-scaffold/pyproject.toml",
    "operator-scaffold/operator_scaffold/__init__.py",
]
PACKAGE_CONTRACT_BOUNDARY = [
    "package not built",
    "package not installed",
    "ascend runtime not executed",
    "operator correctness not validated",
    "customer handoff not completed",
]
PACKAGE_BUILD_CHECK_NAMES = (
    "package_contract_current",
    "package_contract_defined",
    "package_build_dir_available",
    "package_staging_context_closed",
    "package_wheel_build",
    "original_scaffold_unchanged",
)
PACKAGE_BUILD_BOUNDARY = [
    "package not installed",
    "package not imported",
    "package not published",
    "ascend runtime not executed",
    "operator correctness not validated",
    "customer handoff not completed",
]
PACKAGE_BUILD_TIMEOUT_SECONDS = 120


class CliUsageError(ValueError):
    """User-facing CLI usage error handled by argparse."""


def _emit(message: object = "", *, file=None, end: str = "\n") -> None:
    stream = sys.stdout if file is None else file
    stream.write(f"{message}{end}")


def _relative(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def _safe_scaffold_relative_path(path: str) -> str:
    if not path:
        raise ValueError("empty scaffold path is not allowed")
    candidate = Path(path)
    if candidate.is_absolute():
        raise ValueError(f"absolute scaffold path is not allowed: {path}")
    if ".." in candidate.parts:
        raise ValueError(f"parent traversal is not allowed in scaffold path: {path}")
    return str(candidate)


def _is_same_or_child_path(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _is_test_path(path: Path, base_dir: Path) -> bool:
    relative_parts = path.relative_to(base_dir).parts
    lowered_parts = {part.lower() for part in relative_parts}
    name = path.name
    if name == "__init__.py":
        return False
    return (
        "test" in lowered_parts
        or "tests" in lowered_parts
        or name in TEST_FILENAMES
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _is_package_init(path: Path, base_dir: Path) -> bool:
    if path.name != "__init__.py":
        return False
    relative_parts = path.relative_to(base_dir).parts
    parent_parts = {part.lower() for part in relative_parts[:-1]}
    return parent_parts.isdisjoint(TEST_PACKAGE_DIR_NAMES)


def _is_supported_source_file(path: Path, base_dir: Path) -> bool:
    return path.suffix in SOURCE_SUFFIXES and not _is_test_path(path, base_dir)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _strip_c_style_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//.*", "", text)


def _strip_test_comments(text: str) -> str:
    text = _strip_c_style_comments(text)
    return re.sub(r"(?m)#.*$", "", text)


def _source_text(path: Path) -> str:
    text = _read_text(path)
    if path.suffix in SOURCE_SUFFIXES:
        return _strip_c_style_comments(text)
    return text


def _collect_kernel_entries(files: list[Path], base_dir: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for path in files:
        if path.suffix not in SOURCE_SUFFIXES:
            continue
        text = _source_text(path)
        for match in KERNEL_ENTRY_RE.finditer(text):
            name = match.group(1)
            key = (_relative(path, base_dir), name)
            if key in seen:
                continue
            seen.add(key)
            entries.append({"name": name, "file": key[0]})
    return entries


def _collect_sk_markers(files: list[Path]) -> dict[str, bool]:
    joined = "\n".join(
        _source_text(path) for path in files if path.suffix in SOURCE_SUFFIXES
    )
    return {
        "__sk__": "__sk__" in joined,
        "__spk__": "__spk__" in joined,
        "SK_BIND": "SK_BIND" in joined,
        "sk_param_struct": "CommArgs" in joined
        or "SkSystemArgs" in joined
        or "__gm__ uint64_t *param" in joined,
        "ascend_meta_section": ".ascend.meta" in joined,
    }


def _file_evidence(paths: list[Path], base_dir: Path) -> list[dict[str, str]]:
    return [
        {"kind": "file", "path": _relative(path, base_dir), "value": path.name}
        for path in paths
    ]


def _symbol_evidence(items: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"kind": "symbol", "path": item["file"], "value": item["name"]}
        for item in items
    ]


def _marker_evidence(sk_markers: dict[str, bool]) -> list[dict[str, str]]:
    return [
        {"kind": "marker", "path": "", "value": name}
        for name, found in sk_markers.items()
        if found
    ]


def _capability(status: str, evidence: list[dict[str, str]]) -> dict[str, Any]:
    if status == "present":
        confidence = "high" if evidence else "medium"
    elif status == "blocked":
        confidence = "low"
    else:
        confidence = "low"
    return {"status": status, "confidence": confidence, "evidence": evidence}


def _collect_capabilities(
    *,
    source_files: list[Path],
    build_files: list[Path],
    test_files: list[Path],
    package_files: list[Path],
    doc_files: list[Path],
    kernel_entries: list[dict[str, str]],
    sk_markers: dict[str, bool],
    hard_blockers: list[str],
    base_dir: Path,
) -> dict[str, dict[str, Any]]:
    source_status = (
        "blocked" if hard_blockers else ("present" if source_files else "missing")
    )
    entry_status = (
        "blocked"
        if hard_blockers
        else ("present" if kernel_entries or any(sk_markers.values()) else "missing")
    )
    return {
        "source": _capability(source_status, _file_evidence(source_files, base_dir)),
        "entry_binding": _capability(
            entry_status,
            _symbol_evidence(kernel_entries) + _marker_evidence(sk_markers),
        ),
        "build": _capability(
            "present" if build_files else "missing",
            _file_evidence(build_files, base_dir),
        ),
        "test": _capability(
            "present" if test_files else "missing", _file_evidence(test_files, base_dir)
        ),
        "package": _capability(
            "present" if package_files else "missing",
            _file_evidence(package_files, base_dir),
        ),
        "delivery_docs": _capability(
            "present" if doc_files else "missing", _file_evidence(doc_files, base_dir)
        ),
    }


def _all_text(files: list[Path], *, source_comments_stripped: bool = False) -> str:
    chunks: list[str] = []
    for path in files:
        chunks.append(
            _source_text(path)
            if source_comments_stripped and path.suffix in SOURCE_SUFFIXES
            else _read_text(path)
        )
    return "\n".join(chunks)


def _archetype_result(
    name: str, confidence: str, evidence: list[dict[str, str]]
) -> dict[str, Any]:
    return {"name": name, "confidence": confidence, "evidence": evidence}


def _detect_archetype(
    *,
    files: list[Path],
    source_files: list[Path],
    build_files: list[Path],
    package_files: list[Path],
    base_dir: Path,
) -> dict[str, Any]:
    if not source_files:
        return _archetype_result("unknown", "low", [])

    raw_text = _all_text(files)
    source_text = _all_text(source_files, source_comments_stripped=True)
    relative_paths = [_relative(path, base_dir) for path in files]
    source_evidence = _file_evidence(source_files, base_dir)
    build_evidence = _file_evidence(build_files, base_dir)
    package_evidence = _file_evidence(package_files, base_dir)

    ge_evidence: list[dict[str, str]] = []
    if "register_fx_node_ge_converter" in raw_text:
        ge_evidence.append(
            {"kind": "symbol", "path": "", "value": "register_fx_node_ge_converter"}
        )
    if "OpDef" in raw_text or any(
        part in path for path in relative_paths for part in ("op_host/", "op_kernel/")
    ):
        ge_evidence.append(
            {"kind": "directory", "path": "", "value": "op_host/op_kernel"}
        )
    if ge_evidence:
        return _archetype_result(
            "ge-opp-python-converter", "high", ge_evidence + source_evidence
        )

    pybind_evidence: list[dict[str, str]] = []
    if "PYBIND11_MODULE" in source_text or "pybind11" in raw_text:
        pybind_evidence.append(
            {"kind": "symbol", "path": "", "value": "PYBIND11_MODULE/pybind11"}
        )
    if pybind_evidence:
        return _archetype_result(
            "aclgraph-pybind-torch-library",
            "high",
            pybind_evidence + build_evidence + source_evidence,
        )

    torch_evidence: list[dict[str, str]] = []
    if (
        "TORCH_LIBRARY" in source_text
        or "torch.ops" in raw_text
        or "torch/extension.h" in source_text
    ):
        torch_evidence.append(
            {"kind": "symbol", "path": "", "value": "torch extension/operator"}
        )
    if "bisheng" in raw_text and source_files and torch_evidence:
        return _archetype_result(
            "torch-bisheng-extension",
            "high",
            torch_evidence + build_evidence + package_evidence,
        )

    if build_files and source_files:
        return _archetype_result(
            "cann-static-asc-op", "medium", build_evidence + source_evidence
        )

    if source_files:
        return _archetype_result("source-only-ascendc", "medium", source_evidence)

    return _archetype_result("unknown", "low", [])


def _test_quality_result(
    level: str, confidence: str, evidence: list[dict[str, str]]
) -> dict[str, Any]:
    return {"level": level, "confidence": confidence, "evidence": evidence}


def _quality_evidence(path: Path, base_dir: Path, value: str) -> dict[str, str]:
    return {"kind": "symbol", "path": _relative(path, base_dir), "value": value}


def _find_quality_pattern(
    text: str,
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
) -> str | None:
    for value, pattern in patterns:
        if pattern.search(text):
            return value
    return None


def _has_meaningful_assert(text: str) -> bool:
    assert_prefix_len = len("assert ")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("assert "):
            continue
        expression = stripped[assert_prefix_len:].strip()
        if re.fullmatch(r"(?:True|1)(?:\s*,.*)?", expression):
            continue
        return True
    return False


def _collect_test_quality(test_files: list[Path], base_dir: Path) -> dict[str, Any]:
    if not test_files:
        return _test_quality_result("none", "low", [])

    correctness_evidence: list[dict[str, str]] = []
    for path in test_files:
        text = _strip_test_comments(_read_text(path))
        graph_value = _find_quality_pattern(text, GRAPH_SK_TEST_PATTERNS)
        if graph_value:
            return _test_quality_result(
                "graph-sk-integration",
                "high",
                [_quality_evidence(path, base_dir, graph_value)],
            )

        correctness_value = _find_quality_pattern(text, CORRECTNESS_TEST_PATTERNS)
        if correctness_value:
            correctness_evidence.append(
                _quality_evidence(path, base_dir, correctness_value)
            )
        elif _has_meaningful_assert(text):
            correctness_evidence.append(_quality_evidence(path, base_dir, "assert"))

    if correctness_evidence:
        return _test_quality_result(
            "correctness-assertion", "high", correctness_evidence
        )

    return _test_quality_result(
        "smoke-launch", "medium", _file_evidence(test_files, base_dir)
    )


def _build_quality_result(
    level: str, confidence: str, evidence: list[dict[str, str]]
) -> dict[str, Any]:
    return {"level": level, "confidence": confidence, "evidence": evidence}


def _build_command_evidence(path: Path, base_dir: Path, value: str) -> dict[str, str]:
    return {"kind": "command", "path": _relative(path, base_dir), "value": value}


def _collect_build_pattern_evidence(
    build_files: list[Path],
    base_dir: Path,
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for path in build_files:
        text = _strip_test_comments(_read_text(path))
        for value, pattern in patterns:
            if not pattern.search(text):
                continue
            key = (_relative(path, base_dir), value)
            if key in seen:
                continue
            seen.add(key)
            evidence.append(_build_command_evidence(path, base_dir, value))
    return evidence


def _collect_build_quality(build_files: list[Path], base_dir: Path) -> dict[str, Any]:
    if not build_files:
        return _build_quality_result("none", "low", [])

    toolchain_evidence = _collect_build_pattern_evidence(
        build_files, base_dir, TOOLCHAIN_BUILD_PATTERNS
    )
    if toolchain_evidence:
        return _build_quality_result("toolchain-contract", "high", toolchain_evidence)

    command_evidence = _collect_build_pattern_evidence(
        build_files, base_dir, COMMAND_BUILD_PATTERNS
    )
    if command_evidence:
        return _build_quality_result("command-contract", "high", command_evidence)

    return _build_quality_result(
        "build-entry", "medium", _file_evidence(build_files, base_dir)
    )


def _classify_asset(
    source_files: list[Path],
    build_files: list[Path],
    test_files: list[Path],
    package_files: list[Path],
    doc_files: list[Path],
) -> str:
    if not source_files:
        return "unsupported-asset"
    has_complete_delivery = bool(
        build_files and test_files and package_files and doc_files
    )
    if has_complete_delivery:
        return "deliverable-op"
    if test_files:
        return "testable-op"
    if build_files:
        return "buildable-op"
    if len(source_files) > 1:
        return "source-bundle"
    return "source-only"


def _missing_contracts(
    *,
    build_files: list[Path],
    test_files: list[Path],
    package_files: list[Path],
    doc_files: list[Path],
    sk_markers: dict[str, bool],
    kernel_entries: list[dict[str, str]],
) -> list[str]:
    missing: list[str] = []
    if not kernel_entries:
        missing.append("operator_entry_contract")
    if not build_files:
        missing.append("build_contract")
    if not test_files:
        missing.append("test_contract")
    if not package_files:
        missing.append("package_contract")
    if not doc_files:
        missing.append("delivery_docs_contract")
    if not (sk_markers["SK_BIND"] or sk_markers["__sk__"] or sk_markers["__spk__"]):
        missing.append("sk_binding_contract")
    if not test_files and not doc_files:
        missing.append("operator_semantics_contract")
    return missing


def _next_actions(missing_contracts: list[str], asset_level: str) -> list[str]:
    actions: list[str] = [
        f"confirm asset level `{asset_level}` with the operator owner"
    ]
    mapping = {
        "operator_entry_contract": "identify the public kernel entry and expected launch signature",
        "build_contract": "generate or request a minimal CMake/setup build contract",
        "test_contract": "create a minimal eager or C++ correctness test scaffold",
        "package_contract": "plan the Python package or shared-library handoff layout",
        "delivery_docs_contract": "write customer-facing build, run, and limitation notes",
        "sk_binding_contract": "adapt or request SK binding details for __sk__/__spk__/SK_BIND",
        "operator_semantics_contract": "collect shape, dtype, tiling, overflow, and boundary semantics",
    }
    for contract in missing_contracts:
        action = mapping.get(contract)
        if action:
            actions.append(action)
    return actions


def _delivery_action(
    kind: str, target: str, priority: str, reason: str, next_action: str
) -> dict[str, str]:
    return {
        "kind": kind,
        "target": target,
        "priority": priority,
        "reason": reason,
        "next": next_action,
    }


def _missing_contract_action(contract: str) -> dict[str, str] | None:
    mapping = {
        "operator_entry_contract": (
            "entry",
            "high",
            "identify the public kernel entry and expected launch signature",
        ),
        "build_contract": (
            "build",
            "high",
            "generate or request a minimal CMake/setup build contract",
        ),
        "test_contract": (
            "test",
            "high",
            "create a minimal eager or C++ correctness test scaffold",
        ),
        "package_contract": (
            "package",
            "medium",
            "plan the Python package or shared-library handoff layout",
        ),
        "delivery_docs_contract": (
            "docs",
            "medium",
            "write customer-facing build, run, and limitation notes",
        ),
        "sk_binding_contract": (
            "sk_binding",
            "high",
            "adapt or request SK binding details for __sk__/__spk__/SK_BIND",
        ),
        "operator_semantics_contract": (
            "semantics",
            "high",
            "collect shape, dtype, tiling, overflow, and boundary semantics",
        ),
    }
    item = mapping.get(contract)
    if not item:
        return None
    target, priority, next_action = item
    return _delivery_action(
        "contract", target, priority, f"missing_{contract}", next_action
    )


def _missing_contract_actions(missing_contracts: list[str]) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    for contract in missing_contracts:
        action = _missing_contract_action(contract)
        if action:
            actions.append(action)
    return actions


def _delivery_plan_result(
    phase: str, confidence: str, actions: list[dict[str, str]]
) -> dict[str, Any]:
    return {"phase": phase, "confidence": confidence, "actions": actions}


def _collect_delivery_plan(
    *,
    hard_blockers: list[str],
    missing_contracts: list[str],
    asset_level: str,
    build_quality: dict[str, Any],
    test_quality: dict[str, Any],
) -> dict[str, Any]:
    actions: list[dict[str, str]] = []
    if hard_blockers:
        for blocker in hard_blockers:
            if blocker == "no_supported_operator_source_files":
                actions.append(
                    _delivery_action(
                        "blocker",
                        "source",
                        "high",
                        "no_supported_operator_source_files",
                        "provide an Ascend C/C++ operator source file or source directory",
                    )
                )
        actions.extend(_missing_contract_actions(missing_contracts))
        return _delivery_plan_result("blocked", "low", actions)

    if missing_contracts:
        actions.extend(_missing_contract_actions(missing_contracts))
        return _delivery_plan_result("contract-completion", "medium", actions)

    if build_quality["level"] == "build-entry":
        actions.append(
            _delivery_action(
                "scaffold",
                "build",
                "medium",
                "build_entry_without_command_contract",
                "turn the build entry into an explicit command or toolchain build contract",
            )
        )
    if test_quality["level"] == "smoke-launch":
        actions.append(
            _delivery_action(
                "validation",
                "test",
                "medium",
                "smoke_test_without_correctness_or_graph_sk",
                "add correctness assertions or graph/SK integration coverage",
            )
        )
    if actions:
        return _delivery_plan_result("validation-hardening", "medium", actions)

    return _delivery_plan_result(
        "handoff-planning",
        "high",
        [
            _delivery_action(
                "validation",
                "consistency",
                "medium",
                "ready_for_consistency_validation",
                "run build, smoke, correctness, and pre/post SK consistency validation in the target environment",
            ),
            _delivery_action(
                "handoff",
                "package",
                "low",
                "ready_for_package_handoff",
                "prepare customer package and runtime limitation notes after target-environment validation",
            ),
        ],
    )


def _scaffold_plan_result(
    status: str, confidence: str, items: list[dict[str, str]]
) -> dict[str, Any]:
    return {"status": status, "confidence": confidence, "items": items}


def _scaffold_item(spec_key: str) -> dict[str, str]:
    return dict(SCAFFOLD_SPECS[spec_key])


def _append_scaffold_item(
    items: list[dict[str, str]], seen_reasons: set[str], spec_key: str
) -> None:
    item = _scaffold_item(spec_key)
    if item["reason"] in seen_reasons:
        return
    seen_reasons.add(item["reason"])
    items.append(item)


def _collect_scaffold_plan(
    *,
    hard_blockers: list[str],
    missing_contracts: list[str],
    delivery_plan: dict[str, Any],
    build_quality: dict[str, Any],
    test_quality: dict[str, Any],
) -> dict[str, Any]:
    if hard_blockers:
        return _scaffold_plan_result("blocked", "low", [])

    items: list[dict[str, str]] = []
    seen_reasons: set[str] = set()
    missing = set(missing_contracts)
    for contract in SCAFFOLD_CONTRACT_ORDER:
        if contract in missing:
            _append_scaffold_item(items, seen_reasons, contract)

    if build_quality["level"] == "build-entry":
        _append_scaffold_item(
            items, seen_reasons, "build_entry_without_command_contract"
        )
    if test_quality["level"] == "smoke-launch":
        _append_scaffold_item(
            items, seen_reasons, "smoke_test_without_correctness_or_graph_sk"
        )

    if not items:
        confidence = (
            "high" if delivery_plan["phase"] == "handoff-planning" else "medium"
        )
        return _scaffold_plan_result("not-needed", confidence, [])

    reasons = {item["reason"] for item in items}
    has_missing_contract = any(reason.startswith("missing_") for reason in reasons)
    has_weak_build = "build_entry_without_command_contract" in reasons
    status = "needed" if has_missing_contract or has_weak_build else "optional"
    return _scaffold_plan_result(status, "medium", items)


def _build_manifest(asset_path: Path) -> dict[str, Any]:
    if asset_path.is_file():
        files = [asset_path]
        base_dir = asset_path.parent
    else:
        base_dir = asset_path
        files = sorted(
            (path for path in asset_path.rglob("*") if path.is_file()),
            key=lambda path: _relative(path, base_dir),
        )

    source_files = [path for path in files if _is_supported_source_file(path, base_dir)]
    include_files = [path for path in source_files if path.suffix in INCLUDE_SUFFIXES]
    build_files = [path for path in files if path.name in BUILD_FILENAMES]
    test_files = [path for path in files if _is_test_path(path, base_dir)]
    package_files = [
        path
        for path in files
        if path.name in PACKAGE_FILENAMES or _is_package_init(path, base_dir)
    ]
    doc_files = [
        path
        for path in files
        if path.name in DOC_FILENAMES or path.suffix.lower() == ".md"
    ]
    kernel_entries = _collect_kernel_entries(source_files, base_dir)
    sk_markers = _collect_sk_markers(source_files)
    asset_level = _classify_asset(
        source_files, build_files, test_files, package_files, doc_files
    )
    missing = _missing_contracts(
        build_files=build_files,
        test_files=test_files,
        package_files=package_files,
        doc_files=doc_files,
        sk_markers=sk_markers,
        kernel_entries=kernel_entries,
    )
    hard_blockers = [] if source_files else ["no_supported_operator_source_files"]
    capabilities = _collect_capabilities(
        source_files=source_files,
        build_files=build_files,
        test_files=test_files,
        package_files=package_files,
        doc_files=doc_files,
        kernel_entries=kernel_entries,
        sk_markers=sk_markers,
        hard_blockers=hard_blockers,
        base_dir=base_dir,
    )
    test_quality = _collect_test_quality(test_files, base_dir)
    build_quality = _collect_build_quality(build_files, base_dir)
    delivery_plan = _collect_delivery_plan(
        hard_blockers=hard_blockers,
        missing_contracts=missing,
        asset_level=asset_level,
        build_quality=build_quality,
        test_quality=test_quality,
    )
    scaffold_plan = _collect_scaffold_plan(
        hard_blockers=hard_blockers,
        missing_contracts=missing,
        delivery_plan=delivery_plan,
        build_quality=build_quality,
        test_quality=test_quality,
    )
    archetype = _detect_archetype(
        files=files,
        source_files=source_files,
        build_files=build_files,
        package_files=package_files,
        base_dir=base_dir,
    )

    return {
        "asset_path": str(asset_path.resolve()),
        "asset_level": asset_level,
        "base_dir": str(base_dir.resolve()),
        "source_files": [_relative(path, base_dir) for path in source_files],
        "include_files": [_relative(path, base_dir) for path in include_files],
        "kernel_entries": kernel_entries,
        "build_system": [_relative(path, base_dir) for path in build_files],
        "test_system": [_relative(path, base_dir) for path in test_files],
        "package_system": [_relative(path, base_dir) for path in package_files],
        "doc_files": [_relative(path, base_dir) for path in doc_files],
        "sk_markers": sk_markers,
        "capabilities": capabilities,
        "archetype": archetype,
        "test_quality": test_quality,
        "build_quality": build_quality,
        "delivery_plan": delivery_plan,
        "scaffold_plan": scaffold_plan,
        "missing_contracts": missing,
        "supported_next_actions": _next_actions(missing, asset_level),
        "hard_blockers": hard_blockers,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _sk_file_evidence(paths: list[str]) -> list[dict[str, str]]:
    return [{"kind": "file", "path": path, "value": Path(path).name} for path in paths]


def _sk_source_file_fingerprints(
    base_dir: str, source_files: list[str]
) -> list[dict[str, str]]:
    root = Path(base_dir)
    return [
        {"path": rel_path, "sha256": _hash_file(root / rel_path)}
        for rel_path in source_files
    ]


def _support_dir_name(name: str) -> str:
    invalid_name = not name or name in {".", ".."}
    contains_separator = "/" in name or "\\" in name
    unsupported_chars = not SUPPORT_DIR_NAME_RE.fullmatch(name)
    if invalid_name or contains_separator or unsupported_chars:
        raise CliUsageError(f"invalid support directory name: {name}")
    return name


def _parse_support_dir_arg(raw: str) -> tuple[str | None, Path]:
    if "=" in raw:
        name, path = raw.split("=", 1)
        if not path:
            raise CliUsageError("support directory path is empty")
        return _support_dir_name(name), Path(path)
    return None, Path(raw)


def _support_source_files(root: Path) -> list[str]:
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and _is_supported_source_file(path, root)
        ),
        key=lambda path: _relative(path, root),
    )
    return [_relative(path, root) for path in files]


def _normalize_support_directories(
    raw_specs: list[str] | None, asset_base: Path
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    asset_root = asset_base.resolve()
    for raw in raw_specs or []:
        explicit_name, raw_path = _parse_support_dir_arg(raw)
        path = raw_path.resolve()
        if not path.exists():
            raise CliUsageError(f"support directory not found: {raw_path}")
        if not path.is_dir():
            raise CliUsageError(f"support path is not a directory: {raw_path}")
        if _is_same_or_child_path(path, asset_root) or _is_same_or_child_path(
            asset_root, path
        ):
            raise CliUsageError(
                "support directory must be a sibling/shared directory outside the asset root"
            )
        name = explicit_name or _support_dir_name(path.name)
        if name in seen_names:
            raise CliUsageError(f"duplicate support directory name: {name}")
        files = _support_source_files(path)
        if not files:
            raise CliUsageError(
                f"support directory has no supported source files: {path}"
            )
        seen_names.add(name)
        specs.append(
            {
                "source_name": name,
                "source_path": str(path),
                "source_files": files,
                "source_file_fingerprints": _sk_source_file_fingerprints(
                    str(path), files
                ),
            }
        )
    return specs


def _normalize_include_directories(raw_dirs: list[str] | None) -> list[str]:
    include_dirs: list[str] = []
    seen: set[str] = set()
    for raw in raw_dirs or []:
        path = Path(raw).resolve()
        if not path.exists():
            raise CliUsageError(f"include directory not found: {raw}")
        if not path.is_dir():
            raise CliUsageError(f"include path is not a directory: {raw}")
        value = str(path)
        if value not in seen:
            seen.add(value)
            include_dirs.append(value)
    return include_dirs


def _normalize_ascend_compile_options(raw_options: list[str] | None) -> list[str]:
    options: list[str] = []
    for raw in raw_options or []:
        if not raw or any(char in raw for char in "\r\n\0;"):
            raise CliUsageError(f"invalid ascend compile option: {raw}")
        options.append(raw)
    if not any(option.startswith("-std=") for option in options):
        options.insert(0, "-std=c++17")
    return options


def _normalize_ascend_force_includes(raw_paths: list[str] | None) -> list[str]:
    force_includes: list[str] = []
    seen: set[str] = set()
    for raw in raw_paths or []:
        path = Path(raw).resolve()
        if not path.exists():
            raise CliUsageError(f"ascend force include not found: {raw}")
        if not path.is_file():
            raise CliUsageError(f"ascend force include is not a file: {raw}")
        value = str(path)
        if value not in seen:
            seen.add(value)
            force_includes.append(value)
    return force_includes


def _ascend_derived_include_dirs(cann_path: Path) -> list[str]:
    asc_root = cann_path / "aarch64-linux" / "asc"
    return [
        str(asc_root / "include"),
        str(asc_root / "include" / "basic_api"),
        str(asc_root / "impl"),
        str(asc_root / "impl" / "basic_api"),
        str(asc_root),
    ]


def _normalize_ascend_compile_contract(
    *,
    cann_path: str | None = None,
    arch: str | None = None,
    compile_options: list[str] | None = None,
    force_includes: list[str] | None = None,
) -> dict[str, Any] | None:
    has_compile_contract = bool(cann_path or arch or compile_options or force_includes)
    if not has_compile_contract:
        return None
    if not cann_path:
        raise CliUsageError(
            "ascend CANN path is required when ascend compile contract is enabled"
        )
    if not arch:
        raise CliUsageError(
            "ascend arch is required when ascend compile contract is enabled"
        )
    if not ASCEND_ARCH_RE.fullmatch(arch):
        raise CliUsageError(f"invalid ascend arch: {arch}")

    normalized_cann_path = Path(cann_path).resolve()
    if not normalized_cann_path.exists():
        raise CliUsageError(f"ascend CANN path not found: {cann_path}")
    if not normalized_cann_path.is_dir():
        raise CliUsageError(f"ascend CANN path is not a directory: {cann_path}")

    bisheng_path = (
        normalized_cann_path / "tools" / "bisheng_compiler" / "bin" / "bisheng"
    )
    if not bisheng_path.exists():
        raise CliUsageError(
            f"bisheng compiler not found under ascend CANN path: {bisheng_path}"
        )
    if not bisheng_path.is_file():
        raise CliUsageError(f"bisheng compiler is not a file: {bisheng_path}")

    derived_include_dirs = _ascend_derived_include_dirs(normalized_cann_path)
    for include_dir in derived_include_dirs:
        path = Path(include_dir)
        if not path.exists():
            raise CliUsageError(f"ascend include directory not found: {path}")
        if not path.is_dir():
            raise CliUsageError(f"ascend include path is not a directory: {path}")

    return {
        "cann_path": str(normalized_cann_path),
        "bisheng_path": str(bisheng_path),
        "arch": arch,
        "compile_options": _normalize_ascend_compile_options(compile_options),
        "force_includes": _normalize_ascend_force_includes(force_includes),
        "derived_include_dirs": derived_include_dirs,
    }


def _sk_source_copy_items_from_parts(
    source_files: list[str],
    include_files: list[str],
    support_directories: list[dict[str, Any]],
) -> list[str]:
    items: list[str] = []
    for rel_path in source_files:
        if rel_path not in items:
            items.append(rel_path)
    for rel_path in include_files:
        if rel_path not in items:
            items.append(rel_path)
    for support_dir in support_directories:
        for rel_path in support_dir["source_files"]:
            support_path = f"{support_dir['source_name']}/{rel_path}"
            if support_path not in items:
                items.append(support_path)
    return items


def _raw_sk_source_copy_entries(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    base_dir = Path(analysis["base_dir"])
    for rel_path in analysis["source_files"] + analysis["include_files"]:
        entries.append(
            {
                "source": rel_path,
                "source_path": base_dir / rel_path,
                "scaffold_path": f"{SK_SOURCE_SCAFFOLD_DIR}/src/{rel_path}",
            }
        )
    for support_dir in analysis["support_directories"]:
        support_base = Path(support_dir["source_path"])
        for rel_path in support_dir["source_files"]:
            destination_rel = f"{support_dir['source_name']}/{rel_path}"
            entries.append(
                {
                    "source": destination_rel,
                    "source_path": support_base / rel_path,
                    "scaffold_path": f"{SK_SOURCE_SCAFFOLD_DIR}/src/{destination_rel}",
                }
            )
    return entries


def _sk_source_copy_entries(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for entry in _raw_sk_source_copy_entries(analysis):
        scaffold_path = entry["scaffold_path"]
        previous = seen.get(scaffold_path)
        if previous is not None:
            if _hash_file(previous["source_path"]) == _hash_file(entry["source_path"]):
                continue
            raise CliUsageError(
                f"conflicting sk source scaffold destination: {scaffold_path}"
            )
        seen[scaffold_path] = entry
        entries.append(entry)
    return entries


def _validate_sk_source_destination_collisions(analysis: dict[str, Any]) -> None:
    _sk_source_copy_entries(analysis)


def _sk_symbol_evidence(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"kind": "symbol", "path": entry["file"], "value": entry["name"]}
        for entry in entries
    ]


def _sk_marker_evidence(sk_markers: dict[str, bool]) -> list[dict[str, str]]:
    return [
        {"kind": "marker", "path": "", "value": name}
        for name, found in sk_markers.items()
        if found
    ]


def _has_sk_binding_marker(sk_markers: dict[str, bool]) -> bool:
    return bool(sk_markers["SK_BIND"] or sk_markers["__sk__"] or sk_markers["__spk__"])


def _sk_conversion_input(
    name: str, status: str, reason: str, evidence: list[dict[str, str]]
) -> dict[str, Any]:
    return {"name": name, "status": status, "reason": reason, "evidence": evidence}


def _sk_generation_step(
    action: str, status: str, reason: str, evidence: list[dict[str, str]]
) -> dict[str, Any]:
    return {"action": action, "status": status, "reason": reason, "evidence": evidence}


def _sk_conversion_minimal_unit(manifest: dict[str, Any], status: str) -> str:
    if status == "blocked":
        return "unsupported"
    mapping = {
        "source-only": "kernel_source_set",
        "source-bundle": "source_bundle_kernel_set",
        "buildable-op": "buildable_operator_project",
        "testable-op": "testable_operator_project",
        "deliverable-op": "deliverable_operator_project",
    }
    return mapping.get(manifest["asset_level"], "unsupported")


def _sk_conversion_overall_status(manifest: dict[str, Any]) -> tuple[str, list[str]]:
    if not manifest["source_files"]:
        return "blocked", ["no_supported_operator_source_files"]
    if not manifest["kernel_entries"]:
        return "blocked", ["kernel_entries_missing"]
    ready = all(
        [
            _has_sk_binding_marker(manifest["sk_markers"]),
            manifest["build_system"],
            manifest["test_system"],
        ]
    )
    if ready:
        return "ready-for-sk-generation", []
    return "needs-contracts", []


def _sk_conversion_inputs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    source_evidence = _sk_file_evidence(manifest["source_files"])
    entry_evidence = _sk_symbol_evidence(manifest["kernel_entries"])
    marker_evidence = _sk_marker_evidence(manifest["sk_markers"])
    has_source = bool(manifest["source_files"])
    inputs: list[dict[str, Any]] = []

    if has_source:
        inputs.append(
            _sk_conversion_input(
                "source", "present", "source_files_detected", source_evidence
            )
        )
    else:
        inputs.append(
            _sk_conversion_input(
                "source", "blocked", "no_supported_operator_source_files", []
            )
        )

    if manifest["kernel_entries"]:
        inputs.append(
            _sk_conversion_input(
                "entry_signature", "present", "kernel_entries_detected", entry_evidence
            )
        )
    elif has_source:
        inputs.append(
            _sk_conversion_input(
                "entry_signature", "blocked", "kernel_entries_missing", source_evidence
            )
        )
    else:
        inputs.append(
            _sk_conversion_input(
                "entry_signature", "blocked", "no_supported_operator_source_files", []
            )
        )

    if not has_source:
        inputs.append(
            _sk_conversion_input(
                "sk_binding", "blocked", "no_supported_operator_source_files", []
            )
        )
    elif _has_sk_binding_marker(manifest["sk_markers"]):
        inputs.append(
            _sk_conversion_input(
                "sk_binding", "present", "sk_binding_marker_detected", marker_evidence
            )
        )
    else:
        inputs.append(
            _sk_conversion_input(
                "sk_binding", "missing", "sk_binding_contract_missing", []
            )
        )

    for name, manifest_key, present_reason, missing_reason in (
        (
            "build_contract",
            "build_system",
            "build_contract_detected",
            "build_contract_missing",
        ),
        (
            "test_contract",
            "test_system",
            "test_contract_detected",
            "test_contract_missing",
        ),
        (
            "package_contract",
            "package_system",
            "package_contract_detected",
            "package_contract_missing",
        ),
        (
            "delivery_docs",
            "doc_files",
            "delivery_docs_detected",
            "delivery_docs_missing",
        ),
    ):
        evidence = _sk_file_evidence(manifest[manifest_key])
        if not has_source:
            inputs.append(
                _sk_conversion_input(
                    name, "blocked", "no_supported_operator_source_files", []
                )
            )
        elif evidence:
            inputs.append(
                _sk_conversion_input(name, "present", present_reason, evidence)
            )
        else:
            inputs.append(_sk_conversion_input(name, "missing", missing_reason, []))

    if has_source:
        inputs.append(
            _sk_conversion_input(
                "runtime_inputs",
                "not-collected",
                "runtime_input_spec_not_collected",
                [],
            )
        )
        inputs.append(
            _sk_conversion_input(
                "correctness_oracle",
                "not-collected",
                "correctness_oracle_spec_not_collected",
                [],
            )
        )
    else:
        inputs.append(
            _sk_conversion_input(
                "runtime_inputs", "blocked", "no_supported_operator_source_files", []
            )
        )
        inputs.append(
            _sk_conversion_input(
                "correctness_oracle",
                "blocked",
                "no_supported_operator_source_files",
                [],
            )
        )

    return inputs


def _sk_conversion_plan(
    manifest: dict[str, Any],
    status: str,
    blocked_reasons: list[str],
    inputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_evidence = _sk_file_evidence(manifest["source_files"])
    entry_evidence = _sk_symbol_evidence(manifest["kernel_entries"])
    marker_evidence = _sk_marker_evidence(manifest["sk_markers"])
    has_source = bool(manifest["source_files"])
    has_entry = bool(manifest["kernel_entries"])
    plan: list[dict[str, Any]] = []

    if has_entry:
        plan.append(
            _sk_generation_step(
                "identify_kernel_entry_signature",
                "ready",
                "kernel_entries_detected",
                entry_evidence,
            )
        )
    elif has_source:
        plan.append(
            _sk_generation_step(
                "identify_kernel_entry_signature",
                "needed",
                "kernel_entries_missing",
                source_evidence,
            )
        )
    else:
        plan.append(
            _sk_generation_step(
                "identify_kernel_entry_signature",
                "blocked",
                "no_supported_operator_source_files",
                [],
            )
        )

    if not has_source:
        plan.append(
            _sk_generation_step(
                "adapt_sk_binding", "blocked", "no_supported_operator_source_files", []
            )
        )
    elif _has_sk_binding_marker(manifest["sk_markers"]):
        plan.append(
            _sk_generation_step(
                "adapt_sk_binding",
                "ready",
                "sk_binding_marker_detected",
                marker_evidence,
            )
        )
    else:
        plan.append(
            _sk_generation_step(
                "adapt_sk_binding", "needed", "sk_binding_contract_missing", []
            )
        )

    for action, manifest_key, present_reason, missing_reason in (
        (
            "generate_minimal_build_contract",
            "build_system",
            "build_contract_detected",
            "build_contract_missing",
        ),
        (
            "generate_minimal_test_contract",
            "test_system",
            "test_contract_detected",
            "test_contract_missing",
        ),
        (
            "generate_package_contract",
            "package_system",
            "package_contract_detected",
            "package_contract_missing",
        ),
    ):
        evidence = _sk_file_evidence(manifest[manifest_key])
        if not has_source:
            plan.append(
                _sk_generation_step(
                    action, "blocked", "no_supported_operator_source_files", []
                )
            )
        elif evidence:
            plan.append(_sk_generation_step(action, "ready", present_reason, evidence))
        else:
            plan.append(_sk_generation_step(action, "needed", missing_reason, []))

    if has_source and has_entry:
        plan.append(
            _sk_generation_step(
                "collect_runtime_input_spec",
                "needed",
                "runtime_input_spec_not_collected",
                entry_evidence,
            )
        )
        plan.append(
            _sk_generation_step(
                "collect_correctness_oracle_spec",
                "needed",
                "correctness_oracle_spec_not_collected",
                entry_evidence,
            )
        )
    elif has_source:
        plan.append(
            _sk_generation_step(
                "collect_runtime_input_spec",
                "blocked",
                "kernel_entries_missing",
                source_evidence,
            )
        )
        plan.append(
            _sk_generation_step(
                "collect_correctness_oracle_spec",
                "blocked",
                "kernel_entries_missing",
                source_evidence,
            )
        )
    else:
        plan.append(
            _sk_generation_step(
                "collect_runtime_input_spec",
                "blocked",
                "no_supported_operator_source_files",
                [],
            )
        )
        plan.append(
            _sk_generation_step(
                "collect_correctness_oracle_spec",
                "blocked",
                "no_supported_operator_source_files",
                [],
            )
        )

    if status == "ready-for-sk-generation":
        ready_evidence: list[dict[str, str]] = []
        for input_name in (
            "source",
            "entry_signature",
            "sk_binding",
            "build_contract",
            "test_contract",
        ):
            ready_evidence.extend(
                next(item["evidence"] for item in inputs if item["name"] == input_name)
            )
        for input_name in ("package_contract", "delivery_docs"):
            item = next(item for item in inputs if item["name"] == input_name)
            if item["status"] == "present":
                ready_evidence.extend(item["evidence"])
        plan.append(
            _sk_generation_step(
                "generate_sk_source_scaffold",
                "ready",
                "sk_generation_prerequisites_present",
                ready_evidence,
            )
        )
        plan.append(
            _sk_generation_step(
                "run_sk_build_validation", "deferred", "sk_source_not_generated", []
            )
        )
    elif status == "needs-contracts":
        plan.append(
            _sk_generation_step(
                "generate_sk_source_scaffold",
                "blocked",
                "conversion_contracts_missing",
                [],
            )
        )
        plan.append(
            _sk_generation_step(
                "run_sk_build_validation", "deferred", "sk_source_not_generated", []
            )
        )
    else:
        reason = blocked_reasons[0]
        evidence = source_evidence if reason == "kernel_entries_missing" else []
        plan.append(
            _sk_generation_step(
                "generate_sk_source_scaffold", "blocked", reason, evidence
            )
        )
        plan.append(
            _sk_generation_step("run_sk_build_validation", "blocked", reason, evidence)
        )

    return plan


def _sk_conversion_next_actions(
    status: str, blocked_reasons: list[str], plan: list[dict[str, Any]]
) -> list[str]:
    if status == "ready-for-sk-generation":
        return ["generate_sk_source_scaffold"]
    if blocked_reasons == ["no_supported_operator_source_files"]:
        return ["provide_supported_operator_source"]
    if blocked_reasons == ["kernel_entries_missing"]:
        return ["identify_kernel_entry_signature"]
    return [item["action"] for item in plan if item["status"] == "needed"][:3]


class SkConversionAnalysisInput(NamedTuple):
    asset_path: Path
    support_dir_specs: list[str] | None = None
    include_dir_specs: list[str] | None = None
    ascend_cann_path: str | None = None
    ascend_arch: str | None = None
    ascend_compile_options: list[str] | None = None
    ascend_force_includes: list[str] | None = None


def _build_sk_conversion_analysis(request: SkConversionAnalysisInput) -> dict[str, Any]:
    manifest = _build_manifest(request.asset_path)
    support_directories = _normalize_support_directories(
        request.support_dir_specs, Path(manifest["base_dir"])
    )
    include_directories = _normalize_include_directories(request.include_dir_specs)
    ascend_compile_contract = _normalize_ascend_compile_contract(
        cann_path=request.ascend_cann_path,
        arch=request.ascend_arch,
        compile_options=request.ascend_compile_options,
        force_includes=request.ascend_force_includes,
    )
    status, blocked_reasons = _sk_conversion_overall_status(manifest)
    inputs = _sk_conversion_inputs(manifest)
    plan = _sk_conversion_plan(manifest, status, blocked_reasons, inputs)
    analysis = {
        "status": status,
        "asset_path": manifest["asset_path"],
        "base_dir": manifest["base_dir"],
        "asset_level": manifest["asset_level"],
        "archetype": manifest["archetype"],
        "supported_minimal_unit": _sk_conversion_minimal_unit(manifest, status),
        "source_files": manifest["source_files"],
        "source_file_fingerprints": _sk_source_file_fingerprints(
            manifest["base_dir"], manifest["source_files"]
        ),
        "include_files": manifest["include_files"],
        "support_directories": support_directories,
        "include_directories": include_directories,
        "ascend_compile_contract": ascend_compile_contract,
        "kernel_entries": manifest["kernel_entries"],
        "sk_markers": manifest["sk_markers"],
        "build_system": manifest["build_system"],
        "test_system": manifest["test_system"],
        "package_system": manifest["package_system"],
        "doc_files": manifest["doc_files"],
        "conversion_inputs": inputs,
        "generation_plan": plan,
        "blocked_reasons": blocked_reasons,
        "execution_boundary": SK_CONVERSION_BOUNDARY,
        "supported_next_actions": _sk_conversion_next_actions(
            status, blocked_reasons, plan
        ),
    }
    _validate_sk_source_destination_collisions(analysis)
    return analysis


def _render_sk_conversion_analysis_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Operator SK Conversion Analysis",
        "",
        f"- status: `{analysis['status']}`",
        f"- asset level: `{analysis['asset_level']}`",
        f"- supported minimal unit: `{analysis['supported_minimal_unit']}`",
        f"- archetype: `{analysis['archetype']['name']}`",
        "",
        "## Conversion Inputs",
        "",
    ]
    for item in analysis["conversion_inputs"]:
        lines.append(f"- `{item['name']}`: `{item['status']}` ({item['reason']})")
    lines.extend(["", "## Declared Dependencies", ""])
    if analysis["support_directories"]:
        lines.append("Support directories copied into the SK source version:")
        for item in analysis["support_directories"]:
            lines.append(
                f"- `{item['source_name']}`: `{item['source_path']}` ({len(item['source_files'])} files)"
            )
    else:
        lines.append("- support directories: none")
    if analysis["include_directories"]:
        lines.append("Include directories used as build search roots:")
        for item in analysis["include_directories"]:
            lines.append(f"- `{item}`")
    else:
        lines.append("- include directories: none")
    if analysis["ascend_compile_contract"]:
        contract = analysis["ascend_compile_contract"]
        lines.append("Ascend compile contract:")
        lines.append(f"- CANN path: `{contract['cann_path']}`")
        lines.append(f"- bisheng: `{contract['bisheng_path']}`")
        lines.append(f"- arch: `{contract['arch']}`")
        if contract["force_includes"]:
            for item in contract["force_includes"]:
                lines.append(f"- force include: `{item}`")
        else:
            lines.append("- force includes: none")
    else:
        lines.append("- ascend compile contract: none")
    lines.extend(["", "## Generation Plan", ""])
    for item in analysis["generation_plan"]:
        lines.append(f"- `{item['action']}`: `{item['status']}` ({item['reason']})")
    lines.extend(["", "## Supported Next Actions", ""])
    for item in analysis["supported_next_actions"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Execution Boundary", ""])
    for item in analysis["execution_boundary"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This artifact is a static readiness analysis for SK conversion planning.",
            "- It does not prove build, runtime, correctness, package, or customer delivery success.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_sk_conversion_artifacts(output_dir: Path, analysis: dict[str, Any]) -> None:
    _write_json(output_dir / "operator-sk-conversion-analysis.json", analysis)
    _write_text(
        output_dir / "operator-sk-conversion-analysis.md",
        _render_sk_conversion_analysis_markdown(analysis),
    )


def _load_current_sk_conversion_analysis(output_dir: Path) -> dict[str, Any]:
    analysis_path = output_dir / "operator-sk-conversion-analysis.json"
    payload = _load_json_artifact(analysis_path, "operator-sk-conversion-analysis.json")
    analysis = _require_exact_keys(
        payload,
        {
            "status",
            "asset_path",
            "base_dir",
            "asset_level",
            "archetype",
            "supported_minimal_unit",
            "source_files",
            "source_file_fingerprints",
            "include_files",
            "support_directories",
            "include_directories",
            "ascend_compile_contract",
            "kernel_entries",
            "sk_markers",
            "build_system",
            "test_system",
            "package_system",
            "doc_files",
            "conversion_inputs",
            "generation_plan",
            "blocked_reasons",
            "execution_boundary",
            "supported_next_actions",
        },
        "sk conversion analysis",
    )
    asset_path = Path(
        _require_string(analysis["asset_path"], "sk conversion analysis asset_path")
    ).resolve()
    support_directories: list[dict[str, Any]] = []
    for support_dir in _require_list(
        analysis["support_directories"], "sk conversion analysis support_directories"
    ):
        item = _require_exact_keys(
            support_dir,
            {"source_name", "source_path", "source_files", "source_file_fingerprints"},
            "sk conversion analysis support directory",
        )
        source_name = _support_dir_name(
            _require_string(
                item["source_name"],
                "sk conversion analysis support directory source_name",
            )
        )
        source_path = str(
            Path(
                _require_string(
                    item["source_path"],
                    "sk conversion analysis support directory source_path",
                )
            ).resolve()
        )
        source_files = _validate_manifest_path_list(
            item["source_files"],
            "sk conversion analysis support source_files",
            "sk conversion analysis support source file",
        )
        fingerprints: list[dict[str, str]] = []
        for fingerprint in _require_list(
            item["source_file_fingerprints"],
            "sk conversion analysis support source_file_fingerprints",
        ):
            fingerprint_item = _require_exact_keys(
                fingerprint,
                {"path", "sha256"},
                "sk conversion analysis support fingerprint",
            )
            fingerprints.append(
                {
                    "path": _validate_manifest_path(
                        fingerprint_item["path"],
                        "sk conversion analysis support fingerprint path",
                    ),
                    "sha256": _require_string(
                        fingerprint_item["sha256"],
                        "sk conversion analysis support fingerprint sha256",
                    ),
                }
            )
        support_directories.append(
            {
                "source_name": source_name,
                "source_path": source_path,
                "source_files": source_files,
                "source_file_fingerprints": fingerprints,
            }
        )
    analysis["support_directories"] = support_directories
    analysis["include_directories"] = _resolve_string_path_list(
        analysis["include_directories"],
        "sk conversion analysis include_directories",
        "sk conversion analysis include directory",
    )
    raw_contract = analysis["ascend_compile_contract"]
    if raw_contract is None:
        ascend_compile_contract = None
    else:
        contract = _require_exact_keys(
            raw_contract,
            {
                "cann_path",
                "bisheng_path",
                "arch",
                "compile_options",
                "force_includes",
                "derived_include_dirs",
            },
            "sk conversion analysis ascend compile contract",
        )
        ascend_compile_contract = {
            "cann_path": str(
                Path(
                    _require_string(
                        contract["cann_path"], "ascend compile contract cann_path"
                    )
                ).resolve()
            ),
            "bisheng_path": str(
                Path(
                    _require_string(
                        contract["bisheng_path"], "ascend compile contract bisheng_path"
                    )
                ).resolve()
            ),
            "arch": _require_string(contract["arch"], "ascend compile contract arch"),
            "compile_options": _require_string_list(
                contract["compile_options"],
                "ascend compile contract compile_options",
                "ascend compile contract compile option",
            ),
            "force_includes": _resolve_string_path_list(
                contract["force_includes"],
                "ascend compile contract force_includes",
                "ascend compile contract force include",
            ),
            "derived_include_dirs": _resolve_string_path_list(
                contract["derived_include_dirs"],
                "ascend compile contract derived_include_dirs",
                "ascend compile contract derived include dir",
            ),
        }
    analysis["ascend_compile_contract"] = ascend_compile_contract
    expected = _build_sk_conversion_analysis(
        SkConversionAnalysisInput(
            asset_path,
            [
                f"{item['source_name']}={item['source_path']}"
                for item in support_directories
            ],
            analysis["include_directories"],
            ascend_compile_contract["cann_path"] if ascend_compile_contract else None,
            ascend_compile_contract["arch"] if ascend_compile_contract else None,
            ascend_compile_contract["compile_options"]
            if ascend_compile_contract
            else None,
            ascend_compile_contract["force_includes"]
            if ascend_compile_contract
            else None,
        )
    )
    if analysis != expected:
        raise CliUsageError("sk conversion analysis is not current")
    return analysis


def _sk_source_empty_manifest(
    *,
    status: str,
    output_dir: Path,
    analysis: dict[str, Any],
    blocked_reasons: list[str],
    supported_next_actions: list[str],
) -> dict[str, Any]:
    return {
        "status": status,
        "analysis_output_dir": str(output_dir.resolve()),
        "analysis_path": "operator-sk-conversion-analysis.json",
        "asset_path": analysis["asset_path"],
        "source_files": analysis["source_files"],
        "source_file_fingerprints": analysis["source_file_fingerprints"],
        "support_directories": analysis["support_directories"],
        "include_directories": analysis["include_directories"],
        "ascend_compile_contract": analysis["ascend_compile_contract"],
        "kernel_entries": analysis["kernel_entries"],
        "copied_source_files": [],
        "generated_files": [],
        "blocked_reasons": blocked_reasons,
        "execution_boundary": SK_SOURCE_SCAFFOLD_BOUNDARY,
        "supported_next_actions": supported_next_actions,
    }


def _sk_source_needed_next_actions(analysis: dict[str, Any]) -> list[str]:
    return [
        item["action"]
        for item in analysis["generation_plan"]
        if item["status"] == "needed"
    ][:3]


def _raise_if_existing_non_generated_sk_source_scaffold(output_dir: Path) -> None:
    if (output_dir / SK_SOURCE_SCAFFOLD_DIR).exists():
        raise CliUsageError(
            "sk source scaffold directory exists for non-generated result"
        )


def _write_sk_source_artifacts(output_dir: Path, manifest: dict[str, Any]) -> None:
    _write_json(output_dir / "operator-sk-source-scaffold.json", manifest)
    _write_text(
        output_dir / "operator-sk-source-scaffold.md",
        _render_sk_source_scaffold_markdown(manifest),
    )


def _render_sk_source_scaffold_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Operator SK Source Scaffold",
        "",
        "## Status",
        "",
        f"- status: `{manifest['status']}`",
        "",
        "## Copied Sources",
        "",
    ]
    if manifest["copied_source_files"]:
        for item in manifest["copied_source_files"]:
            lines.append(f"- `{item['source']}` -> `{item['scaffold_path']}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Generated Files", ""])
    if manifest["generated_files"]:
        for item in manifest["generated_files"]:
            lines.append(f"- `{item}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Blocked Reasons", ""])
    if manifest["blocked_reasons"]:
        for item in manifest["blocked_reasons"]:
            lines.append(f"- `{item}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Supported Next Actions", ""])
    for item in manifest["supported_next_actions"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Execution Boundary", ""])
    for item in manifest["execution_boundary"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This artifact is a review scaffold for SK source generation.",
            "- It does not prove build, runtime, correctness, package, or customer delivery success.",
        ]
    )
    return "\n".join(lines) + "\n"


def _sk_source_copy_items(analysis: dict[str, Any]) -> list[str]:
    return _sk_source_copy_items_from_parts(
        analysis["source_files"],
        analysis["include_files"],
        analysis["support_directories"],
    )


def _sk_source_copy_source_path(analysis: dict[str, Any], rel_path: str) -> Path:
    for entry in _sk_source_copy_entries(analysis):
        if entry["source"] == rel_path:
            return Path(entry["source_path"])
    return Path(analysis["base_dir"]) / rel_path


def _copy_sk_source_scaffold_files(
    output_dir: Path, analysis: dict[str, Any]
) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for entry in _sk_source_copy_entries(analysis):
        source_path = Path(entry["source_path"])
        scaffold_path = entry["scaffold_path"]
        destination = output_dir / scaffold_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        copied.append({"source": entry["source"], "scaffold_path": scaffold_path})
    return copied


def _sk_source_scaffold_file_entries(analysis: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "source": entry["source"],
            "scaffold_path": entry["scaffold_path"],
        }
        for entry in _sk_source_copy_entries(analysis)
    ]


def _sk_source_scaffold_relative_path(scaffold_path: str) -> str:
    prefix = f"{SK_SOURCE_SCAFFOLD_DIR}/"
    if not scaffold_path.startswith(prefix):
        raise CliUsageError("sk source scaffold mismatch")
    relative_path = scaffold_path.removeprefix(prefix)
    if not relative_path:
        raise CliUsageError("sk source scaffold mismatch")
    return relative_path


def _render_sk_source_cmake(manifest: dict[str, Any]) -> str:
    copied_source_files = [
        {
            "source": item["source"],
            "scaffold_path": _sk_source_scaffold_relative_path(item["scaffold_path"]),
        }
        for item in manifest["copied_source_files"]
    ]
    return _render_scaffold_cmake(manifest, copied_source_files)


def _write_sk_source_scaffold_readme(
    output_dir: Path, manifest: dict[str, Any]
) -> None:
    lines = [
        "# Operator SK Source Scaffold",
        "",
        "This directory contains reviewable SK source scaffold files.",
        "It does not prove build, runtime, correctness, package, or customer delivery success.",
        "",
        "## Copied Sources",
        "",
    ]
    for item in manifest["copied_source_files"]:
        lines.append(f"- `{item['source']}` -> `{item['scaffold_path']}`")
    lines.extend(["", "## Execution Boundary", ""])
    for item in manifest["execution_boundary"]:
        lines.append(f"- {item}")
    _write_text(
        output_dir / SK_SOURCE_SCAFFOLD_DIR / "README.md", "\n".join(lines) + "\n"
    )


def _generate_sk_source_scaffold(
    output_dir: Path, analysis: dict[str, Any]
) -> dict[str, Any]:
    scaffold_dir = output_dir / SK_SOURCE_SCAFFOLD_DIR
    if scaffold_dir.exists():
        raise CliUsageError("sk source scaffold output already exists")

    copied_sources = _copy_sk_source_scaffold_files(output_dir, analysis)
    manifest = {
        "status": "generated",
        "analysis_output_dir": str(output_dir.resolve()),
        "analysis_path": "operator-sk-conversion-analysis.json",
        "asset_path": analysis["asset_path"],
        "source_files": analysis["source_files"],
        "source_file_fingerprints": analysis["source_file_fingerprints"],
        "support_directories": analysis["support_directories"],
        "include_directories": analysis["include_directories"],
        "ascend_compile_contract": analysis["ascend_compile_contract"],
        "kernel_entries": analysis["kernel_entries"],
        "copied_source_files": copied_sources,
        "generated_files": [
            f"{SK_SOURCE_SCAFFOLD_DIR}/README.md",
            f"{SK_SOURCE_SCAFFOLD_DIR}/CMakeLists.txt",
        ],
        "blocked_reasons": [],
        "execution_boundary": SK_SOURCE_SCAFFOLD_BOUNDARY,
        "supported_next_actions": ["run_sk_build_validation"],
    }
    _write_sk_source_scaffold_readme(output_dir, manifest)
    _write_text(
        output_dir / SK_SOURCE_SCAFFOLD_DIR / "CMakeLists.txt",
        _render_sk_source_cmake(manifest),
    )
    return manifest


def _safe_target_name(manifest: dict[str, Any]) -> str:
    raw_name = "operator_scaffold"
    if manifest["kernel_entries"]:
        raw_name = manifest["kernel_entries"][0]["name"]
    target = re.sub(r"\W+", "_", raw_name).strip("_") or "operator_scaffold"
    if target[0].isdigit():
        target = f"op_{target}"
    return target


def _cmake_quoted(value: str) -> str:
    return '"' + value.replace("\\", "/").replace('"', '\\"') + '"'


def _safe_object_name(scaffold_path: str) -> str:
    safe = (
        re.sub(r"[^A-Za-z0-9_.-]+", "_", scaffold_path).strip("_") or "operator_source"
    )
    return f"{safe}.o"


def _render_ascend_scaffold_cmake(
    manifest: dict[str, Any], copied_source_files: list[dict[str, str]]
) -> str:
    contract = manifest["ascend_compile_contract"]
    ascend_sources = [
        item["scaffold_path"]
        for item in copied_source_files
        if Path(item["source"]).suffix in ASCEND_COMPILABLE_SOURCE_SUFFIXES
    ]
    headers = [
        item["scaffold_path"]
        for item in copied_source_files
        if Path(item["source"]).suffix in INCLUDE_SUFFIXES
    ]
    target_name = _safe_target_name(manifest)
    include_dirs = []
    for include_dir in [
        *contract["derived_include_dirs"],
        *manifest.get("include_directories", []),
        "${CMAKE_CURRENT_SOURCE_DIR}/src",
    ]:
        if include_dir not in include_dirs:
            include_dirs.append(include_dir)
    compile_options = [
        "--cce-aicore-lang",
        f"--npu-arch={contract['arch']}",
        *contract["compile_options"],
    ]

    lines = [
        "cmake_minimum_required(VERSION 3.16)",
        f"project({target_name}_scaffold LANGUAGES NONE)",
        "",
        "# Generated Ascend scaffold contract only.",
        "# Build validation compiles .asc sources with the declared Bisheng/CANN contract.",
        "# Runtime, correctness, package, and customer handoff are not validated here.",
        "",
        f"set(ASCEND_BISHENG {_cmake_quoted(contract['bisheng_path'])})",
    ]
    if headers:
        lines.extend(["set(SCAFFOLD_HEADERS"])
        lines.extend(f"  {_cmake_quoted(path)}" for path in headers)
        lines.extend([")", ""])
    if ascend_sources:
        outputs: list[str] = []
        for source in ascend_sources:
            output = (
                f"${{CMAKE_CURRENT_BINARY_DIR}}/objects/{_safe_object_name(source)}"
            )
            outputs.append(output)
            lines.extend(
                [
                    "add_custom_command(",
                    f"  OUTPUT {_cmake_quoted(output)}",
                    "  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/objects",
                    "  COMMAND ${ASCEND_BISHENG}",
                ]
            )
            lines.extend(f"    {_cmake_quoted(option)}" for option in compile_options)
            for force_include in contract["force_includes"]:
                lines.append("    -include")
                lines.append(f"    {_cmake_quoted(force_include)}")
            lines.extend(
                f"    {_cmake_quoted('-I' + include_dir)}"
                for include_dir in include_dirs
            )
            lines.extend(
                [
                    "    -c",
                    f"    {_cmake_quoted('${CMAKE_CURRENT_SOURCE_DIR}/' + source)}",
                    "    -o",
                    f"    {_cmake_quoted(output)}",
                    f"  DEPENDS {_cmake_quoted(source)}",
                    "  VERBATIM",
                    ")",
                    "",
                ]
            )
        lines.extend(["set(SCAFFOLD_OBJECTS"])
        lines.extend(f"  {_cmake_quoted(output)}" for output in outputs)
        lines.extend(
            [
                ")",
                f"add_custom_target({target_name}_objects ALL DEPENDS ${{SCAFFOLD_OBJECTS}})",
            ]
        )
    else:
        lines.extend(
            [
                f"add_custom_target({target_name}_objects ALL",
                f"  COMMAND ${{CMAKE_COMMAND}} -E echo {_cmake_quoted(ASCEND_COMPILE_UNITS_MISSING)}",
                "  COMMAND ${CMAKE_COMMAND} -E false",
                "  VERBATIM",
                ")",
            ]
        )
    return "\n".join(lines) + "\n"


def _render_scaffold_cmake(
    manifest: dict[str, Any], copied_source_files: list[dict[str, str]]
) -> str:
    if manifest.get("ascend_compile_contract"):
        return _render_ascend_scaffold_cmake(manifest, copied_source_files)

    compilable = [
        item["scaffold_path"]
        for item in copied_source_files
        if Path(item["source"]).suffix in COMPILABLE_SOURCE_SUFFIXES
    ]
    headers = [
        item["scaffold_path"]
        for item in copied_source_files
        if Path(item["source"]).suffix in INCLUDE_SUFFIXES
    ]
    lines = [
        "cmake_minimum_required(VERSION 3.16)",
        f"project({_safe_target_name(manifest)}_scaffold LANGUAGES CXX)",
        "",
        "# Generated scaffold contract only.",
        "# build not executed; confirm target Ascend toolchain before using this project.",
        "",
    ]
    if headers:
        lines.extend(["set(SCAFFOLD_HEADERS"])
        lines.extend(f'  "{path}"' for path in headers)
        lines.extend(
            [
                ")",
                "set_source_files_properties(${SCAFFOLD_HEADERS} PROPERTIES HEADER_FILE_ONLY ON)",
                "",
            ]
        )
    if compilable:
        lines.extend(["set(SCAFFOLD_SOURCES"])
        lines.extend(f'  "{path}"' for path in compilable)
        lines.extend(
            [
                ")",
                'set_source_files_properties(${SCAFFOLD_SOURCES} PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++")',
                f"add_library({_safe_target_name(manifest)}_objects OBJECT ${{SCAFFOLD_SOURCES}})",
            ]
        )
        if manifest.get("include_directories"):
            lines.append(
                f"target_include_directories({_safe_target_name(manifest)}_objects PRIVATE"
            )
            lines.extend(f'  "{path}"' for path in manifest["include_directories"])
            lines.append(")")
    else:
        lines.append(
            "# No compilable source file was detected; add an operator build entry before compiling."
        )
    return "\n".join(lines) + "\n"


def _require_exact_keys(
    payload: Any, expected_keys: set[str], label: str
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CliUsageError(f"{label} must be an object")
    keys = set(payload)
    missing = expected_keys - keys
    if missing:
        raise CliUsageError(f"missing {label} field: {sorted(missing)[0]}")
    unexpected = keys - expected_keys
    if unexpected:
        raise CliUsageError(f"unexpected {label} field: {sorted(unexpected)[0]}")
    return payload


def _require_string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise CliUsageError(f"{label} must be a string")
    if not allow_empty and not value:
        raise CliUsageError(
            "empty scaffold path is not allowed"
            if label.endswith("path")
            else f"{label} must be non-empty"
        )
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise CliUsageError(f"{label} must be a list")
    return value


def _require_string_list(value: Any, list_label: str, item_label: str) -> list[str]:
    items = []
    for item in _require_list(value, list_label):
        items.append(_require_string(item, item_label))
    return items


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _load_json_artifact(path: Path, artifact_name: str) -> Any:
    if not path.exists():
        if path.is_symlink():
            raise CliUsageError(f"{artifact_name} is not a regular file")
        raise CliUsageError(f"{artifact_name} not found")
    if path.is_symlink():
        raise CliUsageError(f"{artifact_name} is not a regular file")
    if not path.is_file():
        raise CliUsageError(f"{artifact_name} is not a file")
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise CliUsageError(f"{artifact_name} is not valid UTF-8") from exc
    except OSError as exc:
        raise CliUsageError(f"{artifact_name} cannot be read") from exc
    try:
        return json.loads(content, parse_constant=_reject_json_constant)
    except ValueError as exc:
        raise CliUsageError(f"{artifact_name} is not valid JSON") from exc


def _validate_manifest_path(
    value: Any, label: str, *, allow_empty: bool = False
) -> str:
    path = _require_string(value, label, allow_empty=allow_empty)
    if allow_empty and path == "":
        return path
    try:
        return _safe_scaffold_relative_path(path)
    except ValueError as exc:
        raise CliUsageError(str(exc)) from exc


def _validate_manifest_path_list(value: Any, list_label: str, item_label: str) -> list[str]:
    paths = []
    for item in _require_list(value, list_label):
        paths.append(_validate_manifest_path(item, item_label))
    return paths


def _resolve_string_path_list(value: Any, list_label: str, item_label: str) -> list[str]:
    paths = []
    for item in _require_list(value, list_label):
        paths.append(str(Path(_require_string(item, item_label)).resolve()))
    return paths


def _bounded_pytest_evidence(output: str) -> list[str]:
    lines = [
        line[:300] for line in (item.strip() for item in output.splitlines()) if line
    ]
    if len(lines) <= 20:
        return lines
    return lines[:10] + ["... truncated ..."] + lines[-10:]


def _scaffold_build_check(
    name: str, status: str, reason: str, evidence: list[str] | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "evidence": evidence or [],
    }


def _blocked_scaffold_build_check(
    name: str, reason: str, evidence: list[str] | None = None
) -> dict[str, Any]:
    return _scaffold_build_check(name, "blocked", reason, evidence)


def _build_dir_available_check(
    build_dir: Path, scaffold_output_dir: Path
) -> dict[str, Any]:
    if build_dir.exists():
        return _blocked_scaffold_build_check(
            "build_dir_available",
            "build_dir_already_exists",
            [_relative(build_dir, scaffold_output_dir)],
        )
    return _scaffold_build_check(
        "build_dir_available",
        "passed",
        "build_dir_available",
        [_relative(build_dir, scaffold_output_dir)],
    )


def _bounded_command_evidence(output: str) -> list[str]:
    return _bounded_pytest_evidence(output)


def _run_cmake_command(
    command: list[str], env: dict[str, str]
) -> tuple[int | None, list[str], bool]:
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=CMAKE_EXECUTION_TIMEOUT_SECONDS,
            shell=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return None, _bounded_command_evidence(exc.stdout or ""), True
    return completed.returncode, _bounded_command_evidence(completed.stdout), False


def _expected_sk_source_manifest(
    output_dir: Path, analysis: dict[str, Any]
) -> dict[str, Any]:
    if analysis["status"] == "ready-for-sk-generation":
        return {
            "status": "generated",
            "analysis_output_dir": str(output_dir.resolve()),
            "analysis_path": "operator-sk-conversion-analysis.json",
            "asset_path": analysis["asset_path"],
            "source_files": analysis["source_files"],
            "source_file_fingerprints": analysis["source_file_fingerprints"],
            "support_directories": analysis["support_directories"],
            "include_directories": analysis["include_directories"],
            "ascend_compile_contract": analysis["ascend_compile_contract"],
            "kernel_entries": analysis["kernel_entries"],
            "copied_source_files": _sk_source_scaffold_file_entries(analysis),
            "generated_files": [
                f"{SK_SOURCE_SCAFFOLD_DIR}/README.md",
                f"{SK_SOURCE_SCAFFOLD_DIR}/CMakeLists.txt",
            ],
            "blocked_reasons": [],
            "execution_boundary": SK_SOURCE_SCAFFOLD_BOUNDARY,
            "supported_next_actions": ["run_sk_build_validation"],
        }
    if analysis["status"] == "blocked":
        blocked_reasons = analysis["blocked_reasons"]
        supported_next_actions = analysis["supported_next_actions"]
    else:
        blocked_reasons = ["conversion_contracts_missing"]
        supported_next_actions = _sk_source_needed_next_actions(analysis)
    return _sk_source_empty_manifest(
        status="blocked",
        output_dir=output_dir,
        analysis=analysis,
        blocked_reasons=blocked_reasons,
        supported_next_actions=supported_next_actions,
    )


def _normalize_sk_source_scaffold_manifest(
    payload: Any, output_dir: Path
) -> dict[str, Any]:
    manifest = _require_exact_keys(
        payload,
        {
            "status",
            "analysis_output_dir",
            "analysis_path",
            "asset_path",
            "source_files",
            "source_file_fingerprints",
            "support_directories",
            "include_directories",
            "ascend_compile_contract",
            "kernel_entries",
            "copied_source_files",
            "generated_files",
            "blocked_reasons",
            "execution_boundary",
            "supported_next_actions",
        },
        "sk source scaffold",
    )
    status = _require_string(manifest["status"], "sk source scaffold status")
    if status not in {"generated", "blocked"}:
        raise CliUsageError(f"invalid sk source scaffold status: {status}")
    if _require_string(
        manifest["analysis_output_dir"], "sk source scaffold analysis_output_dir"
    ) != str(output_dir.resolve()):
        raise CliUsageError("sk source scaffold mismatch")
    if (
        _require_string(manifest["analysis_path"], "sk source scaffold analysis_path")
        != "operator-sk-conversion-analysis.json"
    ):
        raise CliUsageError("sk source scaffold mismatch")
    _require_string(manifest["asset_path"], "sk source scaffold asset_path")
    manifest["source_files"] = _validate_manifest_path_list(
        manifest["source_files"],
        "sk source scaffold source_files",
        "sk source scaffold source file",
    )
    fingerprints: list[dict[str, str]] = []
    for fingerprint in _require_list(
        manifest["source_file_fingerprints"],
        "sk source scaffold source_file_fingerprints",
    ):
        item = _require_exact_keys(
            fingerprint, {"path", "sha256"}, "sk source scaffold source fingerprint"
        )
        fingerprints.append(
            {
                "path": _validate_manifest_path(
                    item["path"], "sk source scaffold fingerprint path"
                ),
                "sha256": _require_string(
                    item["sha256"], "sk source scaffold fingerprint sha256"
                ),
            }
        )
    manifest["source_file_fingerprints"] = fingerprints
    support_directories: list[dict[str, Any]] = []
    for support_dir in _require_list(
        manifest["support_directories"], "sk source scaffold support_directories"
    ):
        item = _require_exact_keys(
            support_dir,
            {"source_name", "source_path", "source_files", "source_file_fingerprints"},
            "sk source scaffold support directory",
        )
        fingerprints = []
        for fingerprint in _require_list(
            item["source_file_fingerprints"],
            "sk source scaffold support source_file_fingerprints",
        ):
            fingerprint_item = _require_exact_keys(
                fingerprint,
                {"path", "sha256"},
                "sk source scaffold support fingerprint",
            )
            fingerprints.append(
                {
                    "path": _validate_manifest_path(
                        fingerprint_item["path"],
                        "sk source scaffold support fingerprint path",
                    ),
                    "sha256": _require_string(
                        fingerprint_item["sha256"],
                        "sk source scaffold support fingerprint sha256",
                    ),
                }
            )
        support_directories.append(
            {
                "source_name": _support_dir_name(
                    _require_string(
                        item["source_name"], "sk source scaffold support source_name"
                    )
                ),
                "source_path": str(
                    Path(
                        _require_string(
                            item["source_path"],
                            "sk source scaffold support source_path",
                        )
                    ).resolve()
                ),
                "source_files": _validate_manifest_path_list(
                    item["source_files"],
                    "sk source scaffold support source_files",
                    "sk source scaffold support source file",
                ),
                "source_file_fingerprints": fingerprints,
            }
        )
    manifest["support_directories"] = support_directories
    manifest["include_directories"] = _resolve_string_path_list(
        manifest["include_directories"],
        "sk source scaffold include_directories",
        "sk source scaffold include directory",
    )
    raw_contract = manifest["ascend_compile_contract"]
    if raw_contract is None:
        manifest["ascend_compile_contract"] = None
    else:
        contract = _require_exact_keys(
            raw_contract,
            {
                "cann_path",
                "bisheng_path",
                "arch",
                "compile_options",
                "force_includes",
                "derived_include_dirs",
            },
            "sk source scaffold ascend compile contract",
        )
        manifest["ascend_compile_contract"] = {
            "cann_path": str(
                Path(
                    _require_string(
                        contract["cann_path"], "sk source scaffold ascend cann_path"
                    )
                ).resolve()
            ),
            "bisheng_path": str(
                Path(
                    _require_string(
                        contract["bisheng_path"],
                        "sk source scaffold ascend bisheng_path",
                    )
                ).resolve()
            ),
            "arch": _require_string(contract["arch"], "sk source scaffold ascend arch"),
            "compile_options": _require_string_list(
                contract["compile_options"],
                "sk source scaffold ascend compile_options",
                "sk source scaffold ascend compile option",
            ),
            "force_includes": _resolve_string_path_list(
                contract["force_includes"],
                "sk source scaffold ascend force_includes",
                "sk source scaffold ascend force include",
            ),
            "derived_include_dirs": _resolve_string_path_list(
                contract["derived_include_dirs"],
                "sk source scaffold ascend derived_include_dirs",
                "sk source scaffold ascend derived include dir",
            ),
        }
    entries: list[dict[str, str]] = []
    for entry in _require_list(
        manifest["kernel_entries"], "sk source scaffold kernel_entries"
    ):
        item = _require_exact_keys(
            entry, {"name", "file"}, "sk source scaffold kernel entry"
        )
        entries.append(
            {
                "name": _require_string(
                    item["name"], "sk source scaffold kernel entry name"
                ),
                "file": _validate_manifest_path(
                    item["file"], "sk source scaffold kernel entry file"
                ),
            }
        )
    manifest["kernel_entries"] = entries
    copied: list[dict[str, str]] = []
    for copied_file in _require_list(
        manifest["copied_source_files"], "sk source scaffold copied_source_files"
    ):
        item = _require_exact_keys(
            copied_file, {"source", "scaffold_path"}, "sk source scaffold copied source"
        )
        copied.append(
            {
                "source": _validate_manifest_path(
                    item["source"], "sk source scaffold copied source"
                ),
                "scaffold_path": _validate_manifest_path(
                    item["scaffold_path"], "sk source scaffold copied scaffold path"
                ),
            }
        )
    manifest["copied_source_files"] = copied
    manifest["generated_files"] = _validate_manifest_path_list(
        manifest["generated_files"],
        "sk source scaffold generated_files",
        "sk source scaffold generated file",
    )
    manifest["blocked_reasons"] = _require_string_list(
        manifest["blocked_reasons"],
        "sk source scaffold blocked_reasons",
        "sk source scaffold blocked reason",
    )
    manifest["execution_boundary"] = _require_string_list(
        manifest["execution_boundary"],
        "sk source scaffold execution_boundary",
        "sk source scaffold execution boundary",
    )
    manifest["supported_next_actions"] = _require_string_list(
        manifest["supported_next_actions"],
        "sk source scaffold supported_next_actions",
        "sk source scaffold supported next action",
    )
    return manifest


def _require_sk_source_scaffold_file(
    output_dir: Path, relative_path: str, reason: str
) -> Path:
    path = output_dir / relative_path
    scaffold_dir = (output_dir / SK_SOURCE_SCAFFOLD_DIR).resolve()
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise CliUsageError(reason) from exc
    except OSError as exc:
        raise CliUsageError(reason) from exc
    if not (resolved == scaffold_dir or scaffold_dir in resolved.parents):
        raise CliUsageError("scaffold path escape")
    if not path.is_file():
        raise CliUsageError("copied source not file")
    return path


def _load_current_sk_source_scaffold(
    output_dir: Path, analysis: dict[str, Any]
) -> dict[str, Any]:
    payload = _load_json_artifact(
        output_dir / "operator-sk-source-scaffold.json",
        "operator-sk-source-scaffold.json",
    )
    manifest = _normalize_sk_source_scaffold_manifest(payload, output_dir)
    expected = _expected_sk_source_manifest(output_dir, analysis)
    if manifest != expected:
        raise CliUsageError("sk source scaffold mismatch")
    try:
        markdown = (output_dir / "operator-sk-source-scaffold.md").read_text(
            encoding="utf-8"
        )
    except FileNotFoundError as exc:
        raise CliUsageError("sk source scaffold mismatch") from exc
    except UnicodeDecodeError as exc:
        raise CliUsageError(
            "operator-sk-source-scaffold.md is not valid UTF-8"
        ) from exc
    except OSError as exc:
        raise CliUsageError("sk source scaffold mismatch") from exc
    if markdown != _render_sk_source_scaffold_markdown(expected):
        raise CliUsageError("sk source scaffold mismatch")
    if manifest["status"] == "blocked":
        if (output_dir / SK_SOURCE_SCAFFOLD_DIR).exists():
            raise CliUsageError("sk source scaffold mismatch")
        return manifest
    for item in manifest["copied_source_files"]:
        copied_path = _require_sk_source_scaffold_file(
            output_dir, item["scaffold_path"], "copied source missing"
        )
        try:
            original_bytes = _sk_source_copy_source_path(
                analysis, item["source"]
            ).read_bytes()
            copied_bytes = copied_path.read_bytes()
        except OSError as exc:
            raise CliUsageError("copied source drift") from exc
        if copied_bytes != original_bytes:
            raise CliUsageError("copied source drift")
    for relative_path in manifest["generated_files"]:
        generated_path = _require_sk_source_scaffold_file(
            output_dir, relative_path, "sk source scaffold mismatch"
        )
        if relative_path == f"{SK_SOURCE_SCAFFOLD_DIR}/CMakeLists.txt":
            try:
                content = generated_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise CliUsageError("sk source scaffold mismatch") from exc
            if content != _render_sk_source_cmake(manifest):
                raise CliUsageError("sk source scaffold mismatch")
    return manifest


def _sk_build_validation_check(
    name: str, status: str, reason: str, evidence: list[str] | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "evidence": evidence or [],
    }


def _cmake_evidence_has_uncompiled_source_warning(evidence: list[str]) -> bool:
    return any(UNCOMPILED_SOURCE_WARNING in item for item in evidence)


def _cmake_missing_include_from_evidence(evidence: list[str]) -> str:
    for item in evidence:
        match = MISSING_INCLUDE_RE.search(item)
        if match:
            return match.group(1).strip()
    return ""


def _cmake_missing_include_failure_reason(evidence: list[str], asset_path: Path) -> str:
    if any(ASCEND_COMPILE_UNITS_MISSING in item for item in evidence):
        return ASCEND_COMPILE_UNITS_MISSING
    include_path = _cmake_missing_include_from_evidence(evidence)
    if not include_path:
        return "cmake_build_failed"
    if include_path in TOOLCHAIN_INCLUDE_NAMES or include_path.startswith(
        TOOLCHAIN_INCLUDE_PREFIXES
    ):
        return "cmake_build_missing_toolchain_include"
    asset_root = asset_path.resolve()
    candidate = (asset_root / include_path).resolve(strict=False)
    if _is_same_or_child_path(candidate, asset_root) and candidate.exists():
        return "cmake_build_missing_asset_include"
    return "cmake_build_missing_external_include"


def _blocked_sk_build_validation_check(
    name: str, reason: str, evidence: list[str] | None = None
) -> dict[str, Any]:
    return _sk_build_validation_check(name, "blocked", reason, evidence)


def _fill_blocked_sk_build_validation_checks(
    check_results: dict[str, dict[str, Any]],
    start_name: str,
    reason: str,
    evidence: list[str] | None = None,
) -> None:
    started = False
    for name in SK_BUILD_VALIDATION_CHECK_NAMES:
        if name == start_name:
            started = True
        if started and name not in check_results:
            check_results[name] = _blocked_sk_build_validation_check(
                name, reason, evidence if name == start_name else None
            )


def _sk_build_validation_manifest(
    request: "SkBuildValidationManifestInput",
) -> dict[str, Any]:
    return {
        "status": request.status,
        "analysis_output_dir": str(request.output_dir.resolve()),
        "source_scaffold_manifest_path": "operator-sk-source-scaffold.json",
        "source_scaffold_markdown_path": "operator-sk-source-scaffold.md",
        "build_validation_path": "operator-sk-build-validation.json",
        "build_dir": request.build_dir,
        "commands": request.commands,
        "checks": [
            request.check_results[name] for name in SK_BUILD_VALIDATION_CHECK_NAMES
        ],
        "supported_next_actions": request.supported_next_actions,
        "execution_boundary": SK_BUILD_VALIDATION_BOUNDARY,
    }


class SkBuildValidationManifestInput(NamedTuple):
    output_dir: Path
    status: str
    build_dir: str
    commands: list[list[str]]
    check_results: dict[str, dict[str, Any]]
    supported_next_actions: list[str]


def _normalize_sk_build_validation_checks(
    checks: list[Any],
) -> dict[str, dict[str, Any]]:
    if len(checks) != len(SK_BUILD_VALIDATION_CHECK_NAMES):
        raise CliUsageError("sk build validation checks mismatch")
    normalized: dict[str, dict[str, Any]] = {}
    for expected_name, check in zip(
        SK_BUILD_VALIDATION_CHECK_NAMES, checks, strict=True
    ):
        item = _require_exact_keys(
            check, {"name", "status", "reason", "evidence"}, "sk build validation check"
        )
        name = _require_string(item["name"], "sk build validation check name")
        if name != expected_name:
            raise CliUsageError("sk build validation checks mismatch")
        status = _require_string(item["status"], "sk build validation check status")
        if status not in {"passed", "failed", "blocked"}:
            raise CliUsageError(f"invalid sk build validation check status: {status}")
        evidence = _require_string_list(
            item["evidence"],
            "sk build validation check evidence",
            "sk build validation check evidence item",
        )
        normalized[name] = {
            "name": name,
            "status": status,
            "reason": _require_string(
                item["reason"], "sk build validation check reason"
            ),
            "evidence": evidence,
        }
    return normalized


def _require_sk_build_validation_check(
    checks: dict[str, dict[str, Any]],
    name: str,
    status: str,
    reason: str,
    evidence: list[str] | None = None,
) -> None:
    check = checks[name]
    if check["status"] != status or check["reason"] != reason:
        raise CliUsageError("sk build validation checks semantics mismatch")
    if evidence is not None and check["evidence"] != evidence:
        raise CliUsageError("sk build validation checks semantics mismatch")


def _validate_sk_build_validation_passed_checks(
    checks: dict[str, dict[str, Any]],
    source_scaffold: dict[str, Any],
) -> None:
    _validate_sk_build_validation_passed_prefix(
        checks,
        source_scaffold,
        [
            "sk_source_scaffold_generated",
            "copied_sources_current",
            "cmake_command_available",
            "cmake_contract_current",
            "build_dir_available",
        ],
    )


def _validate_sk_build_validation_passed_prefix(
    checks: dict[str, dict[str, Any]],
    source_scaffold: dict[str, Any],
    names: list[str],
) -> None:
    copied_evidence = [
        item["scaffold_path"] for item in source_scaffold["copied_source_files"]
    ]
    validators = {
        "sk_source_scaffold_generated": lambda: _require_sk_build_validation_check(
            checks,
            "sk_source_scaffold_generated",
            "passed",
            "sk_source_scaffold_generated",
            ["operator-sk-source-scaffold.json"],
        ),
        "copied_sources_current": lambda: _require_sk_build_validation_check(
            checks,
            "copied_sources_current",
            "passed",
            "copied_sources_current",
            copied_evidence,
        ),
        "cmake_command_available": lambda: _require_sk_build_validation_check(
            checks,
            "cmake_command_available",
            "passed",
            "cmake_command_available",
        ),
        "cmake_contract_current": lambda: _require_sk_build_validation_check(
            checks,
            "cmake_contract_current",
            "passed",
            "cmake_contract_current",
            [f"{SK_SOURCE_SCAFFOLD_DIR}/CMakeLists.txt"],
        ),
        "build_dir_available": lambda: _require_sk_build_validation_check(
            checks,
            "build_dir_available",
            "passed",
            "build_dir_available",
            ["operator-sk-build-validation-build"],
        ),
    }
    for name in names:
        validator = validators.get(name)
        if validator is None:
            raise CliUsageError("sk build validation checks semantics mismatch")
        validator()
        check = checks.get(name)
        if (
            name == "cmake_command_available"
            and check is not None
            and not check.get("evidence")
        ):
            raise CliUsageError("sk build validation checks semantics mismatch")


def _validate_sk_build_validation_semantics(
    status: str,
    checks: dict[str, dict[str, Any]],
    source_scaffold: dict[str, Any],
    supported_next_actions: list[str],
) -> None:
    if status == "passed":
        _validate_sk_build_validation_passed_checks(checks, source_scaffold)
        _require_sk_build_validation_check(
            checks, "cmake_configure", "passed", "cmake_configure_passed"
        )
        _require_sk_build_validation_check(
            checks, "cmake_build", "passed", "cmake_build_passed"
        )
        if supported_next_actions != ["collect_runtime_input_spec"]:
            raise CliUsageError("sk build validation supported_next_actions mismatch")
        return

    if status == "failed":
        _validate_sk_build_validation_passed_checks(checks, source_scaffold)
        configure = checks["cmake_configure"]
        build = checks["cmake_build"]
        if configure["status"] == "failed":
            if configure["reason"] not in {
                "cmake_configure_failed",
                "cmake_configure_timeout",
            }:
                raise CliUsageError("sk build validation checks semantics mismatch")
            if (
                build["status"] != "blocked"
                or build["reason"] != "configure_not_passed"
            ):
                raise CliUsageError("sk build validation checks semantics mismatch")
        elif build["status"] == "failed":
            if (
                configure["status"] != "passed"
                or configure["reason"] != "cmake_configure_passed"
            ):
                raise CliUsageError("sk build validation checks semantics mismatch")
            if build["reason"] not in {
                "cmake_build_failed",
                "cmake_build_timeout",
                "cmake_build_uncompiled_source",
                "cmake_build_missing_toolchain_include",
                "cmake_build_missing_external_include",
                "cmake_build_missing_asset_include",
                ASCEND_COMPILE_UNITS_MISSING,
            }:
                raise CliUsageError("sk build validation checks semantics mismatch")
        else:
            raise CliUsageError("sk build validation checks semantics mismatch")
        if supported_next_actions != ["run_sk_build_validation"]:
            raise CliUsageError("sk build validation supported_next_actions mismatch")
        return

    failed_checks = [check for check in checks.values() if check["status"] == "failed"]
    if failed_checks:
        raise CliUsageError("sk build validation checks semantics mismatch")
    blocked_indexes = [
        index
        for index, name in enumerate(SK_BUILD_VALIDATION_CHECK_NAMES)
        if checks[name]["status"] == "blocked"
    ]
    if not blocked_indexes:
        raise CliUsageError("sk build validation checks semantics mismatch")
    first_blocked_index = blocked_indexes[0]
    first_blocked_name = SK_BUILD_VALIDATION_CHECK_NAMES[first_blocked_index]
    blocked_reason = checks[first_blocked_name]["reason"]
    allowed_first_reasons = {
        "sk_source_scaffold_generated": "sk_source_scaffold_not_generated",
        "cmake_command_available": "cmake_command_not_found",
        "build_dir_available": "build_dir_already_exists",
    }
    if allowed_first_reasons.get(first_blocked_name) != blocked_reason:
        raise CliUsageError("sk build validation checks semantics mismatch")
    if (
        source_scaffold["status"] == "generated"
        and first_blocked_name == "sk_source_scaffold_generated"
    ):
        raise CliUsageError("sk build validation checks semantics mismatch")
    if (
        source_scaffold["status"] != "generated"
        and first_blocked_name != "sk_source_scaffold_generated"
    ):
        raise CliUsageError("sk build validation checks semantics mismatch")
    _validate_sk_build_validation_passed_prefix(
        checks,
        source_scaffold,
        list(SK_BUILD_VALIDATION_CHECK_NAMES[:first_blocked_index]),
    )
    remaining_check_start = first_blocked_index + 1
    remaining_check_names = SK_BUILD_VALIDATION_CHECK_NAMES[remaining_check_start:]
    for name in remaining_check_names:
        if (
            checks[name]["status"] != "blocked"
            or checks[name]["reason"] != blocked_reason
        ):
            raise CliUsageError("sk build validation checks semantics mismatch")
    expected_next = (
        source_scaffold["supported_next_actions"]
        if first_blocked_name == "sk_source_scaffold_generated"
        else ["run_sk_build_validation"]
    )
    if supported_next_actions != expected_next:
        raise CliUsageError("sk build validation supported_next_actions mismatch")


def _validate_sk_build_validation_commands(
    status: str,
    commands: list[Any],
    output_dir: Path,
    checks: dict[str, dict[str, Any]],
) -> list[list[str]]:
    normalized: list[list[str]] = []
    for command in commands:
        normalized.append(
            _require_string_list(
                command,
                "sk build validation command",
                "sk build validation command item",
            )
        )

    source_dir = str(output_dir / SK_SOURCE_SCAFFOLD_DIR)
    build_dir = str(output_dir / "operator-sk-build-validation-build")
    if status == "passed":
        expected_count = 2
    elif status == "failed" and checks["cmake_configure"]["status"] == "failed":
        expected_count = 1
    elif status == "failed" and checks["cmake_build"]["status"] == "failed":
        expected_count = 2
    else:
        expected_count = 0

    if status == "blocked":
        if normalized:
            raise CliUsageError("blocked sk build validation commands must be empty")
        return normalized
    if len(normalized) != expected_count:
        raise CliUsageError(f"{status} sk build validation commands mismatch")
    configure = normalized[0]
    if not configure or Path(configure[0]).name != "cmake":
        raise CliUsageError("sk build validation cmake command mismatch")
    if len(configure) != 5 or configure[1:] != ["-S", source_dir, "-B", build_dir]:
        raise CliUsageError("sk build validation configure command mismatch")
    if expected_count == 2:
        build = normalized[1]
        if not build or Path(build[0]).name != "cmake" or build[0] != configure[0]:
            raise CliUsageError("sk build validation cmake command mismatch")
        if len(build) != 3 or build[1:] != ["--build", build_dir]:
            raise CliUsageError("sk build validation build command mismatch")
    return normalized


def _load_current_sk_build_validation(
    output_dir: Path,
    source_scaffold: dict[str, Any],
) -> dict[str, Any]:
    payload = _load_json_artifact(
        output_dir / "operator-sk-build-validation.json",
        "operator-sk-build-validation.json",
    )
    manifest = _require_exact_keys(
        payload,
        {
            "status",
            "analysis_output_dir",
            "source_scaffold_manifest_path",
            "source_scaffold_markdown_path",
            "build_validation_path",
            "build_dir",
            "commands",
            "checks",
            "supported_next_actions",
            "execution_boundary",
        },
        "sk build validation",
    )
    status = _require_string(manifest["status"], "sk build validation status")
    if status not in {"passed", "failed", "blocked"}:
        raise CliUsageError(f"invalid sk build validation status: {status}")
    if _require_string(
        manifest["analysis_output_dir"], "sk build validation analysis_output_dir"
    ) != str(output_dir.resolve()):
        raise CliUsageError("sk build validation analysis_output_dir mismatch")
    if (
        _require_string(
            manifest["source_scaffold_manifest_path"],
            "sk build validation source_scaffold_manifest_path",
        )
        != "operator-sk-source-scaffold.json"
    ):
        raise CliUsageError(
            "sk build validation source_scaffold_manifest_path mismatch"
        )
    if (
        _require_string(
            manifest["source_scaffold_markdown_path"],
            "sk build validation source_scaffold_markdown_path",
        )
        != "operator-sk-source-scaffold.md"
    ):
        raise CliUsageError(
            "sk build validation source_scaffold_markdown_path mismatch"
        )
    if (
        _require_string(
            manifest["build_validation_path"],
            "sk build validation build_validation_path",
        )
        != "operator-sk-build-validation.json"
    ):
        raise CliUsageError("sk build validation build_validation_path mismatch")
    build_dir = _require_string(
        manifest["build_dir"], "sk build validation build_dir", allow_empty=True
    )
    blocked_with_build_dir = status == "blocked" and build_dir
    unblocked_without_expected_build_dir = (
        status != "blocked" and build_dir != "operator-sk-build-validation-build"
    )
    if blocked_with_build_dir or unblocked_without_expected_build_dir:
        raise CliUsageError("sk build validation build_dir mismatch")
    boundary = _require_string_list(
        manifest["execution_boundary"],
        "sk build validation execution_boundary",
        "sk build validation execution boundary",
    )
    if boundary != SK_BUILD_VALIDATION_BOUNDARY:
        raise CliUsageError("sk build validation execution_boundary mismatch")
    checks = _normalize_sk_build_validation_checks(
        _require_list(manifest["checks"], "sk build validation checks")
    )
    supported_next_actions = _require_string_list(
        manifest["supported_next_actions"],
        "sk build validation supported_next_actions",
        "sk build validation supported next action",
    )
    _validate_sk_build_validation_semantics(
        status, checks, source_scaffold, supported_next_actions
    )
    commands = _validate_sk_build_validation_commands(
        status,
        _require_list(manifest["commands"], "sk build validation commands"),
        output_dir,
        checks,
    )
    return {
        "status": status,
        "analysis_output_dir": str(output_dir.resolve()),
        "source_scaffold_manifest_path": "operator-sk-source-scaffold.json",
        "source_scaffold_markdown_path": "operator-sk-source-scaffold.md",
        "build_validation_path": "operator-sk-build-validation.json",
        "build_dir": build_dir,
        "commands": commands,
        "checks": [checks[name] for name in SK_BUILD_VALIDATION_CHECK_NAMES],
        "checks_by_name": checks,
        "supported_next_actions": supported_next_actions,
        "execution_boundary": SK_BUILD_VALIDATION_BOUNDARY,
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cmd_analyze_sk_conversion(args: argparse.Namespace) -> int:
    asset_path = Path(args.asset).resolve()
    if not asset_path.exists():
        raise CliUsageError(f"asset path not found: {asset_path}")

    output_dir = Path(args.output_dir).resolve()
    analysis = _build_sk_conversion_analysis(
        SkConversionAnalysisInput(
            asset_path,
            getattr(args, "support_dir", None),
            getattr(args, "include_dir", None),
            getattr(args, "ascend_cann_path", None),
            getattr(args, "ascend_arch", None),
            getattr(args, "ascend_compile_option", None),
            getattr(args, "ascend_force_include", None),
        )
    )
    _write_sk_conversion_artifacts(output_dir, analysis)
    return 0


def cmd_generate_sk_source_scaffold(args: argparse.Namespace) -> int:
    output_dir = Path(args.analysis_output_dir).resolve()
    if not output_dir.exists():
        raise CliUsageError(f"analysis output directory not found: {output_dir}")
    if not output_dir.is_dir():
        raise CliUsageError(f"analysis output path is not a directory: {output_dir}")

    analysis = _load_current_sk_conversion_analysis(output_dir)

    if analysis["status"] == "ready-for-sk-generation":
        manifest = _generate_sk_source_scaffold(output_dir, analysis)
        _write_sk_source_artifacts(output_dir, manifest)
        return 0

    _raise_if_existing_non_generated_sk_source_scaffold(output_dir)
    if analysis["status"] == "blocked":
        blocked_reasons = analysis["blocked_reasons"]
        supported_next_actions = analysis["supported_next_actions"]
    else:
        blocked_reasons = ["conversion_contracts_missing"]
        supported_next_actions = _sk_source_needed_next_actions(analysis)
    manifest = _sk_source_empty_manifest(
        status="blocked",
        output_dir=output_dir,
        analysis=analysis,
        blocked_reasons=blocked_reasons,
        supported_next_actions=supported_next_actions,
    )
    _write_sk_source_artifacts(output_dir, manifest)
    return 1


def cmd_run_sk_build_validation(args: argparse.Namespace) -> int:
    output_dir = Path(args.analysis_output_dir).resolve()
    if not output_dir.exists():
        raise CliUsageError(f"analysis output directory not found: {output_dir}")
    if not output_dir.is_dir():
        raise CliUsageError(f"analysis output path is not a directory: {output_dir}")

    analysis = _load_current_sk_conversion_analysis(output_dir)
    source_scaffold = _load_current_sk_source_scaffold(output_dir, analysis)
    check_results: dict[str, dict[str, Any]] = {}

    if source_scaffold["status"] != "generated":
        check_results["sk_source_scaffold_generated"] = (
            _blocked_sk_build_validation_check(
                "sk_source_scaffold_generated",
                "sk_source_scaffold_not_generated",
                ["operator-sk-source-scaffold.json"],
            )
        )
        _fill_blocked_sk_build_validation_checks(
            check_results, "copied_sources_current", "sk_source_scaffold_not_generated"
        )
        result = _sk_build_validation_manifest(
            SkBuildValidationManifestInput(
                output_dir,
                "blocked",
                "",
                [],
                check_results,
                source_scaffold["supported_next_actions"],
            )
        )
        _write_json(output_dir / "operator-sk-build-validation.json", result)
        return 1

    check_results["sk_source_scaffold_generated"] = _sk_build_validation_check(
        "sk_source_scaffold_generated",
        "passed",
        "sk_source_scaffold_generated",
        ["operator-sk-source-scaffold.json"],
    )
    check_results["copied_sources_current"] = _sk_build_validation_check(
        "copied_sources_current",
        "passed",
        "copied_sources_current",
        [item["scaffold_path"] for item in source_scaffold["copied_source_files"]],
    )

    cmake_path = shutil.which("cmake")
    if not cmake_path:
        check_results["cmake_command_available"] = _blocked_sk_build_validation_check(
            "cmake_command_available",
            "cmake_command_not_found",
        )
        _fill_blocked_sk_build_validation_checks(
            check_results, "cmake_contract_current", "cmake_command_not_found"
        )
        result = _sk_build_validation_manifest(
            SkBuildValidationManifestInput(
                output_dir, "blocked", "", [], check_results, ["run_sk_build_validation"]
            )
        )
        _write_json(output_dir / "operator-sk-build-validation.json", result)
        return 1
    check_results["cmake_command_available"] = _sk_build_validation_check(
        "cmake_command_available",
        "passed",
        "cmake_command_available",
        [cmake_path],
    )
    check_results["cmake_contract_current"] = _sk_build_validation_check(
        "cmake_contract_current",
        "passed",
        "cmake_contract_current",
        [f"{SK_SOURCE_SCAFFOLD_DIR}/CMakeLists.txt"],
    )

    build_dir = output_dir / "operator-sk-build-validation-build"
    build_dir_check = _build_dir_available_check(build_dir, output_dir)
    check_results["build_dir_available"] = _sk_build_validation_check(
        "build_dir_available",
        build_dir_check["status"],
        build_dir_check["reason"],
        build_dir_check["evidence"],
    )
    if build_dir_check["status"] == "blocked":
        _fill_blocked_sk_build_validation_checks(
            check_results, "cmake_configure", build_dir_check["reason"]
        )
        result = _sk_build_validation_manifest(
            SkBuildValidationManifestInput(
                output_dir, "blocked", "", [], check_results, ["run_sk_build_validation"]
            )
        )
        _write_json(output_dir / "operator-sk-build-validation.json", result)
        return 1

    source_dir = output_dir / SK_SOURCE_SCAFFOLD_DIR
    configure_command = [cmake_path, "-S", str(source_dir), "-B", str(build_dir)]
    build_command = [cmake_path, "--build", str(build_dir)]
    cmake_env = os.environ.copy()
    cmake_env["CMAKE_BUILD_PARALLEL_LEVEL"] = "1"

    configure_code, configure_evidence, configure_timed_out = _run_cmake_command(
        configure_command, cmake_env
    )
    commands = [configure_command]
    if configure_timed_out:
        check_results["cmake_configure"] = _sk_build_validation_check(
            "cmake_configure",
            "failed",
            "cmake_configure_timeout",
            configure_evidence,
        )
        check_results["cmake_build"] = _blocked_sk_build_validation_check(
            "cmake_build", "configure_not_passed"
        )
        result = _sk_build_validation_manifest(
            SkBuildValidationManifestInput(
                output_dir,
                "failed",
                "operator-sk-build-validation-build",
                commands,
                check_results,
                ["run_sk_build_validation"],
            )
        )
        _write_json(output_dir / "operator-sk-build-validation.json", result)
        return 1
    if configure_code != 0:
        check_results["cmake_configure"] = _sk_build_validation_check(
            "cmake_configure",
            "failed",
            "cmake_configure_failed",
            configure_evidence,
        )
        check_results["cmake_build"] = _blocked_sk_build_validation_check(
            "cmake_build", "configure_not_passed"
        )
        result = _sk_build_validation_manifest(
            SkBuildValidationManifestInput(
                output_dir,
                "failed",
                "operator-sk-build-validation-build",
                commands,
                check_results,
                ["run_sk_build_validation"],
            )
        )
        _write_json(output_dir / "operator-sk-build-validation.json", result)
        return 1

    check_results["cmake_configure"] = _sk_build_validation_check(
        "cmake_configure",
        "passed",
        "cmake_configure_passed",
        configure_evidence,
    )
    build_code, build_evidence, build_timed_out = _run_cmake_command(
        build_command, cmake_env
    )
    commands.append(build_command)
    if build_timed_out:
        check_results["cmake_build"] = _sk_build_validation_check(
            "cmake_build",
            "failed",
            "cmake_build_timeout",
            build_evidence,
        )
        result = _sk_build_validation_manifest(
            SkBuildValidationManifestInput(
                output_dir,
                "failed",
                "operator-sk-build-validation-build",
                commands,
                check_results,
                ["run_sk_build_validation"],
            )
        )
        _write_json(output_dir / "operator-sk-build-validation.json", result)
        return 1
    if build_code != 0:
        check_results["cmake_build"] = _sk_build_validation_check(
            "cmake_build",
            "failed",
            _cmake_missing_include_failure_reason(
                build_evidence, Path(source_scaffold["asset_path"])
            ),
            build_evidence,
        )
        result = _sk_build_validation_manifest(
            SkBuildValidationManifestInput(
                output_dir,
                "failed",
                "operator-sk-build-validation-build",
                commands,
                check_results,
                ["run_sk_build_validation"],
            )
        )
        _write_json(output_dir / "operator-sk-build-validation.json", result)
        return 1
    if _cmake_evidence_has_uncompiled_source_warning(build_evidence):
        check_results["cmake_build"] = _sk_build_validation_check(
            "cmake_build",
            "failed",
            "cmake_build_uncompiled_source",
            build_evidence,
        )
        result = _sk_build_validation_manifest(
            SkBuildValidationManifestInput(
                output_dir,
                "failed",
                "operator-sk-build-validation-build",
                commands,
                check_results,
                ["run_sk_build_validation"],
            )
        )
        _write_json(output_dir / "operator-sk-build-validation.json", result)
        return 1

    check_results["cmake_build"] = _sk_build_validation_check(
        "cmake_build",
        "passed",
        "cmake_build_passed",
        build_evidence,
    )
    result = _sk_build_validation_manifest(
        SkBuildValidationManifestInput(
            output_dir,
            "passed",
            "operator-sk-build-validation-build",
            commands,
            check_results,
            ["collect_runtime_input_spec"],
        )
    )
    _write_json(output_dir / "operator-sk-build-validation.json", result)
    return 0


def _sk_source_version_copied_files(
    source_scaffold: dict[str, Any],
) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for item in source_scaffold["copied_source_files"]:
        source_version_path = item["scaffold_path"].replace(
            f"{SK_SOURCE_SCAFFOLD_DIR}/",
            f"{SK_SOURCE_VERSION_DIR}/",
            1,
        )
        copied.append(
            {
                "source": item["source"],
                "scaffold_path": item["scaffold_path"],
                "source_version_path": source_version_path,
            }
        )
    return copied


def _sk_source_version_generated_files(
    copied_source_files: list[dict[str, str]],
) -> list[str]:
    return [
        f"{SK_SOURCE_VERSION_DIR}/README.md",
        f"{SK_SOURCE_VERSION_DIR}/CMakeLists.txt",
        *[item["source_version_path"] for item in copied_source_files],
    ]


def _sk_source_version_manifest(
    output_dir: Path,
    source_scaffold: dict[str, Any],
    build_validation: dict[str, Any],
    *,
    status: str,
) -> dict[str, Any]:
    if status == "generated":
        copied_source_files = _sk_source_version_copied_files(source_scaffold)
        generated_files = _sk_source_version_generated_files(copied_source_files)
        source_version_path = str((output_dir / SK_SOURCE_VERSION_DIR).resolve())
        blocked_reasons: list[str] = []
        execution_boundary = SK_SOURCE_VERSION_GENERATED_BOUNDARY
        supported_next_actions = ["validate_sk_source_version"]
    else:
        copied_source_files = []
        generated_files = []
        source_version_path = ""
        blocked_reasons = ["sk_build_validation_not_passed"]
        execution_boundary = SK_SOURCE_VERSION_BLOCKED_BOUNDARY
        supported_next_actions = list(build_validation["supported_next_actions"])

    return {
        "status": status,
        "analysis_output_dir": str(output_dir.resolve()),
        "source_scaffold_manifest_path": "operator-sk-source-scaffold.json",
        "build_validation_path": "operator-sk-build-validation.json",
        "source_version_path": source_version_path,
        "source_version_dir": SK_SOURCE_VERSION_DIR,
        "asset_path": source_scaffold["asset_path"],
        "source_files": source_scaffold["source_files"],
        "source_file_fingerprints": source_scaffold["source_file_fingerprints"],
        "support_directories": source_scaffold["support_directories"],
        "include_directories": source_scaffold["include_directories"],
        "ascend_compile_contract": source_scaffold["ascend_compile_contract"],
        "kernel_entries": source_scaffold["kernel_entries"],
        "copied_source_files": copied_source_files,
        "generated_files": generated_files,
        "build_validation_status": build_validation["status"],
        "blocked_reasons": blocked_reasons,
        "execution_boundary": execution_boundary,
        "supported_next_actions": supported_next_actions,
    }


def _render_sk_source_version_markdown(manifest: dict[str, Any]) -> str:
    title = (
        "Build-validated SK source version"
        if manifest["status"] == "generated"
        else "SK source version not prepared"
    )
    lines = [
        "# Operator SK Source Version",
        "",
        f"{title}.",
        "",
        "## Status",
        "",
        f"- status: `{manifest['status']}`",
        f"- build validation status: `{manifest['build_validation_status']}`",
        f"- source version dir: `{manifest['source_version_dir']}`",
        f"- build validation path: `{manifest['build_validation_path']}`",
        f"- source version validation path: `{SK_SOURCE_VERSION_VALIDATION_PATH}`",
        "",
        "## Copied Sources",
        "",
    ]
    if manifest["copied_source_files"]:
        for item in manifest["copied_source_files"]:
            lines.append(
                f"- `{item['source']}`: `{item['scaffold_path']}` -> `{item['source_version_path']}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Generated Files", ""])
    if manifest["generated_files"]:
        for item in manifest["generated_files"]:
            lines.append(f"- `{item}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Blocked Reasons", ""])
    if manifest["blocked_reasons"]:
        for item in manifest["blocked_reasons"]:
            lines.append(f"- `{item}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Supported Next Actions", ""])
    for item in manifest["supported_next_actions"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Execution Boundary", ""])
    for item in manifest["execution_boundary"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _render_sk_source_version_readme(manifest: dict[str, Any]) -> str:
    lines = [
        "# Operator SK Source Version",
        "",
        "Build-validated SK source version.",
        "",
        f"- build validation path: `{manifest['build_validation_path']}`",
        f"- build validation status: `{manifest['build_validation_status']}`",
        f"- source version validation path: `{SK_SOURCE_VERSION_VALIDATION_PATH}`",
        "",
        "## Build Validation",
        "",
        "Run from the analysis output directory:",
        "",
        "```bash",
        f"cmake -S {SK_SOURCE_VERSION_DIR} -B {SK_SOURCE_VERSION_BUILD_DIR}",
        f"cmake --build {SK_SOURCE_VERSION_BUILD_DIR}",
        "```",
        "",
        "The repository helper command records this check as:",
        "",
        "```bash",
        "<skills_root>/sk-operator-build-package/scripts/operator_build_package.py "
        "validate-sk-source-version <analysis-output-dir>",
        "```",
        "",
        "## Source Files",
        "",
    ]
    for item in manifest["copied_source_files"]:
        lines.append(f"- `{item['source_version_path']}`")
    lines.extend(["", "## Execution Boundary", ""])
    for item in manifest["execution_boundary"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _write_sk_source_version_readme(output_dir: Path, manifest: dict[str, Any]) -> None:
    _write_text(
        output_dir / SK_SOURCE_VERSION_DIR / "README.md",
        _render_sk_source_version_readme(manifest),
    )


def _raise_if_sk_source_version_outputs_exist(output_dir: Path) -> None:
    if (output_dir / "operator-sk-source-version.json").exists():
        raise CliUsageError("operator-sk-source-version.json already exists")
    if (output_dir / "operator-sk-source-version.md").exists():
        raise CliUsageError("operator-sk-source-version.md already exists")
    if (output_dir / SK_SOURCE_VERSION_DIR).exists():
        raise CliUsageError("operator-sk-source-version directory already exists")


def _copy_sk_source_version_tree(output_dir: Path, manifest: dict[str, Any]) -> None:
    for item in manifest["copied_source_files"]:
        source_path = output_dir / item["scaffold_path"]
        destination = output_dir / item["source_version_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
    cmake_source = output_dir / SK_SOURCE_SCAFFOLD_DIR / "CMakeLists.txt"
    cmake_destination = output_dir / SK_SOURCE_VERSION_DIR / "CMakeLists.txt"
    cmake_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cmake_source, cmake_destination)
    _write_sk_source_version_readme(output_dir, manifest)


def cmd_prepare_sk_source_version(args: argparse.Namespace) -> int:
    output_dir = Path(args.analysis_output_dir).resolve()
    if not output_dir.exists():
        raise CliUsageError(f"analysis output directory not found: {output_dir}")
    if not output_dir.is_dir():
        raise CliUsageError(f"analysis output path is not a directory: {output_dir}")

    analysis = _load_current_sk_conversion_analysis(output_dir)
    source_scaffold = _load_current_sk_source_scaffold(output_dir, analysis)
    build_validation = _load_current_sk_build_validation(output_dir, source_scaffold)
    _raise_if_sk_source_version_outputs_exist(output_dir)

    if build_validation["status"] != "passed":
        manifest = _sk_source_version_manifest(
            output_dir, source_scaffold, build_validation, status="blocked"
        )
        _write_json(output_dir / "operator-sk-source-version.json", manifest)
        _write_text(
            output_dir / "operator-sk-source-version.md",
            _render_sk_source_version_markdown(manifest),
        )
        return 1

    manifest = _sk_source_version_manifest(
        output_dir, source_scaffold, build_validation, status="generated"
    )
    _copy_sk_source_version_tree(output_dir, manifest)
    _write_json(output_dir / "operator-sk-source-version.json", manifest)
    _write_text(
        output_dir / "operator-sk-source-version.md",
        _render_sk_source_version_markdown(manifest),
    )
    return 0


def _validate_current_sk_source_version_tree(
    output_dir: Path,
    source_version: dict[str, Any],
) -> None:
    version_dir = output_dir / SK_SOURCE_VERSION_DIR
    if not version_dir.is_dir():
        raise CliUsageError("operator-sk-source-version directory not found")
    for item in source_version["copied_source_files"]:
        scaffold_path = output_dir / item["scaffold_path"]
        version_path = output_dir / item["source_version_path"]
        try:
            scaffold_bytes = scaffold_path.read_bytes()
            version_bytes = version_path.read_bytes()
        except FileNotFoundError as exc:
            raise CliUsageError("sk source version copied source missing") from exc
        except OSError as exc:
            raise CliUsageError("sk source version copied source drift") from exc
        if scaffold_bytes != version_bytes:
            raise CliUsageError("sk source version copied source drift")

    cmake_source = output_dir / SK_SOURCE_SCAFFOLD_DIR / "CMakeLists.txt"
    cmake_version = output_dir / SK_SOURCE_VERSION_DIR / "CMakeLists.txt"
    try:
        if cmake_source.read_bytes() != cmake_version.read_bytes():
            raise CliUsageError("sk source version mismatch")
    except FileNotFoundError as exc:
        raise CliUsageError("sk source version mismatch") from exc
    except OSError as exc:
        raise CliUsageError("sk source version mismatch") from exc

    readme_path = output_dir / SK_SOURCE_VERSION_DIR / "README.md"
    try:
        readme = readme_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CliUsageError("sk source version mismatch") from exc
    except UnicodeDecodeError as exc:
        raise CliUsageError(
            "operator-sk-source-version/README.md is not valid UTF-8"
        ) from exc
    except OSError as exc:
        raise CliUsageError("sk source version mismatch") from exc
    if readme != _render_sk_source_version_readme(source_version):
        raise CliUsageError("sk source version mismatch")


def _load_current_sk_source_version(
    output_dir: Path,
    source_scaffold: dict[str, Any],
    build_validation: dict[str, Any],
) -> dict[str, Any]:
    payload = _load_json_artifact(
        output_dir / "operator-sk-source-version.json",
        "operator-sk-source-version.json",
    )
    source_version = _require_exact_keys(
        payload,
        {
            "status",
            "analysis_output_dir",
            "source_scaffold_manifest_path",
            "build_validation_path",
            "source_version_path",
            "source_version_dir",
            "asset_path",
            "source_files",
            "source_file_fingerprints",
            "support_directories",
            "include_directories",
            "ascend_compile_contract",
            "kernel_entries",
            "copied_source_files",
            "generated_files",
            "build_validation_status",
            "blocked_reasons",
            "execution_boundary",
            "supported_next_actions",
        },
        "sk source version",
    )
    expected = _sk_source_version_manifest(
        output_dir, source_scaffold, build_validation, status="generated"
    )
    if source_version != expected:
        raise CliUsageError("sk source version mismatch")
    try:
        markdown = (output_dir / "operator-sk-source-version.md").read_text(
            encoding="utf-8"
        )
    except FileNotFoundError as exc:
        raise CliUsageError("operator-sk-source-version.md not found") from exc
    except UnicodeDecodeError as exc:
        raise CliUsageError("operator-sk-source-version.md is not valid UTF-8") from exc
    except OSError as exc:
        raise CliUsageError("sk source version mismatch") from exc
    if markdown != _render_sk_source_version_markdown(expected):
        raise CliUsageError("sk source version mismatch")
    _validate_current_sk_source_version_tree(output_dir, source_version)
    return source_version


def _sk_source_version_validation_check(
    name: str, status: str, reason: str, evidence: list[str] | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "evidence": evidence or [],
    }


def _blocked_sk_source_version_validation_check(
    name: str, reason: str, evidence: list[str] | None = None
) -> dict[str, Any]:
    return _sk_source_version_validation_check(name, "blocked", reason, evidence)


def _fill_blocked_sk_source_version_validation_checks(
    check_results: dict[str, dict[str, Any]],
    start_name: str,
    reason: str,
    evidence: list[str] | None = None,
) -> None:
    started = False
    for name in SK_SOURCE_VERSION_VALIDATION_CHECK_NAMES:
        if name == start_name:
            started = True
        if started and name not in check_results:
            check_results[name] = _blocked_sk_source_version_validation_check(
                name,
                reason,
                evidence if name == start_name else None,
            )


def _sk_source_version_validation_manifest(
    output_dir: Path,
    status: str,
    commands: list[list[str]],
    check_results: dict[str, dict[str, Any]],
    supported_next_actions: list[str],
) -> dict[str, Any]:
    if status == "passed":
        execution_boundary = SK_SOURCE_VERSION_VALIDATION_PASSED_BOUNDARY
    elif status == "blocked":
        execution_boundary = SK_SOURCE_VERSION_VALIDATION_BLOCKED_BOUNDARY
    else:
        execution_boundary = SK_SOURCE_VERSION_VALIDATION_FAILED_BOUNDARY
    return {
        "status": status,
        "analysis_output_dir": str(output_dir.resolve()),
        "source_version_manifest_path": "operator-sk-source-version.json",
        "source_version_markdown_path": "operator-sk-source-version.md",
        "source_version_dir": SK_SOURCE_VERSION_DIR,
        "source_version_validation_path": SK_SOURCE_VERSION_VALIDATION_PATH,
        "build_dir": SK_SOURCE_VERSION_BUILD_DIR,
        "commands": commands,
        "checks": [
            check_results[name] for name in SK_SOURCE_VERSION_VALIDATION_CHECK_NAMES
        ],
        "supported_next_actions": supported_next_actions,
        "execution_boundary": execution_boundary,
    }


def _raise_if_sk_source_version_validation_exists(output_dir: Path) -> None:
    if (output_dir / SK_SOURCE_VERSION_VALIDATION_PATH).exists():
        raise CliUsageError("operator-sk-source-version-validation.json already exists")


def cmd_validate_sk_source_version(args: argparse.Namespace) -> int:
    output_dir = Path(args.analysis_output_dir).resolve()
    if not output_dir.exists():
        raise CliUsageError(f"analysis output directory not found: {output_dir}")
    if not output_dir.is_dir():
        raise CliUsageError(f"analysis output path is not a directory: {output_dir}")

    analysis = _load_current_sk_conversion_analysis(output_dir)
    source_scaffold = _load_current_sk_source_scaffold(output_dir, analysis)
    build_validation = _load_current_sk_build_validation(output_dir, source_scaffold)
    source_version = _load_current_sk_source_version(
        output_dir, source_scaffold, build_validation
    )
    _ = source_version
    _raise_if_sk_source_version_validation_exists(output_dir)

    check_results: dict[str, dict[str, Any]] = {
        "sk_source_version_current": _sk_source_version_validation_check(
            "sk_source_version_current",
            "passed",
            "sk_source_version_current",
            [
                "operator-sk-source-version.json",
                "operator-sk-source-version.md",
                SK_SOURCE_VERSION_DIR,
            ],
        )
    }

    cmake_path = shutil.which("cmake")
    if not cmake_path:
        check_results["cmake_command_available"] = (
            _blocked_sk_source_version_validation_check(
                "cmake_command_available",
                "cmake_command_not_found",
            )
        )
        _fill_blocked_sk_source_version_validation_checks(
            check_results, "build_dir_available", "cmake_command_not_found"
        )
        result = _sk_source_version_validation_manifest(
            output_dir, "blocked", [], check_results, ["validate_sk_source_version"]
        )
        _write_json(output_dir / SK_SOURCE_VERSION_VALIDATION_PATH, result)
        return 1
    check_results["cmake_command_available"] = _sk_source_version_validation_check(
        "cmake_command_available",
        "passed",
        "cmake_command_available",
        [cmake_path],
    )

    build_dir = output_dir / SK_SOURCE_VERSION_BUILD_DIR
    build_dir_check = _build_dir_available_check(build_dir, output_dir)
    check_results["build_dir_available"] = _sk_source_version_validation_check(
        "build_dir_available",
        build_dir_check["status"],
        build_dir_check["reason"],
        build_dir_check["evidence"],
    )
    if build_dir_check["status"] == "blocked":
        _fill_blocked_sk_source_version_validation_checks(
            check_results, "cmake_configure", build_dir_check["reason"]
        )
        result = _sk_source_version_validation_manifest(
            output_dir, "blocked", [], check_results, ["validate_sk_source_version"]
        )
        _write_json(output_dir / SK_SOURCE_VERSION_VALIDATION_PATH, result)
        return 1

    source_dir = output_dir / SK_SOURCE_VERSION_DIR
    configure_command = [cmake_path, "-S", str(source_dir), "-B", str(build_dir)]
    build_command = [cmake_path, "--build", str(build_dir)]
    cmake_env = os.environ.copy()
    cmake_env["CMAKE_BUILD_PARALLEL_LEVEL"] = "1"

    configure_code, configure_evidence, configure_timed_out = _run_cmake_command(
        configure_command, cmake_env
    )
    commands = [configure_command]
    if configure_timed_out:
        check_results["cmake_configure"] = _sk_source_version_validation_check(
            "cmake_configure",
            "failed",
            "cmake_configure_timeout",
            configure_evidence,
        )
        check_results["cmake_build"] = _blocked_sk_source_version_validation_check(
            "cmake_build", "configure_not_passed"
        )
        result = _sk_source_version_validation_manifest(
            output_dir,
            "failed",
            commands,
            check_results,
            ["validate_sk_source_version"],
        )
        _write_json(output_dir / SK_SOURCE_VERSION_VALIDATION_PATH, result)
        return 1
    if configure_code != 0:
        check_results["cmake_configure"] = _sk_source_version_validation_check(
            "cmake_configure",
            "failed",
            "cmake_configure_failed",
            configure_evidence,
        )
        check_results["cmake_build"] = _blocked_sk_source_version_validation_check(
            "cmake_build", "configure_not_passed"
        )
        result = _sk_source_version_validation_manifest(
            output_dir,
            "failed",
            commands,
            check_results,
            ["validate_sk_source_version"],
        )
        _write_json(output_dir / SK_SOURCE_VERSION_VALIDATION_PATH, result)
        return 1

    check_results["cmake_configure"] = _sk_source_version_validation_check(
        "cmake_configure",
        "passed",
        "cmake_configure_passed",
        configure_evidence,
    )
    build_code, build_evidence, build_timed_out = _run_cmake_command(
        build_command, cmake_env
    )
    commands.append(build_command)
    if build_timed_out:
        check_results["cmake_build"] = _sk_source_version_validation_check(
            "cmake_build",
            "failed",
            "cmake_build_timeout",
            build_evidence,
        )
        result = _sk_source_version_validation_manifest(
            output_dir,
            "failed",
            commands,
            check_results,
            ["validate_sk_source_version"],
        )
        _write_json(output_dir / SK_SOURCE_VERSION_VALIDATION_PATH, result)
        return 1
    if build_code != 0:
        check_results["cmake_build"] = _sk_source_version_validation_check(
            "cmake_build",
            "failed",
            _cmake_missing_include_failure_reason(
                build_evidence, Path(source_version["asset_path"])
            ),
            build_evidence,
        )
        result = _sk_source_version_validation_manifest(
            output_dir,
            "failed",
            commands,
            check_results,
            ["validate_sk_source_version"],
        )
        _write_json(output_dir / SK_SOURCE_VERSION_VALIDATION_PATH, result)
        return 1
    if _cmake_evidence_has_uncompiled_source_warning(build_evidence):
        check_results["cmake_build"] = _sk_source_version_validation_check(
            "cmake_build",
            "failed",
            "cmake_build_uncompiled_source",
            build_evidence,
        )
        result = _sk_source_version_validation_manifest(
            output_dir,
            "failed",
            commands,
            check_results,
            ["validate_sk_source_version"],
        )
        _write_json(output_dir / SK_SOURCE_VERSION_VALIDATION_PATH, result)
        return 1

    check_results["cmake_build"] = _sk_source_version_validation_check(
        "cmake_build",
        "passed",
        "cmake_build_passed",
        build_evidence,
    )
    result = _sk_source_version_validation_manifest(
        output_dir, "passed", commands, check_results, ["collect_runtime_input_spec"]
    )
    _write_json(output_dir / SK_SOURCE_VERSION_VALIDATION_PATH, result)
    return 0


def _assert_fresh_sk_source_version_pipeline_output(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise CliUsageError(
            f"source version pipeline output path is not a directory: {output_dir}"
        )
    if output_dir.exists() and output_dir.is_symlink():
        raise CliUsageError(
            f"source version pipeline output path is not a normal directory: {output_dir}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise CliUsageError(
            "source version pipeline output directory must be absent or empty"
        )


def _pipeline_stage_status_from_artifact(artifact: dict[str, Any]) -> str:
    return _require_string(artifact["status"], "source version pipeline stage status")


def _pipeline_stage_supported_next_actions(artifact: dict[str, Any]) -> list[str]:
    return [
        _require_string(item, "source version pipeline stage supported next action")
        for item in _require_list(
            artifact.get("supported_next_actions", []),
            "source version pipeline stage supported_next_actions",
        )
    ]


class SourceVersionPipelineStageInput(NamedTuple):
    name: str
    command: list[str]
    returncode: int
    artifact_path: str
    artifact: dict[str, Any]
    next_action: str


def _source_version_pipeline_stage(
    request: SourceVersionPipelineStageInput,
) -> dict[str, Any]:
    supported_next_actions = _pipeline_stage_supported_next_actions(request.artifact)
    return {
        "name": request.name,
        "command": request.command,
        "returncode": request.returncode,
        "status": _pipeline_stage_status_from_artifact(request.artifact),
        "artifact_path": request.artifact_path,
        "next_action": request.next_action,
        "supported_next_actions": supported_next_actions,
    }


def _source_version_pipeline_status(stage: dict[str, Any]) -> str:
    raw_status = stage["status"]
    if stage["name"] == "validate_sk_source_version" and raw_status == "passed":
        return "passed"
    if raw_status == "failed":
        return "failed"
    return "blocked"


def _source_version_pipeline_boundary(status: str) -> list[str]:
    if status == "passed":
        return SK_SOURCE_VERSION_PIPELINE_PASSED_BOUNDARY
    if status == "failed":
        return SK_SOURCE_VERSION_PIPELINE_FAILED_BOUNDARY
    return SK_SOURCE_VERSION_PIPELINE_BLOCKED_BOUNDARY


def _source_version_pipeline_manifest(
    asset_path: Path,
    output_dir: Path,
    stages: list[dict[str, Any]],
) -> dict[str, Any]:
    last_stage = stages[-1]
    status = _source_version_pipeline_status(last_stage)
    if status == "passed":
        supported_next_actions = ["collect_runtime_input_spec"]
    else:
        supported_next_actions = list(last_stage["supported_next_actions"]) or [
            "prepare_validated_sk_source_version"
        ]
    return {
        "status": status,
        "asset_path": str(asset_path.resolve()),
        "analysis_output_dir": str(output_dir.resolve()),
        "pipeline_path": SK_SOURCE_VERSION_PIPELINE_PATH,
        "stages": stages,
        "source_version_dir": SK_SOURCE_VERSION_DIR,
        "source_version_validation_path": SK_SOURCE_VERSION_VALIDATION_PATH,
        "supported_next_actions": supported_next_actions,
        "execution_boundary": _source_version_pipeline_boundary(status),
    }


def cmd_prepare_validated_sk_source_version(args: argparse.Namespace) -> int:
    asset_path = Path(args.asset).resolve()
    if not asset_path.exists():
        raise CliUsageError(f"asset path not found: {asset_path}")
    raw_output_dir = Path(args.output_dir)
    if raw_output_dir.is_symlink():
        raise CliUsageError(
            f"source version pipeline output path is not a normal directory: {raw_output_dir}"
        )
    output_dir = raw_output_dir.resolve()
    _assert_fresh_sk_source_version_pipeline_output(output_dir)

    stages: list[dict[str, Any]] = []

    analyze_command = [
        "analyze-sk-conversion",
        str(asset_path),
        "--output-dir",
        str(output_dir),
    ]
    for support_dir in getattr(args, "support_dir", None) or []:
        analyze_command.extend(["--support-dir", support_dir])
    for include_dir in getattr(args, "include_dir", None) or []:
        analyze_command.extend(["--include-dir", include_dir])
    if getattr(args, "ascend_cann_path", None):
        analyze_command.extend(["--ascend-cann-path", args.ascend_cann_path])
    if getattr(args, "ascend_arch", None):
        analyze_command.extend(["--ascend-arch", args.ascend_arch])
    for option in getattr(args, "ascend_compile_option", None) or []:
        analyze_command.extend(["--ascend-compile-option", option])
    for force_include in getattr(args, "ascend_force_include", None) or []:
        analyze_command.extend(["--ascend-force-include", force_include])
    analyze_return = cmd_analyze_sk_conversion(
        argparse.Namespace(
            asset=str(asset_path),
            output_dir=str(output_dir),
            support_dir=getattr(args, "support_dir", None),
            include_dir=getattr(args, "include_dir", None),
            ascend_cann_path=getattr(args, "ascend_cann_path", None),
            ascend_arch=getattr(args, "ascend_arch", None),
            ascend_compile_option=getattr(args, "ascend_compile_option", None),
            ascend_force_include=getattr(args, "ascend_force_include", None),
        )
    )
    analysis = _load_json_artifact(
        output_dir / "operator-sk-conversion-analysis.json",
        "operator-sk-conversion-analysis.json",
    )
    stages.append(
        _source_version_pipeline_stage(
            SourceVersionPipelineStageInput(
                "analyze_sk_conversion",
                analyze_command,
                analyze_return,
                "operator-sk-conversion-analysis.json",
                analysis,
                "generate_sk_source_scaffold",
            )
        )
    )

    source_command = ["generate-sk-source-scaffold", str(output_dir)]
    source_return = cmd_generate_sk_source_scaffold(
        argparse.Namespace(analysis_output_dir=str(output_dir))
    )
    source_scaffold = _load_json_artifact(
        output_dir / "operator-sk-source-scaffold.json",
        "operator-sk-source-scaffold.json",
    )
    source_next = (
        "run_sk_build_validation"
        if source_return == 0
        else (
            _pipeline_stage_supported_next_actions(source_scaffold)[0]
            if _pipeline_stage_supported_next_actions(source_scaffold)
            else "prepare_validated_sk_source_version"
        )
    )
    stages.append(
        _source_version_pipeline_stage(
            SourceVersionPipelineStageInput(
                "generate_sk_source_scaffold",
                source_command,
                source_return,
                "operator-sk-source-scaffold.json",
                source_scaffold,
                source_next,
            )
        )
    )
    if source_return != 0:
        result = _source_version_pipeline_manifest(asset_path, output_dir, stages)
        _write_json(output_dir / SK_SOURCE_VERSION_PIPELINE_PATH, result)
        return 1

    build_command = ["run-sk-build-validation", str(output_dir)]
    build_return = cmd_run_sk_build_validation(
        argparse.Namespace(analysis_output_dir=str(output_dir))
    )
    build_validation = _load_json_artifact(
        output_dir / "operator-sk-build-validation.json",
        "operator-sk-build-validation.json",
    )
    if build_return == 0:
        build_next = "prepare_sk_source_version"
    else:
        build_actions = _pipeline_stage_supported_next_actions(build_validation)
        build_next = (
            build_actions[0] if build_actions else "prepare_validated_sk_source_version"
        )
    stages.append(
        _source_version_pipeline_stage(
            SourceVersionPipelineStageInput(
                "run_sk_build_validation",
                build_command,
                build_return,
                "operator-sk-build-validation.json",
                build_validation,
                build_next,
            )
        )
    )
    if build_return != 0:
        result = _source_version_pipeline_manifest(asset_path, output_dir, stages)
        _write_json(output_dir / SK_SOURCE_VERSION_PIPELINE_PATH, result)
        return 1

    version_command = ["prepare-sk-source-version", str(output_dir)]
    version_return = cmd_prepare_sk_source_version(
        argparse.Namespace(analysis_output_dir=str(output_dir))
    )
    source_version = _load_json_artifact(
        output_dir / "operator-sk-source-version.json",
        "operator-sk-source-version.json",
    )
    if version_return == 0:
        version_next = "validate_sk_source_version"
    else:
        version_actions = _pipeline_stage_supported_next_actions(source_version)
        version_next = (
            version_actions[0]
            if version_actions
            else "prepare_validated_sk_source_version"
        )
    stages.append(
        _source_version_pipeline_stage(
            SourceVersionPipelineStageInput(
                "prepare_sk_source_version",
                version_command,
                version_return,
                "operator-sk-source-version.json",
                source_version,
                version_next,
            )
        )
    )
    if version_return != 0:
        result = _source_version_pipeline_manifest(asset_path, output_dir, stages)
        _write_json(output_dir / SK_SOURCE_VERSION_PIPELINE_PATH, result)
        return 1

    validation_command = ["validate-sk-source-version", str(output_dir)]
    validation_return = cmd_validate_sk_source_version(
        argparse.Namespace(analysis_output_dir=str(output_dir))
    )
    source_version_validation = _load_json_artifact(
        output_dir / SK_SOURCE_VERSION_VALIDATION_PATH,
        SK_SOURCE_VERSION_VALIDATION_PATH,
    )
    if validation_return == 0:
        validation_next = "collect_runtime_input_spec"
    else:
        validation_actions = _pipeline_stage_supported_next_actions(
            source_version_validation
        )
        validation_next = (
            validation_actions[0]
            if validation_actions
            else "prepare_validated_sk_source_version"
        )
    stages.append(
        _source_version_pipeline_stage(
            SourceVersionPipelineStageInput(
                "validate_sk_source_version",
                validation_command,
                validation_return,
                SK_SOURCE_VERSION_VALIDATION_PATH,
                source_version_validation,
                validation_next,
            )
        )
    )
    result = _source_version_pipeline_manifest(asset_path, output_dir, stages)
    _write_json(output_dir / SK_SOURCE_VERSION_PIPELINE_PATH, result)
    return 0 if validation_return == 0 else 1


def _render_scaffold_pyproject() -> str:
    return """[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "operator-scaffold"
version = "0.0.0"
description = "Generated operator scaffold package contract. Not a runtime package."
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["."]
include = ["operator_scaffold*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
"""


def _render_package_marker() -> str:
    return '''"""Generated package marker for the operator scaffold contract."""

__version__ = "0.0.0"
'''


def _require_dict_keys(
    payload: Any, expected_keys: set[str], label: str
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CliUsageError(f"{label} must be an object")
    keys = set(payload)
    missing = expected_keys - keys
    if missing:
        raise CliUsageError(f"missing scaffold manifest field: {sorted(missing)[0]}")
    unexpected = keys - expected_keys
    if unexpected:
        if label == "scaffold manifest":
            raise CliUsageError(
                f"unexpected scaffold manifest field: {sorted(unexpected)[0]}"
            )
        raise CliUsageError(f"unexpected {label} field: {sorted(unexpected)[0]}")
    return payload


def _artifact_path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _load_scaffold_manifest_for_validation(scaffold_output_dir: Path) -> dict[str, Any]:
    manifest_path = scaffold_output_dir / "operator-scaffold-manifest.json"
    payload = _load_json_artifact(manifest_path, "operator-scaffold-manifest.json")

    manifest = _require_dict_keys(payload, SCAFFOLD_MANIFEST_KEYS, "scaffold manifest")
    status = _require_string(manifest["status"], "status")
    if status not in {"generated", "blocked"}:
        raise CliUsageError(f"invalid scaffold manifest status: {status}")
    _require_string(manifest["asset_level"], "asset_level", allow_empty=False)

    source_files = _require_list(manifest["source_files"], "source_files")
    manifest["source_files"] = [
        _validate_manifest_path(item, "source_files path") for item in source_files
    ]

    kernel_entries = _require_list(manifest["kernel_entries"], "kernel_entries")
    normalized_entries: list[dict[str, str]] = []
    for entry in kernel_entries:
        item = _require_dict_keys(entry, {"name", "file"}, "kernel entry")
        normalized_entries.append(
            {
                "name": _require_string(item["name"], "kernel entry name"),
                "file": _validate_manifest_path(item["file"], "kernel entry file path"),
            }
        )
    manifest["kernel_entries"] = normalized_entries

    copied_source_files = _require_list(
        manifest["copied_source_files"], "copied_source_files"
    )
    normalized_copied: list[dict[str, str]] = []
    for copied in copied_source_files:
        item = _require_dict_keys(copied, {"source", "scaffold_path"}, "copied source")
        normalized_copied.append(
            {
                "source": _validate_manifest_path(item["source"], "copied source path"),
                "scaffold_path": _validate_manifest_path(
                    item["scaffold_path"], "copied scaffold path"
                ),
            }
        )
    manifest["copied_source_files"] = normalized_copied

    generated_files = _require_list(manifest["generated_files"], "generated_files")
    manifest["generated_files"] = [
        _validate_manifest_path(item, "generated file path") for item in generated_files
    ]

    skipped_items = _require_list(manifest["skipped_items"], "skipped_items")
    normalized_skipped: list[dict[str, str]] = []
    for skipped in skipped_items:
        item = _require_dict_keys(
            skipped, {"reason", "suggested_path", "cause"}, "skipped item"
        )
        normalized_skipped.append(
            {
                "reason": _require_string(item["reason"], "skipped reason"),
                "suggested_path": _validate_manifest_path(
                    item["suggested_path"],
                    "skipped suggested path",
                    allow_empty=True,
                ),
                "cause": _require_string(item["cause"], "skipped cause"),
            }
        )
    manifest["skipped_items"] = normalized_skipped

    blocked_reasons = _require_list(manifest["blocked_reasons"], "blocked_reasons")
    manifest["blocked_reasons"] = [
        _require_string(item, "blocked reason") for item in blocked_reasons
    ]
    boundary = _require_list(manifest["execution_boundary"], "execution_boundary")
    manifest["execution_boundary"] = [
        _require_string(item, "execution boundary") for item in boundary
    ]

    if status == "generated" and manifest["blocked_reasons"]:
        raise CliUsageError(
            "generated scaffold manifest must not include blocked_reasons"
        )
    if status == "blocked":
        if not manifest["blocked_reasons"]:
            raise CliUsageError(
                "blocked scaffold manifest must include blocked_reasons"
            )
        if manifest["generated_files"] or manifest["copied_source_files"]:
            raise CliUsageError(
                "blocked scaffold manifest must not include generated files"
            )
    return manifest


def _load_validation_manifest_for_preflight(
    scaffold_output_dir: Path,
    scaffold_manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = scaffold_output_dir / "operator-validation-manifest.json"
    payload = _load_json_artifact(manifest_path, "operator-validation-manifest.json")

    manifest = _require_exact_keys(
        payload,
        {
            "status",
            "scaffold_output_dir",
            "scaffold_manifest_path",
            "checks",
            "execution_boundary",
        },
        "validation manifest",
    )
    status = _require_string(manifest["status"], "validation status")
    if status not in {"local-ok", "local-failed", "blocked"}:
        raise CliUsageError(f"invalid validation manifest status: {status}")

    expected_output_dir = str(scaffold_output_dir.resolve())
    if (
        _require_string(
            manifest["scaffold_output_dir"], "validation scaffold_output_dir"
        )
        != expected_output_dir
    ):
        raise CliUsageError("validation scaffold_output_dir mismatch")
    if (
        _require_string(
            manifest["scaffold_manifest_path"], "validation scaffold_manifest_path"
        )
        != "operator-scaffold-manifest.json"
    ):
        raise CliUsageError("validation scaffold_manifest_path mismatch")
    boundary = _require_list(
        manifest["execution_boundary"], "validation execution_boundary"
    )
    if [
        _require_string(item, "validation execution boundary") for item in boundary
    ] != VALIDATION_BOUNDARY:
        raise CliUsageError("validation execution_boundary mismatch")

    checks = _require_list(manifest["checks"], "validation checks")
    normalized_checks: dict[str, dict[str, Any]] = {}
    for check in checks:
        item = _require_exact_keys(
            check, {"name", "status", "reason", "evidence"}, "validation check"
        )
        name = _require_string(item["name"], "validation check name")
        if name in normalized_checks:
            raise CliUsageError("duplicate validation check")
        check_status = _require_string(item["status"], "validation check status")
        if check_status not in {"passed", "failed", "skipped", "blocked"}:
            raise CliUsageError(f"invalid validation check status: {check_status}")
        evidence = _require_list(item["evidence"], "validation check evidence")
        normalized_checks[name] = {
            "name": name,
            "status": check_status,
            "reason": _require_string(item["reason"], "validation check reason"),
            "evidence": [
                _require_string(value, "validation check evidence item")
                for value in evidence
            ],
        }
    if set(normalized_checks) != set(VALIDATION_CHECK_NAMES):
        raise CliUsageError("validation checks mismatch")

    scaffold_status = scaffold_manifest["status"]
    check_statuses = [item["status"] for item in normalized_checks.values()]
    if scaffold_status == "blocked" and status != "blocked":
        raise CliUsageError("blocked scaffold requires blocked validation status")
    if scaffold_status == "generated" and status not in {"local-ok", "local-failed"}:
        raise CliUsageError("generated scaffold requires local validation status")
    if status == "local-ok":
        if "failed" in check_statuses:
            raise CliUsageError("local-ok validation must not contain failed checks")
        if "blocked" in check_statuses:
            raise CliUsageError("local-ok validation must not contain blocked checks")
    elif status == "local-failed":
        if "blocked" in check_statuses:
            raise CliUsageError(
                "local-failed validation must not contain blocked checks"
            )
        if "failed" not in check_statuses:
            raise CliUsageError("local-failed validation must contain failed checks")
    else:
        scaffold_project = normalized_checks["scaffold_project_present"]
        if (
            scaffold_project["status"] != "blocked"
            or scaffold_project["reason"] != "scaffold_generation_blocked"
        ):
            raise CliUsageError(
                "blocked validation missing scaffold_generation_blocked check"
            )
        if "failed" in check_statuses:
            raise CliUsageError("blocked validation must not contain failed checks")

    _validate_validation_check_semantics(status, normalized_checks)

    return {
        "status": status,
        "scaffold_output_dir": expected_output_dir,
        "scaffold_manifest_path": "operator-scaffold-manifest.json",
        "checks": [normalized_checks[name] for name in VALIDATION_CHECK_NAMES],
        "checks_by_name": normalized_checks,
        "execution_boundary": VALIDATION_BOUNDARY,
    }


def _validate_validation_check_semantics(
    status: str, checks: dict[str, dict[str, Any]]
) -> None:
    allowed_by_status = {
        "local-ok": {
            "scaffold_manifest_present": {("passed", "scaffold_manifest_loaded")},
            "scaffold_project_present": {("passed", "scaffold_project_present")},
            "copied_sources_present": {("passed", "copied_sources_present")},
            "generated_files_present": {("passed", "generated_files_present")},
            "kernel_entries_visible": {
                ("passed", "kernel_entries_visible"),
                ("skipped", "no_kernel_entries"),
            },
            "pytest_contract_tests": {("skipped", "execution_not_enabled")},
            "cmake_contract_static": {
                ("passed", "cmake_contract_static"),
                ("skipped", "no_cmake_contract"),
            },
        },
        "local-failed": {
            "scaffold_manifest_present": {("passed", "scaffold_manifest_loaded")},
            "scaffold_project_present": {
                ("passed", "scaffold_project_present"),
                ("failed", "scaffold_project_missing"),
            },
            "copied_sources_present": {
                ("passed", "copied_sources_present"),
                ("failed", "scaffold_path_escape"),
                ("failed", "copied_source_missing"),
                ("skipped", "scaffold_project_missing"),
            },
            "generated_files_present": {
                ("passed", "generated_files_present"),
                ("failed", "scaffold_path_escape"),
                ("failed", "generated_file_missing"),
                ("skipped", "scaffold_project_missing"),
            },
            "kernel_entries_visible": {
                ("passed", "kernel_entries_visible"),
                ("failed", "kernel_entry_source_not_copied"),
                ("failed", "scaffold_path_escape"),
                ("failed", "kernel_entry_not_visible"),
                ("skipped", "copied_sources_unavailable"),
                ("skipped", "no_kernel_entries"),
                ("skipped", "scaffold_project_missing"),
            },
            "pytest_contract_tests": {("skipped", "execution_not_enabled")},
            "cmake_contract_static": {
                ("passed", "cmake_contract_static"),
                ("failed", "scaffold_path_escape"),
                ("failed", "cmake_contract_incomplete"),
                ("failed", "header_compiled_as_source"),
                ("skipped", "no_cmake_contract"),
                ("skipped", "cmake_contract_unavailable"),
                ("skipped", "scaffold_project_missing"),
            },
        },
        "blocked": {
            "scaffold_manifest_present": {("passed", "scaffold_manifest_loaded")},
            "scaffold_project_present": {("blocked", "scaffold_generation_blocked")},
            "copied_sources_present": {("skipped", "scaffold_generation_blocked")},
            "generated_files_present": {("skipped", "scaffold_generation_blocked")},
            "kernel_entries_visible": {("skipped", "scaffold_generation_blocked")},
            "pytest_contract_tests": {("skipped", "execution_not_enabled")},
            "cmake_contract_static": {("skipped", "scaffold_generation_blocked")},
        },
    }
    for name, check in checks.items():
        allowed_for_name = allowed_by_status.get(status, {}).get(name)
        if allowed_for_name is None:
            raise CliUsageError(f"invalid validation check semantics for {name}")
        check_status = check.get("status")
        check_reason = check.get("reason")
        if (check_status, check_reason) not in allowed_for_name:
            raise CliUsageError(f"invalid validation check semantics for {name}")


def _normalize_preflight_manifest_checks(
    checks: list[Any],
) -> dict[str, dict[str, Any]]:
    normalized_checks: dict[str, dict[str, Any]] = {}
    for check in checks:
        item = _require_exact_keys(
            check, {"name", "status", "reason", "evidence"}, "preflight check"
        )
        name = _require_string(item["name"], "preflight check name")
        if name in normalized_checks:
            raise CliUsageError("duplicate preflight check")
        check_status = _require_string(item["status"], "preflight check status")
        if check_status not in {"passed", "failed", "skipped", "blocked"}:
            raise CliUsageError(f"invalid preflight check status: {check_status}")
        evidence = _require_list(item["evidence"], "preflight check evidence")
        normalized_checks[name] = {
            "name": name,
            "status": check_status,
            "reason": _require_string(item["reason"], "preflight check reason"),
            "evidence": [
                _require_string(value, "preflight check evidence item")
                for value in evidence
            ],
        }
    if set(normalized_checks) != set(PREFLIGHT_CHECK_NAMES):
        raise CliUsageError("preflight checks mismatch")
    return normalized_checks


def _validate_preflight_check_semantics(
    status: str, checks: dict[str, dict[str, Any]]
) -> None:
    allowed = {
        "validation_manifest_accepted": {
            ("passed", "validation_manifest_local_ok"),
            ("failed", "validation_manifest_local_failed"),
            ("blocked", "scaffold_generation_blocked"),
        },
        "scaffold_project_ready": {
            ("passed", "scaffold_project_ready"),
            ("failed", "scaffold_project_missing"),
            ("blocked", "scaffold_generation_blocked"),
            ("blocked", "validation_manifest_local_failed"),
        },
        "copied_sources_current": {
            ("passed", "copied_sources_current"),
            ("failed", "copied_source_missing"),
            ("failed", "scaffold_path_escape"),
            ("blocked", "scaffold_generation_blocked"),
            ("blocked", "validation_manifest_local_failed"),
        },
        "python_available": {
            ("passed", "python_available"),
            ("failed", "python_executable_unavailable"),
            ("blocked", "scaffold_generation_blocked"),
            ("blocked", "validation_manifest_local_failed"),
        },
        "pytest_contract_present": {
            ("passed", "pytest_contract_present"),
            ("failed", "pytest_contract_missing"),
            ("failed", "scaffold_path_escape"),
            ("blocked", "scaffold_generation_blocked"),
            ("blocked", "validation_manifest_local_failed"),
        },
        "ascend_toolchain_detected": {
            ("passed", "toolchain_command_found"),
            ("failed", "toolchain_command_not_found"),
            ("blocked", "scaffold_generation_blocked"),
            ("blocked", "validation_manifest_local_failed"),
        },
        "cmake_contract_present": {
            ("passed", "cmake_contract_present"),
            ("passed", "cmake_contract_present_present"),
            ("failed", "cmake_contract_missing"),
            ("failed", "scaffold_path_escape"),
            ("skipped", "cmake_contract_not_generated"),
            ("blocked", "scaffold_generation_blocked"),
            ("blocked", "validation_manifest_local_failed"),
        },
        "package_contract_present": {
            ("passed", "package_contract_present"),
            ("passed", "package_contract_present_present"),
            ("failed", "package_contract_missing"),
            ("failed", "scaffold_path_escape"),
            ("skipped", "package_contract_not_generated"),
            ("blocked", "scaffold_generation_blocked"),
            ("blocked", "validation_manifest_local_failed"),
        },
    }
    for name, check in checks.items():
        allowed_for_name = allowed.get(name)
        if allowed_for_name is None:
            raise CliUsageError(f"invalid preflight check semantics for {name}")
        check_status = check.get("status")
        check_reason = check.get("reason")
        if (check_status, check_reason) not in allowed_for_name:
            raise CliUsageError(f"invalid preflight check semantics for {name}")

    check_statuses = [item["status"] for item in checks.values()]
    if status == "ready":
        if "failed" in check_statuses:
            raise CliUsageError("ready preflight must not contain failed checks")
        if "blocked" in check_statuses:
            raise CliUsageError("ready preflight must not contain blocked checks")
        required_checks = set(PREFLIGHT_CHECK_NAMES) - {
            "cmake_contract_present",
            "package_contract_present",
        }
        if any(checks[name]["status"] != "passed" for name in required_checks):
            raise CliUsageError("ready preflight required checks must pass")
        return

    if status == "environment-missing":
        if "blocked" in check_statuses:
            raise CliUsageError(
                "environment-missing preflight must not contain blocked checks"
            )
        if "failed" not in check_statuses:
            raise CliUsageError(
                "environment-missing preflight must contain failed checks"
            )
        if checks["validation_manifest_accepted"]["status"] != "passed":
            raise CliUsageError(
                "environment-missing preflight requires accepted validation manifest"
            )
        return

    validation_check = checks["validation_manifest_accepted"]
    if (
        validation_check["status"] == "blocked"
        and validation_check["reason"] == "scaffold_generation_blocked"
    ):
        upstream_reason = "scaffold_generation_blocked"
    elif (
        validation_check["status"] == "failed"
        and validation_check["reason"] == "validation_manifest_local_failed"
    ):
        upstream_reason = "validation_manifest_local_failed"
    else:
        raise CliUsageError("blocked preflight semantics mismatch")
    for name, check in checks.items():
        if name == "validation_manifest_accepted":
            continue
        if check["status"] != "blocked" or check["reason"] != upstream_reason:
            raise CliUsageError("blocked preflight semantics mismatch")


def _load_preflight_manifest_for_test_run(scaffold_output_dir: Path) -> dict[str, Any]:
    manifest_path = scaffold_output_dir / "operator-execution-preflight.json"
    payload = _load_json_artifact(manifest_path, "operator-execution-preflight.json")

    manifest = _require_exact_keys(
        payload,
        {
            "status",
            "scaffold_output_dir",
            "scaffold_manifest_path",
            "validation_manifest_path",
            "checks",
            "execution_boundary",
        },
        "preflight manifest",
    )
    status = _require_string(manifest["status"], "preflight status")
    if status not in {"ready", "blocked", "environment-missing"}:
        raise CliUsageError(f"invalid preflight manifest status: {status}")
    expected_output_dir = str(scaffold_output_dir.resolve())
    if (
        _require_string(
            manifest["scaffold_output_dir"], "preflight scaffold_output_dir"
        )
        != expected_output_dir
    ):
        raise CliUsageError("preflight scaffold_output_dir mismatch")
    if (
        _require_string(
            manifest["scaffold_manifest_path"], "preflight scaffold_manifest_path"
        )
        != "operator-scaffold-manifest.json"
    ):
        raise CliUsageError("preflight scaffold_manifest_path mismatch")
    if (
        _require_string(
            manifest["validation_manifest_path"], "preflight validation_manifest_path"
        )
        != "operator-validation-manifest.json"
    ):
        raise CliUsageError("preflight validation_manifest_path mismatch")
    boundary = _require_list(
        manifest["execution_boundary"], "preflight execution_boundary"
    )
    if [
        _require_string(item, "preflight execution boundary") for item in boundary
    ] != PREFLIGHT_BOUNDARY:
        raise CliUsageError("preflight execution_boundary mismatch")

    checks = _normalize_preflight_manifest_checks(
        _require_list(manifest["checks"], "preflight checks")
    )
    _validate_preflight_check_semantics(status, checks)
    return {
        "status": status,
        "scaffold_output_dir": expected_output_dir,
        "scaffold_manifest_path": "operator-scaffold-manifest.json",
        "validation_manifest_path": "operator-validation-manifest.json",
        "checks": [checks[name] for name in PREFLIGHT_CHECK_NAMES],
        "checks_by_name": checks,
        "execution_boundary": PREFLIGHT_BOUNDARY,
    }


def _normalize_scaffold_test_result_checks(
    checks: list[Any],
) -> dict[str, dict[str, Any]]:
    normalized_checks: dict[str, dict[str, Any]] = {}
    for check in checks:
        item = _require_exact_keys(
            check,
            {"name", "status", "reason", "evidence"},
            "scaffold test result check",
        )
        name = _require_string(item["name"], "scaffold test result check name")
        if name in normalized_checks:
            raise CliUsageError("duplicate scaffold test result check")
        check_status = _require_string(
            item["status"], "scaffold test result check status"
        )
        if check_status not in {"passed", "failed", "blocked"}:
            raise CliUsageError(
                f"invalid scaffold test result check status: {check_status}"
            )
        evidence = _require_list(
            item["evidence"], "scaffold test result check evidence"
        )
        normalized_checks[name] = {
            "name": name,
            "status": check_status,
            "reason": _require_string(
                item["reason"], "scaffold test result check reason"
            ),
            "evidence": [
                _require_string(value, "scaffold test result check evidence item")
                for value in evidence
            ],
        }
    if set(normalized_checks) != set(SCAFFOLD_TEST_CHECK_NAMES):
        raise CliUsageError("scaffold test result checks mismatch")
    return normalized_checks


def _validate_scaffold_test_result_semantics(
    status: str, checks: dict[str, dict[str, Any]]
) -> None:
    allowed = {
        "preflight_ready": {
            ("passed", "preflight_ready"),
            ("blocked", "preflight_not_ready"),
        },
        "pytest_contract_files_present": {
            ("passed", "pytest_contract_files_present"),
            ("blocked", "preflight_not_ready"),
            ("blocked", "pytest_contract_missing"),
            ("blocked", "scaffold_path_escape"),
        },
        "pytest_contract_content_current": {
            ("passed", "pytest_contract_content_current"),
            ("blocked", "preflight_not_ready"),
            ("blocked", "pytest_contract_missing"),
            ("blocked", "pytest_contract_drift"),
            ("blocked", "scaffold_path_escape"),
        },
        "pytest_import_context_clean": {
            ("passed", "pytest_import_context_clean"),
            ("blocked", "preflight_not_ready"),
            ("blocked", "pytest_contract_missing"),
            ("blocked", "pytest_contract_drift"),
            ("blocked", "pytest_package_init_present"),
            ("blocked", "scaffold_path_escape"),
        },
        "pytest_contract_execution": {
            ("passed", "pytest_contract_passed"),
            ("failed", "pytest_contract_failed"),
            ("failed", "pytest_contract_timeout"),
            ("blocked", "preflight_not_ready"),
            ("blocked", "pytest_contract_missing"),
            ("blocked", "pytest_contract_drift"),
            ("blocked", "pytest_package_init_present"),
            ("blocked", "scaffold_path_escape"),
        },
    }
    for name, check in checks.items():
        allowed_for_name = allowed.get(name)
        if allowed_for_name is None:
            raise CliUsageError(
                f"invalid scaffold test result check semantics for {name}"
            )
        check_status = check.get("status")
        check_reason = check.get("reason")
        if (check_status, check_reason) not in allowed_for_name:
            raise CliUsageError(
                f"invalid scaffold test result check semantics for {name}"
            )

    statuses = [check["status"] for check in checks.values()]
    if status == "passed":
        if any(item != "passed" for item in statuses):
            raise CliUsageError(
                "passed scaffold test result must contain only passed checks"
            )
        if checks["pytest_contract_execution"]["reason"] != "pytest_contract_passed":
            raise CliUsageError(
                "passed scaffold test result missing pytest_contract_passed"
            )
    elif status == "failed":
        if "blocked" in statuses:
            raise CliUsageError(
                "failed scaffold test result must not contain blocked checks"
            )
        if checks["pytest_contract_execution"]["status"] != "failed":
            raise CliUsageError(
                "failed scaffold test result requires failed pytest execution"
            )
    else:
        if "failed" in statuses:
            raise CliUsageError(
                "blocked scaffold test result must not contain failed checks"
            )
        if "blocked" not in statuses:
            raise CliUsageError(
                "blocked scaffold test result must contain blocked checks"
            )


def _expected_scaffold_test_command(scaffold_manifest: dict[str, Any]) -> list[str]:
    relative_paths = [
        contract_path
        for contract_path in PREFLIGHT_TEST_CONTRACTS
        if contract_path in scaffold_manifest["generated_files"]
    ]
    return [
        sys.executable,
        "-I",
        "-B",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--noconftest",
        "-c",
        os.devnull,
        *relative_paths,
    ]


def _load_scaffold_test_result_for_build(
    scaffold_output_dir: Path,
    scaffold_manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = scaffold_output_dir / "operator-scaffold-test-result.json"
    payload = _load_json_artifact(manifest_path, "operator-scaffold-test-result.json")

    manifest = _require_exact_keys(
        payload,
        {
            "status",
            "scaffold_output_dir",
            "preflight_manifest_path",
            "command",
            "checks",
            "execution_boundary",
        },
        "scaffold test result",
    )
    status = _require_string(manifest["status"], "scaffold test result status")
    if status not in {"passed", "failed", "blocked"}:
        raise CliUsageError(f"invalid scaffold test result status: {status}")
    expected_output_dir = str(scaffold_output_dir.resolve())
    if (
        _require_string(
            manifest["scaffold_output_dir"], "scaffold test result scaffold_output_dir"
        )
        != expected_output_dir
    ):
        raise CliUsageError("scaffold test result scaffold_output_dir mismatch")
    if (
        _require_string(
            manifest["preflight_manifest_path"],
            "scaffold test result preflight_manifest_path",
        )
        != "operator-execution-preflight.json"
    ):
        raise CliUsageError("scaffold test result preflight_manifest_path mismatch")
    boundary = _require_list(
        manifest["execution_boundary"], "scaffold test result execution_boundary"
    )
    if [
        _require_string(item, "scaffold test result execution boundary")
        for item in boundary
    ] != SCAFFOLD_TEST_BOUNDARY:
        raise CliUsageError("scaffold test result execution_boundary mismatch")

    command = _require_list(manifest["command"], "scaffold test result command")
    normalized_command = [
        _require_string(item, "scaffold test result command item") for item in command
    ]
    expected_command = _expected_scaffold_test_command(scaffold_manifest)
    if status == "blocked":
        if normalized_command:
            raise CliUsageError("blocked scaffold test result command must be empty")
    elif normalized_command != expected_command:
        raise CliUsageError("scaffold test result command mismatch")

    checks = _normalize_scaffold_test_result_checks(
        _require_list(manifest["checks"], "scaffold test result checks")
    )
    _validate_scaffold_test_result_semantics(status, checks)
    return {
        "status": status,
        "scaffold_output_dir": expected_output_dir,
        "preflight_manifest_path": "operator-execution-preflight.json",
        "command": normalized_command,
        "checks": [checks[name] for name in SCAFFOLD_TEST_CHECK_NAMES],
        "checks_by_name": checks,
        "execution_boundary": SCAFFOLD_TEST_BOUNDARY,
    }


def _normalize_scaffold_build_result_checks(
    checks: list[Any],
) -> dict[str, dict[str, Any]]:
    normalized_checks: dict[str, dict[str, Any]] = {}
    for check in checks:
        item = _require_exact_keys(
            check,
            {"name", "status", "reason", "evidence"},
            "scaffold build result check",
        )
        name = _require_string(item["name"], "scaffold build result check name")
        if name in normalized_checks:
            raise CliUsageError("duplicate scaffold build result check")
        check_status = _require_string(
            item["status"], "scaffold build result check status"
        )
        if check_status not in {"passed", "failed", "blocked"}:
            raise CliUsageError(
                f"invalid scaffold build result check status: {check_status}"
            )
        evidence = _require_list(
            item["evidence"], "scaffold build result check evidence"
        )
        normalized_checks[name] = {
            "name": name,
            "status": check_status,
            "reason": _require_string(
                item["reason"], "scaffold build result check reason"
            ),
            "evidence": [
                _require_string(value, "scaffold build result check evidence item")
                for value in evidence
            ],
        }
    if set(normalized_checks) != set(SCAFFOLD_BUILD_CHECK_NAMES):
        raise CliUsageError("scaffold build result checks mismatch")
    return normalized_checks


def _validate_scaffold_build_result_semantics(
    status: str, checks: dict[str, dict[str, Any]]
) -> None:
    allowed = {
        "preflight_ready": {
            ("passed", "preflight_ready"),
            ("blocked", "preflight_not_ready"),
        },
        "pytest_contract_passed": {
            ("passed", "pytest_contract_passed"),
            ("blocked", "preflight_not_ready"),
            ("blocked", "pytest_contract_not_passed"),
        },
        "copied_sources_current": {
            ("passed", "copied_sources_current"),
            ("blocked", "preflight_not_ready"),
            ("blocked", "pytest_contract_not_passed"),
            ("blocked", "copied_source_missing"),
            ("blocked", "scaffold_path_escape"),
        },
        "cmake_command_available": {
            ("passed", "cmake_command_available"),
            ("blocked", "preflight_not_ready"),
            ("blocked", "pytest_contract_not_passed"),
            ("blocked", "copied_source_missing"),
            ("blocked", "scaffold_path_escape"),
            ("blocked", "cmake_command_not_ready"),
            ("blocked", "cmake_command_not_found"),
        },
        "cmake_contract_current": {
            ("passed", "cmake_contract_current"),
            ("blocked", "preflight_not_ready"),
            ("blocked", "pytest_contract_not_passed"),
            ("blocked", "copied_source_missing"),
            ("blocked", "scaffold_path_escape"),
            ("blocked", "cmake_command_not_ready"),
            ("blocked", "cmake_command_not_found"),
            ("blocked", "cmake_contract_not_ready"),
            ("blocked", "cmake_contract_missing"),
            ("blocked", "cmake_contract_drift"),
        },
        "build_dir_available": {
            ("passed", "build_dir_available"),
            ("blocked", "preflight_not_ready"),
            ("blocked", "pytest_contract_not_passed"),
            ("blocked", "copied_source_missing"),
            ("blocked", "scaffold_path_escape"),
            ("blocked", "cmake_command_not_ready"),
            ("blocked", "cmake_command_not_found"),
            ("blocked", "cmake_contract_not_ready"),
            ("blocked", "cmake_contract_missing"),
            ("blocked", "cmake_contract_drift"),
            ("blocked", "build_dir_already_exists"),
        },
        "cmake_configure": {
            ("passed", "cmake_configure_passed"),
            ("failed", "cmake_configure_failed"),
            ("failed", "cmake_configure_timeout"),
            ("blocked", "preflight_not_ready"),
            ("blocked", "pytest_contract_not_passed"),
            ("blocked", "copied_source_missing"),
            ("blocked", "scaffold_path_escape"),
            ("blocked", "cmake_command_not_ready"),
            ("blocked", "cmake_command_not_found"),
            ("blocked", "cmake_contract_not_ready"),
            ("blocked", "cmake_contract_missing"),
            ("blocked", "cmake_contract_drift"),
            ("blocked", "build_dir_already_exists"),
        },
        "cmake_build": {
            ("passed", "cmake_build_passed"),
            ("failed", "cmake_build_failed"),
            ("failed", "cmake_build_timeout"),
            ("blocked", "configure_not_passed"),
            ("blocked", "preflight_not_ready"),
            ("blocked", "pytest_contract_not_passed"),
            ("blocked", "copied_source_missing"),
            ("blocked", "scaffold_path_escape"),
            ("blocked", "cmake_command_not_ready"),
            ("blocked", "cmake_command_not_found"),
            ("blocked", "cmake_contract_not_ready"),
            ("blocked", "cmake_contract_missing"),
            ("blocked", "cmake_contract_drift"),
            ("blocked", "build_dir_already_exists"),
        },
    }
    for name, check in checks.items():
        allowed_for_name = allowed.get(name)
        if allowed_for_name is None:
            raise CliUsageError(
                f"invalid scaffold build result check semantics for {name}"
            )
        check_status = check.get("status")
        check_reason = check.get("reason")
        if (check_status, check_reason) not in allowed_for_name:
            raise CliUsageError(
                f"invalid scaffold build result check semantics for {name}"
            )

    passed_reasons = {
        "preflight_ready": "preflight_ready",
        "pytest_contract_passed": "pytest_contract_passed",
        "copied_sources_current": "copied_sources_current",
        "cmake_command_available": "cmake_command_available",
        "cmake_contract_current": "cmake_contract_current",
        "build_dir_available": "build_dir_available",
        "cmake_configure": "cmake_configure_passed",
        "cmake_build": "cmake_build_passed",
    }
    direct_blocked_reasons = {
        "preflight_ready": {"preflight_not_ready"},
        "pytest_contract_passed": {"pytest_contract_not_passed"},
        "copied_sources_current": {"copied_source_missing", "scaffold_path_escape"},
        "cmake_command_available": {
            "cmake_command_not_ready",
            "cmake_command_not_found",
        },
        "cmake_contract_current": {
            "cmake_contract_not_ready",
            "cmake_contract_missing",
            "cmake_contract_drift",
        },
        "build_dir_available": {"build_dir_already_exists"},
    }

    def _is_passed_shape(name: str) -> bool:
        check = checks.get(name, {})
        return (
            check.get("status") == "passed"
            and check.get("reason") == passed_reasons.get(name)
        )

    statuses = [check["status"] for check in checks.values()]
    if status == "passed":
        if any(item != "passed" for item in statuses):
            raise CliUsageError(
                "passed scaffold build result must contain only passed checks"
            )
        if checks["cmake_build"]["reason"] != "cmake_build_passed":
            raise CliUsageError(
                "passed scaffold build result missing cmake_build_passed"
            )
    elif status == "failed":
        if "failed" not in statuses:
            raise CliUsageError(
                "failed scaffold build result must contain failed checks"
            )
        for name in SCAFFOLD_BUILD_CHECK_NAMES[:6]:
            if not _is_passed_shape(name):
                raise CliUsageError("failed scaffold build result semantics mismatch")
        configure_check = checks["cmake_configure"]
        build_check = checks["cmake_build"]
        if configure_check["status"] == "failed":
            if (
                build_check["status"] != "blocked"
                or build_check["reason"] != "configure_not_passed"
            ):
                raise CliUsageError("failed scaffold build result semantics mismatch")
        elif build_check["status"] == "failed":
            if (
                configure_check["status"] != "passed"
                or configure_check["reason"] != "cmake_configure_passed"
            ):
                raise CliUsageError("failed scaffold build result semantics mismatch")
        else:
            raise CliUsageError(
                "failed scaffold build result requires configure or build failure"
            )
    else:
        if "failed" in statuses:
            raise CliUsageError(
                "blocked scaffold build result must not contain failed checks"
            )
        if "blocked" not in statuses:
            raise CliUsageError(
                "blocked scaffold build result must contain blocked checks"
            )
        first_blocked_index = next(
            index
            for index, name in enumerate(SCAFFOLD_BUILD_CHECK_NAMES)
            if checks[name]["status"] == "blocked"
        )
        if SCAFFOLD_BUILD_CHECK_NAMES[first_blocked_index] not in set(
            SCAFFOLD_BUILD_CHECK_NAMES[:6]
        ):
            raise CliUsageError("blocked scaffold build result semantics mismatch")
        first_blocked_name = SCAFFOLD_BUILD_CHECK_NAMES[first_blocked_index]
        blocked_reason = checks[first_blocked_name]["reason"]
        if blocked_reason not in direct_blocked_reasons[first_blocked_name]:
            raise CliUsageError("blocked scaffold build result semantics mismatch")
        for name in SCAFFOLD_BUILD_CHECK_NAMES[:first_blocked_index]:
            if not _is_passed_shape(name):
                raise CliUsageError("blocked scaffold build result semantics mismatch")
        remaining_check_start = first_blocked_index + 1
        remaining_check_names = SCAFFOLD_BUILD_CHECK_NAMES[remaining_check_start:]
        for name in remaining_check_names:
            check = checks[name]
            if check["status"] != "blocked" or check["reason"] != blocked_reason:
                raise CliUsageError("blocked scaffold build result semantics mismatch")


def _validate_scaffold_build_commands(
    status: str,
    commands: list[Any],
    scaffold_output_dir: Path,
    checks: dict[str, dict[str, Any]],
) -> list[list[str]]:
    normalized_commands: list[list[str]] = []
    for command in commands:
        command_items = _require_list(command, "scaffold build result command")
        normalized_commands.append(
            [
                _require_string(item, "scaffold build result command item")
                for item in command_items
            ]
        )

    scaffold_dir = str(scaffold_output_dir / "operator-scaffold")
    build_dir = str(scaffold_output_dir / "operator-scaffold-build")
    if status == "passed":
        expected_count = 2
    elif status == "failed" and checks["cmake_configure"]["status"] == "failed":
        expected_count = 1
    elif status == "failed" and checks["cmake_build"]["status"] == "failed":
        expected_count = 2
    else:
        expected_count = 0

    if status == "blocked":
        if normalized_commands:
            raise CliUsageError("blocked scaffold build result commands must be empty")
        return normalized_commands
    if len(normalized_commands) != expected_count:
        raise CliUsageError(f"{status} scaffold build result commands mismatch")
    first = normalized_commands[0]
    if Path(first[0]).name != "cmake":
        raise CliUsageError("scaffold build result cmake command mismatch")
    if len(first) != 5 or first[1:] != ["-S", scaffold_dir, "-B", build_dir]:
        raise CliUsageError("scaffold build result configure command mismatch")
    if expected_count == 2:
        second = normalized_commands[1]
        if second[0] != first[0] or Path(second[0]).name != "cmake":
            raise CliUsageError("scaffold build result cmake command mismatch")
        if len(second) != 3 or second[1:] != ["--build", build_dir]:
            raise CliUsageError("scaffold build result build command mismatch")
    return normalized_commands


def _load_scaffold_build_result_for_readiness(
    scaffold_output_dir: Path,
) -> dict[str, Any]:
    manifest_path = scaffold_output_dir / "operator-scaffold-build-result.json"
    payload = _load_json_artifact(manifest_path, "operator-scaffold-build-result.json")

    manifest = _require_exact_keys(
        payload,
        {
            "status",
            "scaffold_output_dir",
            "preflight_manifest_path",
            "test_result_manifest_path",
            "build_dir",
            "commands",
            "checks",
            "execution_boundary",
        },
        "scaffold build result",
    )
    status = _require_string(manifest["status"], "scaffold build result status")
    if status not in {"passed", "failed", "blocked"}:
        raise CliUsageError(f"invalid scaffold build result status: {status}")
    expected_output_dir = str(scaffold_output_dir.resolve())
    if (
        _require_string(
            manifest["scaffold_output_dir"], "scaffold build result scaffold_output_dir"
        )
        != expected_output_dir
    ):
        raise CliUsageError("scaffold build result scaffold_output_dir mismatch")
    if (
        _require_string(
            manifest["preflight_manifest_path"],
            "scaffold build result preflight_manifest_path",
        )
        != "operator-execution-preflight.json"
    ):
        raise CliUsageError("scaffold build result preflight_manifest_path mismatch")
    if (
        _require_string(
            manifest["test_result_manifest_path"],
            "scaffold build result test_result_manifest_path",
        )
        != "operator-scaffold-test-result.json"
    ):
        raise CliUsageError("scaffold build result test_result_manifest_path mismatch")
    build_dir = _require_string(
        manifest["build_dir"], "scaffold build result build_dir", allow_empty=True
    )
    blocked_with_build_dir = status == "blocked" and build_dir
    unblocked_without_expected_build_dir = (
        status != "blocked" and build_dir != "operator-scaffold-build"
    )
    if blocked_with_build_dir or unblocked_without_expected_build_dir:
        raise CliUsageError("scaffold build result build_dir mismatch")
    boundary = _require_list(
        manifest["execution_boundary"], "scaffold build result execution_boundary"
    )
    if [
        _require_string(item, "scaffold build result execution boundary")
        for item in boundary
    ] != SCAFFOLD_BUILD_BOUNDARY:
        raise CliUsageError("scaffold build result execution_boundary mismatch")

    checks = _normalize_scaffold_build_result_checks(
        _require_list(manifest["checks"], "scaffold build result checks")
    )
    _validate_scaffold_build_result_semantics(status, checks)
    commands = _validate_scaffold_build_commands(
        status,
        _require_list(manifest["commands"], "scaffold build result commands"),
        scaffold_output_dir,
        checks,
    )
    return {
        "status": status,
        "scaffold_output_dir": expected_output_dir,
        "preflight_manifest_path": "operator-execution-preflight.json",
        "test_result_manifest_path": "operator-scaffold-test-result.json",
        "build_dir": build_dir,
        "commands": commands,
        "checks": [checks[name] for name in SCAFFOLD_BUILD_CHECK_NAMES],
        "checks_by_name": checks,
        "execution_boundary": SCAFFOLD_BUILD_BOUNDARY,
    }


def _readiness_check(
    name: str, status: str, reason: str, evidence: list[str] | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "evidence": evidence or [],
    }


class ReadinessManifestInput(NamedTuple):
    scaffold_output_dir: Path
    status: str
    artifact_paths: dict[str, str]
    artifact_statuses: dict[str, str]
    checks: dict[str, dict[str, Any]]
    supported_next_actions: list[str]


def _readiness_manifest(
    request: ReadinessManifestInput,
) -> dict[str, Any]:
    return {
        "status": request.status,
        "scaffold_output_dir": str(request.scaffold_output_dir.resolve()),
        "artifact_paths": request.artifact_paths,
        "artifact_statuses": request.artifact_statuses,
        "readiness_checks": [
            request.checks[name] for name in SCAFFOLD_READINESS_CHECK_NAMES
        ],
        "supported_next_actions": request.supported_next_actions,
        "execution_boundary": SCAFFOLD_READINESS_BOUNDARY,
    }


def _readiness_boundary_checks() -> dict[str, dict[str, Any]]:
    return {
        "package_boundary_open": _readiness_check(
            "package_boundary_open", "open", "package_not_produced"
        ),
        "runtime_boundary_open": _readiness_check(
            "runtime_boundary_open", "open", "runtime_not_executed"
        ),
        "correctness_boundary_open": _readiness_check(
            "correctness_boundary_open", "open", "correctness_not_validated"
        ),
        "handoff_boundary_open": _readiness_check(
            "handoff_boundary_open", "open", "handoff_not_completed"
        ),
    }


def _summarize_scaffold_readiness(
    scaffold_output_dir: Path, scaffold_manifest: dict[str, Any]
) -> dict[str, Any]:
    artifact_paths = {
        "scaffold_manifest": "operator-scaffold-manifest.json",
        "validation_manifest": (
            "operator-validation-manifest.json"
            if _artifact_path_exists(
                scaffold_output_dir / "operator-validation-manifest.json"
            )
            else ""
        ),
        "preflight_manifest": (
            "operator-execution-preflight.json"
            if _artifact_path_exists(
                scaffold_output_dir / "operator-execution-preflight.json"
            )
            else ""
        ),
        "test_result_manifest": (
            "operator-scaffold-test-result.json"
            if _artifact_path_exists(
                scaffold_output_dir / "operator-scaffold-test-result.json"
            )
            else ""
        ),
        "build_result_manifest": (
            "operator-scaffold-build-result.json"
            if _artifact_path_exists(
                scaffold_output_dir / "operator-scaffold-build-result.json"
            )
            else ""
        ),
    }
    artifact_statuses = {
        "scaffold_manifest": scaffold_manifest["status"],
        "validation_manifest": "missing",
        "preflight_manifest": "missing",
        "test_result_manifest": "missing",
        "build_result_manifest": "missing",
    }
    checks = {
        "scaffold_manifest_accepted": _readiness_check(
            "scaffold_manifest_accepted",
            "passed" if scaffold_manifest["status"] == "generated" else "blocked",
            "scaffold_manifest_loaded"
            if scaffold_manifest["status"] == "generated"
            else "scaffold_generation_blocked",
            ["operator-scaffold-manifest.json"],
        ),
        **_readiness_boundary_checks(),
    }
    loaded_artifacts: dict[str, dict[str, Any]] = {}
    if artifact_paths["validation_manifest"]:
        loaded_artifacts["validation_manifest"] = (
            _load_validation_manifest_for_preflight(
                scaffold_output_dir, scaffold_manifest
            )
        )
        artifact_statuses["validation_manifest"] = loaded_artifacts[
            "validation_manifest"
        ]["status"]
    if artifact_paths["preflight_manifest"]:
        loaded_artifacts["preflight_manifest"] = _load_preflight_manifest_for_test_run(
            scaffold_output_dir
        )
        artifact_statuses["preflight_manifest"] = loaded_artifacts[
            "preflight_manifest"
        ]["status"]
    if artifact_paths["test_result_manifest"]:
        loaded_artifacts["test_result_manifest"] = _load_scaffold_test_result_for_build(
            scaffold_output_dir, scaffold_manifest
        )
        artifact_statuses["test_result_manifest"] = loaded_artifacts[
            "test_result_manifest"
        ]["status"]
    if artifact_paths["build_result_manifest"]:
        loaded_artifacts["build_result_manifest"] = (
            _load_scaffold_build_result_for_readiness(scaffold_output_dir)
        )
        artifact_statuses["build_result_manifest"] = loaded_artifacts[
            "build_result_manifest"
        ]["status"]

    if scaffold_manifest["status"] == "blocked":
        for name, reason in (
            ("validation_result_accepted", "validation_manifest_missing"),
            ("preflight_result_accepted", "preflight_manifest_missing"),
            ("pytest_contract_result_accepted", "pytest_result_missing"),
            ("cmake_build_result_accepted", "cmake_build_result_missing"),
        ):
            checks[name] = _readiness_check(name, "missing", reason)
        return _readiness_manifest(
            ReadinessManifestInput(
                scaffold_output_dir,
                "blocked",
                artifact_paths,
                artifact_statuses,
                checks,
                ["provide_supported_operator_source"],
            )
        )

    validation_path = scaffold_output_dir / "operator-validation-manifest.json"
    if not validation_path.exists():
        checks["validation_result_accepted"] = _readiness_check(
            "validation_result_accepted", "missing", "validation_manifest_missing"
        )
        checks["preflight_result_accepted"] = _readiness_check(
            "preflight_result_accepted", "missing", "preflight_manifest_missing"
        )
        checks["pytest_contract_result_accepted"] = _readiness_check(
            "pytest_contract_result_accepted", "missing", "pytest_result_missing"
        )
        checks["cmake_build_result_accepted"] = _readiness_check(
            "cmake_build_result_accepted", "missing", "cmake_build_result_missing"
        )
        return _readiness_manifest(
            ReadinessManifestInput(
                scaffold_output_dir,
                "partial",
                artifact_paths,
                artifact_statuses,
                checks,
                ["run_validate_scaffold"],
            )
        )
    validation_manifest = loaded_artifacts["validation_manifest"]
    if validation_manifest["status"] != "local-ok":
        reason = (
            "validation_local_failed"
            if validation_manifest["status"] == "local-failed"
            else "scaffold_generation_blocked"
        )
        checks["validation_result_accepted"] = _readiness_check(
            "validation_result_accepted",
            "blocked",
            reason,
            ["operator-validation-manifest.json"],
        )
        checks["preflight_result_accepted"] = _readiness_check(
            "preflight_result_accepted", "missing", "preflight_manifest_missing"
        )
        checks["pytest_contract_result_accepted"] = _readiness_check(
            "pytest_contract_result_accepted", "missing", "pytest_result_missing"
        )
        checks["cmake_build_result_accepted"] = _readiness_check(
            "cmake_build_result_accepted", "missing", "cmake_build_result_missing"
        )
        top_status = (
            "blocked"
            if validation_manifest["status"] == "blocked"
            else "local-build-blocked"
        )
        return _readiness_manifest(
            ReadinessManifestInput(
                scaffold_output_dir,
                top_status,
                artifact_paths,
                artifact_statuses,
                checks,
                ["fix_scaffold_validation"],
            )
        )
    checks["validation_result_accepted"] = _readiness_check(
        "validation_result_accepted",
        "passed",
        "validation_local_ok",
        ["operator-validation-manifest.json"],
    )

    preflight_path = scaffold_output_dir / "operator-execution-preflight.json"
    if not preflight_path.exists():
        checks["preflight_result_accepted"] = _readiness_check(
            "preflight_result_accepted", "missing", "preflight_manifest_missing"
        )
        checks["pytest_contract_result_accepted"] = _readiness_check(
            "pytest_contract_result_accepted", "missing", "pytest_result_missing"
        )
        checks["cmake_build_result_accepted"] = _readiness_check(
            "cmake_build_result_accepted", "missing", "cmake_build_result_missing"
        )
        return _readiness_manifest(
            ReadinessManifestInput(
                scaffold_output_dir,
                "partial",
                artifact_paths,
                artifact_statuses,
                checks,
                ["run_preflight_execution"],
            )
        )
    preflight_manifest = loaded_artifacts["preflight_manifest"]
    if preflight_manifest["status"] != "ready":
        checks["preflight_result_accepted"] = _readiness_check(
            "preflight_result_accepted",
            "blocked",
            "preflight_not_ready",
            ["operator-execution-preflight.json"],
        )
        checks["pytest_contract_result_accepted"] = _readiness_check(
            "pytest_contract_result_accepted", "missing", "pytest_result_missing"
        )
        checks["cmake_build_result_accepted"] = _readiness_check(
            "cmake_build_result_accepted", "missing", "cmake_build_result_missing"
        )
        return _readiness_manifest(
            ReadinessManifestInput(
                scaffold_output_dir,
                "local-build-blocked",
                artifact_paths,
                artifact_statuses,
                checks,
                ["fix_preflight_environment_or_drift"],
            )
        )
    checks["preflight_result_accepted"] = _readiness_check(
        "preflight_result_accepted",
        "passed",
        "preflight_ready",
        ["operator-execution-preflight.json"],
    )

    test_path = scaffold_output_dir / "operator-scaffold-test-result.json"
    if not test_path.exists():
        checks["pytest_contract_result_accepted"] = _readiness_check(
            "pytest_contract_result_accepted", "missing", "pytest_result_missing"
        )
        checks["cmake_build_result_accepted"] = _readiness_check(
            "cmake_build_result_accepted", "missing", "cmake_build_result_missing"
        )
        return _readiness_manifest(
            ReadinessManifestInput(
                scaffold_output_dir,
                "partial",
                artifact_paths,
                artifact_statuses,
                checks,
                ["run_scaffold_tests"],
            )
        )
    test_result = loaded_artifacts["test_result_manifest"]
    if test_result["status"] != "passed":
        checks["pytest_contract_result_accepted"] = _readiness_check(
            "pytest_contract_result_accepted",
            "blocked",
            "pytest_contract_not_passed",
            ["operator-scaffold-test-result.json"],
        )
        checks["cmake_build_result_accepted"] = _readiness_check(
            "cmake_build_result_accepted", "missing", "cmake_build_result_missing"
        )
        return _readiness_manifest(
            ReadinessManifestInput(
                scaffold_output_dir,
                "local-build-blocked",
                artifact_paths,
                artifact_statuses,
                checks,
                ["fix_scaffold_pytest_contract"],
            )
        )
    checks["pytest_contract_result_accepted"] = _readiness_check(
        "pytest_contract_result_accepted",
        "passed",
        "pytest_contract_passed",
        ["operator-scaffold-test-result.json"],
    )

    build_path = scaffold_output_dir / "operator-scaffold-build-result.json"
    if not build_path.exists():
        checks["cmake_build_result_accepted"] = _readiness_check(
            "cmake_build_result_accepted", "missing", "cmake_build_result_missing"
        )
        return _readiness_manifest(
            ReadinessManifestInput(
                scaffold_output_dir,
                "partial",
                artifact_paths,
                artifact_statuses,
                checks,
                ["run_scaffold_build"],
            )
        )
    build_result = loaded_artifacts["build_result_manifest"]
    if build_result["status"] == "passed":
        checks["cmake_build_result_accepted"] = _readiness_check(
            "cmake_build_result_accepted",
            "passed",
            "cmake_build_passed",
            ["operator-scaffold-build-result.json"],
        )
        actions = [
            "define_package_contract",
            "define_runtime_inputs",
            "define_correctness_oracle",
            "run_target_runtime_validation",
            "prepare_customer_handoff_notes",
        ]
        return _readiness_manifest(
            ReadinessManifestInput(
                scaffold_output_dir,
                "local-build-passed",
                artifact_paths,
                artifact_statuses,
                checks,
                actions,
            )
        )
    if build_result["status"] == "failed":
        reason = "cmake_build_failed"
        if build_result["checks_by_name"]["cmake_configure"]["status"] == "failed":
            reason = "cmake_configure_failed"
        checks["cmake_build_result_accepted"] = _readiness_check(
            "cmake_build_result_accepted",
            "failed",
            reason,
            ["operator-scaffold-build-result.json"],
        )
        return _readiness_manifest(
            ReadinessManifestInput(
                scaffold_output_dir,
                "local-build-failed",
                artifact_paths,
                artifact_statuses,
                checks,
                ["fix_cmake_build_failure"],
            )
        )
    checks["cmake_build_result_accepted"] = _readiness_check(
        "cmake_build_result_accepted",
        "blocked",
        "cmake_build_blocked",
        ["operator-scaffold-build-result.json"],
    )
    return _readiness_manifest(
        ReadinessManifestInput(
            scaffold_output_dir,
            "local-build-blocked",
            artifact_paths,
            artifact_statuses,
            checks,
            ["fix_cmake_build_gate"],
        )
    )


def _load_readiness_manifest_for_package_contract(
    scaffold_output_dir: Path, scaffold_manifest: dict[str, Any]
) -> dict[str, Any]:
    manifest_path = scaffold_output_dir / "operator-scaffold-readiness.json"
    manifest = _load_json_artifact(manifest_path, "operator-scaffold-readiness.json")
    expected = _summarize_scaffold_readiness(scaffold_output_dir, scaffold_manifest)
    if manifest != expected:
        raise CliUsageError(
            "operator-scaffold-readiness.json does not match current scaffold artifacts"
        )
    return expected


def _package_contract_check(
    name: str, status: str, reason: str, evidence: list[str] | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "evidence": evidence or [],
    }


def _fill_blocked_package_checks(
    check_results: dict[str, dict[str, Any]],
    start_name: str,
    reason: str,
    evidence: list[str] | None = None,
) -> None:
    started = False
    for name in PACKAGE_CONTRACT_CHECK_NAMES:
        if name == start_name:
            started = True
        if started and name not in check_results:
            check_results[name] = _package_contract_check(
                name, "blocked", reason, evidence if name == start_name else None
            )


def _package_contract_manifest(
    scaffold_output_dir: Path,
    status: str,
    check_results: dict[str, dict[str, Any]],
    supported_next_actions: list[str],
) -> dict[str, Any]:
    return {
        "status": status,
        "scaffold_output_dir": str(scaffold_output_dir.resolve()),
        "readiness_manifest_path": "operator-scaffold-readiness.json",
        "package_contract_path": "operator-scaffold/pyproject.toml",
        "package_files": PACKAGE_CONTRACT_FILES,
        "checks": [check_results[name] for name in PACKAGE_CONTRACT_CHECK_NAMES],
        "supported_next_actions": supported_next_actions,
        "execution_boundary": PACKAGE_CONTRACT_BOUNDARY,
    }


def _summarize_package_contract(
    scaffold_output_dir: Path,
    scaffold_manifest: dict[str, Any],
    readiness_manifest: dict[str, Any],
) -> dict[str, Any]:
    check_results: dict[str, dict[str, Any]] = {}

    if readiness_manifest["status"] != "local-build-passed":
        check_results["readiness_local_build_passed"] = _package_contract_check(
            "readiness_local_build_passed",
            "blocked",
            "readiness_not_local_build_passed",
            ["operator-scaffold-readiness.json"],
        )
        _fill_blocked_package_checks(
            check_results, "package_files_declared", "readiness_not_local_build_passed"
        )
        return _package_contract_manifest(
            scaffold_output_dir,
            "blocked",
            check_results,
            ["fix_scaffold_pipeline_before_package_contract"],
        )
    check_results["readiness_local_build_passed"] = _package_contract_check(
        "readiness_local_build_passed",
        "passed",
        "readiness_local_build_passed",
        ["operator-scaffold-readiness.json"],
    )

    generated_files = set(scaffold_manifest["generated_files"])
    expected_generated = {"pyproject.toml", "operator_scaffold/__init__.py"}
    if not expected_generated <= generated_files:
        _fill_blocked_package_checks(
            check_results,
            "package_files_declared",
            "package_contract_not_ready",
            sorted(expected_generated - generated_files),
        )
        return _package_contract_manifest(
            scaffold_output_dir,
            "blocked",
            check_results,
            ["fix_package_contract_files"],
        )
    check_results["package_files_declared"] = _package_contract_check(
        "package_files_declared",
        "passed",
        "package_files_declared",
        sorted(expected_generated),
    )

    scaffold_dir = scaffold_output_dir / "operator-scaffold"
    pyproject_check = _package_file_current_check(
        scaffold_dir,
        "pyproject.toml",
        _render_scaffold_pyproject(),
        check_name="pyproject_contract_current",
        reason_prefix="pyproject_contract",
    )
    check_results["pyproject_contract_current"] = pyproject_check
    if pyproject_check["status"] == "blocked":
        _fill_blocked_package_checks(
            check_results, "package_marker_current", pyproject_check["reason"]
        )
        return _package_contract_manifest(
            scaffold_output_dir,
            "blocked",
            check_results,
            ["fix_package_contract_files"],
        )

    marker_check = _package_file_current_check(
        scaffold_dir,
        "operator_scaffold/__init__.py",
        _render_package_marker(),
        check_name="package_marker_current",
        reason_prefix="package_marker",
    )
    check_results["package_marker_current"] = marker_check
    if marker_check["status"] == "blocked":
        return _package_contract_manifest(
            scaffold_output_dir,
            "blocked",
            check_results,
            ["fix_package_contract_files"],
        )

    return _package_contract_manifest(
        scaffold_output_dir,
        "defined",
        check_results,
        [
            "run_package_build",
            "define_runtime_inputs",
            "define_correctness_oracle",
            "run_target_runtime_validation",
            "prepare_customer_handoff_notes",
        ],
    )


def _package_file_current_check(
    scaffold_dir: Path,
    relative_path: str,
    expected_content: str,
    *,
    check_name: str,
    reason_prefix: str,
) -> dict[str, Any]:
    evidence = [f"operator-scaffold/{relative_path}"]
    if scaffold_dir.is_symlink():
        return _package_contract_check(
            check_name, "blocked", f"{reason_prefix}_path_escape", evidence
        )
    if not scaffold_dir.exists():
        return _package_contract_check(
            check_name, "blocked", f"{reason_prefix}_missing", evidence
        )
    if not scaffold_dir.is_dir():
        return _package_contract_check(
            check_name, "blocked", f"{reason_prefix}_not_file", evidence
        )

    current_parent = scaffold_dir
    for part in Path(relative_path).parent.parts:
        current_parent = current_parent / part
        if current_parent.is_symlink():
            return _package_contract_check(
                check_name, "blocked", f"{reason_prefix}_path_escape", evidence
            )
        if not current_parent.exists():
            return _package_contract_check(
                check_name, "blocked", f"{reason_prefix}_missing", evidence
            )
        if not current_parent.is_dir():
            return _package_contract_check(
                check_name, "blocked", f"{reason_prefix}_not_file", evidence
            )

    path = scaffold_dir / relative_path
    if path.is_symlink():
        return _package_contract_check(
            check_name, "blocked", f"{reason_prefix}_not_regular_file", evidence
        )
    if not path.exists():
        return _package_contract_check(
            check_name, "blocked", f"{reason_prefix}_missing", evidence
        )
    if not path.is_file():
        return _package_contract_check(
            check_name, "blocked", f"{reason_prefix}_not_file", evidence
        )
    try:
        path.resolve().relative_to(scaffold_dir.resolve())
    except ValueError:
        return _package_contract_check(
            check_name, "blocked", f"{reason_prefix}_path_escape", evidence
        )
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return _package_contract_check(
            check_name, "blocked", f"{reason_prefix}_not_utf8", evidence
        )
    except OSError:
        return _package_contract_check(
            check_name, "blocked", f"{reason_prefix}_unreadable", evidence
        )
    if content != expected_content:
        return _package_contract_check(
            check_name, "blocked", f"{reason_prefix}_drift", evidence
        )
    return _package_contract_check(
        check_name, "passed", f"{reason_prefix}_current", evidence
    )


def _package_build_check(
    name: str, status: str, reason: str, evidence: list[str] | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "evidence": evidence or [],
    }


def _package_build_result_manifest(
    scaffold_output_dir: Path,
    status: str,
    check_results: dict[str, dict[str, Any]],
    commands: list[dict[str, Any]] | None = None,
    artifacts: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "scaffold_output_dir": str(scaffold_output_dir.resolve()),
        "package_contract_manifest_path": "operator-package-contract.json",
        "build_dir": "operator-package-build",
        "source_dir": "operator-package-build/source",
        "wheel_dir": "operator-package-build/wheels",
        "checks": [check_results[name] for name in PACKAGE_BUILD_CHECK_NAMES],
        "commands": commands or [],
        "artifacts": artifacts or [],
        "execution_boundary": PACKAGE_BUILD_BOUNDARY,
    }


def _write_package_build_result(
    scaffold_output_dir: Path,
    status: str,
    check_results: dict[str, dict[str, Any]],
    commands: list[dict[str, Any]] | None = None,
    artifacts: list[str] | None = None,
) -> None:
    _write_json(
        scaffold_output_dir / "operator-package-build-result.json",
        _package_build_result_manifest(
            scaffold_output_dir, status, check_results, commands, artifacts
        ),
    )


def _fill_blocked_package_build_checks(
    check_results: dict[str, dict[str, Any]],
    start_name: str,
    reason: str,
    evidence: list[str] | None = None,
) -> None:
    started = False
    for name in PACKAGE_BUILD_CHECK_NAMES:
        if name == start_name:
            started = True
        if started and name not in check_results:
            check_results[name] = _package_build_check(
                name, "blocked", reason, evidence if name == start_name else None
            )


def _load_package_contract_for_build(
    scaffold_output_dir: Path, scaffold_manifest: dict[str, Any]
) -> dict[str, Any]:
    manifest_path = scaffold_output_dir / "operator-package-contract.json"
    manifest = _load_json_artifact(manifest_path, "operator-package-contract.json")
    readiness_manifest = _load_readiness_manifest_for_package_contract(
        scaffold_output_dir, scaffold_manifest
    )
    expected = _summarize_package_contract(
        scaffold_output_dir, scaffold_manifest, readiness_manifest
    )
    if manifest != expected:
        raise CliUsageError(
            "operator-package-contract.json does not match current scaffold artifacts"
        )
    return expected


def _snapshot_tree(root: Path) -> list[str]:
    if not root.exists():
        return ["<missing>"]
    snapshot: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root))):
        rel = str(path.relative_to(root))
        if path.is_symlink():
            snapshot.append(f"symlink:{rel}->{os.readlink(path)}")
        elif path.is_dir():
            snapshot.append(f"dir:{rel}")
        elif path.is_file():
            snapshot.append(f"file:{rel}:{_hash_file(path)}")
        else:
            snapshot.append(f"other:{rel}")
    return snapshot


def _staging_relative_path(contract_path: str) -> Path:
    prefix = "operator-scaffold/"
    if not contract_path.startswith(prefix):
        raise CliUsageError(
            "operator-package-contract.json package file path is outside operator-scaffold"
        )
    prefix_len = len(prefix)
    relative = contract_path[prefix_len:]
    if not relative:
        raise CliUsageError("operator-package-contract.json package file path is empty")
    return Path(_safe_scaffold_relative_path(relative))


def _copy_package_staging_source(
    scaffold_output_dir: Path,
    package_contract: dict[str, Any],
    source_dir: Path,
) -> dict[str, Any]:
    scaffold_dir = scaffold_output_dir / "operator-scaffold"
    expected_files = [
        _staging_relative_path(path) for path in package_contract["package_files"]
    ]
    for relative in expected_files:
        source_path = scaffold_dir / relative
        target_path = source_dir / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    actual_files = sorted(
        str(path.relative_to(source_dir))
        for path in source_dir.rglob("*")
        if path.is_file()
    )
    expected_relative = sorted(str(path) for path in expected_files)
    if actual_files != expected_relative:
        return _package_build_check(
            "package_staging_context_closed",
            "blocked",
            "package_staging_file_set_mismatch",
            actual_files,
        )
    content_checks = {
        "pyproject.toml": _render_scaffold_pyproject(),
        "operator_scaffold/__init__.py": _render_package_marker(),
    }
    for relative, expected_content in content_checks.items():
        if (source_dir / relative).read_text(encoding="utf-8") != expected_content:
            return _package_build_check(
                "package_staging_context_closed",
                "blocked",
                "package_staging_content_drift",
                [relative],
            )
    return _package_build_check(
        "package_staging_context_closed",
        "passed",
        "package_staging_context_closed",
        actual_files,
    )


def _sanitized_package_build_env() -> tuple[dict[str, str], dict[str, Any]]:
    keep_names = {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "TEMP",
        "TMP",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "LD_LIBRARY_PATH",
        "TORCH_DEVICE_BACKEND_AUTOLOAD",
    }
    env = {name: value for name, value in os.environ.items() if name in keep_names}
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_NO_INPUT"] = "1"
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    removed_pip_env = sorted(
        name
        for name in os.environ
        if name.startswith("PIP_")
        and name not in {"PIP_DISABLE_PIP_VERSION_CHECK", "PIP_NO_INPUT"}
    )
    removed_python_env = sorted(
        name for name in os.environ if name in {"PYTHONPATH", "PYTHONHOME"}
    )
    summary = {
        "PIP_DISABLE_PIP_VERSION_CHECK": env["PIP_DISABLE_PIP_VERSION_CHECK"],
        "PIP_NO_INPUT": env["PIP_NO_INPUT"],
        "TORCH_DEVICE_BACKEND_AUTOLOAD": env["TORCH_DEVICE_BACKEND_AUTOLOAD"],
        "removed_pip_env": removed_pip_env,
        "removed_python_env": removed_python_env,
    }
    return env, summary


def _tail_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    lines = value.splitlines()
    return "\n".join(lines[-40:])


def _run_package_wheel_command(
    build_dir: Path, source_dir: Path, wheel_dir: Path
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    env, env_summary = _sanitized_package_build_env()
    command = [
        sys.executable,
        "-I",
        "-B",
        "-m",
        "pip",
        "--isolated",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "--no-index",
        "--no-cache-dir",
        "--wheel-dir",
        str(wheel_dir),
        str(source_dir),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=build_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=PACKAGE_BUILD_TIMEOUT_SECONDS,
            shell=False,
            env=env,
        )
        command_result = {
            "name": "pip_wheel",
            "argv": command,
            "cwd": str(build_dir.resolve()),
            "timeout_seconds": PACKAGE_BUILD_TIMEOUT_SECONDS,
            "returncode": completed.returncode,
            "timed_out": False,
            "stdout_tail": _tail_text(completed.stdout),
            "stderr_tail": _tail_text(completed.stderr),
            "env": env_summary,
        }
        artifacts = sorted(
            _relative(path, build_dir)
            for path in wheel_dir.glob("*.whl")
            if path.is_file()
        )
        if completed.returncode == 0 and artifacts:
            check = _package_build_check(
                "package_wheel_build", "passed", "wheel_build_passed", artifacts
            )
        elif completed.returncode == 0:
            check = _package_build_check(
                "package_wheel_build",
                "failed",
                "wheel_artifact_missing",
                [str(wheel_dir)],
            )
        else:
            check = _package_build_check(
                "package_wheel_build",
                "failed",
                "wheel_build_failed",
                [f"returncode={completed.returncode}"],
            )
        return command_result, artifacts, check
    except subprocess.TimeoutExpired as exc:
        command_result = {
            "name": "pip_wheel",
            "argv": command,
            "cwd": str(build_dir.resolve()),
            "timeout_seconds": PACKAGE_BUILD_TIMEOUT_SECONDS,
            "returncode": None,
            "timed_out": True,
            "stdout_tail": _tail_text(exc.stdout),
            "stderr_tail": _tail_text(exc.stderr),
            "env": env_summary,
        }
        check = _package_build_check(
            "package_wheel_build",
            "failed",
            "wheel_build_timeout",
            [f"timeout={PACKAGE_BUILD_TIMEOUT_SECONDS}"],
        )
        return command_result, [], check


def cmd_define_package_contract(args: argparse.Namespace) -> int:
    scaffold_output_dir = Path(args.scaffold_output_dir).resolve()
    if not scaffold_output_dir.exists():
        raise CliUsageError(
            f"scaffold output directory not found: {scaffold_output_dir}"
        )
    if not scaffold_output_dir.is_dir():
        raise CliUsageError(
            f"scaffold output path is not a directory: {scaffold_output_dir}"
        )

    scaffold_manifest = _load_scaffold_manifest_for_validation(scaffold_output_dir)
    readiness_manifest = _load_readiness_manifest_for_package_contract(
        scaffold_output_dir, scaffold_manifest
    )
    package_contract = _summarize_package_contract(
        scaffold_output_dir, scaffold_manifest, readiness_manifest
    )
    _write_json(
        scaffold_output_dir / "operator-package-contract.json", package_contract
    )
    return 0 if package_contract["status"] == "defined" else 1


def cmd_run_package_build(args: argparse.Namespace) -> int:
    scaffold_output_dir = Path(args.scaffold_output_dir).resolve()
    if not scaffold_output_dir.exists():
        raise CliUsageError(
            f"scaffold output directory not found: {scaffold_output_dir}"
        )
    if not scaffold_output_dir.is_dir():
        raise CliUsageError(
            f"scaffold output path is not a directory: {scaffold_output_dir}"
        )

    scaffold_manifest = _load_scaffold_manifest_for_validation(scaffold_output_dir)
    package_contract = _load_package_contract_for_build(
        scaffold_output_dir, scaffold_manifest
    )
    check_results: dict[str, dict[str, Any]] = {
        "package_contract_current": _package_build_check(
            "package_contract_current",
            "passed",
            "package_contract_matches_current_scaffold",
            ["operator-package-contract.json"],
        )
    }

    if package_contract["status"] != "defined":
        check_results["package_contract_defined"] = _package_build_check(
            "package_contract_defined",
            "blocked",
            "package_contract_not_defined",
            ["operator-package-contract.json"],
        )
        build_dir = scaffold_output_dir / "operator-package-build"
        if build_dir.exists():
            check_results["package_build_dir_available"] = _package_build_check(
                "package_build_dir_available",
                "blocked",
                "package_build_dir_exists",
                ["operator-package-build"],
            )
            _fill_blocked_package_build_checks(
                check_results,
                "package_staging_context_closed",
                "package_build_dir_exists",
            )
        else:
            check_results["package_build_dir_available"] = _package_build_check(
                "package_build_dir_available",
                "passed",
                "package_build_dir_available",
                ["operator-package-build"],
            )
            _fill_blocked_package_build_checks(
                check_results,
                "package_staging_context_closed",
                "package_contract_not_defined",
            )
        _write_package_build_result(scaffold_output_dir, "blocked", check_results)
        return 1
    check_results["package_contract_defined"] = _package_build_check(
        "package_contract_defined",
        "passed",
        "package_contract_defined",
        ["operator-package-contract.json"],
    )

    build_dir = scaffold_output_dir / "operator-package-build"
    if build_dir.exists():
        check_results["package_build_dir_available"] = _package_build_check(
            "package_build_dir_available",
            "blocked",
            "package_build_dir_exists",
            ["operator-package-build"],
        )
        _fill_blocked_package_build_checks(
            check_results, "package_staging_context_closed", "package_build_dir_exists"
        )
        _write_package_build_result(scaffold_output_dir, "blocked", check_results)
        return 1
    check_results["package_build_dir_available"] = _package_build_check(
        "package_build_dir_available",
        "passed",
        "package_build_dir_available",
        ["operator-package-build"],
    )

    scaffold_snapshot_before = _snapshot_tree(scaffold_output_dir / "operator-scaffold")
    source_dir = build_dir / "source"
    wheel_dir = build_dir / "wheels"
    try:
        source_dir.mkdir(parents=True)
        wheel_dir.mkdir(parents=True)
        staging_check = _copy_package_staging_source(
            scaffold_output_dir, package_contract, source_dir
        )
    except OSError as exc:
        staging_check = _package_build_check(
            "package_staging_context_closed",
            "blocked",
            "package_staging_create_failed",
            [str(exc)],
        )
    check_results["package_staging_context_closed"] = staging_check
    if staging_check["status"] != "passed":
        check_results["package_wheel_build"] = _package_build_check(
            "package_wheel_build",
            "blocked",
            "package_staging_context_not_closed",
        )
        check_results["original_scaffold_unchanged"] = _package_build_check(
            "original_scaffold_unchanged",
            "blocked",
            "wheel_build_not_started",
        )
        _write_package_build_result(scaffold_output_dir, "blocked", check_results)
        return 1

    command_result, artifacts, wheel_check = _run_package_wheel_command(
        build_dir, source_dir, wheel_dir
    )
    check_results["package_wheel_build"] = wheel_check
    scaffold_snapshot_after = _snapshot_tree(scaffold_output_dir / "operator-scaffold")
    if scaffold_snapshot_after == scaffold_snapshot_before:
        check_results["original_scaffold_unchanged"] = _package_build_check(
            "original_scaffold_unchanged",
            "passed",
            "original_scaffold_unchanged",
        )
    else:
        check_results["original_scaffold_unchanged"] = _package_build_check(
            "original_scaffold_unchanged",
            "failed",
            "original_scaffold_changed",
        )

    status = "passed"
    if (
        wheel_check["status"] != "passed"
        or check_results["original_scaffold_unchanged"]["status"] != "passed"
    ):
        status = "failed"
    _write_package_build_result(
        scaffold_output_dir, status, check_results, [command_result], artifacts
    )
    return 0 if status == "passed" else 1


def _pybind_lib():
    import sys as _sys

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in _sys.path:
        _sys.path.insert(0, str(script_dir))
    import sk_pybind_lib

    return sk_pybind_lib


def cmd_generate_pybind_binding(args: argparse.Namespace) -> int:
    lib = _pybind_lib()
    adapted_root = Path(args.adapted_output_dir).resolve()
    if not adapted_root.exists() or not adapted_root.is_dir():
        raise CliUsageError(f"adapted output directory not found: {adapted_root}")
    manifest_path = adapted_root / "operator-sk-adapted.json"
    if not manifest_path.exists():
        raise CliUsageError(
            f"operator-sk-adapted.json not found in {adapted_root}; run codegen.adapt-sk-from-global first"
        )
    adapted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    adapted_dir = adapted_root / "operator-sk-adapted"
    if not adapted_dir.exists():
        raise CliUsageError(f"adapted source tree not found: {adapted_dir}")

    binding_manifest_path = adapted_root / "operator-sk-pybind-binding.json"
    if binding_manifest_path.exists():
        raise CliUsageError(
            f"pybind binding manifest already exists: {binding_manifest_path}"
        )
    binding_manifest = lib.generate_pybind_artifacts(
        adapted_dir, adapted_manifest, adapted_root
    )
    binding_manifest_path.write_text(
        json.dumps(binding_manifest, indent=2), encoding="utf-8"
    )
    _emit(
        json.dumps(
            {
                "package": binding_manifest["package_name"],
                "files": len(binding_manifest["written_files"]),
            }
        )
    )
    return 0


def _split_arches_for_manifest(raw_arches: str | None) -> list[str]:
    arches: list[str] = []
    seen: set[str] = set()
    for item in str(raw_arches or "").replace(";", ",").split(","):
        arch = item.strip().lower().replace("_", "-")
        if not arch or arch in seen:
            continue
        seen.add(arch)
        arches.append(arch)
    return arches


def _target_chips_to_arches(raw_chips: str | None) -> list[str]:
    try:
        from operator_target_arch import resolve_target_chips

        resolved, _ = resolve_target_chips(raw_chips)
        return _split_arches_for_manifest(",".join(item["arch"] for item in resolved))
    except Exception:
        return []


def _count_expected_extensions(
    binding_manifest: dict[str, Any], env: dict[str, str]
) -> int:
    entries = binding_manifest.get("kernel_entries", [])
    if env.get("SK_TARGET_CHIPS"):
        requested_arches = _target_chips_to_arches(env.get("SK_TARGET_CHIPS"))
    elif env.get("SK_NPU_ARCHS"):
        requested_arches = _split_arches_for_manifest(env.get("SK_NPU_ARCHS"))
    else:
        requested_arches = []
    if not isinstance(entries, list):
        return 0
    expected = 0
    for entry in entries:
        declared = _split_arches_for_manifest(
            ",".join(entry.get("supported_arches", []))
        )
        if requested_arches and declared:
            expected += len([arch for arch in requested_arches if arch in declared])
        elif requested_arches:
            expected += len(requested_arches)
    return expected


def _count_wheel_extensions(
    wheel_dir: Path, wheels: list[str], python_package: str = "op_extension"
) -> int | None:
    if not wheels:
        return None
    prefix = f"{python_package}/"
    try:
        with zipfile.ZipFile(wheel_dir / wheels[0]) as zf:
            return sum(
                1
                for name in zf.namelist()
                if name.startswith(prefix) and name.endswith((".so", ".pyd"))
            )
    except (OSError, zipfile.BadZipFile):
        return None


def cmd_build_native_wheel(args: argparse.Namespace) -> int:
    adapted_root = Path(args.adapted_output_dir).resolve()
    binding_path = adapted_root / "operator-sk-pybind-binding.json"
    if not binding_path.exists():
        raise CliUsageError(
            f"pybind binding manifest not found: {binding_path}; run generate-pybind-binding first"
        )
    binding_manifest = json.loads(binding_path.read_text(encoding="utf-8"))
    pybind_root = Path(binding_manifest["package_root"])
    if not pybind_root.exists():
        raise CliUsageError(f"pybind source root missing: {pybind_root}")

    build_dir = adapted_root / "operator-sk-native-wheel-build"
    if build_dir.exists():
        raise CliUsageError(f"native wheel build directory already exists: {build_dir}")
    build_dir.mkdir(parents=True)
    wheel_dir = build_dir / "wheels"
    wheel_dir.mkdir()

    cmd = [
        sys.executable,
        "-I",
        "-B",
        "-m",
        "pip",
        "--isolated",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "--no-cache-dir",
        "--wheel-dir",
        str(wheel_dir),
        str(pybind_root),
    ]
    env = os.environ.copy()
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    requested_jobs = int(args.jobs) if getattr(args, "jobs", None) else None
    if requested_jobs is not None:
        if requested_jobs <= 0:
            raise CliUsageError("--jobs must be a positive integer")
        env["SK_BISHENG_JOBS"] = str(requested_jobs)
    if getattr(args, "target_chip", ""):
        env["SK_TARGET_CHIPS"] = str(args.target_chip)
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        cwd=str(build_dir),
        env=env,
        timeout=600,
    )
    log_path = build_dir / "pip-wheel.log"
    log_path.write_text(completed.stdout or "", encoding="utf-8")

    wheels = sorted(p.name for p in wheel_dir.glob("*.whl"))
    status = "passed" if (completed.returncode == 0 and wheels) else "failed"
    structural_toolchain = env.get("SK_OPERATOR_STRUCTURAL_TOOLCHAIN") == "1"
    if status == "passed" and structural_toolchain:
        status = "structural-passed"
    actual_extension_count = _count_wheel_extensions(
        wheel_dir,
        wheels,
        str(
            binding_manifest.get("python_package")
            or binding_manifest.get("package_name")
            or "op_extension"
        ).replace("-", "_"),
    )
    result_manifest = {
        "status": status,
        "package_name": binding_manifest["package_name"],
        "python_package": binding_manifest.get("python_package", "op_extension"),
        "pybind_source_root": str(pybind_root),
        "build_dir": str(build_dir),
        "command": cmd,
        "jobs": requested_jobs,
        "sk_bisheng_jobs": env.get("SK_BISHENG_JOBS", ""),
        "target_chips": env.get("SK_TARGET_CHIPS", ""),
        "torch_device_backend_autoload": env.get("TORCH_DEVICE_BACKEND_AUTOLOAD", ""),
        "structural": structural_toolchain,
        "extension_module_pattern": binding_manifest.get(
            "extension_module_pattern", ""
        ),
        "extension_count": (
            actual_extension_count
            if actual_extension_count is not None
            else _count_expected_extensions(binding_manifest, env)
        ),
        "return_code": completed.returncode,
        "wheels": wheels,
        "log_path": str(log_path),
    }
    (adapted_root / "operator-sk-native-wheel.json").write_text(
        json.dumps(result_manifest, indent=2), encoding="utf-8"
    )
    _emit(json.dumps({"status": status, "wheels": wheels}))
    return 0 if status == "passed" else 1


def cmd_build_baseline(args: argparse.Namespace) -> int:
    from operator_baseline_build import build_baseline

    asset = Path(args.asset).resolve()
    if not asset.exists():
        raise CliUsageError(f"asset path not found: {asset}")
    if not args.entry_name:
        raise CliUsageError("--entry-name is required")
    manifest = build_baseline(
        asset,
        Path(args.output_dir).resolve(),
        backend=args.backend,
        entry_name=args.entry_name,
        structural=args.structural,
    )
    _emit(
        json.dumps({"status": manifest["status"], "entry_name": manifest["entry_name"]})
    )
    return 0 if manifest.get("status") == "passed" else 1


def cmd_build_standalone_executable(args: argparse.Namespace) -> int:
    output_root = Path(args.standalone_output_dir).resolve()
    manifest_path = output_root / "operator-sk-standalone-verify.json"
    if not manifest_path.exists():
        raise CliUsageError(
            "operator-sk-standalone-verify.json not found in "
            f"{output_root}; run codegen.generate-standalone-compare first"
        )
    verify_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_root = output_root / "operator-sk-standalone-verify"
    cmake_path = source_root / "CMakeLists.txt"
    runtime_sources_manifest = verify_manifest.get("runtime_sources")
    if isinstance(runtime_sources_manifest, dict) and runtime_sources_manifest:
        runtime_sources = {
            str(target): output_root / str(source)
            for target, source in runtime_sources_manifest.items()
        }
    else:
        runtime_sources = {"runtime_compare": source_root / "runtime_compare.asc"}
    if not cmake_path.is_file() or any(
        not source.is_file() for source in runtime_sources.values()
    ):
        raise CliUsageError(f"standalone CMake project missing under {source_root}")

    build_dir = source_root / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    configure_cmd = ["cmake", "-S", str(source_root), "-B", str(build_dir)]
    build_cmd = ["cmake", "--build", str(build_dir)]
    log_chunks: list[str] = []
    configure = subprocess.run(
        configure_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=300,
    )
    log_chunks.append("$ " + " ".join(configure_cmd))
    log_chunks.append(configure.stdout or "")
    if configure.returncode == 0:
        build = subprocess.run(
            build_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=600,
        )
    else:
        build = subprocess.CompletedProcess(
            build_cmd, returncode=configure.returncode, stdout=""
        )
    log_chunks.append("$ " + " ".join(build_cmd))
    log_chunks.append(build.stdout or "")
    log_path = output_root / "build-log.txt"
    log_path.write_text("\n".join(log_chunks), encoding="utf-8")

    executable_targets = {target: build_dir / target for target in runtime_sources}
    structural_toolchain = os.environ.get("SK_OPERATOR_STRUCTURAL_TOOLCHAIN") == "1"
    status = (
        "passed" if configure.returncode == 0 and build.returncode == 0 else "failed"
    )
    if status == "passed" and structural_toolchain:
        status = "structural-passed"
    entry_targets = (
        verify_manifest.get("entry_targets")
        if isinstance(verify_manifest.get("entry_targets"), dict)
        else {}
    )
    entries = verify_manifest.get("entries", [])
    allow_mock_executable = (
        os.environ.get("SK_OPERATOR_ALLOW_STANDALONE_MOCK_EXECUTABLE") == "1"
    )
    missing_executables: list[str] = []
    if status in {"passed", "structural-passed"}:
        for target, executable in executable_targets.items():
            if executable.exists():
                continue
            if not allow_mock_executable:
                missing_executables.append(str(executable))
                continue
            target_entries = [
                entry
                for entry in entries
                if entry_targets.get(
                    entry.get("entry_name"), next(iter(executable_targets))
                )
                == target
            ]
            stdout_payload = {
                "backend": "standalone",
                "status": "structural-passed" if structural_toolchain else "passed",
                "outputs": {
                    entry["entry_name"]: {"baseline": [0], "sk": [0]}
                    for entry in target_entries
                },
                "calls": {
                    entry["entry_name"]: [
                        f"launch_{entry['entry_name']}_baseline",
                        f"launch_{entry['entry_name']}_sk",
                    ]
                    for entry in target_entries
                },
                "statuses": {
                    entry["entry_name"]: {
                        "status": "structural-passed"
                        if structural_toolchain
                        else "passed",
                        "reason": "standalone_mock_bind_target_route_matched",
                    }
                    for entry in target_entries
                },
            }
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.write_text(
                "#!/bin/sh\nprintf '%s\\n' "
                + repr(json.dumps(stdout_payload, separators=(",", ":")))
                + "\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
    if missing_executables:
        status = "failed"
    executable = next(iter(executable_targets.values()))
    entry_executables = {
        entry["entry_name"]: str(
            executable_targets[
                entry_targets.get(entry["entry_name"], next(iter(executable_targets)))
            ]
        )
        for entry in entries
    }
    (output_root / "executable-path.txt").write_text(
        str(executable) + "\n", encoding="utf-8"
    )
    result_manifest = {
        "status": status,
        "standalone_verify_dir": verify_manifest["standalone_verify_dir"],
        "target_chip": args.target_chip or "",
        "target_cann": getattr(args, "target_cann", "") or "",
        "build_dir": str(build_dir),
        "configure_command": configure_cmd,
        "build_command": build_cmd,
        "configure_return_code": configure.returncode,
        "build_return_code": build.returncode,
        "executable": str(executable),
        "executable_targets": {
            target: str(path) for target, path in executable_targets.items()
        },
        "executables": entry_executables,
        "missing_executables": missing_executables,
        "mock_executable_allowed": allow_mock_executable,
        "structural": structural_toolchain,
        "log_path": str(log_path),
    }
    (output_root / "operator-sk-standalone-build.json").write_text(
        json.dumps(result_manifest, indent=2), encoding="utf-8"
    )
    _emit(json.dumps({"status": status, "executable": str(executable)}))
    return 0 if status == "passed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Top-level CLI for sk-operator-build-package"
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    sk_source_pipeline = subparsers.add_parser(
        "prepare-validated-sk-source-version",
        help="Run the first-step pipeline from operator source to validated SK source version",
    )
    sk_source_pipeline.add_argument("asset", help="Operator asset directory or file")
    sk_source_pipeline.add_argument(
        "--output-dir",
        default="operator-delivery-output",
        help="Fresh output directory",
    )
    sk_source_pipeline.add_argument(
        "--support-dir",
        action="append",
        default=[],
        help="Support source directory to copy into SK source, optionally name=/path",
    )
    sk_source_pipeline.add_argument(
        "--include-dir",
        action="append",
        default=[],
        help="External include directory to use during generated CMake builds",
    )
    sk_source_pipeline.add_argument(
        "--ascend-cann-path",
        help="CANN root containing tools/bisheng_compiler/bin/bisheng for Ascend build validation",
    )
    sk_source_pipeline.add_argument(
        "--ascend-arch",
        help="Ascend NPU arch passed as --npu-arch=<value> when Ascend build validation is enabled",
    )
    sk_source_pipeline.add_argument(
        "--ascend-compile-option",
        action="append",
        default=[],
        help="Extra Bisheng compile option for generated Ascend CMake builds",
    )
    sk_source_pipeline.add_argument(
        "--ascend-force-include",
        action="append",
        default=[],
        help="Header file to force include before generated Ascend source compilation",
    )
    sk_source_pipeline.set_defaults(func=cmd_prepare_validated_sk_source_version)

    sk_build = subparsers.add_parser(
        "run-sk-build-validation",
        help="Run controlled CMake configure/build for an exact-current generated SK source scaffold",
    )
    sk_build.add_argument(
        "analysis_output_dir",
        help="Output directory produced by generate-sk-source-scaffold",
    )
    sk_build.set_defaults(func=cmd_run_sk_build_validation)

    sk_source_version = subparsers.add_parser(
        "prepare-sk-source-version",
        help="Prepare a build-validated SK source version directory after SK build validation",
    )
    sk_source_version.add_argument(
        "analysis_output_dir",
        help="Output directory produced by run-sk-build-validation",
    )
    sk_source_version.set_defaults(func=cmd_prepare_sk_source_version)

    sk_source_version_validation = subparsers.add_parser(
        "validate-sk-source-version",
        help="Run controlled CMake configure/build for the exported SK source version directory",
    )
    sk_source_version_validation.add_argument(
        "analysis_output_dir",
        help="Output directory produced by prepare-sk-source-version",
    )
    sk_source_version_validation.set_defaults(func=cmd_validate_sk_source_version)

    gen_pybind = subparsers.add_parser(
        "generate-pybind-binding",
        help="Emit pybind11 wrapper + Python package + setup.py from codegen's adapted output",
    )
    gen_pybind.add_argument(
        "adapted_output_dir",
        help="Output directory containing operator-sk-adapted/ and operator-sk-adapted.json",
    )
    gen_pybind.set_defaults(func=cmd_generate_pybind_binding)

    native_wheel = subparsers.add_parser(
        "build-native-wheel",
        help="Run `pip wheel` against the generated pybind package; bisheng on PATH compiles the kernel into a .so",
    )
    native_wheel.add_argument(
        "adapted_output_dir",
        help="Output directory containing operator-sk-pybind-binding.json",
    )
    native_wheel.add_argument(
        "--jobs",
        type=int,
        help="Maximum parallel bisheng compile jobs for generated native extensions",
    )
    native_wheel.add_argument(
        "--target-chip",
        default="",
        help="Comma/semicolon separated target chips used by sparse generated setup.py",
    )
    native_wheel.set_defaults(func=cmd_build_native_wheel)

    baseline = subparsers.add_parser(
        "build-baseline",
        help="Build or record the non-SK baseline artifact for one normalized operator unit",
    )
    baseline.add_argument(
        "asset", help="Normalized operator unit asset directory or source file"
    )
    baseline.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for operator-baseline-build.json",
    )
    baseline.add_argument("--entry-name", required=True, help="Kernel entry name")
    baseline.add_argument(
        "--backend", default="baseline_direct_asc", help="Baseline backend id"
    )
    baseline.add_argument(
        "--structural",
        action="store_true",
        help="Use structural placeholder artifacts without numerical execution",
    )
    baseline.set_defaults(func=cmd_build_baseline)

    standalone_build = subparsers.add_parser(
        "build-standalone-executable",
        help="Run CMake configure/build for a generated standalone ASC comparison executable",
    )
    standalone_build.add_argument(
        "standalone_output_dir",
        help="Output directory containing operator-sk-standalone-verify.json",
    )
    standalone_build.add_argument(
        "--target-chip", default="", help="Target chip recorded in the build manifest"
    )
    standalone_build.add_argument("--target-cann", default="", help=argparse.SUPPRESS)
    standalone_build.set_defaults(func=cmd_build_standalone_executable)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(args.func(args))
    except CliUsageError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
