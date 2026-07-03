# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""SK Operator Sample Generation CLI -- runnable SK target runtime sample and scoped correctness verdict."""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
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
            "Define the operator entry contract before build or test scaffolds can target a stable callable unit."
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
ASCEND_COMPILE_UNITS_MISSING = "ascend_compile_units_missing"
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
SK_RUNTIME_INPUT_SPEC_CHECK_NAMES = (
    "sk_build_validation_current",
    "sk_build_validation_passed",
    "kernel_entries_declared",
    "copied_sources_current",
    "runtime_input_specs_defined",
    "runtime_execution_boundary_open",
    "correctness_boundary_open",
)
SK_RUNTIME_INPUT_SPEC_BOUNDARY = [
    "runtime not executed",
    "operator correctness not validated",
    "customer handoff not completed",
]
SK_RUNTIME_INPUT_VALUES_CHECK_NAMES = (
    "sk_runtime_input_spec_current",
    "input_value_spec_schema",
    "input_value_sets_complete",
    "runtime_input_values_declared",
    "runtime_execution_boundary_open",
    "correctness_boundary_open",
)
SK_RUNTIME_INPUT_VALUES_BOUNDARY = SK_RUNTIME_INPUT_SPEC_BOUNDARY
SK_CORRECTNESS_ORACLE_SPEC_CHECK_NAMES = (
    "sk_runtime_input_spec_current",
    "sk_runtime_input_values_current",
    "oracle_spec_schema",
    "oracle_sets_complete",
    "golden_outputs_declared",
    "correctness_comparator_declared",
    "runtime_execution_boundary_open",
    "correctness_boundary_open",
)
SK_CORRECTNESS_ORACLE_SPEC_BOUNDARY = SK_RUNTIME_INPUT_SPEC_BOUNDARY
SK_TARGET_RUNTIME_VALIDATION_CHECK_NAMES = (
    "sk_correctness_oracle_spec_current",
    "sk_correctness_oracle_spec_defined",
    "sk_runtime_input_values_current",
    "sk_runtime_input_values_defined",
    "runtime_command_spec_schema",
    "runtime_command_cwd_safe",
    "runtime_command_executable_resolved",
    "runtime_input_values_manifest_bound_as_argv",
    "runtime_command_execution",
    "output_comparison_boundary_open",
    "correctness_boundary_open",
    "handoff_boundary_open",
)
SK_TARGET_RUNTIME_VALIDATION_BOUNDARY = [
    "output comparison not executed",
    "operator correctness not validated",
    "customer handoff not completed",
]
SK_RUNTIME_OUTPUT_COMPARISON_CHECK_NAMES = (
    "sk_correctness_oracle_spec_current",
    "sk_correctness_oracle_spec_defined",
    "sk_target_runtime_validation_current",
    "sk_target_runtime_validation_passed",
    "runtime_output_spec_schema",
    "runtime_output_sets_complete",
    "declared_runtime_outputs_compared",
    "runtime_output_provenance_boundary_open",
    "correctness_boundary_open",
    "handoff_boundary_open",
)
SK_RUNTIME_OUTPUT_COMPARISON_BOUNDARY = [
    "runtime output provenance not validated",
    "operator correctness not validated",
    "customer handoff not completed",
]
SK_OPERATOR_CORRECTNESS_VERDICT_CHECK_NAMES = (
    "sk_runtime_output_comparison_current",
    "sk_runtime_output_comparison_matched",
    "operator_correctness_validated",
    "handoff_boundary_open",
)
SK_OPERATOR_CORRECTNESS_VERDICT_BOUNDARY = [
    "customer handoff not completed",
]
TARGET_RUNTIME_ENV_ALLOWLIST = (
    "PATH",
    "LD_LIBRARY_PATH",
    "ASCEND_HOME_PATH",
    "PYTHONPATH",
)
TARGET_RUNTIME_OUTPUT_TAIL_LINES = 40
TARGET_RUNTIME_OUTPUT_TAIL_CHARS = 16 * 1024
SUPPORT_DIR_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


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
    source_texts = []
    for path in files:
        if path.suffix in SOURCE_SUFFIXES:
            source_texts.append(_source_text(path))
    joined = "\n".join(source_texts)
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
    has_op_dev_path = False
    for path in relative_paths:
        for part in ("op_host/", "op_kernel/"):
            if part in path:
                has_op_dev_path = True
                break
        if has_op_dev_path:
            break
    if "OpDef" in raw_text or has_op_dev_path:
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
    ascendc_highlevel_root = (
        cann_path / "aarch64-linux" / "ascendc" / "include" / "highlevel_api"
    )
    candidates = [
        asc_root / "include",
        asc_root / "include" / "basic_api",
        ascendc_highlevel_root,
        asc_root / "impl",
        asc_root / "impl" / "basic_api",
        asc_root,
    ]
    return [str(path) for path in candidates if path.is_dir()]


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
        manifest_value = _require_mapping_value(
            manifest, manifest_key, "sk conversion manifest"
        )
        evidence = _sk_file_evidence(manifest_value)
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
        manifest_value = _require_mapping_value(
            manifest, manifest_key, "sk conversion manifest"
        )
        evidence = _sk_file_evidence(manifest_value)
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


def _split_sk_param_sources(param_text: str) -> list[str] | None:
    stripped = param_text.strip()
    if not stripped or stripped == "void":
        return []
    if "(" in stripped or ")" in stripped:
        return None
    return [item.strip() for item in stripped.split(",") if item.strip()]


def _parse_sk_scaffold_params(param_text: str) -> list[dict[str, str]] | None:
    sources = _split_sk_param_sources(param_text)
    if sources is None:
        return None
    params: list[dict[str, str]] = []
    for source in sources:
        if "=" in source or "[" in source or "]" in source:
            return None
        parts = source.rsplit(None, 1)
        if len(parts) != 2:
            return None
        type_part, name_part = parts
        pointer_prefix = ""
        while name_part.startswith(("*", "&")):
            pointer_prefix += name_part[0]
            name_part = name_part[1:]
        if pointer_prefix:
            type_part = f"{type_part} {pointer_prefix}"
        if not re.fullmatch(r"[A-Za-z_]\w*", name_part):
            return None
        base_type = (
            type_part.replace("const", "").replace("*", "").replace("&", "").strip()
        )
        alignment = (
            "alignas(4)"
            if base_type in {"int8_t", "uint8_t", "int16_t", "uint16_t"}
            else ""
        )
        params.append(
            {
                "type": type_part,
                "name": name_part,
                "field": name_part,
                "alignment": alignment,
                "source": source,
            }
        )
    return params


def _find_matching_brace(text: str, open_brace_index: int) -> int | None:
    depth = 0
    for index in range(open_brace_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _find_kernel_signature_for_entry(
    asset_base_dir: Path, entry: dict[str, str]
) -> dict[str, Any] | None:
    source_path = asset_base_dir / entry["file"]
    if not source_path.exists() or not source_path.is_file():
        return None
    text = _source_text(source_path)
    pattern = re.compile(
        r'(?:extern\s+"C"\s+)?(?P<qualifiers>(?:__global__|__spk__|__sk__)[\w\s_()*,:&<>]*?)\s+void\s+'
        r"(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^()]*)\)\s*\{",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        if match.group("name") != entry["name"]:
            continue
        open_brace = match.end() - 1
        close_brace = _find_matching_brace(text, open_brace)
        if close_brace is None:
            return None
        signature_start = match.start()
        body_start = open_brace + 1
        signature = text[signature_start:open_brace].strip()
        body = text[body_start:close_brace]
        return {
            "signature": signature,
            "qualifiers": match.group("qualifiers")
            .replace("__global__", "__sk__", 1)
            .strip(),
            "params": match.group("params"),
            "body": body,
        }
    return None


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


def _require_mapping_value(mapping: dict[Any, Any], key: Any, label: str) -> Any:
    value = mapping.get(key)
    if value is None:
        raise CliUsageError(f"{label} missing required key: {key!r}")
    return value


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


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CliUsageError(f"{label} must be an object")
    return value


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


def _validate_manifest_path_list(
    value: Any, list_label: str, item_label: str
) -> list[str]:
    paths = []
    for item in _require_list(value, list_label):
        paths.append(_validate_manifest_path(item, item_label))
    return paths


def _resolve_string_path_list(
    value: Any, list_label: str, item_label: str
) -> list[str]:
    paths = []
    for item in _require_list(value, list_label):
        paths.append(str(Path(_require_string(item, item_label)).resolve()))
    return paths


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
    check = _require_mapping_value(checks, name, "sk build validation checks")
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
    copied_evidence = []
    for item in source_scaffold["copied_source_files"]:
        copied_evidence.append(item["scaffold_path"])
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
        configure = _require_mapping_value(
            checks, "cmake_configure", "sk build validation checks"
        )
        build = _require_mapping_value(
            checks, "cmake_build", "sk build validation checks"
        )
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

    for check in checks.values():
        if check["status"] == "failed":
            raise CliUsageError("sk build validation checks semantics mismatch")
    blocked_indexes = []
    for index, name in enumerate(SK_BUILD_VALIDATION_CHECK_NAMES):
        check = _require_mapping_value(checks, name, "sk build validation checks")
        if check["status"] == "blocked":
            blocked_indexes.append(index)
    if not blocked_indexes:
        raise CliUsageError("sk build validation checks semantics mismatch")
    first_blocked_index = blocked_indexes[0]
    first_blocked_name = SK_BUILD_VALIDATION_CHECK_NAMES[first_blocked_index]
    first_blocked_check = _require_mapping_value(
        checks, first_blocked_name, "sk build validation checks"
    )
    blocked_reason = first_blocked_check["reason"]
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
        check = _require_mapping_value(checks, name, "sk build validation checks")
        if check["status"] != "blocked" or check["reason"] != blocked_reason:
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


def _sk_runtime_input_spec_check(
    name: str, status: str, reason: str, evidence: list[str] | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "evidence": evidence or [],
    }


def _sk_entry_runtime_parameters(
    base_dir: Path, entry: dict[str, str]
) -> list[dict[str, Any]] | None:
    parsed = _find_kernel_signature_for_entry(base_dir, entry)
    if parsed is None:
        return None
    raw_params = _parse_sk_scaffold_params(parsed["params"])
    if not raw_params:
        return None
    seen: set[str] = set()
    params: list[dict[str, Any]] = []
    for index, param in enumerate(raw_params):
        name = param["name"]
        if name in seen:
            return None
        seen.add(name)
        params.append(
            {
                "index": index,
                "name": name,
                "type": param["type"],
                "source": param["source"],
            }
        )
    return params


def _sk_runtime_input_specs(
    base_dir: Path,
    kernel_entries: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    input_specs: list[dict[str, Any]] = []
    unresolved_inputs: list[dict[str, Any]] = []
    parameters_ready = True
    for entry in kernel_entries:
        spec_id = f"entry:{entry['name']}:default"
        parameters = _sk_entry_runtime_parameters(base_dir, entry)
        if parameters is None:
            parameters_ready = False
            parameters = []
            reasons = ["runtime_parameters_not_inferred"]
        else:
            reasons = ["runtime_input_values_not_declared", "golden_output_not_defined"]
        input_specs.append(
            {
                "id": spec_id,
                "entry_name": entry["name"],
                "source_file": entry["file"],
                "parameters": parameters,
                "requires_user_input": True,
            }
        )
        unresolved_inputs.append(
            {
                "id": spec_id,
                "entry_name": entry["name"],
                "source_file": entry["file"],
                "reasons": reasons,
            }
        )
    return input_specs, unresolved_inputs, parameters_ready


class RuntimeInputSpecManifestInput(NamedTuple):
    output_dir: Path
    status: str
    kernel_entries: list[dict[str, str]]
    input_specs: list[dict[str, Any]]
    unresolved_inputs: list[dict[str, Any]]
    checks: dict[str, dict[str, Any]]
    supported_next_actions: list[str]


def _sk_runtime_input_spec_manifest(
    request: RuntimeInputSpecManifestInput,
) -> dict[str, Any]:
    return {
        "status": request.status,
        "analysis_output_dir": str(request.output_dir.resolve()),
        "source_scaffold_manifest_path": "operator-sk-source-scaffold.json",
        "build_validation_path": "operator-sk-build-validation.json",
        "source_version_manifest_path": "operator-sk-source-version.json",
        "source_version_markdown_path": "operator-sk-source-version.md",
        "source_version_dir": SK_SOURCE_VERSION_DIR,
        "source_version_validation_path": SK_SOURCE_VERSION_VALIDATION_PATH,
        "runtime_input_spec_path": "operator-sk-runtime-input-spec.json",
        "kernel_entries": request.kernel_entries,
        "input_specs": request.input_specs,
        "unresolved_inputs": request.unresolved_inputs,
        "checks": [request.checks[name] for name in SK_RUNTIME_INPUT_SPEC_CHECK_NAMES],
        "supported_next_actions": request.supported_next_actions,
        "execution_boundary": SK_RUNTIME_INPUT_SPEC_BOUNDARY,
    }


def _sk_runtime_boundary_checks(checks: dict[str, dict[str, Any]]) -> None:
    checks["runtime_execution_boundary_open"] = _sk_runtime_input_spec_check(
        "runtime_execution_boundary_open",
        "open",
        "runtime_not_executed",
    )
    checks["correctness_boundary_open"] = _sk_runtime_input_spec_check(
        "correctness_boundary_open",
        "open",
        "operator_correctness_not_validated",
    )


class RuntimeInputSpecSummaryInput(NamedTuple):
    output_dir: Path
    analysis: dict[str, Any]
    source_scaffold: dict[str, Any]
    build_validation: dict[str, Any]
    source_version: dict[str, Any]
    source_version_validation: dict[str, Any]


def _summarize_sk_runtime_input_spec(
    request: RuntimeInputSpecSummaryInput,
) -> dict[str, Any]:
    output_dir = request.output_dir
    analysis = request.analysis
    source_scaffold = request.source_scaffold
    build_validation = request.build_validation
    _ = request.source_version, request.source_version_validation
    base_dir = Path(
        _require_string(analysis["base_dir"], "sk conversion analysis base_dir")
    )
    kernel_entries = [
        {"name": entry["name"], "file": entry["file"]}
        for entry in source_scaffold["kernel_entries"]
    ]
    checks: dict[str, dict[str, Any]] = {
        "sk_build_validation_current": _sk_runtime_input_spec_check(
            "sk_build_validation_current",
            "passed",
            "sk_build_validation_current",
            ["operator-sk-build-validation.json"],
        )
    }

    if build_validation["status"] != "passed":
        checks["sk_build_validation_passed"] = _sk_runtime_input_spec_check(
            "sk_build_validation_passed",
            "blocked",
            "sk_build_validation_not_passed",
            ["operator-sk-build-validation.json"],
        )
        for name in (
            "kernel_entries_declared",
            "copied_sources_current",
            "runtime_input_specs_defined",
        ):
            checks[name] = _sk_runtime_input_spec_check(
                name, "blocked", "sk_build_validation_not_passed"
            )
        _sk_runtime_boundary_checks(checks)
        return _sk_runtime_input_spec_manifest(
            RuntimeInputSpecManifestInput(
                output_dir,
                "blocked",
                kernel_entries,
                [],
                [],
                checks,
                ["run_sk_build_validation"],
            )
        )

    checks["sk_build_validation_passed"] = _sk_runtime_input_spec_check(
        "sk_build_validation_passed",
        "passed",
        "sk_build_validation_passed",
        ["operator-sk-build-validation.json"],
    )
    if not kernel_entries:
        for name in (
            "kernel_entries_declared",
            "copied_sources_current",
            "runtime_input_specs_defined",
        ):
            checks[name] = _sk_runtime_input_spec_check(
                name, "blocked", "kernel_entries_missing"
            )
        _sk_runtime_boundary_checks(checks)
        return _sk_runtime_input_spec_manifest(
            RuntimeInputSpecManifestInput(
                output_dir, "blocked", [], [], [], checks, ["analyze_sk_conversion"]
            )
        )

    input_specs, unresolved_inputs, parameters_ready = _sk_runtime_input_specs(
        base_dir, kernel_entries
    )
    checks["kernel_entries_declared"] = _sk_runtime_input_spec_check(
        "kernel_entries_declared",
        "passed",
        "kernel_entries_declared",
        [entry["name"] for entry in kernel_entries],
    )
    checks["copied_sources_current"] = _sk_runtime_input_spec_check(
        "copied_sources_current",
        "passed",
        "copied_sources_current",
        [item["scaffold_path"] for item in source_scaffold["copied_source_files"]],
    )
    if not parameters_ready:
        checks["runtime_input_specs_defined"] = _sk_runtime_input_spec_check(
            "runtime_input_specs_defined",
            "blocked",
            "runtime_parameters_not_inferred",
            [
                item["id"]
                for item in unresolved_inputs
                if "runtime_parameters_not_inferred" in item["reasons"]
            ],
        )
        _sk_runtime_boundary_checks(checks)
        return _sk_runtime_input_spec_manifest(
            RuntimeInputSpecManifestInput(
                output_dir,
                "blocked",
                kernel_entries,
                input_specs,
                unresolved_inputs,
                checks,
                ["identify_kernel_entry_signature"],
            )
        )
    checks["runtime_input_specs_defined"] = _sk_runtime_input_spec_check(
        "runtime_input_specs_defined",
        "passed",
        "runtime_input_specs_defined",
        [item["id"] for item in input_specs],
    )
    _sk_runtime_boundary_checks(checks)
    return _sk_runtime_input_spec_manifest(
        RuntimeInputSpecManifestInput(
            output_dir,
            "defined",
            kernel_entries,
            input_specs,
            unresolved_inputs,
            checks,
            ["provide_sk_runtime_input_values"],
        )
    )


def _load_current_sk_runtime_input_spec(output_dir: Path) -> dict[str, Any]:
    analysis = _load_current_sk_conversion_analysis(output_dir)
    source_scaffold = _load_current_sk_source_scaffold(output_dir, analysis)
    build_validation = _load_current_sk_build_validation(output_dir, source_scaffold)
    source_version = _load_current_sk_source_version(
        output_dir, source_scaffold, build_validation
    )
    source_version_validation = _load_current_sk_source_version_validation(
        output_dir, source_version
    )
    runtime_spec = _load_json_artifact(
        output_dir / "operator-sk-runtime-input-spec.json",
        "operator-sk-runtime-input-spec.json",
    )
    runtime_spec = _require_exact_keys(
        runtime_spec,
        {
            "status",
            "analysis_output_dir",
            "source_scaffold_manifest_path",
            "build_validation_path",
            "source_version_manifest_path",
            "source_version_markdown_path",
            "source_version_dir",
            "source_version_validation_path",
            "runtime_input_spec_path",
            "kernel_entries",
            "input_specs",
            "unresolved_inputs",
            "checks",
            "supported_next_actions",
            "execution_boundary",
        },
        "sk runtime input spec",
    )
    status = _require_string(runtime_spec["status"], "sk runtime input spec status")
    if status not in {"defined", "blocked"}:
        raise CliUsageError(f"invalid sk runtime input spec status: {status}")
    if status != "defined":
        raise CliUsageError("sk runtime input spec not defined")
    expected = _summarize_sk_runtime_input_spec(
        RuntimeInputSpecSummaryInput(
            output_dir,
            analysis,
            source_scaffold,
            build_validation,
            source_version,
            source_version_validation,
        )
    )
    if runtime_spec != expected:
        raise CliUsageError("sk runtime input spec mismatch")
    return runtime_spec


def _sk_runtime_input_values_check(
    name: str, status: str, reason: str, evidence: list[str] | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "evidence": evidence or [],
    }


def _normalize_sk_input_value_spec(spec: Any) -> list[dict[str, Any]]:
    payload = _require_exact_keys(spec, {"input_values"}, "sk input value spec")
    raw_values = _require_list(
        payload["input_values"], "sk input value spec input_values"
    )
    normalized: list[dict[str, Any]] = []
    seen_input_ids: set[str] = set()
    for raw_item in raw_values:
        item = _require_exact_keys(
            raw_item, {"input_set_id", "parameter_values"}, "sk input value"
        )
        input_set_id = _require_string(
            item["input_set_id"], "sk input value input_set_id"
        )
        if input_set_id in seen_input_ids:
            raise CliUsageError("duplicate sk input value spec input_set_id")
        seen_input_ids.add(input_set_id)
        raw_parameter_values = _require_list(
            item["parameter_values"], "sk input value parameter_values"
        )
        parameter_values: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for raw_parameter in raw_parameter_values:
            parameter = _require_exact_keys(
                raw_parameter,
                {"name", "shape", "dtype", "layout", "value_source"},
                "sk parameter value",
            )
            name = _require_string(parameter["name"], "sk parameter value name")
            if not name:
                raise CliUsageError("sk parameter value name must be non-empty")
            if name in seen_names:
                raise CliUsageError("duplicate sk parameter value name")
            seen_names.add(name)
            shape = _require_list(parameter["shape"], "sk parameter value shape")
            for dim in shape:
                if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
                    raise CliUsageError(
                        "sk parameter value shape dimension must be a non-negative integer"
                    )
            dtype = _require_string(parameter["dtype"], "sk parameter value dtype")
            if not dtype:
                raise CliUsageError("sk parameter value dtype must be non-empty")
            layout = _require_string(parameter["layout"], "sk parameter value layout")
            if not layout:
                raise CliUsageError("sk parameter value layout must be non-empty")
            value_source = _require_exact_keys(
                parameter["value_source"],
                {"kind", "value"},
                "sk parameter value source",
            )
            kind = _require_string(
                value_source["kind"], "sk parameter value source kind"
            )
            if kind != "inline_json":
                raise CliUsageError("unsupported sk parameter value source kind")
            if value_source["value"] is None:
                raise CliUsageError("sk parameter value source value must not be null")
            _require_json_finite(
                value_source["value"], "sk parameter value source value"
            )
            parameter_values.append(
                {
                    "name": name,
                    "shape": shape,
                    "dtype": dtype,
                    "layout": layout,
                    "value_source": {"kind": kind, "value": value_source["value"]},
                }
            )
        normalized.append(
            {"input_set_id": input_set_id, "parameter_values": parameter_values}
        )
    return normalized


def _sk_artifact_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RuntimeInputValuesManifestInput(NamedTuple):
    output_dir: Path
    runtime_spec_sha256: str
    spec_path: Path
    spec_bytes: bytes
    input_values: list[dict[str, Any]]
    check_results: dict[str, dict[str, Any]]


def _sk_runtime_input_values_manifest(
    request: RuntimeInputValuesManifestInput,
) -> dict[str, Any]:
    runtime_input_value_checks = []
    for name in SK_RUNTIME_INPUT_VALUES_CHECK_NAMES:
        runtime_input_value_checks.append(request.check_results[name])
    return {
        "status": "defined",
        "analysis_output_dir": str(request.output_dir.resolve()),
        "runtime_input_spec_path": "operator-sk-runtime-input-spec.json",
        "runtime_input_spec_sha256": request.runtime_spec_sha256,
        "runtime_input_values_path": "operator-sk-runtime-input-values.json",
        "input_value_spec_path": str(request.spec_path.resolve()),
        "input_value_spec_sha256": hashlib.sha256(request.spec_bytes).hexdigest(),
        "input_values": request.input_values,
        "unresolved_inputs": [],
        "checks": runtime_input_value_checks,
        "supported_next_actions": ["collect_correctness_oracle_spec"],
        "execution_boundary": SK_RUNTIME_INPUT_VALUES_BOUNDARY,
    }


def _summarize_sk_runtime_input_values(
    output_dir: Path,
    runtime_spec: dict[str, Any],
    spec_path: Path,
    spec_bytes: bytes,
    spec: Any,
) -> dict[str, Any]:
    spec_values = _normalize_sk_input_value_spec(spec)
    by_input_set_id = {item["input_set_id"]: item for item in spec_values}
    expected_ids = [item["id"] for item in runtime_spec["input_specs"]]
    if set(by_input_set_id) != set(expected_ids):
        raise CliUsageError("sk input value set mismatch")
    input_values: list[dict[str, Any]] = []
    for input_spec in runtime_spec["input_specs"]:
        input_spec_id = input_spec["id"]
        spec_item = _require_mapping_value(
            by_input_set_id, input_spec_id, "sk input value spec"
        )
        by_parameter_name = {}
        for item in spec_item["parameter_values"]:
            by_parameter_name[item["name"]] = item
        expected_names = [item["name"] for item in input_spec["parameters"]]
        if set(by_parameter_name) != set(expected_names):
            raise CliUsageError("sk parameter value set mismatch")
        parameter_values: list[dict[str, Any]] = []
        for parameter in input_spec["parameters"]:
            parameter_name = parameter["name"]
            value = _require_mapping_value(
                by_parameter_name, parameter_name, "sk parameter value spec"
            )
            parameter_values.append(
                {
                    "index": parameter["index"],
                    "name": parameter["name"],
                    "type": parameter["type"],
                    "source": parameter["source"],
                    "shape": value["shape"],
                    "dtype": value["dtype"],
                    "layout": value["layout"],
                    "value_source": value["value_source"],
                }
            )
        input_values.append(
            {
                "input_set_id": input_spec_id,
                "entry_name": input_spec["entry_name"],
                "source_file": input_spec["source_file"],
                "parameter_values": parameter_values,
                "requires_user_input": False,
            }
        )
    runtime_spec_sha256 = _sk_artifact_sha256(
        output_dir / "operator-sk-runtime-input-spec.json"
    )
    check_results: dict[str, dict[str, Any]] = {
        "sk_runtime_input_spec_current": _sk_runtime_input_values_check(
            "sk_runtime_input_spec_current",
            "passed",
            "sk_runtime_input_spec_defined_current",
            ["operator-sk-runtime-input-spec.json"],
        ),
        "input_value_spec_schema": _sk_runtime_input_values_check(
            "input_value_spec_schema",
            "passed",
            "sk_input_value_spec_valid",
            [str(spec_path.resolve())],
        ),
        "input_value_sets_complete": _sk_runtime_input_values_check(
            "input_value_sets_complete",
            "passed",
            "sk_input_value_sets_complete",
            expected_ids,
        ),
        "runtime_input_values_declared": _sk_runtime_input_values_check(
            "runtime_input_values_declared",
            "passed",
            "sk_runtime_input_values_declared",
            expected_ids,
        ),
        "runtime_execution_boundary_open": _sk_runtime_input_values_check(
            "runtime_execution_boundary_open",
            "open",
            "runtime_not_executed",
        ),
        "correctness_boundary_open": _sk_runtime_input_values_check(
            "correctness_boundary_open",
            "open",
            "operator_correctness_not_validated",
        ),
    }
    return _sk_runtime_input_values_manifest(
        RuntimeInputValuesManifestInput(
            output_dir,
            runtime_spec_sha256,
            spec_path,
            spec_bytes,
            input_values,
            check_results,
        )
    )


def _load_current_sk_runtime_input_values(
    output_dir: Path, runtime_spec: dict[str, Any]
) -> dict[str, Any]:
    values = _load_json_artifact(
        output_dir / "operator-sk-runtime-input-values.json",
        "operator-sk-runtime-input-values.json",
    )
    values = _require_exact_keys(
        values,
        {
            "status",
            "analysis_output_dir",
            "runtime_input_spec_path",
            "runtime_input_spec_sha256",
            "runtime_input_values_path",
            "input_value_spec_path",
            "input_value_spec_sha256",
            "input_values",
            "unresolved_inputs",
            "checks",
            "supported_next_actions",
            "execution_boundary",
        },
        "sk runtime input values",
    )
    if _require_string(values["status"], "sk runtime input values status") != "defined":
        raise CliUsageError("sk runtime input values not defined")
    if _require_string(
        values["analysis_output_dir"], "sk runtime input values analysis_output_dir"
    ) != str(output_dir.resolve()):
        raise CliUsageError("sk runtime input values mismatch")
    if (
        _require_string(
            values["runtime_input_spec_path"],
            "sk runtime input values runtime_input_spec_path",
        )
        != "operator-sk-runtime-input-spec.json"
    ):
        raise CliUsageError("sk runtime input values mismatch")
    if (
        _require_string(
            values["runtime_input_values_path"],
            "sk runtime input values runtime_input_values_path",
        )
        != "operator-sk-runtime-input-values.json"
    ):
        raise CliUsageError("sk runtime input values mismatch")
    if _require_string(
        values["runtime_input_spec_sha256"],
        "sk runtime input values runtime_input_spec_sha256",
    ) != _sk_artifact_sha256(output_dir / "operator-sk-runtime-input-spec.json"):
        raise CliUsageError("sk runtime input values mismatch")
    spec_path_text = _require_string(
        values["input_value_spec_path"], "sk runtime input values input_value_spec_path"
    )
    spec_path = Path(spec_path_text)
    if not spec_path.is_absolute():
        raise CliUsageError("sk runtime input values mismatch")
    spec_path, spec_bytes, spec = _read_input_value_spec(spec_path)
    if (
        _require_string(
            values["input_value_spec_sha256"],
            "sk runtime input values input_value_spec_sha256",
        )
        != hashlib.sha256(spec_bytes).hexdigest()
    ):
        raise CliUsageError("sk runtime input values mismatch")
    try:
        expected = _summarize_sk_runtime_input_values(
            output_dir, runtime_spec, spec_path, spec_bytes, spec
        )
    except CliUsageError as exc:
        raise CliUsageError("sk runtime input values mismatch") from exc
    if values != expected:
        raise CliUsageError("sk runtime input values mismatch")
    return values


def _sk_correctness_oracle_spec_check(
    name: str, status: str, reason: str, evidence: list[str] | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "evidence": evidence or [],
    }


class CorrectnessOracleSpecManifestInput(NamedTuple):
    output_dir: Path
    runtime_values_sha256: str
    spec_path: Path
    spec_bytes: bytes
    oracle_specs: list[dict[str, Any]]
    check_results: dict[str, dict[str, Any]]


def _sk_correctness_oracle_spec_manifest(
    request: CorrectnessOracleSpecManifestInput,
) -> dict[str, Any]:
    return {
        "status": "defined",
        "analysis_output_dir": str(request.output_dir.resolve()),
        "runtime_input_spec_path": "operator-sk-runtime-input-spec.json",
        "runtime_input_values_path": "operator-sk-runtime-input-values.json",
        "runtime_input_values_sha256": request.runtime_values_sha256,
        "correctness_oracle_spec_path": str(request.spec_path.resolve()),
        "correctness_oracle_spec_sha256": hashlib.sha256(
            request.spec_bytes
        ).hexdigest(),
        "oracle_specs": request.oracle_specs,
        "checks": [
            request.check_results[name]
            for name in SK_CORRECTNESS_ORACLE_SPEC_CHECK_NAMES
        ],
        "supported_next_actions": ["run_sk_target_runtime_validation"],
        "execution_boundary": SK_CORRECTNESS_ORACLE_SPEC_BOUNDARY,
    }


class CorrectnessOracleSpecSummaryInput(NamedTuple):
    output_dir: Path
    runtime_spec: dict[str, Any]
    runtime_values: dict[str, Any]
    spec_path: Path
    spec_bytes: bytes
    spec: Any


def _summarize_sk_correctness_oracle_spec(
    request: CorrectnessOracleSpecSummaryInput,
) -> dict[str, Any]:
    output_dir = request.output_dir
    runtime_spec = request.runtime_spec
    runtime_values = request.runtime_values
    spec_path = request.spec_path
    spec_bytes = request.spec_bytes
    spec = request.spec
    raw_specs = _normalize_correctness_oracle_spec(spec)
    by_input_set_id = {item["input_set_id"]: item for item in raw_specs}
    expected_ids = [item["input_set_id"] for item in runtime_values["input_values"]]
    if set(by_input_set_id) != set(expected_ids):
        raise CliUsageError("sk correctness oracle set mismatch")
    input_specs_by_id = {item["id"]: item for item in runtime_spec["input_specs"]}
    oracle_specs: list[dict[str, Any]] = []
    for input_value in runtime_values["input_values"]:
        input_set_id = input_value["input_set_id"]
        input_spec = _require_mapping_value(
            input_specs_by_id, input_set_id, "sk runtime input spec"
        )
        spec_item = _require_mapping_value(
            by_input_set_id, input_set_id, "sk correctness oracle spec"
        )
        oracle_specs.append(
            {
                "oracle_set_id": f"oracle:{input_set_id}",
                "input_set_id": input_set_id,
                "entry_name": input_spec["entry_name"],
                "source_file": input_spec["source_file"],
                "expected_output": spec_item["expected_output"],
                "comparator": spec_item["comparator"],
                "tolerance": spec_item["tolerance"],
                "requires_user_input": False,
            }
        )
    runtime_values_sha256 = _sk_artifact_sha256(
        output_dir / "operator-sk-runtime-input-values.json"
    )
    check_results: dict[str, dict[str, Any]] = {
        "sk_runtime_input_spec_current": _sk_correctness_oracle_spec_check(
            "sk_runtime_input_spec_current",
            "passed",
            "sk_runtime_input_spec_defined_current",
            ["operator-sk-runtime-input-spec.json"],
        ),
        "sk_runtime_input_values_current": _sk_correctness_oracle_spec_check(
            "sk_runtime_input_values_current",
            "passed",
            "sk_runtime_input_values_defined_current",
            ["operator-sk-runtime-input-values.json"],
        ),
        "oracle_spec_schema": _sk_correctness_oracle_spec_check(
            "oracle_spec_schema",
            "passed",
            "oracle_spec_valid",
            [str(spec_path.resolve())],
        ),
        "oracle_sets_complete": _sk_correctness_oracle_spec_check(
            "oracle_sets_complete",
            "passed",
            "oracle_sets_complete",
            expected_ids,
        ),
        "golden_outputs_declared": _sk_correctness_oracle_spec_check(
            "golden_outputs_declared",
            "passed",
            "golden_outputs_declared",
            expected_ids,
        ),
        "correctness_comparator_declared": _sk_correctness_oracle_spec_check(
            "correctness_comparator_declared",
            "passed",
            "correctness_comparator_declared",
            expected_ids,
        ),
        "runtime_execution_boundary_open": _sk_correctness_oracle_spec_check(
            "runtime_execution_boundary_open",
            "open",
            "runtime_not_executed",
        ),
        "correctness_boundary_open": _sk_correctness_oracle_spec_check(
            "correctness_boundary_open",
            "open",
            "operator_correctness_not_validated",
        ),
    }
    return _sk_correctness_oracle_spec_manifest(
        CorrectnessOracleSpecManifestInput(
            output_dir,
            runtime_values_sha256,
            spec_path,
            spec_bytes,
            oracle_specs,
            check_results,
        )
    )


def _load_current_sk_correctness_oracle_spec(
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_spec = _load_current_sk_runtime_input_spec(output_dir)
    runtime_values = _load_current_sk_runtime_input_values(output_dir, runtime_spec)
    oracle_spec = _load_json_artifact(
        output_dir / "operator-sk-correctness-oracle-spec.json",
        "operator-sk-correctness-oracle-spec.json",
    )
    oracle_spec = _require_exact_keys(
        oracle_spec,
        {
            "status",
            "analysis_output_dir",
            "runtime_input_spec_path",
            "runtime_input_values_path",
            "runtime_input_values_sha256",
            "correctness_oracle_spec_path",
            "correctness_oracle_spec_sha256",
            "oracle_specs",
            "checks",
            "supported_next_actions",
            "execution_boundary",
        },
        "sk correctness oracle spec",
    )
    if (
        _require_string(oracle_spec["status"], "sk correctness oracle spec status")
        != "defined"
    ):
        raise CliUsageError("sk correctness oracle spec not defined")
    if (
        _require_string(
            oracle_spec["runtime_input_values_path"],
            "sk correctness oracle spec runtime_input_values_path",
        )
        != "operator-sk-runtime-input-values.json"
    ):
        raise CliUsageError("sk correctness oracle spec mismatch")
    if _require_string(
        oracle_spec["runtime_input_values_sha256"],
        "sk correctness oracle spec runtime_input_values_sha256",
    ) != _sk_artifact_sha256(output_dir / "operator-sk-runtime-input-values.json"):
        raise CliUsageError("sk correctness oracle spec mismatch")
    spec_path_text = _require_string(
        oracle_spec["correctness_oracle_spec_path"],
        "sk correctness oracle spec correctness_oracle_spec_path",
    )
    spec_path = Path(spec_path_text)
    if not spec_path.is_absolute():
        raise CliUsageError("sk correctness oracle spec mismatch")
    spec_path, spec_bytes, spec = _read_correctness_oracle_spec(spec_path)
    try:
        expected = _summarize_sk_correctness_oracle_spec(
            CorrectnessOracleSpecSummaryInput(
                output_dir, runtime_spec, runtime_values, spec_path, spec_bytes, spec
            )
        )
    except CliUsageError as exc:
        raise CliUsageError("sk correctness oracle spec mismatch") from exc
    if oracle_spec != expected:
        raise CliUsageError("sk correctness oracle spec mismatch")
    return oracle_spec, runtime_spec


class TargetRuntimeResultManifestInput(NamedTuple):
    output_dir: Path
    status: str
    runtime_values_sha256: str
    input_binding: dict[str, Any]
    spec_path: Path
    spec_bytes: bytes
    check_results: dict[str, dict[str, Any]]
    commands: list[dict[str, Any]]
    supported_next_actions: list[str]


def _sk_target_runtime_result_manifest(
    request: TargetRuntimeResultManifestInput,
) -> dict[str, Any]:
    return {
        "status": request.status,
        "analysis_output_dir": str(request.output_dir.resolve()),
        "correctness_oracle_spec_path": "operator-sk-correctness-oracle-spec.json",
        "runtime_input_values_path": "operator-sk-runtime-input-values.json",
        "runtime_input_values_sha256": request.runtime_values_sha256,
        "target_runtime_validation_path": "operator-sk-target-runtime-validation.json",
        "runtime_command_spec_path": str(request.spec_path.resolve()),
        "runtime_command_spec_sha256": hashlib.sha256(request.spec_bytes).hexdigest(),
        "input_binding": request.input_binding,
        "commands": request.commands,
        "checks": [
            request.check_results[name]
            for name in SK_TARGET_RUNTIME_VALIDATION_CHECK_NAMES
        ],
        "supported_next_actions": request.supported_next_actions,
        "execution_boundary": SK_TARGET_RUNTIME_VALIDATION_BOUNDARY,
    }


def _sk_target_runtime_boundary_checks(
    check_results: dict[str, dict[str, Any]],
) -> None:
    check_results["output_comparison_boundary_open"] = _target_runtime_check(
        "output_comparison_boundary_open",
        "open",
        "output_comparison_not_executed",
    )
    check_results["correctness_boundary_open"] = _target_runtime_check(
        "correctness_boundary_open",
        "open",
        "operator_correctness_not_validated",
    )
    check_results["handoff_boundary_open"] = _target_runtime_check(
        "handoff_boundary_open",
        "open",
        "handoff_not_completed",
    )


def _load_current_sk_target_runtime_validation(
    output_dir: Path, oracle_spec: dict[str, Any]
) -> dict[str, Any]:
    try:
        target = _load_json_artifact(
            output_dir / "operator-sk-target-runtime-validation.json",
            "operator-sk-target-runtime-validation.json",
        )
        target = _require_exact_keys(
            target,
            {
                "status",
                "analysis_output_dir",
                "correctness_oracle_spec_path",
                "runtime_input_values_path",
                "runtime_input_values_sha256",
                "target_runtime_validation_path",
                "runtime_command_spec_path",
                "runtime_command_spec_sha256",
                "input_binding",
                "commands",
                "checks",
                "supported_next_actions",
                "execution_boundary",
            },
            "sk target runtime validation",
        )
        status = _require_string(
            target["status"], "sk target runtime validation status"
        )
        if status not in {"passed", "failed"}:
            raise CliUsageError("sk target runtime validation status is invalid")
        if _require_string(
            target["analysis_output_dir"],
            "sk target runtime validation analysis_output_dir",
        ) != str(output_dir.resolve()):
            raise CliUsageError(
                "sk target runtime validation analysis_output_dir mismatch"
            )
        if (
            _require_string(
                target["correctness_oracle_spec_path"],
                "sk target runtime validation correctness_oracle_spec_path",
            )
            != "operator-sk-correctness-oracle-spec.json"
        ):
            raise CliUsageError(
                "sk target runtime validation correctness_oracle_spec_path mismatch"
            )
        if (
            _require_string(
                target["runtime_input_values_path"],
                "sk target runtime validation runtime_input_values_path",
            )
            != "operator-sk-runtime-input-values.json"
        ):
            raise CliUsageError(
                "sk target runtime validation runtime_input_values_path mismatch"
            )
        runtime_values_sha256 = _sk_artifact_sha256(
            output_dir / "operator-sk-runtime-input-values.json"
        )
        if (
            _require_string(
                target["runtime_input_values_sha256"],
                "sk target runtime validation runtime_input_values_sha256",
            )
            != runtime_values_sha256
        ):
            raise CliUsageError(
                "sk target runtime validation runtime_input_values_sha256 mismatch"
            )
        if (
            _require_string(
                target["target_runtime_validation_path"],
                "sk target runtime validation target_runtime_validation_path",
            )
            != "operator-sk-target-runtime-validation.json"
        ):
            raise CliUsageError(
                "sk target runtime validation target_runtime_validation_path mismatch"
            )
        if (
            _require_string(oracle_spec["status"], "sk correctness oracle spec status")
            != "defined"
        ):
            raise CliUsageError("sk target runtime validation mismatch")
        spec_path_text = _require_string(
            target["runtime_command_spec_path"],
            "sk target runtime validation runtime_command_spec_path",
        )
        spec_path = Path(spec_path_text)
        if not spec_path.is_absolute():
            raise CliUsageError(
                "sk target runtime validation runtime_command_spec_path must be absolute"
            )
        spec_path, spec_bytes, spec = _read_runtime_command_spec(spec_path)
        if (
            _require_string(
                target["runtime_command_spec_sha256"],
                "sk target runtime validation runtime_command_spec_sha256",
            )
            != hashlib.sha256(spec_bytes).hexdigest()
        ):
            raise CliUsageError(
                "sk target runtime validation runtime_command_spec_sha256 mismatch"
            )
        command = _normalize_sk_runtime_command_spec(spec)
        cwd = _safe_target_runtime_cwd(output_dir, command["cwd"])
        input_binding = _sk_runtime_values_manifest_binding(output_dir, cwd, command)
        if target["input_binding"] != input_binding:
            raise CliUsageError("sk target runtime validation input_binding mismatch")
        effective_env, env_summary = _target_runtime_effective_env(command["env"])
        executable = _resolve_target_runtime_executable(
            output_dir, command["argv"][0], effective_env
        )
        commands = _require_list(
            target["commands"], "sk target runtime validation commands"
        )
        if len(commands) != 1:
            raise CliUsageError("sk target runtime validation commands mismatch")
        command_result = _require_exact_keys(
            commands[0],
            {
                "name",
                "argv",
                "cwd",
                "timeout_seconds",
                "returncode",
                "timed_out",
                "stdout_tail",
                "stdout_tail_truncated",
                "stderr_tail",
                "stderr_tail_truncated",
                "env",
            },
            "sk target runtime validation command",
        )
        if (
            _require_string(
                command_result["name"], "sk target runtime validation command name"
            )
            != "target_runtime_validation"
        ):
            raise CliUsageError("sk target runtime validation command name mismatch")
        expected_argv = [executable, *command["argv"][1:]]
        if command_result["argv"] != expected_argv:
            raise CliUsageError("sk target runtime validation argv mismatch")
        if _require_string(
            command_result["cwd"], "sk target runtime validation command cwd"
        ) != str(cwd):
            raise CliUsageError("sk target runtime validation cwd mismatch")
        command_timeout = command_result["timeout_seconds"]
        if (
            not isinstance(command_timeout, int)
            or isinstance(command_timeout, bool)
            or command_timeout != command["timeout_seconds"]
        ):
            raise CliUsageError("sk target runtime validation timeout mismatch")
        if command_result["env"] != env_summary:
            raise CliUsageError("sk target runtime validation env summary mismatch")
        stdout_tail = _require_string(
            command_result["stdout_tail"],
            "sk target runtime validation stdout_tail",
            allow_empty=True,
        )
        stderr_tail = _require_string(
            command_result["stderr_tail"],
            "sk target runtime validation stderr_tail",
            allow_empty=True,
        )
        if not isinstance(command_result["stdout_tail_truncated"], bool):
            raise CliUsageError(
                "sk target runtime validation stdout_tail_truncated mismatch"
            )
        if not isinstance(command_result["stderr_tail_truncated"], bool):
            raise CliUsageError(
                "sk target runtime validation stderr_tail_truncated mismatch"
            )
        if (
            len(stdout_tail) > TARGET_RUNTIME_OUTPUT_TAIL_CHARS
            or len(stderr_tail) > TARGET_RUNTIME_OUTPUT_TAIL_CHARS
        ):
            raise CliUsageError(
                "sk target runtime validation output tail exceeds bound"
            )
        check_results: dict[str, dict[str, Any]] = {
            "sk_correctness_oracle_spec_current": _target_runtime_check(
                "sk_correctness_oracle_spec_current",
                "passed",
                "sk_correctness_oracle_spec_defined_current",
                ["operator-sk-correctness-oracle-spec.json"],
            ),
            "sk_correctness_oracle_spec_defined": _target_runtime_check(
                "sk_correctness_oracle_spec_defined",
                "passed",
                "sk_correctness_oracle_spec_defined",
                ["operator-sk-correctness-oracle-spec.json"],
            ),
            "sk_runtime_input_values_current": _target_runtime_check(
                "sk_runtime_input_values_current",
                "passed",
                "sk_runtime_input_values_defined_current",
                ["operator-sk-runtime-input-values.json"],
            ),
            "sk_runtime_input_values_defined": _target_runtime_check(
                "sk_runtime_input_values_defined",
                "passed",
                "sk_runtime_input_values_defined",
                ["operator-sk-runtime-input-values.json"],
            ),
            "runtime_command_spec_schema": _target_runtime_check(
                "runtime_command_spec_schema",
                "passed",
                "runtime_command_spec_valid",
                [str(spec_path.resolve())],
            ),
            "runtime_command_cwd_safe": _target_runtime_check(
                "runtime_command_cwd_safe",
                "passed",
                "runtime_command_cwd_safe",
                [str(cwd)],
            ),
            "runtime_command_executable_resolved": _target_runtime_check(
                "runtime_command_executable_resolved",
                "passed",
                "runtime_command_executable_resolved",
                [executable],
            ),
            "runtime_input_values_manifest_bound_as_argv": _target_runtime_check(
                "runtime_input_values_manifest_bound_as_argv",
                "passed",
                "runtime_input_values_manifest_bound_as_argv",
                [input_binding["path"]],
            ),
        }
        returncode = command_result["returncode"]
        timed_out = command_result["timed_out"]
        if not isinstance(timed_out, bool):
            raise CliUsageError("sk target runtime validation timed_out mismatch")
        if status == "passed":
            invalid_returncode = (
                not isinstance(returncode, int)
                or isinstance(returncode, bool)
                or returncode != 0
            )
            if invalid_returncode or timed_out is not False:
                raise CliUsageError(
                    "sk target runtime validation passed command mismatch"
                )
            execution_check = _target_runtime_check(
                "runtime_command_execution", "passed", "target_runtime_passed"
            )
        elif timed_out is True:
            if returncode is not None:
                raise CliUsageError(
                    "sk target runtime validation timeout command mismatch"
                )
            execution_check = _target_runtime_check(
                "runtime_command_execution", "failed", "target_runtime_timeout"
            )
        else:
            if (
                not isinstance(returncode, int)
                or isinstance(returncode, bool)
                or returncode == 0
            ):
                raise CliUsageError(
                    "sk target runtime validation failed command mismatch"
                )
            execution_check = _target_runtime_check(
                "runtime_command_execution",
                "failed",
                "target_runtime_failed",
                [f"returncode={returncode}"],
            )
        check_results["runtime_command_execution"] = execution_check
        _sk_target_runtime_boundary_checks(check_results)
        expected = _sk_target_runtime_result_manifest(
            TargetRuntimeResultManifestInput(
                output_dir,
                status,
                runtime_values_sha256,
                input_binding,
                spec_path,
                spec_bytes,
                check_results,
                [command_result],
                ["compare_sk_runtime_outputs"]
                if status == "passed"
                else [
                    "fix_sk_target_runtime_command",
                    "run_sk_target_runtime_validation",
                ],
            )
        )
        if target != expected:
            raise CliUsageError("sk target runtime validation mismatch")
        return target
    except CliUsageError as exc:
        if str(exc) == "sk target runtime validation mismatch":
            raise
        raise CliUsageError("sk target runtime validation mismatch") from exc


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_input_value_spec(path: Path) -> tuple[Path, bytes, Any]:
    spec_path = path.resolve()
    if not path.exists():
        if path.is_symlink():
            raise CliUsageError("input value spec is not a regular file")
        raise CliUsageError("input value spec not found")
    if path.is_symlink():
        raise CliUsageError("input value spec is not a regular file")
    if not path.is_file():
        raise CliUsageError("input value spec is not a file")
    try:
        spec_bytes = path.read_bytes()
    except OSError as exc:
        raise CliUsageError("input value spec cannot be read") from exc
    try:
        spec_text = spec_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CliUsageError("input value spec is not valid UTF-8") from exc
    try:
        spec = json.loads(spec_text, parse_constant=_reject_json_constant)
    except ValueError as exc:
        raise CliUsageError("input value spec is not valid JSON") from exc
    return spec_path, spec_bytes, spec


def _require_json_finite(value: Any, label: str) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CliUsageError(f"{label} must not contain non-finite numbers")
        return
    if isinstance(value, str):
        return
    if isinstance(value, list):
        for item in value:
            _require_json_finite(item, label)
        return
    if isinstance(value, dict):
        for item in value.values():
            _require_json_finite(item, label)
        return
    raise CliUsageError(f"{label} contains unsupported JSON value")


def _read_correctness_oracle_spec(path: Path) -> tuple[Path, bytes, Any]:
    spec_path = path.resolve()
    if not path.exists():
        if path.is_symlink():
            raise CliUsageError("correctness oracle spec is not a regular file")
        raise CliUsageError("correctness oracle spec not found")
    if path.is_symlink():
        raise CliUsageError("correctness oracle spec is not a regular file")
    if not path.is_file():
        raise CliUsageError("correctness oracle spec is not a file")
    try:
        spec_bytes = path.read_bytes()
    except OSError as exc:
        raise CliUsageError("correctness oracle spec cannot be read") from exc
    try:
        spec_text = spec_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CliUsageError("correctness oracle spec is not valid UTF-8") from exc
    try:
        spec = json.loads(spec_text, parse_constant=_reject_json_constant)
    except ValueError as exc:
        raise CliUsageError("correctness oracle spec is not valid JSON") from exc
    return spec_path, spec_bytes, spec


def _normalize_correctness_tolerance(
    value: Any, comparator: str
) -> dict[str, int | float]:
    tolerance = _require_exact_keys(value, {"rtol", "atol"}, "correctness tolerance")
    normalized: dict[str, int | float] = {}
    for name in ("rtol", "atol"):
        raw_value = tolerance[name]
        invalid_number = not isinstance(raw_value, (int, float)) or isinstance(
            raw_value, bool
        )
        invalid_range = not invalid_number and (
            not math.isfinite(raw_value) or raw_value < 0
        )
        if invalid_number or invalid_range:
            raise CliUsageError(
                "correctness tolerance must be a non-negative finite number"
            )
        normalized[name] = raw_value
    if comparator == "exact" and (normalized["rtol"] != 0 or normalized["atol"] != 0):
        raise CliUsageError("exact comparator requires zero tolerance")
    return normalized


def _normalize_correctness_oracle_spec(spec: Any) -> list[dict[str, Any]]:
    payload = _require_exact_keys(spec, {"oracle_specs"}, "correctness oracle spec")
    raw_specs = _require_list(
        payload["oracle_specs"], "correctness oracle spec oracle_specs"
    )
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_item in raw_specs:
        item = _require_exact_keys(
            raw_item,
            {"input_set_id", "expected_output", "comparator", "tolerance"},
            "correctness oracle spec item",
        )
        input_set_id = _require_string(
            item["input_set_id"], "correctness oracle spec input_set_id"
        )
        if input_set_id in seen_ids:
            raise CliUsageError("duplicate correctness oracle spec input_set_id")
        seen_ids.add(input_set_id)
        expected_output = _require_exact_keys(
            item["expected_output"], {"kind", "value"}, "expected output"
        )
        expected_output_kind = _require_string(
            expected_output["kind"], "expected output kind"
        )
        if expected_output_kind != "inline_json":
            raise CliUsageError("unsupported expected output kind")
        if expected_output["value"] is None:
            raise CliUsageError("expected output value must not be null")
        _require_json_finite(expected_output["value"], "expected output value")
        comparator = _require_string(item["comparator"], "correctness comparator")
        if comparator not in {"exact", "allclose"}:
            raise CliUsageError("unsupported correctness comparator")
        tolerance = _normalize_correctness_tolerance(item["tolerance"], comparator)
        normalized.append(
            {
                "input_set_id": input_set_id,
                "expected_output": {
                    "kind": expected_output_kind,
                    "value": expected_output["value"],
                },
                "comparator": comparator,
                "tolerance": tolerance,
            }
        )
    return normalized


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


def _normalize_sk_source_version_validation_checks(
    checks: list[Any],
) -> dict[str, dict[str, Any]]:
    if len(checks) != len(SK_SOURCE_VERSION_VALIDATION_CHECK_NAMES):
        raise CliUsageError("sk source version validation checks mismatch")
    normalized: dict[str, dict[str, Any]] = {}
    for expected_name, check in zip(
        SK_SOURCE_VERSION_VALIDATION_CHECK_NAMES, checks, strict=True
    ):
        item = _require_exact_keys(
            check,
            {"name", "status", "reason", "evidence"},
            "sk source version validation check",
        )
        name = _require_string(item["name"], "sk source version validation check name")
        if name != expected_name:
            raise CliUsageError("sk source version validation checks mismatch")
        status = _require_string(
            item["status"], f"sk source version validation check {name} status"
        )
        if status not in {"passed", "failed", "blocked"}:
            raise CliUsageError(
                f"invalid sk source version validation check status: {status}"
            )
        normalized[name] = {
            "name": name,
            "status": status,
            "reason": _require_string(
                item["reason"], f"sk source version validation check {name} reason"
            ),
            "evidence": _require_string_list(
                item["evidence"],
                f"sk source version validation check {name} evidence",
                f"sk source version validation check {name} evidence",
            ),
        }
    return normalized


def _validate_passed_sk_source_version_validation(
    output_dir: Path,
    validation: dict[str, Any],
) -> None:
    checks = _normalize_sk_source_version_validation_checks(validation["checks"])
    expected_statuses = {
        "sk_source_version_current": ("passed", "sk_source_version_current"),
        "cmake_command_available": ("passed", "cmake_command_available"),
        "build_dir_available": ("passed", "build_dir_available"),
        "cmake_configure": ("passed", "cmake_configure_passed"),
        "cmake_build": ("passed", "cmake_build_passed"),
    }
    for name, (status, reason) in expected_statuses.items():
        check = checks[name]
        if check["status"] != status or check["reason"] != reason:
            raise CliUsageError("sk source version validation mismatch")
    if checks["sk_source_version_current"]["evidence"] != [
        "operator-sk-source-version.json",
        "operator-sk-source-version.md",
        SK_SOURCE_VERSION_DIR,
    ]:
        raise CliUsageError("sk source version validation mismatch")
    cmake_evidence = checks["cmake_command_available"]["evidence"]
    if len(cmake_evidence) != 1 or Path(cmake_evidence[0]).name != "cmake":
        raise CliUsageError("sk source version validation mismatch")
    if checks["build_dir_available"]["evidence"] != [SK_SOURCE_VERSION_BUILD_DIR]:
        raise CliUsageError("sk source version validation mismatch")

    commands = _require_list(
        validation["commands"], "sk source version validation commands"
    )
    if len(commands) != 2:
        raise CliUsageError("sk source version validation mismatch")
    normalized_commands: list[list[str]] = []
    for command in commands:
        argv = _require_string_list(
            command,
            "sk source version validation command",
            "sk source version validation command arg",
        )
        if not argv:
            raise CliUsageError("sk source version validation mismatch")
        normalized_commands.append(argv)
    source_dir = str(output_dir / SK_SOURCE_VERSION_DIR)
    build_dir = str(output_dir / SK_SOURCE_VERSION_BUILD_DIR)
    if normalized_commands[0][1:] != ["-S", source_dir, "-B", build_dir]:
        raise CliUsageError("sk source version validation mismatch")
    if normalized_commands[1][1:] != ["--build", build_dir]:
        raise CliUsageError("sk source version validation mismatch")
    if normalized_commands[0][0] != normalized_commands[1][0]:
        raise CliUsageError("sk source version validation mismatch")
    if normalized_commands[0][0] != cmake_evidence[0]:
        raise CliUsageError("sk source version validation mismatch")
    if Path(normalized_commands[0][0]).name != "cmake":
        raise CliUsageError("sk source version validation mismatch")
    try:
        resolved_build_dir = (output_dir / SK_SOURCE_VERSION_BUILD_DIR).resolve(
            strict=True
        )
    except FileNotFoundError as exc:
        raise CliUsageError("sk source version validation mismatch") from exc
    except OSError as exc:
        raise CliUsageError("sk source version validation mismatch") from exc
    if (
        not resolved_build_dir.is_dir()
        or resolved_build_dir.name != SK_SOURCE_VERSION_BUILD_DIR
    ):
        raise CliUsageError("sk source version validation mismatch")
    if output_dir.resolve() not in resolved_build_dir.parents:
        raise CliUsageError("sk source version validation mismatch")


def _load_current_sk_source_version_validation(
    output_dir: Path,
    source_version: dict[str, Any],
) -> dict[str, Any]:
    _ = source_version
    payload = _load_json_artifact(
        output_dir / SK_SOURCE_VERSION_VALIDATION_PATH,
        SK_SOURCE_VERSION_VALIDATION_PATH,
    )
    validation = _require_exact_keys(
        payload,
        {
            "status",
            "analysis_output_dir",
            "source_version_manifest_path",
            "source_version_markdown_path",
            "source_version_dir",
            "source_version_validation_path",
            "build_dir",
            "commands",
            "checks",
            "supported_next_actions",
            "execution_boundary",
        },
        "sk source version validation",
    )
    if (
        _require_string(validation["status"], "sk source version validation status")
        != "passed"
    ):
        raise CliUsageError("sk source version validation not passed")
    if _require_string(
        validation["analysis_output_dir"],
        "sk source version validation analysis_output_dir",
    ) != str(output_dir.resolve()):
        raise CliUsageError("sk source version validation mismatch")
    fixed_fields = {
        "source_version_manifest_path": "operator-sk-source-version.json",
        "source_version_markdown_path": "operator-sk-source-version.md",
        "source_version_dir": SK_SOURCE_VERSION_DIR,
        "source_version_validation_path": SK_SOURCE_VERSION_VALIDATION_PATH,
        "build_dir": SK_SOURCE_VERSION_BUILD_DIR,
    }
    for field, expected in fixed_fields.items():
        if (
            _require_string(validation[field], f"sk source version validation {field}")
            != expected
        ):
            raise CliUsageError("sk source version validation mismatch")
    supported_next_actions = _require_string_list(
        validation["supported_next_actions"],
        "sk source version validation supported_next_actions",
        "sk source version validation supported next action",
    )
    if supported_next_actions != ["collect_runtime_input_spec"]:
        raise CliUsageError("sk source version validation mismatch")
    execution_boundary = _require_string_list(
        validation["execution_boundary"],
        "sk source version validation execution_boundary",
        "sk source version validation execution boundary",
    )
    if execution_boundary != SK_SOURCE_VERSION_VALIDATION_PASSED_BOUNDARY:
        raise CliUsageError("sk source version validation mismatch")
    _validate_passed_sk_source_version_validation(output_dir, validation)
    return validation


def cmd_collect_runtime_input_spec(args: argparse.Namespace) -> int:
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
    source_version_validation = _load_current_sk_source_version_validation(
        output_dir, source_version
    )
    result = _summarize_sk_runtime_input_spec(
        RuntimeInputSpecSummaryInput(
            output_dir,
            analysis,
            source_scaffold,
            build_validation,
            source_version,
            source_version_validation,
        )
    )
    _write_json(output_dir / "operator-sk-runtime-input-spec.json", result)
    return 0 if result["status"] == "defined" else 1


def cmd_provide_sk_runtime_input_values(args: argparse.Namespace) -> int:
    output_dir = Path(args.analysis_output_dir).resolve()
    if not output_dir.exists():
        raise CliUsageError(f"analysis output directory not found: {output_dir}")
    if not output_dir.is_dir():
        raise CliUsageError(f"analysis output path is not a directory: {output_dir}")

    spec_path, spec_bytes, spec = _read_input_value_spec(
        Path(args.input_value_spec_json)
    )
    _normalize_sk_input_value_spec(spec)
    runtime_spec = _load_current_sk_runtime_input_spec(output_dir)
    values = _summarize_sk_runtime_input_values(
        output_dir, runtime_spec, spec_path, spec_bytes, spec
    )
    _write_json(output_dir / "operator-sk-runtime-input-values.json", values)
    return 0


def cmd_collect_correctness_oracle_spec(args: argparse.Namespace) -> int:
    output_dir = Path(args.analysis_output_dir).resolve()
    if not output_dir.exists():
        raise CliUsageError(f"analysis output directory not found: {output_dir}")
    if not output_dir.is_dir():
        raise CliUsageError(f"analysis output path is not a directory: {output_dir}")

    spec_path, spec_bytes, spec = _read_correctness_oracle_spec(
        Path(args.oracle_spec_json)
    )
    _normalize_correctness_oracle_spec(spec)
    runtime_spec = _load_current_sk_runtime_input_spec(output_dir)
    runtime_values = _load_current_sk_runtime_input_values(output_dir, runtime_spec)
    oracle_spec = _summarize_sk_correctness_oracle_spec(
        CorrectnessOracleSpecSummaryInput(
            output_dir, runtime_spec, runtime_values, spec_path, spec_bytes, spec
        )
    )
    _write_json(output_dir / "operator-sk-correctness-oracle-spec.json", oracle_spec)
    return 0


def cmd_run_sk_target_runtime_validation(args: argparse.Namespace) -> int:
    output_dir = Path(args.analysis_output_dir).resolve()
    if not output_dir.exists():
        raise CliUsageError(f"analysis output directory not found: {output_dir}")
    if not output_dir.is_dir():
        raise CliUsageError(f"analysis output path is not a directory: {output_dir}")

    spec_path, spec_bytes, spec = _read_runtime_command_spec(
        Path(args.runtime_command_spec_json)
    )
    command = _normalize_sk_runtime_command_spec(spec)
    cwd = _safe_target_runtime_cwd(output_dir, command["cwd"])
    _load_current_sk_correctness_oracle_spec(output_dir)
    input_binding = _sk_runtime_values_manifest_binding(output_dir, cwd, command)
    runtime_values_sha256 = _sk_artifact_sha256(
        output_dir / "operator-sk-runtime-input-values.json"
    )
    effective_env, env_summary = _target_runtime_effective_env(command["env"])
    executable = _resolve_target_runtime_executable(
        output_dir, command["argv"][0], effective_env
    )

    check_results: dict[str, dict[str, Any]] = {
        "sk_correctness_oracle_spec_current": _target_runtime_check(
            "sk_correctness_oracle_spec_current",
            "passed",
            "sk_correctness_oracle_spec_defined_current",
            ["operator-sk-correctness-oracle-spec.json"],
        ),
        "sk_correctness_oracle_spec_defined": _target_runtime_check(
            "sk_correctness_oracle_spec_defined",
            "passed",
            "sk_correctness_oracle_spec_defined",
            ["operator-sk-correctness-oracle-spec.json"],
        ),
        "sk_runtime_input_values_current": _target_runtime_check(
            "sk_runtime_input_values_current",
            "passed",
            "sk_runtime_input_values_defined_current",
            ["operator-sk-runtime-input-values.json"],
        ),
        "sk_runtime_input_values_defined": _target_runtime_check(
            "sk_runtime_input_values_defined",
            "passed",
            "sk_runtime_input_values_defined",
            ["operator-sk-runtime-input-values.json"],
        ),
        "runtime_command_spec_schema": _target_runtime_check(
            "runtime_command_spec_schema",
            "passed",
            "runtime_command_spec_valid",
            [str(spec_path.resolve())],
        ),
        "runtime_command_cwd_safe": _target_runtime_check(
            "runtime_command_cwd_safe",
            "passed",
            "runtime_command_cwd_safe",
            [str(cwd)],
        ),
        "runtime_command_executable_resolved": _target_runtime_check(
            "runtime_command_executable_resolved",
            "passed",
            "runtime_command_executable_resolved",
            [executable],
        ),
        "runtime_input_values_manifest_bound_as_argv": _target_runtime_check(
            "runtime_input_values_manifest_bound_as_argv",
            "passed",
            "runtime_input_values_manifest_bound_as_argv",
            [input_binding["path"]],
        ),
    }
    command_result, execution_check = _run_target_runtime_command(
        command, executable, cwd, effective_env, env_summary
    )
    check_results["runtime_command_execution"] = execution_check
    _sk_target_runtime_boundary_checks(check_results)
    status = "passed" if execution_check["status"] == "passed" else "failed"
    result = _sk_target_runtime_result_manifest(
        TargetRuntimeResultManifestInput(
            output_dir,
            status,
            runtime_values_sha256,
            input_binding,
            spec_path,
            spec_bytes,
            check_results,
            [command_result],
            ["compare_sk_runtime_outputs"]
            if status == "passed"
            else [
                "fix_sk_target_runtime_command",
                "run_sk_target_runtime_validation",
            ],
        )
    )
    _write_json(output_dir / "operator-sk-target-runtime-validation.json", result)
    return 0 if status == "passed" else 1


def _target_runtime_check(
    name: str, status: str, reason: str, evidence: list[str] | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "evidence": evidence or [],
    }


def _read_runtime_command_spec(path: Path) -> tuple[Path, bytes, Any]:
    spec_path = path.resolve()
    if not path.exists():
        if path.is_symlink():
            raise CliUsageError("runtime command spec is not a regular file")
        raise CliUsageError("runtime command spec not found")
    if path.is_symlink():
        raise CliUsageError("runtime command spec is not a regular file")
    if not path.is_file():
        raise CliUsageError("runtime command spec is not a file")
    try:
        spec_bytes = path.read_bytes()
    except OSError as exc:
        raise CliUsageError("runtime command spec cannot be read") from exc
    try:
        spec_text = spec_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CliUsageError("runtime command spec is not valid UTF-8") from exc
    try:
        spec = json.loads(spec_text, parse_constant=_reject_json_constant)
    except ValueError as exc:
        raise CliUsageError("runtime command spec is not valid JSON") from exc
    return spec_path, spec_bytes, spec


def _reject_nul(value: str) -> None:
    if "\0" in value:
        raise CliUsageError("runtime command spec must not contain NUL")


def _normalize_runtime_command_spec(spec: Any) -> dict[str, Any]:
    payload = _require_exact_keys(spec, {"runtime_command"}, "runtime command spec")
    command = _require_exact_keys(
        payload["runtime_command"],
        {"argv", "cwd", "timeout_seconds", "env"},
        "runtime command",
    )
    raw_argv = _require_list(command["argv"], "runtime command argv")
    if not raw_argv:
        raise CliUsageError("runtime command argv must not be empty")
    argv: list[str] = []
    for item in raw_argv:
        value = _require_string(item, "runtime command argv item")
        _reject_nul(value)
        if not value:
            raise CliUsageError("runtime command argv item must be non-empty")
        argv.append(value)
    if not argv[0].strip():
        raise CliUsageError("runtime command executable must be non-empty")
    cwd = _require_string(command["cwd"], "runtime command cwd")
    _reject_nul(cwd)
    if not cwd:
        raise CliUsageError("runtime command cwd must be non-empty")
    if Path(cwd).is_absolute():
        raise CliUsageError("runtime command cwd must be relative")
    timeout_seconds = command["timeout_seconds"]
    invalid_timeout_type = not isinstance(timeout_seconds, int) or isinstance(
        timeout_seconds, bool
    )
    invalid_timeout_range = not invalid_timeout_type and (
        timeout_seconds < 1 or timeout_seconds > 600
    )
    if invalid_timeout_type or invalid_timeout_range:
        raise CliUsageError(
            "runtime command timeout_seconds must be an integer between 1 and 600"
        )
    raw_env = command["env"]
    if not isinstance(raw_env, dict):
        raise CliUsageError("runtime command env must be an object")
    env: dict[str, str] = {}
    for raw_key, raw_value in raw_env.items():
        key = _require_string(raw_key, "runtime command env key")
        _reject_nul(key)
        if not key:
            raise CliUsageError("runtime command env key must be non-empty")
        if "=" in key:
            raise CliUsageError("runtime command env key must not contain '='")
        value = _require_string(
            raw_value, "runtime command env value", allow_empty=True
        )
        _reject_nul(value)
        env[key] = value
    return {"argv": argv, "cwd": cwd, "timeout_seconds": timeout_seconds, "env": env}


def _normalize_sk_runtime_command_spec(spec: Any) -> dict[str, Any]:
    payload = _require_exact_keys(spec, {"runtime_command"}, "sk runtime command spec")
    command = _require_exact_keys(
        payload["runtime_command"],
        {"argv", "cwd", "timeout_seconds", "env", "input_binding"},
        "sk runtime command",
    )
    base_command = _normalize_runtime_command_spec(
        {
            "runtime_command": {
                "argv": command["argv"],
                "cwd": command["cwd"],
                "timeout_seconds": command["timeout_seconds"],
                "env": command["env"],
            }
        }
    )
    binding = _require_exact_keys(
        command["input_binding"],
        {"kind", "argv_index"},
        "sk runtime command input_binding",
    )
    kind = _require_string(binding["kind"], "sk runtime command input_binding kind")
    if kind != "values_manifest_argv":
        raise CliUsageError("unsupported sk runtime command input binding kind")
    argv_index = binding["argv_index"]
    invalid_argv_index_type = not isinstance(argv_index, int) or isinstance(
        argv_index, bool
    )
    invalid_argv_index_range = not invalid_argv_index_type and (
        argv_index < 1 or argv_index >= len(base_command["argv"])
    )
    if invalid_argv_index_type or invalid_argv_index_range:
        raise CliUsageError(
            "sk runtime command input binding argv_index is out of range"
        )
    return {**base_command, "input_binding": {"kind": kind, "argv_index": argv_index}}


def _sk_runtime_values_manifest_binding(
    output_dir: Path,
    cwd: Path,
    command: dict[str, Any],
) -> dict[str, Any]:
    binding = command["input_binding"]
    argv_index = binding["argv_index"]
    value = command["argv"][argv_index]
    _reject_nul(value)
    if not value:
        raise CliUsageError("sk runtime command input binding path must be non-empty")
    if Path(value).is_absolute():
        raise CliUsageError("sk runtime command input binding path must be relative")
    candidate = cwd / value
    if candidate.is_symlink():
        raise CliUsageError(
            "sk runtime command input binding path is not a regular file"
        )
    expected = (output_dir / "operator-sk-runtime-input-values.json").resolve()
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(output_dir.resolve())
    except ValueError as exc:
        raise CliUsageError(
            "sk runtime command input binding path escapes analysis output"
        ) from exc
    if resolved != expected:
        raise CliUsageError(
            "sk runtime command input binding must reference operator-sk-runtime-input-values.json"
        )
    if not expected.exists() or not expected.is_file():
        raise CliUsageError(
            "sk runtime command input binding values manifest not found"
        )
    return {"kind": binding["kind"], "argv_index": argv_index, "path": str(expected)}


def _safe_target_runtime_cwd(scaffold_output_dir: Path, cwd: str) -> Path:
    path = scaffold_output_dir / cwd
    try:
        resolved = path.resolve(strict=False)
        resolved.relative_to(scaffold_output_dir.resolve())
    except ValueError as exc:
        raise CliUsageError("runtime command cwd escapes scaffold output") from exc
    if path.is_symlink():
        raise CliUsageError("runtime command cwd escapes scaffold output")
    if not path.exists():
        raise CliUsageError("runtime command cwd not found")
    if not path.is_dir():
        raise CliUsageError("runtime command cwd is not a directory")
    return path.resolve()


def _target_runtime_effective_env(
    spec_env: dict[str, str],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    inherited = {
        key: os.environ[key]
        for key in TARGET_RUNTIME_ENV_ALLOWLIST
        if key in os.environ
    }
    effective = {**inherited, **spec_env}
    summary = {
        "inherited_env_keys": sorted(inherited),
        "provided_env_keys": sorted(spec_env),
    }
    return effective, summary


def _has_path_separator(value: str) -> bool:
    return "/" in value or "\\" in value


def _resolve_target_runtime_executable(
    scaffold_output_dir: Path, argv0: str, effective_env: dict[str, str]
) -> str:
    if _has_path_separator(argv0) or Path(argv0).is_absolute():
        candidate = Path(argv0)
        if not candidate.is_absolute():
            candidate = scaffold_output_dir / candidate
        try:
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(scaffold_output_dir.resolve())
        except ValueError as exc:
            raise CliUsageError(
                "runtime command executable escapes scaffold output"
            ) from exc
        if candidate.is_symlink():
            raise CliUsageError("runtime command executable is not a regular file")
        if not candidate.exists():
            raise CliUsageError("runtime command executable not found")
        if not candidate.is_file():
            raise CliUsageError("runtime command executable is not a file")
        if not os.access(candidate, os.X_OK):
            raise CliUsageError("runtime command executable is not executable")
        return str(candidate.resolve())
    resolved_text = shutil.which(argv0, path=effective_env.get("PATH", ""))
    if not resolved_text:
        raise CliUsageError("runtime command executable not found")
    resolved = Path(resolved_text)
    if resolved.is_symlink():
        raise CliUsageError("runtime command executable is not a regular file")
    if not resolved.exists():
        raise CliUsageError("runtime command executable not found")
    if not resolved.is_file():
        raise CliUsageError("runtime command executable is not a file")
    if not os.access(resolved, os.X_OK):
        raise CliUsageError("runtime command executable is not executable")
    return str(resolved.resolve())


def _bounded_target_runtime_output(output: str | None) -> tuple[str, bool]:
    if not output:
        return "", False
    lines = output.splitlines()
    truncated = (
        len(lines) > TARGET_RUNTIME_OUTPUT_TAIL_LINES
        or len(output) > TARGET_RUNTIME_OUTPUT_TAIL_CHARS
    )
    if not truncated:
        return output, False
    tail_lines = lines[-TARGET_RUNTIME_OUTPUT_TAIL_LINES:]
    tail = "\n".join(tail_lines)
    if output.endswith("\n") and tail:
        tail += "\n"
    if len(tail) > TARGET_RUNTIME_OUTPUT_TAIL_CHARS:
        marker = "...<truncated>\n"
        tail_start = -(TARGET_RUNTIME_OUTPUT_TAIL_CHARS - len(marker))
        tail = marker + tail[tail_start:]
    return tail, True


def _run_target_runtime_command(
    command: dict[str, Any],
    executable: str,
    cwd: Path,
    effective_env: dict[str, str],
    env_summary: dict[str, list[str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    argv = [executable, *command["argv"][1:]]
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=command["timeout_seconds"],
            shell=False,
            env=effective_env,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_tail, stdout_truncated = _bounded_target_runtime_output(
            exc.stdout if isinstance(exc.stdout, str) else ""
        )
        stderr_tail, stderr_truncated = _bounded_target_runtime_output(
            exc.stderr if isinstance(exc.stderr, str) else ""
        )
        command_result = {
            "name": "target_runtime_validation",
            "argv": argv,
            "cwd": str(cwd),
            "timeout_seconds": command["timeout_seconds"],
            "returncode": None,
            "timed_out": True,
            "stdout_tail": stdout_tail,
            "stdout_tail_truncated": stdout_truncated,
            "stderr_tail": stderr_tail,
            "stderr_tail_truncated": stderr_truncated,
            "env": env_summary,
        }
        return command_result, _target_runtime_check(
            "runtime_command_execution", "failed", "target_runtime_timeout"
        )
    except OSError as exc:
        stderr_tail, stderr_truncated = _bounded_target_runtime_output(str(exc))
        command_result = {
            "name": "target_runtime_validation",
            "argv": argv,
            "cwd": str(cwd),
            "timeout_seconds": command["timeout_seconds"],
            "returncode": None,
            "timed_out": False,
            "stdout_tail": "",
            "stdout_tail_truncated": False,
            "stderr_tail": stderr_tail,
            "stderr_tail_truncated": stderr_truncated,
            "env": env_summary,
        }
        return command_result, _target_runtime_check(
            "runtime_command_execution", "failed", "target_runtime_failed", [str(exc)]
        )
    stdout_tail, stdout_truncated = _bounded_target_runtime_output(completed.stdout)
    stderr_tail, stderr_truncated = _bounded_target_runtime_output(completed.stderr)
    command_result = {
        "name": "target_runtime_validation",
        "argv": argv,
        "cwd": str(cwd),
        "timeout_seconds": command["timeout_seconds"],
        "returncode": completed.returncode,
        "timed_out": False,
        "stdout_tail": stdout_tail,
        "stdout_tail_truncated": stdout_truncated,
        "stderr_tail": stderr_tail,
        "stderr_tail_truncated": stderr_truncated,
        "env": env_summary,
    }
    if completed.returncode == 0:
        return command_result, _target_runtime_check(
            "runtime_command_execution", "passed", "target_runtime_passed"
        )
    return command_result, _target_runtime_check(
        "runtime_command_execution",
        "failed",
        "target_runtime_failed",
        [f"returncode={completed.returncode}"],
    )


def _runtime_output_check(
    name: str, status: str, reason: str, evidence: list[str] | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "evidence": evidence or [],
    }


def _read_runtime_output_spec(path: Path) -> tuple[Path, bytes, Any]:
    spec_path = path.resolve()
    if not path.exists():
        if path.is_symlink():
            raise CliUsageError("runtime output spec is not a regular file")
        raise CliUsageError("runtime output spec not found")
    if path.is_symlink():
        raise CliUsageError("runtime output spec is not a regular file")
    if not path.is_file():
        raise CliUsageError("runtime output spec is not a file")
    try:
        spec_bytes = path.read_bytes()
    except OSError as exc:
        raise CliUsageError("runtime output spec cannot be read") from exc
    try:
        spec_text = spec_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CliUsageError("runtime output spec is not valid UTF-8") from exc
    try:
        spec = json.loads(spec_text, parse_constant=_reject_json_constant)
    except ValueError as exc:
        raise CliUsageError("runtime output spec is not valid JSON") from exc
    return spec_path, spec_bytes, spec


def _reject_runtime_output_nul(value: str) -> None:
    if "\0" in value:
        raise CliUsageError("runtime output spec must not contain NUL")


def _normalize_runtime_output_spec(spec: Any) -> list[dict[str, Any]]:
    payload = _require_exact_keys(spec, {"actual_outputs"}, "runtime output spec")
    raw_outputs = _require_list(
        payload["actual_outputs"], "runtime output actual_outputs"
    )
    seen: set[str] = set()
    outputs: list[dict[str, Any]] = []
    for raw_output in raw_outputs:
        item = _require_exact_keys(
            raw_output, {"oracle_set_id", "actual_output"}, "runtime output item"
        )
        oracle_set_id = _require_string(
            item["oracle_set_id"], "runtime output oracle_set_id"
        )
        _reject_runtime_output_nul(oracle_set_id)
        if not oracle_set_id:
            raise CliUsageError("runtime output oracle_set_id must be non-empty")
        if oracle_set_id in seen:
            raise CliUsageError("duplicate runtime output oracle_set_id")
        seen.add(oracle_set_id)
        actual_output = _require_exact_keys(
            item["actual_output"], {"kind", "value"}, "runtime output actual_output"
        )
        kind = _require_string(actual_output["kind"], "runtime output kind")
        if kind != "inline_json":
            raise CliUsageError("unsupported runtime output kind")
        if actual_output["value"] is None:
            raise CliUsageError("runtime output value must not be null")
        _require_json_finite(actual_output["value"], "runtime output value")
        outputs.append(
            {
                "oracle_set_id": oracle_set_id,
                "actual_output": {"kind": kind, "value": actual_output["value"]},
            }
        )
    return outputs


def _normalize_runtime_output_payload(
    spec: Any, oracle_spec: dict[str, Any]
) -> list[dict[str, Any]]:
    _reject_nonpassed_runtime_statuses(spec)
    if isinstance(spec, dict) and "actual_outputs" in spec:
        return _normalize_runtime_output_spec(spec)
    if isinstance(spec, dict) and "outputs" in spec:
        raw_outputs = _require_dict(spec["outputs"], "runtime output outputs")
    elif isinstance(spec, dict):
        raw_outputs = _require_dict(spec, "runtime output outputs")
    else:
        raise CliUsageError("runtime output spec must be an object")
    entry_to_oracle = {
        item["entry_name"]: item["oracle_set_id"]
        for item in oracle_spec.get("oracle_specs", [])
    }
    actual_outputs: list[dict[str, Any]] = []
    for entry_name, value in raw_outputs.items():
        if not isinstance(entry_name, str) or not entry_name:
            raise CliUsageError("runtime output entry name must be non-empty")
        if not isinstance(value, dict):
            raise CliUsageError(
                "runtime output nested item must contain baseline and sk"
            )
        if set(value) != {"baseline", "sk"}:
            raise CliUsageError(
                "runtime output nested item must contain baseline and sk"
            )
        baseline_value = value["baseline"]
        oracle_set_id = entry_to_oracle.get(entry_name)
        if oracle_set_id is None:
            raise CliUsageError("runtime output set mismatch")
        actual_outputs.append(
            {
                "oracle_set_id": oracle_set_id,
                "actual_output": {
                    "kind": "inline_json",
                    "value": {"baseline": baseline_value, "sk": value["sk"]},
                },
            }
        )
    return _normalize_runtime_output_spec({"actual_outputs": actual_outputs})


def _reject_nonpassed_runtime_statuses(spec: Any) -> None:
    if not isinstance(spec, dict):
        return
    statuses = spec.get("statuses")
    if not isinstance(statuses, dict):
        return
    bad: list[str] = []
    for entry_name, item in sorted(statuses.items()):
        if not isinstance(item, dict):
            bad.append(f"{entry_name}:invalid-status")
            continue
        status = item.get("status")
        if status != "passed":
            reason = item.get("reason", status)
            bad.append(f"{entry_name}:{status}:{reason}")
    if bad:
        raise CliUsageError(
            "runtime output status not passed: " + "; ".join(str(item) for item in bad)
        )


def _json_pointer(path: list[str]) -> str:
    if not path:
        return ""
    return "/" + "/".join(item.replace("~", "~0").replace("/", "~1") for item in path)


def _json_number(value: Any) -> bool:
    return (isinstance(value, int) or isinstance(value, float)) and not isinstance(
        value, bool
    )


def _compare_declared_json(
    actual: Any,
    expected: Any,
    comparator: str,
    tolerance: dict[str, Any],
    path: list[str] | None = None,
) -> tuple[bool, str, str | None]:
    path = path or []
    if isinstance(actual, dict) or isinstance(expected, dict):
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            return False, "declared_output_type_mismatch", _json_pointer(path)
        if set(actual) != set(expected):
            return False, "declared_output_shape_mismatch", _json_pointer(path)
        for key in sorted(actual):
            matched, reason, mismatch_path = _compare_declared_json(
                actual[key], expected[key], comparator, tolerance, [*path, key]
            )
            if not matched:
                return matched, reason, mismatch_path
        return True, "declared_output_matched", None
    if isinstance(actual, list) or isinstance(expected, list):
        if not isinstance(actual, list) or not isinstance(expected, list):
            return False, "declared_output_type_mismatch", _json_pointer(path)
        if len(actual) != len(expected):
            return False, "declared_output_shape_mismatch", _json_pointer(path)
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            matched, reason, mismatch_path = _compare_declared_json(
                actual_item, expected_item, comparator, tolerance, [*path, str(index)]
            )
            if not matched:
                return matched, reason, mismatch_path
        return True, "declared_output_matched", None
    if actual is None or expected is None:
        return (
            (True, "declared_output_matched", None)
            if actual is expected
            else (False, "declared_output_type_mismatch", _json_pointer(path))
        )
    if isinstance(actual, bool) or isinstance(expected, bool):
        if not isinstance(actual, bool) or not isinstance(expected, bool):
            return False, "declared_output_type_mismatch", _json_pointer(path)
        return (
            (True, "declared_output_matched", None)
            if actual == expected
            else (False, "declared_output_value_mismatch", _json_pointer(path))
        )
    if _json_number(actual) or _json_number(expected):
        if not _json_number(actual) or not _json_number(expected):
            return False, "declared_output_type_mismatch", _json_pointer(path)
        if comparator == "allclose":
            rtol = float(tolerance["rtol"])
            atol = float(tolerance["atol"])
            if abs(float(actual) - float(expected)) <= atol + rtol * abs(
                float(expected)
            ):
                return True, "declared_output_matched", None
            return False, "declared_output_value_mismatch", _json_pointer(path)
        return (
            (True, "declared_output_matched", None)
            if actual == expected
            else (False, "declared_output_value_mismatch", _json_pointer(path))
        )
    if isinstance(actual, str) or isinstance(expected, str):
        if not isinstance(actual, str) or not isinstance(expected, str):
            return False, "declared_output_type_mismatch", _json_pointer(path)
        return (
            (True, "declared_output_matched", None)
            if actual == expected
            else (False, "declared_output_value_mismatch", _json_pointer(path))
        )
    return False, "declared_output_type_mismatch", _json_pointer(path)


def _validate_runtime_output_binding(
    oracle: dict[str, Any], actual_outputs: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    expected_ids = [item["id"] for item in oracle["oracle_sets"]]
    actual_ids = [item["oracle_set_id"] for item in actual_outputs]
    if set(actual_ids) != set(expected_ids):
        raise CliUsageError("runtime output set mismatch")
    return {item["oracle_set_id"]: item for item in actual_outputs}


def _compare_runtime_outputs(
    oracle: dict[str, Any], actual_outputs_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for oracle_set in oracle["oracle_sets"]:
        oracle_set_id = oracle_set["id"]
        actual_output_item = _require_mapping_value(
            actual_outputs_by_id, oracle_set_id, "runtime output binding"
        )
        actual_output = actual_output_item["actual_output"]
        expected_output = oracle_set["expected_output"]
        actual_value = actual_output["value"]
        expected_value = expected_output["value"]
        actual_is_bind_target_pair = isinstance(actual_value, dict) and set(
            actual_value
        ) == {"baseline", "sk"}
        expected_is_bind_target = (
            isinstance(expected_value, dict)
            and expected_value.get("source") == "bind-target-on-wheel"
        )
        if actual_is_bind_target_pair and expected_is_bind_target:
            compare_actual = actual_value["sk"]
            compare_expected = actual_value["baseline"]
        else:
            compare_actual = actual_value
            compare_expected = expected_value
        matched, reason, mismatch_path = _compare_declared_json(
            compare_actual,
            compare_expected,
            oracle_set["comparator"],
            oracle_set["tolerance"],
        )
        comparisons.append(
            {
                "oracle_set_id": oracle_set_id,
                "input_set_id": oracle_set["input_set_id"],
                "entry_name": oracle_set["entry_name"],
                "comparator": oracle_set["comparator"],
                "tolerance": oracle_set["tolerance"],
                "actual_output_kind": actual_output["kind"],
                "expected_output_kind": expected_output["kind"],
                "status": "matched" if matched else "mismatched",
                "reason": reason,
                "mismatch_path": mismatch_path,
            }
        )
    return comparisons


class RuntimeOutputResultManifestInput(NamedTuple):
    output_dir: Path
    status: str
    spec_path: Path
    spec_bytes: bytes
    check_results: dict[str, dict[str, Any]]
    comparisons: list[dict[str, Any]] | None = None
    supported_next_actions: list[str] | None = None


def _sk_runtime_output_result_manifest(
    request: RuntimeOutputResultManifestInput,
) -> dict[str, Any]:
    return {
        "status": request.status,
        "analysis_output_dir": str(request.output_dir.resolve()),
        "correctness_oracle_spec_path": "operator-sk-correctness-oracle-spec.json",
        "target_runtime_validation_path": "operator-sk-target-runtime-validation.json",
        "runtime_output_comparison_path": "operator-sk-runtime-output-comparison.json",
        "runtime_output_spec_path": str(request.spec_path.resolve()),
        "runtime_output_spec_sha256": hashlib.sha256(request.spec_bytes).hexdigest(),
        "comparisons": request.comparisons or [],
        "checks": [
            request.check_results[name]
            for name in SK_RUNTIME_OUTPUT_COMPARISON_CHECK_NAMES
        ],
        "supported_next_actions": request.supported_next_actions or [],
        "execution_boundary": SK_RUNTIME_OUTPUT_COMPARISON_BOUNDARY,
    }


def _sk_runtime_output_boundary_checks(
    check_results: dict[str, dict[str, Any]],
) -> None:
    check_results["runtime_output_provenance_boundary_open"] = _runtime_output_check(
        "runtime_output_provenance_boundary_open",
        "open",
        "runtime_output_provenance_not_validated",
    )
    check_results["correctness_boundary_open"] = _runtime_output_check(
        "correctness_boundary_open",
        "open",
        "operator_correctness_not_validated",
    )
    check_results["handoff_boundary_open"] = _runtime_output_check(
        "handoff_boundary_open",
        "open",
        "handoff_not_completed",
    )


def _sk_oracle_spec_as_runtime_oracle(oracle_spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": oracle_spec["status"],
        "oracle_sets": [
            {
                "id": item["oracle_set_id"],
                "input_set_id": item["input_set_id"],
                "entry_name": item["entry_name"],
                "source_file": item["source_file"],
                "expected_output": item["expected_output"],
                "comparator": item["comparator"],
                "tolerance": item["tolerance"],
            }
            for item in oracle_spec["oracle_specs"]
        ],
    }


def _summarize_sk_runtime_output_comparison(
    output_dir: Path,
    oracle_spec: dict[str, Any],
    spec_path: Path,
    spec_bytes: bytes,
    actual_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    runtime_oracle = _sk_oracle_spec_as_runtime_oracle(oracle_spec)
    actual_outputs_by_id = _validate_runtime_output_binding(
        runtime_oracle, actual_outputs
    )
    target_runtime = _load_current_sk_target_runtime_validation(output_dir, oracle_spec)
    check_results: dict[str, dict[str, Any]] = {
        "sk_correctness_oracle_spec_current": _runtime_output_check(
            "sk_correctness_oracle_spec_current",
            "passed",
            "sk_correctness_oracle_spec_defined_current",
            ["operator-sk-correctness-oracle-spec.json"],
        ),
        "sk_correctness_oracle_spec_defined": _runtime_output_check(
            "sk_correctness_oracle_spec_defined",
            "passed",
            "sk_correctness_oracle_spec_defined",
            ["operator-sk-correctness-oracle-spec.json"],
        ),
        "sk_target_runtime_validation_current": _runtime_output_check(
            "sk_target_runtime_validation_current",
            "passed",
            "sk_target_runtime_validation_current",
            ["operator-sk-target-runtime-validation.json"],
        ),
        "runtime_output_spec_schema": _runtime_output_check(
            "runtime_output_spec_schema",
            "passed",
            "runtime_output_spec_valid",
            [str(spec_path.resolve())],
        ),
        "runtime_output_sets_complete": _runtime_output_check(
            "runtime_output_sets_complete",
            "passed",
            "runtime_output_sets_complete",
        ),
    }
    if target_runtime["status"] != "passed":
        check_results["sk_target_runtime_validation_passed"] = _runtime_output_check(
            "sk_target_runtime_validation_passed",
            "blocked",
            "sk_target_runtime_validation_not_passed",
            ["operator-sk-target-runtime-validation.json"],
        )
        check_results["declared_runtime_outputs_compared"] = _runtime_output_check(
            "declared_runtime_outputs_compared",
            "blocked",
            "sk_target_runtime_validation_not_passed",
        )
        _sk_runtime_output_boundary_checks(check_results)
        return _sk_runtime_output_result_manifest(
            RuntimeOutputResultManifestInput(
                output_dir,
                "blocked",
                spec_path,
                spec_bytes,
                check_results,
                [],
                ["run_sk_target_runtime_validation"],
            )
        )

    check_results["sk_target_runtime_validation_passed"] = _runtime_output_check(
        "sk_target_runtime_validation_passed",
        "passed",
        "sk_target_runtime_validation_passed",
        ["operator-sk-target-runtime-validation.json"],
    )
    comparisons = _compare_runtime_outputs(runtime_oracle, actual_outputs_by_id)
    status = (
        "matched"
        if all(item["status"] == "matched" for item in comparisons)
        else "mismatched"
    )
    check_results["declared_runtime_outputs_compared"] = _runtime_output_check(
        "declared_runtime_outputs_compared",
        "passed" if status == "matched" else "failed",
        "declared_runtime_outputs_matched"
        if status == "matched"
        else "declared_runtime_outputs_mismatched",
    )
    _sk_runtime_output_boundary_checks(check_results)
    return _sk_runtime_output_result_manifest(
        RuntimeOutputResultManifestInput(
            output_dir,
            status,
            spec_path,
            spec_bytes,
            check_results,
            comparisons,
            ["review_sk_runtime_output_comparison"]
            if status == "matched"
            else ["fix_sk_runtime_outputs_or_oracle", "compare_sk_runtime_outputs"],
        )
    )


def _summarize_standalone_bind_target_comparison(
    output_dir: Path,
    spec_path: Path,
    spec_bytes: bytes,
    spec: Any,
) -> dict[str, Any]:
    if not isinstance(spec, dict) or "outputs" not in spec:
        raise CliUsageError("operator correctness inputs are not current")
    backend = spec.get("backend")
    if backend not in {None, "standalone", "bind-target-on-wheel", "wheel"}:
        raise CliUsageError("runtime output spec is not bind-target output")
    statuses = spec.get("statuses")
    if isinstance(statuses, dict):
        bad_statuses = []
        for entry_name, item in sorted(statuses.items()):
            if not isinstance(item, dict) or item.get("status") != "passed":
                reason = (
                    item.get("reason", item.get("status"))
                    if isinstance(item, dict)
                    else "invalid-status"
                )
                bad_statuses.append(
                    (
                        entry_name,
                        item.get("status", "invalid-status")
                        if isinstance(item, dict)
                        else "invalid-status",
                        reason,
                    )
                )
        if bad_statuses:
            comparisons = [
                {
                    "oracle_set_id": f"oracle:entry:{entry_name}:default",
                    "input_set_id": f"entry:{entry_name}:default",
                    "entry_name": entry_name,
                    "comparator": "allclose",
                    "tolerance": {"rtol": 0, "atol": 0},
                    "actual_output_kind": "inline_json",
                    "expected_output_kind": "bind-target-on-wheel",
                    "status": "mismatched",
                    "reason": str(reason),
                    "mismatch_path": None,
                }
                for entry_name, _status, reason in bad_statuses
            ]
            return {
                "status": "mismatched",
                "analysis_output_dir": str(output_dir.resolve()),
                "correctness_oracle_spec_path": "operator-sk-auto-oracle-spec.json",
                "target_runtime_validation_path": "runner-stdout.json",
                "runtime_output_comparison_path": "operator-sk-runtime-output-comparison.json",
                "runtime_output_spec_path": str(spec_path.resolve()),
                "runtime_output_spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
                "comparisons": comparisons,
                "checks": [
                    _runtime_output_check(
                        "runtime_output_spec_schema",
                        "passed",
                        "runtime_output_spec_valid",
                        [str(spec_path.resolve())],
                    ),
                    _runtime_output_check(
                        "declared_runtime_outputs_compared",
                        "failed",
                        "standalone_runtime_status_not_passed",
                    ),
                ],
                "supported_next_actions": ["fix_standalone_runtime_status"],
                "execution_boundary": SK_RUNTIME_OUTPUT_COMPARISON_BOUNDARY,
            }
    outputs = _require_dict(spec["outputs"], "runtime output outputs")
    comparisons: list[dict[str, Any]] = []
    status = "matched"
    for entry_name, value in sorted(outputs.items()):
        if not isinstance(entry_name, str) or not entry_name:
            raise CliUsageError("runtime output entry name must be non-empty")
        if not isinstance(value, dict):
            raise CliUsageError(
                "runtime output nested item must contain baseline and sk"
            )
        if set(value) != {"baseline", "sk"}:
            raise CliUsageError(
                "runtime output nested item must contain baseline and sk"
            )
        baseline_value = value["baseline"]
        matched, reason, mismatch_path = _compare_declared_json(
            value["sk"],
            baseline_value,
            "allclose",
            {"rtol": 0, "atol": 0},
        )
        comparisons.append(
            {
                "oracle_set_id": f"oracle:entry:{entry_name}:default",
                "input_set_id": f"entry:{entry_name}:default",
                "entry_name": entry_name,
                "comparator": "allclose",
                "tolerance": {"rtol": 0, "atol": 0},
                "actual_output_kind": "inline_json",
                "expected_output_kind": "bind-target-on-wheel",
                "status": "matched" if matched else "mismatched",
                "reason": reason,
                "mismatch_path": mismatch_path,
            }
        )
        if not matched:
            status = "mismatched"
    return {
        "status": status,
        "analysis_output_dir": str(output_dir.resolve()),
        "correctness_oracle_spec_path": "operator-sk-auto-oracle-spec.json",
        "target_runtime_validation_path": "runner-stdout.json",
        "runtime_output_comparison_path": "operator-sk-runtime-output-comparison.json",
        "runtime_output_spec_path": str(spec_path.resolve()),
        "runtime_output_spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "comparisons": comparisons,
        "checks": [
            _runtime_output_check(
                "runtime_output_spec_schema",
                "passed",
                "runtime_output_spec_valid",
                [str(spec_path.resolve())],
            ),
            _runtime_output_check(
                "declared_runtime_outputs_compared",
                "passed" if status == "matched" else "failed",
                "declared_runtime_outputs_matched"
                if status == "matched"
                else "declared_runtime_outputs_mismatched",
            ),
        ],
        "supported_next_actions": ["review_sk_runtime_output_comparison"]
        if status == "matched"
        else ["fix_sk_runtime_outputs_or_oracle"],
        "execution_boundary": SK_RUNTIME_OUTPUT_COMPARISON_BOUNDARY,
    }


def _load_current_sk_runtime_output_comparison(
    output_dir: Path, oracle_spec: dict[str, Any]
) -> dict[str, Any]:
    comparison = _load_json_artifact(
        output_dir / "operator-sk-runtime-output-comparison.json",
        "operator-sk-runtime-output-comparison.json",
    )
    try:
        comparison = _require_exact_keys(
            comparison,
            {
                "status",
                "analysis_output_dir",
                "correctness_oracle_spec_path",
                "target_runtime_validation_path",
                "runtime_output_comparison_path",
                "runtime_output_spec_path",
                "runtime_output_spec_sha256",
                "comparisons",
                "checks",
                "supported_next_actions",
                "execution_boundary",
            },
            "sk runtime output comparison",
        )
        if _require_string(
            comparison["analysis_output_dir"],
            "sk runtime output comparison analysis_output_dir",
        ) != str(output_dir.resolve()):
            raise CliUsageError(
                "sk runtime output comparison analysis_output_dir mismatch"
            )
        if (
            _require_string(
                comparison["correctness_oracle_spec_path"],
                "sk runtime output comparison correctness_oracle_spec_path",
            )
            != "operator-sk-correctness-oracle-spec.json"
        ):
            raise CliUsageError(
                "sk runtime output comparison correctness_oracle_spec_path mismatch"
            )
        if (
            _require_string(
                comparison["target_runtime_validation_path"],
                "sk runtime output comparison target_runtime_validation_path",
            )
            != "operator-sk-target-runtime-validation.json"
        ):
            raise CliUsageError(
                "sk runtime output comparison target_runtime_validation_path mismatch"
            )
        if (
            _require_string(
                comparison["runtime_output_comparison_path"],
                "sk runtime output comparison runtime_output_comparison_path",
            )
            != "operator-sk-runtime-output-comparison.json"
        ):
            raise CliUsageError(
                "sk runtime output comparison runtime_output_comparison_path mismatch"
            )
        spec_path_text = _require_string(
            comparison["runtime_output_spec_path"],
            "sk runtime output comparison runtime_output_spec_path",
        )
        spec_path = Path(spec_path_text)
        if not spec_path.is_absolute():
            raise CliUsageError(
                "sk runtime output comparison runtime_output_spec_path must be absolute"
            )
        spec_path, spec_bytes, spec = _read_runtime_output_spec(spec_path)
        if (
            _require_string(
                comparison["runtime_output_spec_sha256"],
                "sk runtime output comparison runtime_output_spec_sha256",
            )
            != hashlib.sha256(spec_bytes).hexdigest()
        ):
            raise CliUsageError(
                "sk runtime output comparison runtime_output_spec_sha256 mismatch"
            )
        actual_outputs = _normalize_runtime_output_spec(spec)
        expected = _summarize_sk_runtime_output_comparison(
            output_dir, oracle_spec, spec_path, spec_bytes, actual_outputs
        )
        if comparison != expected:
            raise CliUsageError("sk runtime output comparison mismatch")
        return comparison
    except CliUsageError as exc:
        if str(exc) == "sk runtime output comparison mismatch":
            raise
        raise CliUsageError("sk runtime output comparison mismatch") from exc


def _load_current_sk_runtime_output_comparison_context(
    output_dir: Path,
    oracle_spec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    comparison = _load_current_sk_runtime_output_comparison(output_dir, oracle_spec)
    spec_path = Path(
        _require_string(
            comparison["runtime_output_spec_path"],
            "sk runtime output comparison runtime_output_spec_path",
        )
    )
    _spec_path, _spec_bytes, spec = _read_runtime_output_spec(spec_path)
    actual_outputs = _normalize_runtime_output_spec(spec)
    actual_outputs_by_id = _validate_runtime_output_binding(
        _sk_oracle_spec_as_runtime_oracle(oracle_spec), actual_outputs
    )
    target_runtime = _load_current_sk_target_runtime_validation(output_dir, oracle_spec)
    return comparison, actual_outputs_by_id, target_runtime


def _read_sk_runtime_output_extraction_spec(path: Path) -> tuple[Path, bytes, Any]:
    spec_path = path.resolve()
    if not path.exists():
        if path.is_symlink():
            raise CliUsageError(
                "sk runtime output extraction spec is not a regular file"
            )
        raise CliUsageError("sk runtime output extraction spec not found")
    if path.is_symlink():
        raise CliUsageError("sk runtime output extraction spec is not a regular file")
    if not path.is_file():
        raise CliUsageError("sk runtime output extraction spec is not a file")
    try:
        spec_bytes = path.read_bytes()
    except OSError as exc:
        raise CliUsageError("sk runtime output extraction spec cannot be read") from exc
    try:
        spec_text = spec_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CliUsageError(
            "sk runtime output extraction spec is not valid UTF-8"
        ) from exc
    try:
        spec = json.loads(spec_text, parse_constant=_reject_json_constant)
    except ValueError as exc:
        raise CliUsageError(
            "sk runtime output extraction spec is not valid JSON"
        ) from exc
    return spec_path, spec_bytes, spec


def _reject_sk_runtime_output_extraction_nul(value: str) -> None:
    if "\0" in value:
        raise CliUsageError("sk runtime output extraction spec must not contain NUL")


def _normalize_sk_runtime_output_extraction_spec(spec: Any) -> list[dict[str, Any]]:
    payload = _require_exact_keys(
        spec, {"output_extractions"}, "sk runtime output extraction spec"
    )
    raw_items = _require_list(
        payload["output_extractions"], "sk runtime output extraction output_extractions"
    )
    if not raw_items:
        raise CliUsageError(
            "sk runtime output extraction output_extractions must not be empty"
        )
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for raw_item in raw_items:
        item = _require_exact_keys(
            raw_item, {"oracle_set_id", "source"}, "sk runtime output extraction item"
        )
        oracle_set_id = _require_string(
            item["oracle_set_id"], "sk runtime output extraction oracle_set_id"
        )
        _reject_sk_runtime_output_extraction_nul(oracle_set_id)
        if not oracle_set_id:
            raise CliUsageError(
                "sk runtime output extraction oracle_set_id must be non-empty"
            )
        if oracle_set_id in seen:
            raise CliUsageError("duplicate sk runtime output extraction oracle_set_id")
        seen.add(oracle_set_id)
        source_raw = item["source"]
        if not isinstance(source_raw, dict):
            raise CliUsageError("sk runtime output extraction source must be an object")
        source = _require_exact_keys(
            source_raw, {"kind", "json_pointer"}, "sk runtime output extraction source"
        )
        kind = _require_string(
            source["kind"], "sk runtime output extraction source kind"
        )
        if kind not in {"stdout_json_pointer", "stderr_json_pointer"}:
            raise CliUsageError("unsupported sk runtime output extraction source kind")
        pointer = _require_string(
            source["json_pointer"],
            "sk runtime output extraction source json_pointer",
            allow_empty=True,
        )
        _reject_sk_runtime_output_extraction_nul(pointer)
        _validate_json_pointer(pointer)
        items.append(
            {
                "oracle_set_id": oracle_set_id,
                "source": {"kind": kind, "json_pointer": pointer},
            }
        )
    return items


def _validate_sk_runtime_output_extraction_binding(
    oracle_spec: dict[str, Any],
    items: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected_ids = [item["oracle_set_id"] for item in oracle_spec["oracle_specs"]]
    item_ids = [item["oracle_set_id"] for item in items]
    if set(item_ids) != set(expected_ids):
        raise CliUsageError("sk runtime output extraction set mismatch")
    return {item["oracle_set_id"]: item for item in items}


def _extract_sk_runtime_output_value(
    target_runtime: dict[str, Any], source: dict[str, Any]
) -> Any:
    command = target_runtime["commands"][0]
    channel = "stdout" if source["kind"] == "stdout_json_pointer" else "stderr"
    if command[f"{channel}_tail_truncated"] is True:
        raise CliUsageError("sk runtime output extraction source stream is truncated")
    try:
        payload = _parse_source_stream_json(command[f"{channel}_tail"])
    except CliUsageError as exc:
        raise CliUsageError(
            "sk runtime output extraction source stream is not valid JSON"
        ) from exc
    try:
        value = _resolve_json_pointer(payload, source["json_pointer"])
    except KeyError as exc:
        raise CliUsageError("sk runtime output extraction pointer unresolved") from exc
    if value is None:
        raise CliUsageError("sk runtime output extraction value must not be null")
    _require_json_finite(value, "sk runtime output extraction value")
    return value


def _sk_runtime_output_spec_from_captured(
    output_dir: Path,
    oracle_spec: dict[str, Any],
    extraction_items: list[dict[str, Any]],
) -> dict[str, Any]:
    target_runtime = _load_current_sk_target_runtime_validation(output_dir, oracle_spec)
    if target_runtime["status"] != "passed":
        raise CliUsageError("sk target runtime validation not passed")
    items_by_id = _validate_sk_runtime_output_extraction_binding(
        oracle_spec, extraction_items
    )
    actual_outputs: list[dict[str, Any]] = []
    for oracle_item in oracle_spec["oracle_specs"]:
        extraction = items_by_id[oracle_item["oracle_set_id"]]
        value = _extract_sk_runtime_output_value(target_runtime, extraction["source"])
        actual_outputs.append(
            {
                "oracle_set_id": oracle_item["oracle_set_id"],
                "actual_output": {"kind": "inline_json", "value": value},
            }
        )
    return {"actual_outputs": actual_outputs}


def cmd_extract_sk_runtime_outputs(args: argparse.Namespace) -> int:
    output_dir = Path(args.analysis_output_dir).resolve()
    if not output_dir.exists():
        raise CliUsageError(f"analysis output directory not found: {output_dir}")
    if not output_dir.is_dir():
        raise CliUsageError(f"analysis output path is not a directory: {output_dir}")

    _spec_path, _spec_bytes, spec = _read_sk_runtime_output_extraction_spec(
        Path(args.runtime_output_extraction_spec_json)
    )
    extraction_items = _normalize_sk_runtime_output_extraction_spec(spec)
    oracle_spec, _runtime_spec = _load_current_sk_correctness_oracle_spec(output_dir)
    runtime_output_spec = _sk_runtime_output_spec_from_captured(
        output_dir, oracle_spec, extraction_items
    )
    _write_json(
        output_dir / "operator-sk-runtime-output-spec.json", runtime_output_spec
    )
    return 0


def _sk_operator_correctness_verdict_check(
    name: str, status: str, reason: str, evidence: list[str] | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "evidence": evidence or [],
    }


def _sk_operator_correctness_verdict_result_manifest(
    output_dir: Path,
    status: str,
    verdict_items: list[dict[str, Any]],
    check_results: dict[str, dict[str, Any]],
    supported_next_actions: list[str],
) -> dict[str, Any]:
    correctness_verdict_checks = []
    for name in SK_OPERATOR_CORRECTNESS_VERDICT_CHECK_NAMES:
        correctness_verdict_checks.append(
            _require_mapping_value(check_results, name, "sk correctness verdict checks")
        )
    return {
        "status": status,
        "analysis_output_dir": str(output_dir.resolve()),
        "runtime_output_comparison_path": "operator-sk-runtime-output-comparison.json",
        "correctness_verdict_path": "operator-sk-correctness-verdict.json",
        "verdict_scope": "declared_oracle_and_captured_runtime_evidence",
        "verdict_items": verdict_items,
        "checks": correctness_verdict_checks,
        "supported_next_actions": supported_next_actions,
        "execution_boundary": SK_OPERATOR_CORRECTNESS_VERDICT_BOUNDARY,
    }


def _sk_operator_correctness_verdict_items(
    oracle_spec: dict[str, Any],
    comparison: dict[str, Any],
    actual_outputs_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    comparison_by_id = {}
    for item in comparison["comparisons"]:
        comparison_by_id[item["oracle_set_id"]] = item
    items: list[dict[str, Any]] = []
    for oracle_item in oracle_spec["oracle_specs"]:
        oracle_set_id = oracle_item["oracle_set_id"]
        comparison_item = _require_mapping_value(
            comparison_by_id, oracle_set_id, "runtime output comparison"
        )
        actual_output_item = _require_mapping_value(
            actual_outputs_by_id, oracle_set_id, "runtime output binding"
        )
        actual_output = actual_output_item["actual_output"]
        expected_output = oracle_item["expected_output"]
        comp_status = comparison_item["status"]
        if comp_status == "matched":
            item_status, item_reason = "passed", "operator_output_correct"
        elif comp_status == "mismatched":
            item_status, item_reason = "failed", "operator_output_mismatch"
        else:
            item_status = "blocked"
            item_reason = (
                comparison_item.get("reason") or "runtime_output_comparison_blocked"
            )
        items.append(
            {
                "oracle_set_id": oracle_set_id,
                "input_set_id": oracle_item["input_set_id"],
                "entry_name": oracle_item["entry_name"],
                "status": item_status,
                "reason": item_reason,
                "comparator": oracle_item["comparator"],
                "tolerance": oracle_item["tolerance"],
                "comparison_status": comparison_item["status"],
                "comparison_reason": comparison_item["reason"],
                "expected_output_kind": expected_output["kind"],
                "actual_output_kind": actual_output["kind"],
                "mismatch_path": comparison_item["mismatch_path"],
            }
        )
    return items


def _current_sk_operator_correctness_verdict_manifest(
    output_dir: Path, oracle_spec: dict[str, Any]
) -> dict[str, Any]:
    comparison, actual_outputs_by_id, _target_runtime = (
        _load_current_sk_runtime_output_comparison_context(output_dir, oracle_spec)
    )
    check_results: dict[str, dict[str, Any]] = {
        "sk_runtime_output_comparison_current": _sk_operator_correctness_verdict_check(
            "sk_runtime_output_comparison_current",
            "passed",
            "sk_runtime_output_comparison_current",
            ["operator-sk-runtime-output-comparison.json"],
        )
    }
    if comparison["status"] != "matched":
        block_reason = "sk_runtime_output_comparison_" + comparison["status"]
        check_results["sk_runtime_output_comparison_matched"] = (
            _sk_operator_correctness_verdict_check(
                "sk_runtime_output_comparison_matched",
                "blocked",
                block_reason,
                ["operator-sk-runtime-output-comparison.json"],
            )
        )
        check_results["operator_correctness_validated"] = (
            _sk_operator_correctness_verdict_check(
                "operator_correctness_validated",
                "blocked",
                block_reason,
                ["operator-sk-runtime-output-comparison.json"],
            )
        )
        check_results["handoff_boundary_open"] = _sk_operator_correctness_verdict_check(
            "handoff_boundary_open", "open", "handoff_not_completed"
        )
        return _sk_operator_correctness_verdict_result_manifest(
            output_dir,
            "blocked",
            [],
            check_results,
            list(comparison["supported_next_actions"]),
        )

    check_results["sk_runtime_output_comparison_matched"] = (
        _sk_operator_correctness_verdict_check(
            "sk_runtime_output_comparison_matched",
            "passed",
            "sk_runtime_output_comparison_matched",
            ["operator-sk-runtime-output-comparison.json"],
        )
    )
    check_results["operator_correctness_validated"] = (
        _sk_operator_correctness_verdict_check(
            "operator_correctness_validated",
            "passed",
            "operator_correctness_validated",
            ["operator-sk-runtime-output-comparison.json"],
        )
    )
    check_results["handoff_boundary_open"] = _sk_operator_correctness_verdict_check(
        "handoff_boundary_open", "open", "handoff_not_completed"
    )
    return _sk_operator_correctness_verdict_result_manifest(
        output_dir,
        "passed",
        _sk_operator_correctness_verdict_items(
            oracle_spec, comparison, actual_outputs_by_id
        ),
        check_results,
        [],
    )


def cmd_validate_sk_operator_correctness(args: argparse.Namespace) -> int:
    output_dir = Path(args.analysis_output_dir).resolve()
    if not output_dir.exists():
        raise CliUsageError(f"analysis output directory not found: {output_dir}")
    if not output_dir.is_dir():
        raise CliUsageError(f"analysis output path is not a directory: {output_dir}")

    oracle_spec, _runtime_spec = _load_current_sk_correctness_oracle_spec(output_dir)
    result = _current_sk_operator_correctness_verdict_manifest(output_dir, oracle_spec)
    _write_json(output_dir / "operator-sk-correctness-verdict.json", result)
    return 0 if result["status"] == "passed" else 1


def cmd_compare_sk_runtime_outputs(args: argparse.Namespace) -> int:
    output_dir = Path(args.analysis_output_dir).resolve()
    if not output_dir.exists():
        raise CliUsageError(f"analysis output directory not found: {output_dir}")
    if not output_dir.is_dir():
        raise CliUsageError(f"analysis output path is not a directory: {output_dir}")

    spec_path, spec_bytes, spec = _read_runtime_output_spec(
        Path(args.runtime_output_spec_json)
    )
    try:
        oracle_spec, _runtime_spec = _load_current_sk_correctness_oracle_spec(
            output_dir
        )
        actual_outputs = _normalize_runtime_output_payload(spec, oracle_spec)
        result = _summarize_sk_runtime_output_comparison(
            output_dir, oracle_spec, spec_path, spec_bytes, actual_outputs
        )
    except CliUsageError as exc:
        try:
            result = _summarize_standalone_bind_target_comparison(
                output_dir, spec_path, spec_bytes, spec
            )
        except CliUsageError as fallback_exc:
            raise exc from fallback_exc
    _write_json(output_dir / "operator-sk-runtime-output-comparison.json", result)
    return 0 if result["status"] == "matched" else 1


def _validate_json_pointer(pointer: str) -> None:
    if pointer == "":
        return
    if not pointer.startswith("/"):
        raise CliUsageError("runtime output provenance json_pointer is invalid")
    for segment in pointer.split("/")[1:]:
        index = 0
        while index < len(segment):
            if segment[index] == "~":
                if index + 1 >= len(segment) or segment[index + 1] not in {"0", "1"}:
                    raise CliUsageError(
                        "runtime output provenance json_pointer is invalid"
                    )
                index += 2
            else:
                index += 1


def _decode_json_pointer_segment(segment: str) -> str:
    return segment.replace("~1", "/").replace("~0", "~")


def _resolve_json_pointer(payload: Any, pointer: str) -> Any:
    if pointer == "":
        return payload
    current = payload
    for raw_segment in pointer.split("/")[1:]:
        segment = _decode_json_pointer_segment(raw_segment)
        if isinstance(current, dict):
            if segment not in current:
                raise KeyError(segment)
            current = current[segment]
        elif isinstance(current, list):
            if not segment.isdigit():
                raise KeyError(segment)
            index = int(segment)
            if index >= len(current):
                raise KeyError(segment)
            current = current[index]
        else:
            raise KeyError(segment)
    return current


def _parse_source_stream_json(text: str) -> Any:
    try:
        payload = json.loads(text, parse_constant=_reject_json_constant)
        _require_json_finite(payload, "runtime output provenance source stream")
        return payload
    except (ValueError, CliUsageError) as exc:
        raise CliUsageError("runtime output source stream is not valid JSON") from exc


def _sample_lib():
    import sys as _sys

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in _sys.path:
        _sys.path.insert(0, str(script_dir))
    import sk_sample_gen_lib

    return sk_sample_gen_lib


def _parse_shape(text: str) -> list[int]:
    if not text:
        return []
    dims: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if not token.isdigit():
            raise CliUsageError(
                f"shape dimension must be a non-negative integer, got {token!r}"
            )
        dims.append(int(token))
    return dims


def cmd_auto_construct_runtime_input_values(args: argparse.Namespace) -> int:
    lib = _sample_lib()
    output_dir = Path(args.output_dir).resolve()
    spec_path = output_dir / "operator-sk-runtime-input-spec.json"
    if not spec_path.exists():
        raise CliUsageError(f"runtime input spec not found: {spec_path}")
    runtime_input_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    shape = _parse_shape(args.shape)
    closed_spec = lib.build_input_values_spec(
        runtime_input_spec, shape=shape, dtype=args.dtype, fill=args.fill
    )
    out_path = output_dir / "operator-sk-auto-input-values-spec.json"
    out_path.write_text(json.dumps(closed_spec, indent=2), encoding="utf-8")
    _emit(
        json.dumps(
            {"written": out_path.name, "input_sets": len(closed_spec["input_values"])}
        )
    )
    return 0


def cmd_auto_build_correctness_oracle(args: argparse.Namespace) -> int:
    lib = _sample_lib()
    output_dir = Path(args.output_dir).resolve()
    values_path = output_dir / "operator-sk-runtime-input-values.json"
    spec_path = output_dir / "operator-sk-runtime-input-spec.json"
    if not values_path.exists():
        raise CliUsageError(f"runtime input values not found: {values_path}")
    if not spec_path.exists():
        raise CliUsageError(f"runtime input spec not found: {spec_path}")
    input_values_artifact = json.loads(values_path.read_text(encoding="utf-8"))
    runtime_input_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if args.oracle_source == "bind-target-on-wheel":
        oracle_spec = lib.build_bind_target_oracle_spec(
            input_values_artifact, runtime_input_spec
        )
    else:
        reference_dir = Path(__file__).resolve().parent.parent / "reference_impls"
        registry = lib.ReferenceImplRegistry.load(reference_dir)
        oracle_spec, missing = lib.build_oracle_spec(
            input_values_artifact, runtime_input_spec, registry
        )
        if missing:
            raise CliUsageError(
                "no reference implementation for kernel entries behind input sets: "
                + ", ".join(missing)
                + f"; add reference_impls/<entry>.py (known: {registry.entry_names()})"
            )
    out_path = output_dir / "operator-sk-auto-oracle-spec.json"
    out_path.write_text(json.dumps(oracle_spec, indent=2), encoding="utf-8")
    _emit(
        json.dumps(
            {
                "written": out_path.name,
                "oracle_sets": len(oracle_spec["oracle_specs"]),
                "oracle_source": args.oracle_source,
            }
        )
    )
    return 0


def cmd_generate_runner_script(args: argparse.Namespace) -> int:
    lib = _sample_lib()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runner_rel = "operator-sample-runner.py"
    values_rel = "operator-sk-runtime-input-values.json"
    (output_dir / runner_rel).write_text(
        lib.render_runner_script(args.oracle_source), encoding="utf-8"
    )
    command_spec = lib.build_target_runtime_command_spec(runner_rel, values_rel)
    (output_dir / "operator-sk-target-runtime-command-spec.json").write_text(
        json.dumps(command_spec, indent=2), encoding="utf-8"
    )
    _emit(
        json.dumps(
            {
                "runner": runner_rel,
                "command_spec": "operator-sk-target-runtime-command-spec.json",
            }
        )
    )
    return 0


def cmd_build_single_op_verification_contract(args: argparse.Namespace) -> int:
    lib = _sample_lib()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_contract = _load_json_artifact(
        Path(args.runtime_contract_json), "operator-runtime-contract.json"
    )
    try:
        contract = lib.build_single_op_verification_contract(runtime_contract)
    except ValueError as exc:
        raise CliUsageError(str(exc)) from exc
    out_path = output_dir / "operator-single-op-verification-contract.json"
    out_path.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _emit(
        json.dumps(
            {
                "written": out_path.name,
                "status": contract["status"],
                "requires_wheel": contract["requires_wheel"],
            }
        )
    )
    return 0 if contract["status"] == "available" else 1


def cmd_collect_network_sample_contract(args: argparse.Namespace) -> int:
    lib = _sample_lib()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_contract = _load_json_artifact(
        Path(args.network_contract_json), "operator network sample contract"
    )
    try:
        contract = lib.normalize_network_sample_contract(raw_contract)
    except ValueError as exc:
        raise CliUsageError(str(exc)) from exc
    contract_path = output_dir / "operator-network-sample-contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    try:
        fusion_expectation = lib.build_network_fusion_expectation(contract)
    except ValueError as exc:
        raise CliUsageError(str(exc)) from exc
    fusion_path = output_dir / "operator-network-fusion-expectation.json"
    fusion_path.write_text(
        json.dumps(fusion_expectation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _emit(
        json.dumps(
            {
                "written": contract_path.name,
                "fusion_expectation": fusion_path.name,
                "status": contract["status"],
            }
        )
    )
    return 0 if contract["status"] == "available" else 1


def cmd_generate_network_runner_script(args: argparse.Namespace) -> int:
    lib = _sample_lib()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "operator-network-sample-contract.json"
    if not contract_path.exists():
        raise CliUsageError(f"network sample contract not found: {contract_path}")
    try:
        contract = lib.normalize_network_sample_contract(
            _load_json_artifact(contract_path, "operator-network-sample-contract.json")
        )
    except ValueError as exc:
        raise CliUsageError(str(exc)) from exc
    if contract["status"] != "available":
        raise CliUsageError("network sample contract is not available")
    runner_rel = "operator-network-sample-runner.py"
    contract_rel = "operator-network-sample-contract.json"
    (output_dir / runner_rel).write_text(
        lib.render_network_runner_script(), encoding="utf-8"
    )
    command_spec = lib.build_network_target_runtime_command_spec(
        runner_rel, contract_rel
    )
    (output_dir / "operator-network-target-runtime-command-spec.json").write_text(
        json.dumps(command_spec, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _emit(
        json.dumps(
            {
                "runner": runner_rel,
                "command_spec": "operator-network-target-runtime-command-spec.json",
            }
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Top-level CLI for sk-operator-sample-gen"
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    sk_runtime_inputs = subparsers.add_parser(
        "collect-runtime-input-spec",
        help="Define SK runtime input placeholders after a current passed SK source version validation",
    )
    sk_runtime_inputs.add_argument(
        "analysis_output_dir",
        help="Output directory produced by validate-sk-source-version",
    )
    sk_runtime_inputs.set_defaults(func=cmd_collect_runtime_input_spec)

    sk_runtime_input_values = subparsers.add_parser(
        "provide-sk-runtime-input-values",
        help="Declare SK runtime input values after exact-current SK runtime input specs",
    )
    sk_runtime_input_values.add_argument(
        "analysis_output_dir",
        help="Output directory produced by collect-runtime-input-spec",
    )
    sk_runtime_input_values.add_argument(
        "input_value_spec_json", help="JSON file declaring SK runtime input values"
    )
    sk_runtime_input_values.set_defaults(func=cmd_provide_sk_runtime_input_values)

    sk_oracle_spec = subparsers.add_parser(
        "collect-correctness-oracle-spec",
        help="Collect non-executing SK correctness oracle specs after SK runtime input specs",
    )
    sk_oracle_spec.add_argument(
        "analysis_output_dir",
        help="Output directory produced by collect-runtime-input-spec",
    )
    sk_oracle_spec.add_argument(
        "oracle_spec_json", help="JSON file declaring SK correctness oracle specs"
    )
    sk_oracle_spec.set_defaults(func=cmd_collect_correctness_oracle_spec)

    sk_target_runtime = subparsers.add_parser(
        "run-sk-target-runtime-validation",
        help="Run a controlled SK target runtime command after SK correctness oracle specs",
    )
    sk_target_runtime.add_argument(
        "analysis_output_dir",
        help="Output directory produced by collect-correctness-oracle-spec",
    )
    sk_target_runtime.add_argument(
        "runtime_command_spec_json",
        help="JSON file declaring the SK target runtime command",
    )
    sk_target_runtime.set_defaults(func=cmd_run_sk_target_runtime_validation)

    sk_runtime_output_extraction = subparsers.add_parser(
        "extract-sk-runtime-outputs",
        help="Extract actual SK runtime outputs from captured target runtime stdout/stderr JSON",
    )
    sk_runtime_output_extraction.add_argument(
        "analysis_output_dir",
        help="Output directory with passed SK target runtime validation and SK correctness oracle spec artifacts",
    )
    sk_runtime_output_extraction.add_argument(
        "runtime_output_extraction_spec_json",
        help="JSON file declaring where actual SK runtime outputs appear in captured stdout/stderr",
    )
    sk_runtime_output_extraction.set_defaults(func=cmd_extract_sk_runtime_outputs)

    sk_runtime_output_comparison = subparsers.add_parser(
        "compare-sk-runtime-outputs",
        help="Compare declared SK runtime outputs against SK correctness oracle expected outputs",
    )
    sk_runtime_output_comparison.add_argument(
        "analysis_output_dir",
        help="Output directory with SK target runtime validation and SK correctness oracle spec artifacts",
    )
    sk_runtime_output_comparison.add_argument(
        "runtime_output_spec_json",
        help="JSON file declaring actual SK runtime outputs",
    )
    sk_runtime_output_comparison.set_defaults(func=cmd_compare_sk_runtime_outputs)

    sk_operator_correctness = subparsers.add_parser(
        "validate-sk-operator-correctness",
        help="Validate SK operator correctness against exact-current declared oracle and captured runtime evidence",
    )
    sk_operator_correctness.add_argument(
        "analysis_output_dir",
        help="Output directory with SK correctness verdict inputs",
    )
    sk_operator_correctness.set_defaults(func=cmd_validate_sk_operator_correctness)

    auto_inputs = subparsers.add_parser(
        "auto-construct-runtime-input-values",
        help="Synthesise a closed input-values spec (zero-filled) from operator-sk-runtime-input-spec.json",
    )
    auto_inputs.add_argument(
        "output_dir",
        help="Output directory containing operator-sk-runtime-input-spec.json",
    )
    auto_inputs.add_argument(
        "--shape",
        default="16",
        help="Tensor shape for GM_ADDR parameters, comma-separated (default '16')",
    )
    auto_inputs.add_argument(
        "--dtype",
        default="float16",
        help="Tensor dtype for GM_ADDR parameters (default 'float16')",
    )
    auto_inputs.add_argument(
        "--fill",
        default="zero",
        choices=["zero"],
        help="Fill mode (only 'zero' supported currently)",
    )
    auto_inputs.set_defaults(func=cmd_auto_construct_runtime_input_values)

    auto_oracle = subparsers.add_parser(
        "auto-build-correctness-oracle",
        help="Compute expected outputs via reference_impls/<entry>.py and emit a closed oracle spec",
    )
    auto_oracle.add_argument(
        "output_dir",
        help="Output directory containing the canonical operator-sk-runtime-input-values.json",
    )
    auto_oracle.add_argument(
        "--oracle-source",
        default="bind-target-on-wheel",
        choices=["bind-target-on-wheel", "reference-impls-numpy"],
        help="Oracle source mode (default: bind-target-on-wheel)",
    )
    auto_oracle.set_defaults(func=cmd_auto_build_correctness_oracle)

    gen_runner = subparsers.add_parser(
        "generate-runner-script",
        help="Emit operator-sample-runner.py + operator-sk-target-runtime-command-spec.json",
    )
    gen_runner.add_argument("output_dir", help="Output directory")
    gen_runner.add_argument(
        "--oracle-source",
        default="bind-target-on-wheel",
        choices=["bind-target-on-wheel", "reference-impls-numpy"],
        help="Runner/oracle source mode (default: bind-target-on-wheel)",
    )
    gen_runner.set_defaults(func=cmd_generate_runner_script)

    single_op_contract = subparsers.add_parser(
        "build-single-op-verification-contract",
        help="Build a single-op standalone differential verification contract from operator-runtime-contract.json",
    )
    single_op_contract.add_argument(
        "output_dir",
        help="Output directory for operator-single-op-verification-contract.json",
    )
    single_op_contract.add_argument(
        "runtime_contract_json",
        help="operator-runtime-contract.json with explicit comparable outputs",
    )
    single_op_contract.set_defaults(func=cmd_build_single_op_verification_contract)

    network_contract = subparsers.add_parser(
        "collect-network-sample-contract",
        help="Collect and canonicalise a network-level wheel verification contract",
    )
    network_contract.add_argument(
        "output_dir", help="Output directory for network sample artifacts"
    )
    network_contract.add_argument(
        "network_contract_json",
        help="JSON file declaring network topology, wheel package, outputs, and fusion expectation",
    )
    network_contract.set_defaults(func=cmd_collect_network_sample_contract)

    network_runner = subparsers.add_parser(
        "generate-network-runner-script",
        help="Emit a network-level runner and command spec from operator-network-sample-contract.json",
    )
    network_runner.add_argument(
        "output_dir",
        help="Output directory containing operator-network-sample-contract.json",
    )
    network_runner.set_defaults(func=cmd_generate_network_runner_script)

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
