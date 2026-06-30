# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""SK Operator Codegen CLI -- operator asset intake and SK source scaffold generation."""

from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

WORKFLOW_STEPS = [
    "operator-intake",
    "asset-maturity-classification",
    "sk-readiness-check",
    "build-contract-check",
    "test-contract-check",
    "consistency-validation-plan",
    "package-and-handoff-plan",
]
DEPENDENCY_HINTS = [
    "skills_root/sk-operator-codegen/references/workflow.md",
    "skills_root/sk-operator-codegen/references/sk-adaptation-cookbook.md",
    "skills_root/sk-operator-codegen/templates/",
]
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
_CPP_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
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
        "description": "Define the operator entry contract before build or test scaffolds can target a stable callable unit.",
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
SK_BINDING_SCAFFOLD_BOUNDARY = [
    "original asset not modified",
    "build not executed",
    "runtime not executed",
    "operator correctness not validated",
    "customer handoff not completed",
]
SK_BINDING_SCAFFOLD_DIR = "operator-sk-binding-scaffold"
SK_SOURCE_SCAFFOLD_BOUNDARY = [
    "original asset not modified",
    "build not executed",
    "runtime not executed",
    "operator correctness not validated",
    "customer handoff not completed",
]
SK_SOURCE_SCAFFOLD_DIR = "operator-sk-source-scaffold"
SUPPORT_DIR_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class CliUsageError(ValueError):
    """User-facing CLI usage error handled by argparse."""


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


def _read_pybind_io_contract(path_text: str) -> dict[str, dict[str, Any]]:
    if not path_text:
        return {}
    contract_path = Path(path_text).expanduser().resolve()
    if not contract_path.is_file():
        raise CliUsageError(f"IO contract file not found: {contract_path}")
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CliUsageError(
            f"IO contract must be valid JSON: {contract_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CliUsageError("IO contract must be a JSON object")
    try:
        schema_version = int(payload.get("schema_version", 1))
    except (TypeError, ValueError) as exc:
        raise CliUsageError("IO contract schema_version must be an integer") from exc
    if schema_version != 1:
        raise CliUsageError(f"unsupported IO contract schema_version: {schema_version}")
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise CliUsageError("IO contract must contain an 'entries' object")

    def _name_list(value: object, field_name: str, entry_name: str) -> list[str]:
        if value in (None, ""):
            return []
        if not isinstance(value, list):
            raise CliUsageError(
                f"IO contract entry {entry_name!r} field {field_name!r} must be a list"
            )
        names = [str(item).strip() for item in value]
        if any(not item for item in names):
            raise CliUsageError(
                f"IO contract entry {entry_name!r} field {field_name!r} contains an empty name"
            )
        return names

    def _bool_value(value: object, field_name: str, entry_name: str) -> bool:
        if isinstance(value, bool):
            return value
        raise CliUsageError(
            f"IO contract entry {entry_name!r} field {field_name!r} must be a boolean"
        )

    def _string_list(value: object, field_name: str, entry_name: str) -> list[str]:
        if value in (None, ""):
            return []
        if not isinstance(value, list):
            raise CliUsageError(
                f"IO contract entry {entry_name!r} field {field_name!r} must be a list"
            )
        rendered = [str(item).strip() for item in value]
        if any(not item for item in rendered):
            raise CliUsageError(
                f"IO contract entry {entry_name!r} field {field_name!r} contains an empty string"
            )
        if any("\x00" in item for item in rendered):
            raise CliUsageError(
                f"IO contract entry {entry_name!r} field {field_name!r} contains a NUL byte"
            )
        return rendered

    def _param_map(
        value: object, field_name: str, entry_name: str
    ) -> dict[str, dict[str, Any]]:
        if value in (None, ""):
            return {}
        if not isinstance(value, dict):
            raise CliUsageError(
                f"IO contract entry {entry_name!r} field {field_name!r} must be an object"
            )
        normalized: dict[str, dict[str, Any]] = {}
        for raw_name, raw_spec in value.items():
            param_name = str(raw_name).strip()
            if not param_name:
                raise CliUsageError(
                    f"IO contract entry {entry_name!r} field {field_name!r} contains an empty parameter name"
                )
            if isinstance(raw_spec, str):
                raw_spec = {"kind": raw_spec}
            if not isinstance(raw_spec, dict):
                raise CliUsageError(
                    f"IO contract entry {entry_name!r} field {field_name!r} parameter {param_name!r} "
                    "must be a string or object"
                )
            kind = str(raw_spec.get("kind") or "").strip()
            if not kind:
                raise CliUsageError(
                    f"IO contract entry {entry_name!r} field {field_name!r} parameter {param_name!r} "
                    "must declare kind"
                )
            if kind not in {"tensor", "tensor_list", "scalar", "host_struct"}:
                raise CliUsageError(
                    f"IO contract entry {entry_name!r} field {field_name!r} parameter {param_name!r} "
                    f"has unsupported kind {kind!r}"
                )
            if raw_spec.get("nullable") is not None and not isinstance(
                raw_spec.get("nullable"), bool
            ):
                raise CliUsageError(
                    f"IO contract entry {entry_name!r} field {field_name!r} parameter {param_name!r} "
                    "nullable must be a boolean"
                )
            normalized[param_name] = {
                key: value for key, value in raw_spec.items() if key != "kind"
            }
            normalized[param_name]["kind"] = kind
        return normalized

    def _launch_contract(value: object, entry_name: str) -> dict[str, Any]:
        if value in (None, ""):
            return {}
        if not isinstance(value, dict):
            raise CliUsageError(
                f"IO contract entry {entry_name!r} field 'launch' must be an object"
            )
        normalized: dict[str, Any] = {}
        if "block_dim" in value:
            try:
                block_dim = int(value["block_dim"])
            except (TypeError, ValueError) as exc:
                raise CliUsageError(
                    f"IO contract entry {entry_name!r} launch.block_dim must be a positive integer"
                ) from exc
            if block_dim <= 0:
                raise CliUsageError(
                    f"IO contract entry {entry_name!r} launch.block_dim must be a positive integer"
                )
            normalized["block_dim"] = block_dim
        return normalized

    def _compile_contract(
        value: object, raw_spec: dict[str, Any], entry_name: str
    ) -> dict[str, Any]:
        if value in (None, ""):
            value = {}
        if not isinstance(value, dict):
            raise CliUsageError(
                f"IO contract entry {entry_name!r} field 'compile' must be an object"
            )
        raw_defines = (
            value["defines"] if "defines" in value else raw_spec.get("compile_defines")
        )
        raw_options = (
            value["options"] if "options" in value else raw_spec.get("compile_options")
        )
        defines = []
        for define in _string_list(raw_defines, "compile.defines", entry_name):
            defines.append(define if define.startswith("-D") else f"-D{define}")
        options = _string_list(raw_options, "compile.options", entry_name)
        return {"defines": defines, "options": options}

    def _migration_contract(value: object, entry_name: str) -> dict[str, Any]:
        if value in (None, ""):
            return {}
        if not isinstance(value, dict):
            raise CliUsageError(
                f"IO contract entry {entry_name!r} field 'migration' must be an object"
            )
        normalized: dict[str, Any] = {}
        if "confirm_legacy_mix_semantics" in value:
            normalized["confirm_legacy_mix_semantics"] = _bool_value(
                value["confirm_legacy_mix_semantics"],
                "migration.confirm_legacy_mix_semantics",
                entry_name,
            )
        return normalized

    def _runtime_wrapper_contract(value: object, entry_name: str) -> dict[str, Any]:
        if value in (None, ""):
            return {}
        if not isinstance(value, dict):
            raise CliUsageError(
                f"IO contract entry {entry_name!r} field 'runtime_wrapper' must be an object"
            )
        source = str(value.get("source") or "").strip()
        wrapper_entry = str(value.get("entry") or "").strip()
        if not source:
            raise CliUsageError(
                f"IO contract entry {entry_name!r} runtime_wrapper.source must be non-empty"
            )
        if "\x00" in source:
            raise CliUsageError(
                f"IO contract entry {entry_name!r} runtime_wrapper.source contains a NUL byte"
            )
        source_path = Path(source)
        if source_path.is_absolute() or ".." in source_path.parts:
            raise CliUsageError(
                f"IO contract entry {entry_name!r} runtime_wrapper.source must be a relative path inside the asset"
            )
        if not wrapper_entry or not _CPP_IDENTIFIER_RE.match(wrapper_entry):
            raise CliUsageError(
                f"IO contract entry {entry_name!r} runtime_wrapper.entry must be a C++ identifier"
            )
        normalized: dict[str, Any] = {
            "source": source_path.as_posix(),
            "entry": wrapper_entry,
        }
        strategy = str(value.get("tensor_list_descriptor_strategy") or "").strip()
        if strategy:
            if strategy != "prepared_workspace_tail":
                raise CliUsageError(
                    f"IO contract entry {entry_name!r} runtime_wrapper.tensor_list_descriptor_strategy "
                    f"has unsupported value {strategy!r}"
                )
            normalized["tensor_list_descriptor_strategy"] = strategy
            prepare_entry = str(value.get("prepare_entry") or "").strip()
            if not prepare_entry:
                raise CliUsageError(
                    f"IO contract entry {entry_name!r} runtime_wrapper.prepare_entry must be non-empty "
                    "when tensor_list_descriptor_strategy is declared"
                )
            normalized["prepare_entry"] = prepare_entry
            try:
                descriptor_bytes = int(value.get("descriptor_bytes"))
            except (TypeError, ValueError) as exc:
                raise CliUsageError(
                    f"IO contract entry {entry_name!r} runtime_wrapper.descriptor_bytes must be a positive integer"
                ) from exc
            if descriptor_bytes <= 0:
                raise CliUsageError(
                    f"IO contract entry {entry_name!r} runtime_wrapper.descriptor_bytes must be a positive integer"
                )
            normalized["descriptor_bytes"] = descriptor_bytes
            normalized["descriptor_order"] = _name_list(
                value.get("descriptor_order"),
                "runtime_wrapper.descriptor_order",
                entry_name,
            )
        return normalized

    contract: dict[str, dict[str, Any]] = {}
    for raw_name, raw_spec in entries.items():
        entry_name = str(raw_name).strip()
        if not entry_name:
            raise CliUsageError("IO contract entry name must not be empty")
        if isinstance(raw_spec, str):
            return_tensor = raw_spec.strip()
            input_tensors: list[str] = []
            output_tensors: list[str] = [return_tensor] if return_tensor else []
        elif isinstance(raw_spec, dict):
            return_tensor = str(
                raw_spec.get("pybind_return_tensor")
                or raw_spec.get("return_tensor")
                or ""
            ).strip()
            raw_inputs = (
                raw_spec["inputs"]
                if "inputs" in raw_spec
                else raw_spec.get("input_tensors")
            )
            raw_outputs = (
                raw_spec["outputs"]
                if "outputs" in raw_spec
                else raw_spec.get("output_tensors")
            )
            raw_workspaces = (
                raw_spec["workspaces"]
                if "workspaces" in raw_spec
                else raw_spec.get("workspace_tensors")
            )
            input_tensors = _name_list(raw_inputs, "inputs", entry_name)
            output_tensors = _name_list(raw_outputs, "outputs", entry_name)
            workspace_tensors = _name_list(raw_workspaces, "workspaces", entry_name)
            raw_params = (
                raw_spec["parameters"]
                if "parameters" in raw_spec
                else raw_spec.get("params")
            )
            parameters = _param_map(raw_params, "parameters", entry_name)
            launch = _launch_contract(raw_spec.get("launch"), entry_name)
            compile_contract = _compile_contract(
                raw_spec.get("compile"), raw_spec, entry_name
            )
            migration_contract = _migration_contract(
                raw_spec.get("migration"), entry_name
            )
            runtime_wrapper_contract = _runtime_wrapper_contract(
                raw_spec.get("runtime_wrapper"), entry_name
            )
            public_entry_name = str(
                raw_spec.get("public_entry_name") or raw_spec.get("entry_name") or ""
            ).strip()
            source_entry_name = str(raw_spec.get("source_entry_name") or "").strip()
            bind_target = str(raw_spec.get("bind_target") or "").strip()
            if not return_tensor and len(output_tensors) == 1:
                return_tensor = output_tensors[0]
        else:
            raise CliUsageError(
                f"IO contract entry {entry_name!r} must be a string or object"
            )
        if not return_tensor:
            raise CliUsageError(
                f"IO contract entry {entry_name!r} must declare pybind_return_tensor "
                "or exactly one output tensor"
            )
        if output_tensors and return_tensor not in output_tensors:
            raise CliUsageError(
                f"IO contract entry {entry_name!r} pybind_return_tensor {return_tensor!r} "
                "must be listed in outputs"
            )
        contract[entry_name] = {
            "pybind_return_tensor": return_tensor,
            "input_tensors": input_tensors,
            "output_tensors": output_tensors or [return_tensor],
            "workspace_tensors": workspace_tensors
            if isinstance(raw_spec, dict)
            else [],
            "parameters": parameters if isinstance(raw_spec, dict) else {},
            "launch": launch if isinstance(raw_spec, dict) else {},
            "compile": compile_contract
            if isinstance(raw_spec, dict)
            else {"defines": [], "options": []},
            "migration": migration_contract if isinstance(raw_spec, dict) else {},
            "runtime_wrapper": runtime_wrapper_contract
            if isinstance(raw_spec, dict)
            else {},
            "public_entry_name": public_entry_name
            if isinstance(raw_spec, dict)
            else "",
            "source_entry_name": source_entry_name
            if isinstance(raw_spec, dict)
            else "",
            "bind_target": bind_target if isinstance(raw_spec, dict) else "",
            "contract_path": str(contract_path),
        }
    return contract


def _is_tensor_c_type(c_type: str) -> bool:
    return "GM_ADDR" in c_type or "__gm__" in c_type or "*" in c_type


def _is_scalar_c_type(c_type: str) -> bool:
    scalar_markers = (
        "int",
        "uint",
        "float",
        "double",
        "half",
        "bool",
        "size_t",
    )
    return not _is_tensor_c_type(c_type) and any(
        marker in c_type for marker in scalar_markers
    )


def _entry_param_kinds(entry: dict[str, Any]) -> dict[str, str]:
    kinds: dict[str, str] = {}
    for param in entry.get("parameters", []):
        name = str(param.get("name") or "")
        c_type = re.sub(r"\s+", " ", str(param.get("c_type", ""))).strip()
        if not name:
            continue
        if _is_tensor_c_type(c_type):
            kinds[name] = "tensor"
        elif _is_scalar_c_type(c_type):
            kinds[name] = "scalar"
        else:
            kinds[name] = "host_struct"
    return kinds


def _is_declared_kind_compatible(
    detected_kind: str | None, declared_kind: str | None
) -> bool:
    if detected_kind == declared_kind:
        return True
    if detected_kind == "tensor" and declared_kind in {"tensor", "tensor_list"}:
        return True
    return False


def _io_contract_runtime_tensor_param_names(entry: dict[str, Any]) -> list[str]:
    declared = entry.get("io_contract", {}).get("parameters", {}) or {}
    names: list[str] = []
    for name, kind in _entry_param_kinds(entry).items():
        if kind != "tensor":
            continue
        declared_kind = (
            declared.get(name, {}).get("kind")
            if isinstance(declared.get(name), dict)
            else ""
        )
        if declared_kind == "tensor_list":
            continue
        names.append(name)
    return names


def _io_contract_tensor_param_names(entry: dict[str, Any]) -> list[str]:
    return [
        name for name, kind in _entry_param_kinds(entry).items() if kind == "tensor"
    ]


def _io_contract_host_struct_param_names(entry: dict[str, Any]) -> list[str]:
    return [
        name
        for name, kind in _entry_param_kinds(entry).items()
        if kind == "host_struct"
    ]


def _entry_contract_bind_target(entry: dict[str, Any]) -> str:
    io_contract = entry.get("io_contract", {})
    if isinstance(io_contract, dict):
        return str(io_contract.get("bind_target") or "").strip()
    return ""


def _resolve_runtime_wrapper_source(
    asset_path: Path,
    runtime_wrapper: dict[str, Any],
    extra_roots: list[Path] | None = None,
) -> tuple[Path, Path]:
    source = str(runtime_wrapper.get("source") or "").strip()
    asset_root = asset_path if asset_path.is_dir() else asset_path.parent
    candidate_roots = [asset_root, *(extra_roots or [])]
    for root in candidate_roots:
        resolved_root = root.resolve()
        candidate = (resolved_root / source).resolve()
        if _is_same_or_child_path(candidate, resolved_root) and candidate.is_file():
            return candidate, resolved_root
    return (asset_root / source).resolve(), asset_root.resolve()


def _runtime_wrapper_entry_exists(source_text: str, wrapper_entry: str) -> bool:
    return bool(re.search(rf"\b{re.escape(wrapper_entry)}\s*\(", source_text))


def _entry_export_key(entry: dict[str, Any]) -> str:
    return str(entry.get("entry_name") or entry.get("public_entry_name") or "").strip()


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
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("assert "):
            continue
        expression = stripped[len("assert ") :].strip()
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
    if build_files and test_files and package_files and doc_files:
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


def _collect_asset_summary(asset_path: Path) -> dict[str, Any]:
    if asset_path.is_file():
        files = [asset_path]
        base_dir = asset_path.parent
    else:
        base_dir = asset_path
        files = sorted(
            (path for path in asset_path.rglob("*") if path.is_file()),
            key=lambda path: _relative(path, base_dir),
        )
    suffix_count: dict[str, int] = {}
    for path in files:
        suffix = path.suffix or "<no-suffix>"
        suffix_count[suffix] = suffix_count.get(suffix, 0) + 1
    return {
        "asset_path": str(asset_path.resolve()),
        "file_count": len(files),
        "suffix_count": dict(sorted(suffix_count.items())),
        "sample_files": [str(path.relative_to(base_dir)) for path in files[:20]],
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


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Operator Delivery Report",
        "",
        f"- asset path: `{summary['asset']['asset_path']}`",
        f"- file count: `{summary['asset']['file_count']}`",
        f"- asset level: `{summary['manifest']['asset_level']}`",
        f"- archetype: `{summary['manifest']['archetype']['name']}`",
        f"- test quality: `{summary['manifest']['test_quality']['level']}`",
        f"- build quality: `{summary['manifest']['build_quality']['level']}`",
        f"- delivery phase: `{summary['manifest']['delivery_plan']['phase']}` (planning only; build/test not executed)",
        f"- scaffold plan: `{summary['manifest']['scaffold_plan']['status']}` "
        f"({len(summary['manifest']['scaffold_plan']['items'])} items)",
        "- execution plan: `operator-delivery-execution-plan.md`",
        f"- base mode: `{summary['execution']['base_mode']}`",
        f"- ai requested: `{summary['execution']['ai_requested']}`",
        "",
        "## Asset Manifest",
        "",
        f"- source files: `{len(summary['manifest']['source_files'])}`",
        f"- kernel entries: `{len(summary['manifest']['kernel_entries'])}`",
        f"- build entries: `{len(summary['manifest']['build_system'])}`",
        f"- test entries: `{len(summary['manifest']['test_system'])}`",
        f"- package entries: `{len(summary['manifest']['package_system'])}`",
        "",
        "## Missing Contracts",
        "",
    ]
    if summary["manifest"]["missing_contracts"]:
        for item in summary["manifest"]["missing_contracts"]:
            lines.append(f"- `{item}`")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Supported Next Actions",
            "",
        ]
    )
    for item in summary["manifest"]["supported_next_actions"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Workflow Plan",
            "",
        ]
    )
    for step in summary["workflow_steps"]:
        lines.append(f"- `{step}`")
    lines.extend(
        [
            "",
            "## Dependency Hints",
            "",
        ]
    )
    for item in summary["dependency_hints"]:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "## Current Note",
            "",
            "- 当前 CLI 负责资产扫描、分层、缺失合同和交付计划输出。",
            "- 完整 codegen、编译执行、样例执行和打包自动化仍属于后续执行期子任务。",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_execution_plan(summary: dict[str, Any]) -> str:
    manifest = summary["manifest"]
    delivery_plan = manifest["delivery_plan"]
    lines = [
        "# Operator Delivery Execution Plan",
        "",
        f"- delivery phase: `{delivery_plan['phase']}` (planning only; build/test not executed)",
        f"- asset level: `{manifest['asset_level']}`",
        f"- build quality: `{manifest['build_quality']['level']}`",
        f"- test quality: `{manifest['test_quality']['level']}`",
        f"- scaffold plan: `{manifest['scaffold_plan']['status']}` "
        f"({len(manifest['scaffold_plan']['items'])} items)",
        "",
        "## Ordered Actions",
        "",
    ]
    if delivery_plan["actions"]:
        for action in delivery_plan["actions"]:
            lines.append(
                f"- [{action['priority']}] {action['kind']}/{action['target']} "
                f"({action['reason']}): {action['next']}"
            )
    else:
        lines.append(
            "- no immediate action generated; confirm target-environment validation requirements"
        )
    lines.extend(
        [
            "",
            "## Execution Boundary",
            "",
            "- planning only: build/test/package/consistency validation not executed",
            "- not passed: this artifact does not prove build, test, validation, or handoff success",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_checklist(summary: dict[str, Any]) -> str:
    manifest = summary["manifest"]
    lines = [
        "# Operator Delivery Checklist",
        "",
        f"- [x] identify input asset: `{manifest['asset_path']}`",
        f"- [x] classify asset level: `{manifest['asset_level']}`",
    ]
    checks = [
        (
            "kernel entry contract",
            not any(
                item == "operator_entry_contract"
                for item in manifest["missing_contracts"]
            ),
        ),
        ("build contract", bool(manifest["build_system"])),
        ("test contract", bool(manifest["test_system"])),
        ("package contract", bool(manifest["package_system"])),
        ("delivery docs contract", bool(manifest["doc_files"])),
        (
            "SK binding contract",
            not any(
                item == "sk_binding_contract" for item in manifest["missing_contracts"]
            ),
        ),
    ]
    for label, ok in checks:
        marker = "x" if ok else " "
        lines.append(f"- [{marker}] {label}")
    lines.extend(["", "## Next Actions", ""])
    for item in manifest["supported_next_actions"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _render_ai_hints(summary: dict[str, Any]) -> str:
    lines = [
        "# Operator Delivery AI Hints",
        "",
        f"- requested: `{summary['execution']['ai_requested']}`",
        "- status: `not-configured`",
        "- 当前仓已经把 AI 层作为可选增强合同固定下来，但还没有接入真实模型后端。",
        "",
        "## Ready Inputs For Future AI Layer",
        "",
        f"- asset path: `{summary['asset']['asset_path']}`",
        f"- file count: `{summary['asset']['file_count']}`",
        f"- suffix count: `{summary['asset']['suffix_count']}`",
        f"- asset level: `{summary['manifest']['asset_level']}`",
        f"- test quality: `{summary['manifest']['test_quality']['level']}`",
        f"- build quality: `{summary['manifest']['build_quality']['level']}`",
        f"- delivery phase: `{summary['manifest']['delivery_plan']['phase']}` (planning only; build/test not executed)",
        f"- scaffold plan: `{summary['manifest']['scaffold_plan']['status']}` "
        f"({len(summary['manifest']['scaffold_plan']['items'])} items)",
        "- execution plan: `operator-delivery-execution-plan.md`",
        f"- missing contracts: `{summary['manifest']['missing_contracts']}`",
        "",
        "## Manual / Future AI Focus",
        "",
        "- 先确认当前 asset level 是否符合客户输入事实。",
        "- 对照 skill-local references 和 templates 选改造起点。",
        "- 优先补齐缺失合同，再进入 codegen、检查、一致性验证和样例骨架。",
    ]
    return "\n".join(lines) + "\n"


def _build_delivery_summary(asset_path: Path, *, with_ai: bool) -> dict[str, Any]:
    manifest = _build_manifest(asset_path)
    return {
        "skill": "sk-operator-codegen",
        "execution": {
            "entry": "operator_codegen.py",
            "base_mode": True,
            "ai_requested": with_ai,
            "ai_status": "not-configured" if with_ai else "not-requested",
        },
        "asset": _collect_asset_summary(asset_path),
        "manifest": manifest,
        "workflow_steps": WORKFLOW_STEPS,
        "dependency_hints": DEPENDENCY_HINTS,
    }


def _write_delivery_artifacts(output_dir: Path, summary: dict[str, Any]) -> None:
    _write_json(output_dir / "operator-asset-manifest.json", summary["manifest"])
    _write_json(output_dir / "operator-delivery-summary.json", summary)
    _write_text(output_dir / "operator-delivery-report.md", _render_report(summary))
    _write_text(
        output_dir / "operator-delivery-execution-plan.md",
        _render_execution_plan(summary),
    )
    _write_text(
        output_dir / "operator-delivery-checklist.md", _render_checklist(summary)
    )
    if summary["execution"]["ai_requested"]:
        _write_text(
            output_dir / "operator-delivery-ai-hints.md", _render_ai_hints(summary)
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
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or not SUPPORT_DIR_NAME_RE.fullmatch(name)
    ):
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
    if not cann_path and not arch and not compile_options and not force_includes:
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


def _build_sk_conversion_analysis(
    asset_path: Path,
    support_dir_specs: list[str] | None = None,
    include_dir_specs: list[str] | None = None,
    ascend_cann_path: str | None = None,
    ascend_arch: str | None = None,
    ascend_compile_options: list[str] | None = None,
    ascend_force_includes: list[str] | None = None,
) -> dict[str, Any]:
    manifest = _build_manifest(asset_path)
    support_directories = _normalize_support_directories(
        support_dir_specs, Path(manifest["base_dir"])
    )
    include_directories = _normalize_include_directories(include_dir_specs)
    ascend_compile_contract = _normalize_ascend_compile_contract(
        cann_path=ascend_cann_path,
        arch=ascend_arch,
        compile_options=ascend_compile_options,
        force_includes=ascend_force_includes,
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


def _pascal_case(value: str) -> str:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", value) if part]
    return "".join(part[:1].upper() + part[1:] for part in parts) or "Operator"


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
        source_files = [
            _validate_manifest_path(path, "sk conversion analysis support source file")
            for path in _require_list(
                item["source_files"], "sk conversion analysis support source_files"
            )
        ]
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
    analysis["include_directories"] = [
        str(
            Path(
                _require_string(item, "sk conversion analysis include directory")
            ).resolve()
        )
        for item in _require_list(
            analysis["include_directories"],
            "sk conversion analysis include_directories",
        )
    ]
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
            "compile_options": [
                _require_string(item, "ascend compile contract compile option")
                for item in _require_list(
                    contract["compile_options"],
                    "ascend compile contract compile_options",
                )
            ],
            "force_includes": [
                str(
                    Path(
                        _require_string(item, "ascend compile contract force include")
                    ).resolve()
                )
                for item in _require_list(
                    contract["force_includes"], "ascend compile contract force_includes"
                )
            ],
            "derived_include_dirs": [
                str(
                    Path(
                        _require_string(
                            item, "ascend compile contract derived include dir"
                        )
                    ).resolve()
                )
                for item in _require_list(
                    contract["derived_include_dirs"],
                    "ascend compile contract derived_include_dirs",
                )
            ],
        }
    analysis["ascend_compile_contract"] = ascend_compile_contract
    expected = _build_sk_conversion_analysis(
        asset_path,
        [
            f"{item['source_name']}={item['source_path']}"
            for item in support_directories
        ],
        analysis["include_directories"],
        ascend_compile_contract["cann_path"] if ascend_compile_contract else None,
        ascend_compile_contract["arch"] if ascend_compile_contract else None,
        ascend_compile_contract["compile_options"] if ascend_compile_contract else None,
        ascend_compile_contract["force_includes"] if ascend_compile_contract else None,
    )
    if analysis != expected:
        raise CliUsageError("sk conversion analysis is not current")
    return analysis


def _sk_conversion_input_by_name(analysis: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in analysis["conversion_inputs"] if item["name"] == name]
    if len(matches) != 1:
        raise CliUsageError("sk conversion analysis is not current")
    return matches[0]


def _sk_binding_empty_manifest(
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
        "kernel_entries": analysis["kernel_entries"],
        "adapted_entries": [],
        "copied_source_files": [],
        "generated_files": [],
        "blocked_reasons": blocked_reasons,
        "execution_boundary": SK_BINDING_SCAFFOLD_BOUNDARY,
        "supported_next_actions": supported_next_actions,
    }


def _sk_binding_remaining_next_actions(analysis: dict[str, Any]) -> list[str]:
    actions = [
        item["action"]
        for item in analysis["generation_plan"]
        if item["status"] == "needed" and item["action"] != "adapt_sk_binding"
    ][:3]
    return actions or ["generate_sk_source_scaffold"]


def _raise_if_existing_non_generated_scaffold(output_dir: Path) -> None:
    if (output_dir / SK_BINDING_SCAFFOLD_DIR).exists():
        raise CliUsageError(
            "sk binding scaffold directory exists for non-generated result"
        )


def _write_sk_binding_artifacts(output_dir: Path, manifest: dict[str, Any]) -> None:
    _write_json(output_dir / "operator-sk-binding-scaffold.json", manifest)
    _write_text(
        output_dir / "operator-sk-binding-scaffold.md",
        _render_sk_binding_scaffold_markdown(manifest),
    )


def _render_sk_binding_scaffold_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Operator SK Binding Scaffold",
        "",
        "## Status",
        "",
        f"- status: `{manifest['status']}`",
        "",
        "## Adapted Entries",
        "",
    ]
    if manifest["adapted_entries"]:
        for entry in manifest["adapted_entries"]:
            lines.append(
                f"- `{entry['entry_name']}` -> `{entry['generated_path']}` ({entry['reason']})"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Copied Sources", ""])
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
            "- This artifact is a review scaffold for SK binding adaptation.",
            "- It does not prove build, runtime, correctness, package, or customer delivery success.",
        ]
    )
    return "\n".join(lines) + "\n"


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
    index = open_brace_index
    state = "code"
    quote = ""
    while index < len(text):
        char = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if state == "line-comment":
            if char == "\n":
                state = "code"
            index += 1
            continue
        if state == "block-comment":
            if char == "*" and nxt == "/":
                state = "code"
                index += 2
                continue
            index += 1
            continue
        if state == "string":
            if char == "\\":
                index += 2
                continue
            if char == quote:
                state = "code"
                quote = ""
            index += 1
            continue
        if char == "/" and nxt == "/":
            state = "line-comment"
            index += 2
            continue
        if char == "/" and nxt == "*":
            state = "block-comment"
            index += 2
            continue
        if char in {"'", '"'}:
            state = "string"
            quote = char
            index += 1
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
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
        return {
            "signature": text[match.start() : open_brace].strip(),
            "qualifiers": match.group("qualifiers")
            .replace("__global__", "__sk__", 1)
            .strip(),
            "params": match.group("params"),
            "body": text[open_brace + 1 : close_brace],
        }
    return None


def _replace_param_tokens(body: str, params: list[dict[str, str]]) -> str:
    result = body
    for param in params:
        result = re.sub(
            rf"\b{re.escape(param['name'])}\b", f"args->{param['field']}", result
        )
    return result.strip("\n")


def _render_sk_binding_source(
    entry: dict[str, str], parsed: dict[str, Any], params: list[dict[str, str]]
) -> tuple[str, dict[str, Any]]:
    entry_name = entry["name"]
    args_struct = f"{_pascal_case(entry_name)}SkArgs"
    sk_function = f"{entry_name}_sk"
    bind_macro = f"SK_BIND({entry_name}, 4, {sk_function}<0>, {sk_function}<1>, {sk_function}<2>, {sk_function}<3>);"
    lines = [
        "// Generated SK binding scaffold for review.",
        "// original asset not modified; build not executed.",
        f"// original signature: {parsed['signature']}",
        '#include "kernel_operator.h"',
        "",
        f"struct {args_struct} {{",
    ]
    if params:
        for param in params:
            prefix = f"{param['alignment']} " if param["alignment"] else ""
            lines.append(f"    {prefix}{param['type']} {param['field']};")
    lines.extend(
        [
            "};",
            "",
            "template<uint32_t split_index>",
            f"{parsed['qualifiers']} void {sk_function}(const {args_struct} *args, const sk::SkSystemArgs *sysArgs)",
            "{",
            "    (void)sysArgs;",
        ]
    )
    for param in params:
        lines.append(f"    {param['type']} {param['name']} = args->{param['field']};")
    body = _replace_param_tokens(parsed["body"], params)
    if body.strip():
        for line in body.splitlines():
            lines.append(f"    {line.strip()}" if line.strip() else "")
    else:
        lines.append("    // Original kernel body is empty in the detected source.")
    lines.extend(["}", "", bind_macro, ""])
    adapted_entry = {
        "entry_name": entry_name,
        "source_file": entry["file"],
        "generated_path": f"{SK_BINDING_SCAFFOLD_DIR}/src/{entry_name}_sk_binding.asc",
        "args_struct": args_struct,
        "sk_function": sk_function,
        "bind_macro": bind_macro,
        "params": params,
        "status": "generated",
        "reason": "sk_binding_scaffold_generated",
    }
    return "\n".join(lines), adapted_entry


def _copy_sk_binding_sources(
    output_dir: Path, analysis: dict[str, Any]
) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    base_dir = Path(analysis["base_dir"])
    for rel_path in analysis["source_files"]:
        source_path = base_dir / rel_path
        scaffold_path = f"{SK_BINDING_SCAFFOLD_DIR}/original/{rel_path}"
        destination = output_dir / scaffold_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        copied.append({"source": rel_path, "scaffold_path": scaffold_path})
    return copied


def _write_sk_binding_scaffold_readme(
    output_dir: Path, manifest: dict[str, Any]
) -> None:
    lines = [
        "# Operator SK Binding Scaffold",
        "",
        "This directory contains reviewable SK binding source scaffolds.",
        "It does not prove build, runtime, correctness, package, or customer delivery success.",
        "",
        "## Generated Files",
        "",
    ]
    for item in manifest["generated_files"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Execution Boundary", ""])
    for item in manifest["execution_boundary"]:
        lines.append(f"- {item}")
    _write_text(
        output_dir / SK_BINDING_SCAFFOLD_DIR / "README.md", "\n".join(lines) + "\n"
    )


def _generate_sk_binding_scaffold(
    output_dir: Path, analysis: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    scaffold_dir = output_dir / SK_BINDING_SCAFFOLD_DIR
    if scaffold_dir.exists():
        raise CliUsageError("sk binding scaffold output already exists")

    base_dir = Path(analysis["base_dir"])
    adapted_entries: list[dict[str, Any]] = []
    generated_sources: list[tuple[str, str]] = []
    for entry in analysis["kernel_entries"]:
        parsed = _find_kernel_signature_for_entry(base_dir, entry)
        if parsed is None:
            manifest = _sk_binding_empty_manifest(
                status="blocked",
                output_dir=output_dir,
                analysis=analysis,
                blocked_reasons=["sk_binding_scaffold_entry_parse_failed"],
                supported_next_actions=["identify_kernel_entry_signature"],
            )
            return manifest, 1
        params = _parse_sk_scaffold_params(parsed["params"])
        if params is None:
            manifest = _sk_binding_empty_manifest(
                status="blocked",
                output_dir=output_dir,
                analysis=analysis,
                blocked_reasons=["sk_binding_scaffold_entry_parse_failed"],
                supported_next_actions=["identify_kernel_entry_signature"],
            )
            return manifest, 1
        content, adapted_entry = _render_sk_binding_source(entry, parsed, params)
        adapted_entries.append(adapted_entry)
        generated_sources.append((adapted_entry["generated_path"], content))

    copied_sources = _copy_sk_binding_sources(output_dir, analysis)
    generated_files = [f"{SK_BINDING_SCAFFOLD_DIR}/README.md"] + [
        path for path, _ in generated_sources
    ]
    manifest = {
        "status": "generated",
        "analysis_output_dir": str(output_dir.resolve()),
        "analysis_path": "operator-sk-conversion-analysis.json",
        "asset_path": analysis["asset_path"],
        "source_files": analysis["source_files"],
        "kernel_entries": analysis["kernel_entries"],
        "adapted_entries": adapted_entries,
        "copied_source_files": copied_sources,
        "generated_files": generated_files,
        "blocked_reasons": [],
        "execution_boundary": SK_BINDING_SCAFFOLD_BOUNDARY,
        "supported_next_actions": _sk_binding_remaining_next_actions(analysis),
    }
    _write_sk_binding_scaffold_readme(output_dir, manifest)
    for rel_path, content in generated_sources:
        _write_text(output_dir / rel_path, content)
    return manifest, 0


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
    except (json.JSONDecodeError, ValueError) as exc:
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


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cmd_intake(args: argparse.Namespace) -> int:
    asset_path = Path(args.asset).resolve()
    if not asset_path.exists():
        raise FileNotFoundError(f"asset path not found: {asset_path}")

    output_dir = Path(args.output_dir).resolve()
    summary = _build_delivery_summary(asset_path, with_ai=args.with_ai)
    _write_delivery_artifacts(output_dir, summary)
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    args.with_ai = args.with_ai
    return cmd_intake(args)


def cmd_analyze_sk_conversion(args: argparse.Namespace) -> int:
    asset_path = Path(args.asset).resolve()
    if not asset_path.exists():
        raise CliUsageError(f"asset path not found: {asset_path}")

    output_dir = Path(args.output_dir).resolve()
    analysis = _build_sk_conversion_analysis(
        asset_path,
        getattr(args, "support_dir", None),
        getattr(args, "include_dir", None),
        getattr(args, "ascend_cann_path", None),
        getattr(args, "ascend_arch", None),
        getattr(args, "ascend_compile_option", None),
        getattr(args, "ascend_force_include", None),
    )
    _write_sk_conversion_artifacts(output_dir, analysis)
    return 0


def cmd_adapt_sk_binding_scaffold(args: argparse.Namespace) -> int:
    output_dir = Path(args.analysis_output_dir).resolve()
    if not output_dir.exists():
        raise CliUsageError(f"analysis output directory not found: {output_dir}")
    if not output_dir.is_dir():
        raise CliUsageError(f"analysis output path is not a directory: {output_dir}")

    analysis = _load_current_sk_conversion_analysis(output_dir)
    sk_binding = _sk_conversion_input_by_name(analysis, "sk_binding")

    if analysis["status"] == "blocked":
        _raise_if_existing_non_generated_scaffold(output_dir)
        manifest = _sk_binding_empty_manifest(
            status="blocked",
            output_dir=output_dir,
            analysis=analysis,
            blocked_reasons=analysis["blocked_reasons"],
            supported_next_actions=analysis["supported_next_actions"],
        )
        _write_sk_binding_artifacts(output_dir, manifest)
        return 1

    if sk_binding["status"] == "present":
        _raise_if_existing_non_generated_scaffold(output_dir)
        manifest = _sk_binding_empty_manifest(
            status="not-needed",
            output_dir=output_dir,
            analysis=analysis,
            blocked_reasons=["sk_binding_already_present"],
            supported_next_actions=analysis["supported_next_actions"]
            or ["generate_sk_source_scaffold"],
        )
        _write_sk_binding_artifacts(output_dir, manifest)
        return 0

    manifest, return_code = _generate_sk_binding_scaffold(output_dir, analysis)
    _write_sk_binding_artifacts(output_dir, manifest)
    return return_code


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


_SK_ADAPT_SUFFIXES = {".asc", ".cpp"}
_SK_FORM_SIGNAL_RE = re.compile(
    r"__global__\b|__spk__\b|__sk__\b|\bSK_BIND\s*\(|\.ascend\.meta"
)
_SK_NON_GLOBAL_FORM_SIGNAL_RE = re.compile(
    r"__spk__\b|__sk__\b|\bSK_BIND\s*\(|\.ascend\.meta|FunLevel(?:MixCoreType|KType)"
)
_QUOTED_INCLUDE_RE = re.compile(r'(?m)^\s*#\s*include\s+"([^"]+)"')
_COPIED_INCLUDE_SUFFIXES = INCLUDE_SUFFIXES | {".inc", ".inl", ".ipp", ".tpp"}


def _is_test_relative_path(rel: Path) -> bool:
    return any(part in TEST_PACKAGE_DIR_NAMES for part in rel.parts)


def _has_global_kernel_definition(source_text: str) -> bool:
    from sk_codegen_lib import parse_global_entries

    return bool(parse_global_entries(source_text))


def _collect_sk_form_sources(
    asset_path: Path,
) -> tuple[list[tuple[Path, Path]], list[dict[str, str]]]:
    if asset_path.is_file():
        return [(asset_path, Path(asset_path.name))], []
    files = sorted(
        path
        for path in asset_path.rglob("*")
        if path.is_file() and path.suffix in _SK_ADAPT_SUFFIXES
    )
    candidates: list[tuple[Path, Path]] = []
    ignored: list[dict[str, str]] = []
    for path in files:
        rel = path.relative_to(asset_path)
        text = path.read_text(encoding="utf-8", errors="replace")
        if _is_test_relative_path(rel):
            ignored.append({"file": str(rel), "reason": "test_support_file"})
            continue
        if not _SK_FORM_SIGNAL_RE.search(text):
            ignored.append({"file": str(rel), "reason": "no_kernel_signal"})
            continue
        if not _has_global_kernel_definition(
            text
        ) and not _SK_NON_GLOBAL_FORM_SIGNAL_RE.search(text):
            ignored.append({"file": str(rel), "reason": "no_kernel_definition"})
            continue
        candidates.append((path, rel))
    return candidates, ignored


def _resolve_quoted_include(
    include_path: str,
    including_source: Path,
    asset_path: Path,
    support_directories: list[dict[str, Any]],
) -> Path | None:
    include_rel = Path(include_path)
    if include_rel.is_absolute() or not include_rel.parts:
        return None
    if include_rel.suffix and include_rel.suffix not in _COPIED_INCLUDE_SUFFIXES:
        return None
    asset_root = asset_path if asset_path.is_dir() else asset_path.parent
    search_roots = [including_source.parent, asset_root]
    for root in search_roots:
        candidate = (root / include_rel).resolve()
        if candidate.is_file():
            return candidate
    for support_dir in support_directories:
        source_name = support_dir["source_name"]
        if include_rel.parts[0] != source_name:
            continue
        support_rel = (
            Path(*include_rel.parts[1:]) if len(include_rel.parts) > 1 else Path()
        )
        candidate = (Path(support_dir["source_path"]) / support_rel).resolve()
        if candidate.is_file():
            return candidate
    candidate = (asset_root.parent / include_rel).resolve()
    if candidate.is_file():
        return candidate
    return None


def _include_destination(
    include_path: str, including_dest: Path, adapted_dir: Path
) -> Path | None:
    include_rel = Path(include_path)
    if include_rel.is_absolute() or not include_rel.parts:
        return None
    destination = (including_dest.parent / include_rel).resolve()
    adapted_root = adapted_dir.resolve()
    if not _is_same_or_child_path(destination, adapted_root):
        return None
    return destination


def _copy_aclgraph_quoted_include_dependencies(
    *,
    source_path: Path,
    source_text: str,
    csrc_path: Path,
    adapted_dir: Path,
    asset_path: Path,
    support_directories: list[dict[str, Any]],
) -> list[str]:
    copied: list[str] = []
    queue = [
        (include, source_path, csrc_path)
        for include in _QUOTED_INCLUDE_RE.findall(source_text)
    ]
    seen_destinations: set[Path] = set()
    while queue:
        include_path, including_source, including_dest = queue.pop(0)
        dependency = _resolve_quoted_include(
            include_path, including_source, asset_path, support_directories
        )
        destination = _include_destination(include_path, including_dest, adapted_dir)
        if dependency is None or destination is None:
            continue
        if destination in seen_destinations:
            continue
        seen_destinations.add(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != dependency.read_bytes():
                raise CliUsageError(
                    f"conflicting generated include dependency: {destination}"
                )
            continue
        shutil.copy2(dependency, destination)
        copied.append(
            f"operator-sk-adapted/{destination.relative_to(adapted_dir).as_posix()}"
        )
        dependency_text = dependency.read_text(encoding="utf-8", errors="replace")
        queue.extend(
            (nested_include, dependency, destination)
            for nested_include in _QUOTED_INCLUDE_RE.findall(dependency_text)
        )
    return copied


def _unique_aclgraph_csrc_name(name: str, used_names: set[str]) -> str:
    def _is_pybind_source(candidate_name: str) -> bool:
        return candidate_name == "pybind11.asc" or (
            candidate_name.startswith("pybind11_") and candidate_name.endswith(".asc")
        )

    candidate = Path(name).name
    if not candidate:
        candidate = "runtime_wrapper.cpp"
    if candidate not in used_names and not _is_pybind_source(candidate):
        used_names.add(candidate)
        return candidate
    stem = Path(candidate).stem or "runtime_wrapper"
    suffix = Path(candidate).suffix or ".cpp"
    index = 1
    while True:
        candidate = f"{stem}_{index}{suffix}"
        if candidate not in used_names and not _is_pybind_source(candidate):
            used_names.add(candidate)
            return candidate
        index += 1


def _copy_runtime_wrapper_source(
    *,
    entry: dict[str, Any],
    csrc_dir: Path,
    adapted_dir: Path,
    asset_path: Path,
    support_directories: list[dict[str, Any]],
    used_csrc_names: set[str],
) -> list[str]:
    runtime_wrapper = entry.get("runtime_wrapper")
    if not isinstance(runtime_wrapper, dict) or not runtime_wrapper:
        return []
    source_path = Path(str(runtime_wrapper.get("source_path") or "")).resolve()
    if not source_path.is_file():
        raise CliUsageError(f"runtime wrapper source not found: {source_path}")
    source_text = source_path.read_text(encoding="utf-8", errors="replace")
    csrc_name = _unique_aclgraph_csrc_name(source_path.name, used_csrc_names)
    csrc_path = csrc_dir / csrc_name
    csrc_path.write_text(source_text, encoding="utf-8")
    runtime_wrapper["csrc_file"] = csrc_name
    written = [f"operator-sk-adapted/csrc/{csrc_name}"]
    written.extend(
        _copy_aclgraph_quoted_include_dependencies(
            source_path=source_path,
            source_text=source_text,
            csrc_path=csrc_path,
            adapted_dir=adapted_dir,
            asset_path=asset_path,
            support_directories=support_directories,
        )
    )
    return written


def _codegen_human_finding(
    finding_id: str, message: str, evidence: list[str] | None = None
) -> dict[str, Any]:
    return {
        "finding_id": finding_id,
        "rule_id": finding_id,
        "severity": "blocker",
        "category": "codegen",
        "actionable_by": ["human"],
        "remediation_hint": {"kind": "human-decision"},
        "evidence": evidence or [],
        "message": message,
    }


def cmd_detect_sk_form(args: argparse.Namespace) -> int:
    from sk_codegen_lib import detect_sk_form

    asset_path = Path(args.asset).resolve()
    if not asset_path.exists():
        raise CliUsageError(f"asset path not found: {asset_path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources, ignored = _collect_sk_form_sources(asset_path)
    per_file: list[dict[str, Any]] = []
    for source_path, rel in sources:
        if not source_path.is_file():
            continue
        text = source_path.read_text(encoding="utf-8", errors="replace")
        analysis = detect_sk_form(text)
        per_file.append(
            {
                "file": str(rel),
                "form": analysis.form,
                "has_global": analysis.has_global,
                "has_spk_keyword": analysis.has_spk_keyword,
                "has_sk_keyword": analysis.has_sk_keyword,
                "has_sk_bind": analysis.has_sk_bind,
                "has_legacy_meta_struct": analysis.has_legacy_meta_struct,
                "notes": analysis.notes,
            }
        )
    if not per_file:
        raise CliUsageError(
            f"no kernel .asc or .cpp source files found under {asset_path}"
        )
    forms = {entry["form"] for entry in per_file}
    if forms == {"none"}:
        overall = "none"
    elif forms == {"current-sk-bind"}:
        overall = "current-sk-bind"
    elif forms == {"legacy-spk"}:
        overall = "legacy-spk"
    elif "unknown" in forms and len(forms) == 1:
        overall = "unknown"
    else:
        overall = "mixed"
    manifest = {
        "status": "analyzed",
        "asset_path": str(asset_path),
        "overall_form": overall,
        "per_file": per_file,
        "ignored_support_files": ignored,
    }
    out_path = output_dir / "operator-sk-form-analysis.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


def cmd_scan_operator_assets(args: argparse.Namespace) -> int:
    from operator_asset_contracts import (
        build_adapter_report,
        build_default_build_context,
        build_default_verify_context,
        write_json,
    )
    from operator_asset_layout import build_inventory, build_layout_from_inventory

    asset_path = Path(args.asset).resolve()
    if not asset_path.exists():
        raise CliUsageError(f"asset path not found: {asset_path}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory(asset_path)
    layout = build_layout_from_inventory(inventory)
    build_context = build_default_build_context(layout, inventory)
    verify_context = build_default_verify_context(layout)
    adapter_report = build_adapter_report(
        asset_root=asset_path,
        inventory=inventory,
        layout=layout,
        build_context=build_context,
        verify_context=verify_context,
    )
    write_json(output_dir / "operator-asset-inventory.json", inventory)
    write_json(output_dir / "operator-asset-layout.json", layout)
    write_json(output_dir / "operator-build-context.json", build_context)
    write_json(output_dir / "operator-verify-context.json", verify_context)
    write_json(output_dir / "adapter-report.json", adapter_report)
    return 0 if layout.get("status") == "ready" else 2


def cmd_analyze_operator_asset(args: argparse.Namespace) -> int:
    from operator_asset_analyzer import analyze_asset, analyze_asset_from_layout_file
    from operator_asset_contract import write_understanding
    from operator_target_arch import build_target_resolution

    asset_path = Path(args.asset).resolve()
    if not asset_path.exists():
        raise CliUsageError(f"asset path not found: {asset_path}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "asset_layout", ""):
        manifest = analyze_asset_from_layout_file(Path(args.asset_layout).resolve())
    else:
        manifest = analyze_asset(asset_path)
    if getattr(args, "target_chip", ""):
        updated_units = []
        for unit in manifest.operator_units:
            target_resolution = build_target_resolution(
                unit.supported_soc_versions, args.target_chip
            )
            updated_units.append(
                unit.__class__(
                    **{
                        **unit.__dict__,
                        "supported_arches": target_resolution.get("arches")
                        or unit.supported_arches,
                        "target_resolution": target_resolution,
                    }
                )
            )
        manifest = manifest.__class__(
            **{
                **manifest.__dict__,
                "operator_units": updated_units,
            }
        )
    write_understanding(output_dir / "operator-asset-understanding.json", manifest)
    return 0


def cmd_normalize_operator_asset(args: argparse.Namespace) -> int:
    from operator_asset_analyzer import (
        analyze_asset,
        analyze_asset_from_layout_file,
        materialize_operator_units,
    )
    from operator_asset_contract import read_understanding, write_understanding
    from operator_target_arch import build_target_resolution

    asset_path = Path(args.asset).resolve()
    if not asset_path.exists():
        raise CliUsageError(f"asset path not found: {asset_path}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "asset_layout", ""):
        manifest = analyze_asset_from_layout_file(Path(args.asset_layout).resolve())
    elif args.understanding:
        manifest = read_understanding(Path(args.understanding).resolve())
    else:
        manifest = analyze_asset(asset_path)
    if getattr(args, "target_chip", ""):
        updated_units = []
        for unit in manifest.operator_units:
            target_resolution = build_target_resolution(
                unit.supported_soc_versions, args.target_chip
            )
            updated_units.append(
                unit.__class__(
                    **{
                        **unit.__dict__,
                        "supported_arches": target_resolution.get("arches")
                        or unit.supported_arches,
                        "target_resolution": target_resolution,
                    }
                )
            )
        manifest = manifest.__class__(
            **{
                **manifest.__dict__,
                "operator_units": updated_units,
            }
        )
    write_understanding(output_dir / "operator-asset-understanding.json", manifest)
    materialize_operator_units(asset_path, manifest, output_dir)
    return 0


def cmd_adapt_sk_from_global(args: argparse.Namespace) -> int:
    from sk_codegen_lib import (
        _aclgraph_entry_pybind_name,
        _run_spec_clean_loop_inline,
        adapt_source_text,
        detect_sk_form,
        migrate_legacy_spk_to_sk_bind,
        render_aclgraph_kernel_source,
        render_arch_selector_py,
        render_op_extension_init_py,
        render_pybind11_asc,
        render_pybind11_entry_asc,
        render_setup_py_aclgraph_style,
        render_torch_library_py,
        _safe_identifier,
    )
    from operator_target_arch import build_target_resolution

    asset_path = Path(args.asset).resolve()
    if not asset_path.exists():
        raise CliUsageError(f"asset path not found: {asset_path}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    package_name = getattr(args, "package_name", "op_extension") or "op_extension"
    package_version = getattr(args, "package_version", "0.1.0") or "0.1.0"
    package_dir_name = _safe_identifier(package_name, "op_extension")
    source_asset_override = (
        Path(args.source_asset).resolve() if getattr(args, "source_asset", "") else None
    )
    name_resolution_index: dict[tuple[str, str], dict[str, Any]] = {}
    name_resolution_payload: dict[str, Any] | None = None
    name_resolution_path = getattr(args, "name_resolution", "") or ""
    if name_resolution_path:
        from operator_name_resolution import (
            public_name_resolution_payload,
            read_name_resolution,
            resolution_by_asset_and_entry,
        )

        name_resolution_payload = read_name_resolution(
            Path(name_resolution_path).resolve()
        )
        name_resolution_index = resolution_by_asset_and_entry(name_resolution_payload)
    public_name_resolution = (
        public_name_resolution_payload(name_resolution_payload)
        if name_resolution_payload is not None
        else None
    )

    io_contract_by_entry = _read_pybind_io_contract(
        getattr(args, "io_contract", "") or ""
    )
    used_io_contract_entries: set[str] = set()

    def apply_name_resolution(
        entry: dict[str, Any], source_asset: object | None
    ) -> None:
        source_entry_name = str(entry.get("entry_name", ""))
        source_asset_path = str(
            Path(str(source_asset_override or source_asset or asset_path)).resolve()
        )
        resolution = name_resolution_index.get((source_asset_path, source_entry_name))
        entry["source_entry_name"] = source_entry_name
        if resolution:
            entry["entry_name"] = resolution["public_entry_name"]
            entry["public_entry_name"] = resolution["public_entry_name"]
            entry["internal_symbol_name"] = resolution["internal_symbol_name"]
            entry["bind_target"] = resolution["bind_target"]
            entry["name_resolution"] = {
                "renamed": bool(resolution.get("renamed")),
                "rename_reason": resolution.get("rename_reason", ""),
                "asset_namespace": resolution.get("asset_namespace", ""),
                "collision_group": resolution.get("collision_group", ""),
            }
        else:
            entry.setdefault("public_entry_name", source_entry_name)
            entry.setdefault(
                "internal_symbol_name", _safe_identifier(source_entry_name, "op")
            )
            entry.setdefault("bind_target", source_entry_name)
            entry.setdefault(
                "name_resolution",
                {
                    "renamed": False,
                    "rename_reason": "",
                    "asset_namespace": "",
                    "collision_group": "",
                },
            )

    def find_io_contract(
        entry_name: str, public_entry_name: str = ""
    ) -> tuple[str, dict[str, Any] | None]:
        for key in (public_entry_name, entry_name):
            if key and key in io_contract_by_entry:
                return key, io_contract_by_entry[key]
        for key, value in io_contract_by_entry.items():
            if (
                value.get("source_entry_name") == entry_name
                or value.get("public_entry_name") == public_entry_name
            ):
                return key, value
        return "", None

    def contract_bind_targets_for_global_entry(
        global_entry_names: list[str],
    ) -> dict[str, str]:
        selected: dict[str, str] = {}
        global_name_set = {name for name in global_entry_names if name}
        for key, value in io_contract_by_entry.items():
            bind_target = str(value.get("bind_target") or "").strip()
            if not bind_target:
                continue
            bind_global_name = bind_target.split("<", 1)[0].strip()
            candidates = {
                str(key),
                str(value.get("source_entry_name") or ""),
                str(value.get("public_entry_name") or ""),
                bind_global_name,
            }
            matched_names = sorted((candidates | {bind_global_name}) & global_name_set)
            for matched_name in matched_names:
                selected[matched_name] = bind_target
        return selected

    def apply_io_contract(entry: dict[str, Any]) -> dict[str, Any] | None:
        source_entry_name = str(
            entry.get("source_entry_name") or entry.get("entry_name") or ""
        )
        public_entry_name = str(entry.get("entry_name") or "")
        io_contract_key, io_contract = find_io_contract(
            source_entry_name, public_entry_name
        )
        param_kinds = _entry_param_kinds(entry)
        tensor_names = _io_contract_tensor_param_names(entry)
        host_struct_names = _io_contract_host_struct_param_names(entry)
        if io_contract is None:
            entry["pybind_return_source"] = (
                "single-tensor-default"
                if len(tensor_names) == 1
                else "no-tensor-default"
                if not tensor_names
                else "unresolved"
            )
            if host_struct_names:
                return _codegen_human_finding(
                    "codegen.runtime-parameter-contract-required",
                    (
                        f"operator entry {source_entry_name or public_entry_name!r} has host-struct parameter(s) "
                        f"{host_struct_names}; provide --io-contract parameters with kind='host_struct' so the runtime boundary is explicit."
                    ),
                    [
                        f"entry={source_entry_name or public_entry_name}",
                        f"host_struct_parameters={host_struct_names}",
                        "contract_schema={'schema_version':1,'entries':{'entry_name':{'parameters':{'tiling':{'kind':'host_struct'}}}}}",
                    ],
                )
            if len(tensor_names) <= 1:
                return None
            return _codegen_human_finding(
                "codegen.pybind-return-tensor-unresolved",
                (
                    f"operator entry {source_entry_name or public_entry_name!r} has multiple tensor-like parameters "
                    f"{tensor_names}; provide --io-contract with pybind_return_tensor instead of relying on name guesses."
                ),
                [
                    f"entry={source_entry_name or public_entry_name}",
                    f"tensor_parameters={tensor_names}",
                    "contract_schema={'schema_version':1,'entries':{'entry_name':{'inputs':[...],'outputs':[...],'pybind_return_tensor':'...'}}}",
                ],
            )
        declared_params = io_contract.get("parameters", {})
        contract_bind_target = str(io_contract.get("bind_target") or "").strip()
        entry_bind_target = str(entry.get("bind_target") or "").strip()
        if (
            contract_bind_target
            and entry_bind_target
            and contract_bind_target != entry_bind_target
        ):
            return _codegen_human_finding(
                "codegen.io-contract-bind-target-mismatch",
                (
                    f"operator entry {source_entry_name or public_entry_name!r} matched IO contract, but its "
                    f"bind_target {entry_bind_target!r} does not match contract bind_target {contract_bind_target!r}."
                ),
                [
                    f"entry={source_entry_name or public_entry_name}",
                    f"entry_bind_target={entry_bind_target}",
                    f"contract_bind_target={contract_bind_target}",
                ],
            )
        unknown_params = sorted(set(declared_params) - set(param_kinds))
        if unknown_params:
            return _codegen_human_finding(
                "codegen.io-contract-parameter-invalid",
                (
                    f"operator entry {source_entry_name or public_entry_name!r} declares unknown runtime parameter(s) "
                    f"{unknown_params}; detected parameters are {sorted(param_kinds)}."
                ),
                [
                    f"entry={source_entry_name or public_entry_name}",
                    f"unknown_parameters={unknown_params}",
                    f"parameters={sorted(param_kinds)}",
                ],
            )
        kind_mismatches = [
            f"{name}: declared={spec.get('kind')} detected={param_kinds.get(name)}"
            for name, spec in declared_params.items()
            if not _is_declared_kind_compatible(param_kinds.get(name), spec.get("kind"))
        ]
        if kind_mismatches:
            return _codegen_human_finding(
                "codegen.io-contract-parameter-kind-mismatch",
                (
                    f"operator entry {source_entry_name or public_entry_name!r} has runtime parameter kind mismatch: "
                    f"{kind_mismatches}."
                ),
                [
                    f"entry={source_entry_name or public_entry_name}",
                    f"kind_mismatches={kind_mismatches}",
                    f"detected_parameter_kinds={param_kinds}",
                ],
            )
        tensor_list_params_in_order = [
            name
            for name, spec in declared_params.items()
            if spec.get("kind") == "tensor_list"
        ]
        tensor_list_params = sorted(tensor_list_params_in_order)
        runtime_wrapper = dict(io_contract.get("runtime_wrapper", {}) or {})
        if tensor_list_params and not runtime_wrapper:
            return _codegen_human_finding(
                "codegen.tensor-list-runtime-wrapper-required",
                (
                    f"operator entry {source_entry_name or public_entry_name!r} declares TensorList runtime parameter(s) "
                    f"{tensor_list_params}; generated pybind cannot safely synthesize descriptor pointers. "
                    "Provide an adapter/user wrapper that constructs the exact TensorList descriptor."
                ),
                [
                    f"entry={source_entry_name or public_entry_name}",
                    f"tensor_list_parameters={tensor_list_params}",
                    "reason=TensorList descriptor layout is an external runtime contract, not a raw Tensor data pointer",
                ],
            )
        if runtime_wrapper:
            asset_root = asset_path if asset_path.is_dir() else asset_path.parent
            extra_wrapper_roots = []
            if source_asset_override is not None:
                source_asset_root = (
                    source_asset_override
                    if source_asset_override.is_dir()
                    else source_asset_override.parent
                )
                extra_wrapper_roots.append(source_asset_root)
                extra_wrapper_roots.append(source_asset_root.parent)
            wrapper_source, wrapper_root = _resolve_runtime_wrapper_source(
                asset_path, runtime_wrapper, extra_wrapper_roots
            )
            if (
                not _is_same_or_child_path(wrapper_source, wrapper_root)
                or not wrapper_source.is_file()
            ):
                return _codegen_human_finding(
                    "codegen.runtime-wrapper-source-missing",
                    (
                        f"operator entry {source_entry_name or public_entry_name!r} declares runtime wrapper source "
                        f"{runtime_wrapper.get('source')!r}, but it was not found inside the asset."
                    ),
                    [
                        f"entry={source_entry_name or public_entry_name}",
                        f"runtime_wrapper_source={runtime_wrapper.get('source')}",
                        f"asset_root={asset_root}",
                        f"source_asset={source_asset_override or ''}",
                    ],
                )
            wrapper_text = wrapper_source.read_text(encoding="utf-8", errors="replace")
            wrapper_entry = str(runtime_wrapper.get("entry") or "").strip()
            if not _runtime_wrapper_entry_exists(wrapper_text, wrapper_entry):
                return _codegen_human_finding(
                    "codegen.runtime-wrapper-entry-missing",
                    (
                        f"operator entry {source_entry_name or public_entry_name!r} declares runtime wrapper entry "
                        f"{wrapper_entry!r}, but that function was not found in {runtime_wrapper.get('source')!r}."
                    ),
                    [
                        f"entry={source_entry_name or public_entry_name}",
                        f"runtime_wrapper_entry={wrapper_entry}",
                        f"runtime_wrapper_source={runtime_wrapper.get('source')}",
                    ],
                )
            runtime_wrapper["source_path"] = str(wrapper_source)
            if tensor_list_params:
                strategy = str(
                    runtime_wrapper.get("tensor_list_descriptor_strategy") or ""
                ).strip()
                if strategy != "prepared_workspace_tail":
                    return _codegen_human_finding(
                        "codegen.tensor-list-prepared-runtime-state-required",
                        (
                            f"operator entry {source_entry_name or public_entry_name!r} declares TensorList runtime parameter(s) "
                            f"{tensor_list_params}; runtime wrappers must declare a graph-safe prepared descriptor strategy."
                        ),
                        [
                            f"entry={source_entry_name or public_entry_name}",
                            f"tensor_list_parameters={tensor_list_params}",
                            "contract_schema={'runtime_wrapper':{'tensor_list_descriptor_strategy':'prepared_workspace_tail','prepare_entry':'prepare_for_capture','descriptor_order':[...]}}",
                        ],
                    )
                descriptor_order = list(runtime_wrapper.get("descriptor_order") or [])
                missing_descriptors = sorted(
                    set(tensor_list_params_in_order) - set(descriptor_order)
                )
                extra_descriptors = sorted(
                    set(descriptor_order) - set(tensor_list_params_in_order)
                )
                if (
                    missing_descriptors
                    or extra_descriptors
                    or descriptor_order != tensor_list_params_in_order
                ):
                    return _codegen_human_finding(
                        "codegen.tensor-list-descriptor-order-mismatch",
                        (
                            f"operator entry {source_entry_name or public_entry_name!r} descriptor_order must match "
                            f"TensorList parameter declaration order exactly; missing={missing_descriptors}, extra={extra_descriptors}."
                        ),
                        [
                            f"entry={source_entry_name or public_entry_name}",
                            f"tensor_list_parameters={tensor_list_params_in_order}",
                            f"descriptor_order={descriptor_order}",
                        ],
                    )
        undeclared_host_structs = sorted(set(host_struct_names) - set(declared_params))
        if undeclared_host_structs:
            return _codegen_human_finding(
                "codegen.runtime-parameter-contract-required",
                (
                    f"operator entry {source_entry_name or public_entry_name!r} has host-struct parameter(s) "
                    f"{undeclared_host_structs} that are not declared in the IO contract."
                ),
                [
                    f"entry={source_entry_name or public_entry_name}",
                    f"host_struct_parameters={host_struct_names}",
                    f"declared_parameters={sorted(declared_params)}",
                ],
            )
        declared_tensors = (
            set(io_contract.get("input_tensors", []))
            | set(io_contract.get("output_tensors", []))
            | set(io_contract.get("workspace_tensors", []))
        )
        unknown_tensors = sorted(declared_tensors - set(tensor_names))
        if unknown_tensors:
            return _codegen_human_finding(
                "codegen.io-contract-tensor-invalid",
                (
                    f"operator entry {source_entry_name or public_entry_name!r} declares unknown IO tensor(s) "
                    f"{unknown_tensors}; tensor-like parameters are {tensor_names}."
                ),
                [
                    f"entry={source_entry_name or public_entry_name}",
                    f"unknown_io_tensors={unknown_tensors}",
                    f"tensor_parameters={tensor_names}",
                ],
            )
        undeclared_tensors = sorted(set(tensor_names) - declared_tensors)
        if undeclared_tensors:
            return _codegen_human_finding(
                "codegen.io-contract-tensor-incomplete",
                (
                    f"operator entry {source_entry_name or public_entry_name!r} leaves tensor-like parameter(s) "
                    f"{undeclared_tensors} out of the IO contract; classify each as input, output, or workspace."
                ),
                [
                    f"entry={source_entry_name or public_entry_name}",
                    f"undeclared_io_tensors={undeclared_tensors}",
                    f"tensor_parameters={tensor_names}",
                ],
            )
        return_tensor = io_contract["pybind_return_tensor"]
        if return_tensor not in tensor_names:
            return _codegen_human_finding(
                "codegen.pybind-return-tensor-invalid",
                (
                    f"operator entry {source_entry_name or public_entry_name!r} declares pybind_return_tensor "
                    f"{return_tensor!r}, but tensor-like parameters are {tensor_names}."
                ),
                [
                    f"entry={source_entry_name or public_entry_name}",
                    f"pybind_return_tensor={return_tensor}",
                    f"tensor_parameters={tensor_names}",
                ],
            )
        used_io_contract_entries.add(io_contract_key)
        entry["pybind_return_tensor"] = return_tensor
        entry["pybind_return_source"] = "explicit-io-contract"
        entry["io_contract"] = {
            "inputs": io_contract.get("input_tensors", []),
            "outputs": io_contract.get("output_tensors", []),
            "workspaces": io_contract.get("workspace_tensors", []),
            "pybind_return_tensor": return_tensor,
            "parameters": declared_params,
            "launch": io_contract.get("launch", {}),
            "compile": io_contract.get("compile", {}),
            "migration": io_contract.get("migration", {}),
            "runtime_wrapper": runtime_wrapper,
            "bind_target": contract_bind_target,
            "contract_path": io_contract.get("contract_path", ""),
        }
        if runtime_wrapper:
            entry["runtime_wrapper"] = runtime_wrapper
        if contract_bind_target:
            entry["bind_target"] = contract_bind_target
        entry["compile"] = io_contract.get("compile", {})
        return None

    sources, ignored = _collect_sk_form_sources(asset_path)
    if not sources:
        raise CliUsageError(
            f"no kernel .asc or .cpp source files found under {asset_path}"
        )
    support_directories = _normalize_support_directories(
        getattr(args, "support_dir", None),
        asset_path if asset_path.is_dir() else asset_path.parent,
    )
    target_chips = getattr(args, "target_chip", "") or ""
    unit_metadata_by_entry: dict[str, dict[str, Any]] = {}
    understanding_path = getattr(args, "understanding", "") or ""
    if understanding_path:
        from operator_asset_contract import read_understanding

        understanding = read_understanding(Path(understanding_path).resolve())
        for unit in understanding.operator_units:
            target_resolution = build_target_resolution(
                unit.supported_soc_versions, target_chips
            )
            unit_metadata_by_entry[unit.entry_name] = {
                "supported_soc_versions": unit.supported_soc_versions,
                "supported_arches": target_resolution.get("arches")
                or unit.supported_arches,
                "target_resolution": target_resolution,
                "support_source": unit.support_source,
                "source_asset": unit.source_asset,
            }

    adapted_dir = output_dir / "operator-sk-adapted"
    if adapted_dir.exists():
        raise CliUsageError(f"adapted output already exists: {adapted_dir}")
    adapted_dir.mkdir(parents=True)

    per_file_results: list[dict[str, Any]] = []
    all_escalations: list[dict[str, Any]] = []
    canonical_sources: list[tuple[str, str, list[dict[str, Any]], Path]] = []
    canonical_entries: list[dict[str, Any]] = []
    for source_path, rel in sources:
        text = source_path.read_text(encoding="utf-8", errors="replace")
        form = detect_sk_form(text)
        target_path = adapted_dir / rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if form.form == "none":
            cleaned_text, inline_remediations, escalations = (
                _run_spec_clean_loop_inline(text, rel_path=str(rel))
            )
            if escalations:
                all_escalations.extend(escalations)
                per_file_results.append(
                    {
                        "file": str(rel),
                        "status": "needs-human",
                        "form_before": "none",
                        "form_after": "none",
                        "entries": [],
                        "inline_remediations": inline_remediations,
                        "escalations": escalations,
                    }
                )
                continue
            new_text, metas = adapt_source_text(
                cleaned_text,
                mask=int(args.mask),
                num_splits=int(args.num_splits),
                sys_args_mode=args.with_sys_args,
            )
            for meta in metas:
                metadata = unit_metadata_by_entry.get(meta.get("entry_name"), {})
                meta.update(metadata)
                apply_name_resolution(meta, metadata.get("source_asset"))
                io_escalation = apply_io_contract(meta)
                if io_escalation is not None:
                    escalations.append(io_escalation)
            if escalations:
                all_escalations.extend(escalations)
                per_file_results.append(
                    {
                        "file": str(rel),
                        "status": "needs-human",
                        "form_before": "none",
                        "form_after": "none",
                        "entries": metas,
                        "inline_remediations": inline_remediations,
                        "escalations": escalations,
                    }
                )
                continue
            target_path.write_text(new_text, encoding="utf-8")
            canonical_name = f"{metas[0]['entry_name']}.asc" if metas else rel.name
            canonical_sources.append((canonical_name, new_text, metas, source_path))
            canonical_entries.extend(metas)
            per_file_results.append(
                {
                    "file": str(rel),
                    "status": "adapted",
                    "form_before": "none",
                    "form_after": "current-sk-bind",
                    "pybind_layout": "aclgraph-canonical",
                    "entries": metas,
                    "inline_remediations": inline_remediations,
                    "escalations": [],
                }
            )
        elif form.form == "legacy-spk":
            migration_contract: dict[str, Any] = {}
            possible_contract_names = [
                match.group(1) for match in KERNEL_ENTRY_RE.finditer(text)
            ]
            for possible_name in possible_contract_names:
                _, matched_contract = find_io_contract(possible_name)
                if matched_contract and matched_contract.get("migration"):
                    migration_contract.update(matched_contract.get("migration", {}))
            bind_targets_by_stem = contract_bind_targets_for_global_entry(
                possible_contract_names
            )
            new_text, migration_meta = migrate_legacy_spk_to_sk_bind(
                text,
                mask=int(args.mask),
                sys_args_mode=args.with_sys_args,
                migration_contract=migration_contract,
                bind_targets_by_stem=bind_targets_by_stem,
            )
            escalations = migration_meta.get("escalations", [])
            if escalations:
                all_escalations.extend(escalations)
                per_file_results.append(
                    {
                        "file": str(rel),
                        "status": "needs-human",
                        "form_before": "legacy-spk",
                        "form_after": "legacy-spk",
                        "entries": [],
                        "migration_meta": migration_meta,
                        "inline_remediations": [],
                        "escalations": escalations,
                    }
                )
                continue
            entries = migration_meta.get("entries", [])
            for entry in entries:
                metadata = unit_metadata_by_entry.get(entry.get("entry_name"), {})
                entry.update(metadata)
                apply_name_resolution(entry, metadata.get("source_asset"))
                io_escalation = apply_io_contract(entry)
                if io_escalation is not None:
                    escalations.append(io_escalation)
            if escalations:
                all_escalations.extend(escalations)
                per_file_results.append(
                    {
                        "file": str(rel),
                        "status": "needs-human",
                        "form_before": "legacy-spk",
                        "form_after": "legacy-spk",
                        "entries": entries,
                        "migration_meta": migration_meta,
                        "inline_remediations": [],
                        "escalations": escalations,
                    }
                )
                continue
            target_path.write_text(new_text, encoding="utf-8")
            canonical_name = f"{entries[0]['entry_name']}.asc" if entries else rel.name
            canonical_sources.append((canonical_name, new_text, entries, source_path))
            canonical_entries.extend(entries)
            per_file_results.append(
                {
                    "file": str(rel),
                    "status": "adapted",
                    "form_before": "legacy-spk",
                    "form_after": "current-sk-bind",
                    "pybind_layout": "aclgraph-canonical",
                    "entries": entries,
                    "migration_meta": migration_meta,
                    "inline_remediations": [],
                    "escalations": [],
                }
            )
        elif form.form == "current-sk-bind":
            target_path.write_bytes(source_path.read_bytes())
            per_file_results.append(
                {
                    "file": str(rel),
                    "status": "already_current",
                    "form_before": "current-sk-bind",
                    "form_after": "current-sk-bind",
                    "pybind_layout": "source-passthrough",
                    "entries": [],
                    "inline_remediations": [],
                    "escalations": [],
                }
            )
        else:
            escalation = _codegen_human_finding(
                "codegen.unknown-sk-form",
                f"source file {rel} has unsupported SK form {form.form!r}; human migration is required.",
                form.notes,
            )
            all_escalations.append(escalation)
            per_file_results.append(
                {
                    "file": str(rel),
                    "status": "needs-human",
                    "form_before": form.form,
                    "form_after": form.form,
                    "entries": [],
                    "inline_remediations": [],
                    "escalations": [escalation],
                }
            )
    unresolved_entries = [
        str(entry.get("source_entry_name") or entry.get("entry_name"))
        for entry in canonical_entries
        if entry.get("pybind_return_source") == "unresolved"
    ]
    if io_contract_by_entry and unresolved_entries and not used_io_contract_entries:
        escalation = _codegen_human_finding(
            "codegen.io-contract-entry-unused",
            "IO contract did not match any adapted operator entry.",
            [
                f"unresolved_entries={unresolved_entries}",
                f"contract_entries={sorted(io_contract_by_entry)}",
            ],
        )
        all_escalations.append(escalation)
    entries_by_export_name: dict[str, list[dict[str, Any]]] = {}
    for entry in canonical_entries:
        export_name = _entry_export_key(entry)
        if export_name:
            entries_by_export_name.setdefault(export_name, []).append(entry)
    for export_name, entries in sorted(entries_by_export_name.items()):
        bind_targets = sorted(
            {str(entry.get("bind_target") or "") for entry in entries}
        )
        if len(entries) <= 1 or len(bind_targets) <= 1:
            continue
        selected_targets = sorted(
            {
                _entry_contract_bind_target(entry)
                for entry in entries
                if _entry_contract_bind_target(entry)
            }
        )
        if len(selected_targets) == 1 and selected_targets[0] in bind_targets:
            continue
        escalation = _codegen_human_finding(
            "codegen.pybind-entry-specialization-ambiguous",
            (
                f"operator entry {export_name!r} maps to multiple bind targets {bind_targets}; "
                "provide --io-contract with bind_target, or assign distinct public entry names before packaging."
            ),
            [
                f"entry={export_name}",
                f"bind_targets={bind_targets}",
                "contract_schema={'schema_version':1,'entries':{'entry_name':{'bind_target':'entry<TilingType>',...}}}",
            ],
        )
        all_escalations.append(escalation)
    canonical_written: list[str] = []
    module_name = None
    if not all_escalations and canonical_entries:
        source_entry_names: set[str] = set()
        unique_sources: list[tuple[str, str, list[dict[str, Any]], Path]] = []
        for csrc_name, source_text, entries, source_path in canonical_sources:
            source_entry_name = (
                entries[0]["entry_name"] if entries else Path(csrc_name).stem
            )
            if source_entry_name in source_entry_names:
                continue
            source_entry_names.add(source_entry_name)
            unique_sources.append((csrc_name, source_text, entries, source_path))
        pybind_entries_by_name: dict[str, dict[str, Any]] = {}
        for entry in canonical_entries:
            pybind_entries_by_name.setdefault(entry["entry_name"], entry)
        pybind_entries = list(pybind_entries_by_name.values())

        safe_operator_id = re.sub(
            r"[^A-Za-z0-9_]", "_", pybind_entries[0]["entry_name"]
        )
        module_name = f"{safe_operator_id}_sk_pkg_lib"
        csrc_dir = adapted_dir / "csrc"
        csrc_dir.mkdir(parents=True, exist_ok=True)
        used_csrc_names: set[str] = set()
        for csrc_name, source_text, entries, source_path in unique_sources:
            final_name = csrc_name
            if final_name in used_csrc_names:
                suffix = len(used_csrc_names)
                final_name = f"{Path(csrc_name).stem}_{suffix}.asc"
            used_csrc_names.add(final_name)
            csrc_path = csrc_dir / final_name
            csrc_path.write_text(
                render_aclgraph_kernel_source(source_text, entries), encoding="utf-8"
            )
            for entry in entries:
                entry["csrc_file"] = final_name
            canonical_written.append(f"operator-sk-adapted/csrc/{final_name}")
            canonical_written.extend(
                _copy_aclgraph_quoted_include_dependencies(
                    source_path=source_path,
                    source_text=source_text,
                    csrc_path=csrc_path,
                    adapted_dir=adapted_dir,
                    asset_path=asset_path,
                    support_directories=support_directories,
                )
            )
            for entry in entries:
                canonical_written.extend(
                    _copy_runtime_wrapper_source(
                        entry=entry,
                        csrc_dir=csrc_dir,
                        adapted_dir=adapted_dir,
                        asset_path=asset_path,
                        support_directories=support_directories,
                        used_csrc_names=used_csrc_names,
                    )
                )
        (csrc_dir / "pybind11.asc").write_text(
            render_pybind11_asc(module_name, pybind_entries), encoding="utf-8"
        )
        canonical_written.append("operator-sk-adapted/csrc/pybind11.asc")
        for entry in pybind_entries:
            pybind_name = _aclgraph_entry_pybind_name(entry)
            (csrc_dir / pybind_name).write_text(
                render_pybind11_entry_asc(entry), encoding="utf-8"
            )
            canonical_written.append(f"operator-sk-adapted/csrc/{pybind_name}")
        op_extension = adapted_dir / package_dir_name
        op_extension.mkdir(parents=True, exist_ok=True)
        (op_extension / "__init__.py").write_text(
            render_op_extension_init_py(module_name, pybind_entries), encoding="utf-8"
        )
        (op_extension / "_arch_selector.py").write_text(
            render_arch_selector_py(module_name, pybind_entries), encoding="utf-8"
        )
        (op_extension / "_torch_library.py").write_text(
            render_torch_library_py(module_name, pybind_entries), encoding="utf-8"
        )
        if public_name_resolution is not None:
            (op_extension / "_name_resolution.json").write_text(
                json.dumps(public_name_resolution, indent=2), encoding="utf-8"
            )
        (adapted_dir / "setup.py").write_text(
            render_setup_py_aclgraph_style(
                module_name,
                pybind_entries,
                package_name=package_name,
                package_version=package_version,
            ),
            encoding="utf-8",
        )
        canonical_written.extend(
            [
                f"operator-sk-adapted/{package_dir_name}/__init__.py",
                f"operator-sk-adapted/{package_dir_name}/_arch_selector.py",
                f"operator-sk-adapted/{package_dir_name}/_torch_library.py",
                *(
                    [f"operator-sk-adapted/{package_dir_name}/_name_resolution.json"]
                    if public_name_resolution is not None
                    else []
                ),
                "operator-sk-adapted/setup.py",
            ]
        )
    manifest = {
        "status": "needs-human" if all_escalations else "completed",
        "asset_path": str(asset_path),
        "adapted_dir": str(adapted_dir),
        "pybind_layout": "aclgraph-canonical"
        if canonical_entries and not all_escalations
        else None,
        "pybind_module": module_name,
        "package_name": package_name,
        "package_version": package_version,
        "python_package": package_dir_name,
        "canonical_written_files": canonical_written,
        "legacy_migration_body_sources": [
            {
                "entry_name": entry.get("entry_name"),
                "body_source": entry.get("body_source"),
            }
            for entry in canonical_entries
            if entry.get("body_source")
        ],
        "legacy_helper_evidence": [
            {
                "entry_name": entry.get("entry_name"),
                "mode": entry.get("legacy_helper_evidence"),
            }
            for entry in canonical_entries
            if entry.get("legacy_helper_evidence")
        ],
        "template_specializations": [
            entry["template_specialization"]
            for entry in canonical_entries
            if entry.get("template_specialization")
        ],
        "mask": int(args.mask),
        "num_splits": int(args.num_splits),
        "with_sys_args": args.with_sys_args,
        "target_chips": target_chips,
        "name_resolution": public_name_resolution
        or {"policy": "none", "renamed_entry_count": 0, "resolutions": []},
        "per_file": per_file_results,
        "support_directories": support_directories,
        "ignored_support_files": ignored,
        "escalations": all_escalations,
    }
    (output_dir / "operator-sk-adapted.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


def cmd_aggregate_sk_adapted(args: argparse.Namespace) -> int:
    from sk_codegen_lib import aggregate_aclgraph_adapted_trees

    adapted_roots = [Path(item).resolve() for item in args.adapted_output_dir]
    for adapted_root in adapted_roots:
        if not adapted_root.exists() or not adapted_root.is_dir():
            raise CliUsageError(f"adapted output directory not found: {adapted_root}")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise CliUsageError(f"aggregate output directory not empty: {output_dir}")
    manifest = aggregate_aclgraph_adapted_trees(
        adapted_roots,
        output_dir,
        package_name=args.aggregate_wheel_name,
        package_version=args.package_version,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "package_name": manifest["package_name"],
                "package_version": manifest["package_version"],
                "entries": sum(
                    len(item.get("entries", [])) for item in manifest["per_file"]
                ),
            }
        )
    )
    return 0


def cmd_generate_standalone_compare(args: argparse.Namespace) -> int:
    from sk_codegen_lib import generate_standalone_compare_artifacts

    aggregate_output_dir = Path(args.aggregate_output_dir).resolve()
    if not aggregate_output_dir.exists() or not aggregate_output_dir.is_dir():
        raise CliUsageError(
            f"aggregate output directory not found: {aggregate_output_dir}"
        )
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise CliUsageError(
            f"standalone compare output directory not empty: {output_dir}"
        )
    fixture_dir = (
        Path(args.runtime_fixture_dir).resolve() if args.runtime_fixture_dir else None
    )
    manifest = generate_standalone_compare_artifacts(
        aggregate_output_dir,
        output_dir,
        runtime_fixture_dir=fixture_dir,
        target_chip=args.target_chip or "",
        npu_arch=args.npu_arch or "",
    )
    print(
        json.dumps({"status": manifest["status"], "entries": len(manifest["entries"])})
    )
    return 0


def cmd_apply_remediation(args: argparse.Namespace) -> int:
    from sk_codegen_lib import apply_remediation

    asset_dir = Path(args.asset_dir).resolve()
    if not asset_dir.exists() or not asset_dir.is_dir():
        raise CliUsageError(f"asset directory not found: {asset_dir}")
    findings_path = Path(args.findings_json).resolve()
    if not findings_path.exists():
        raise CliUsageError(f"findings file not found: {findings_path}")

    envelope = json.loads(findings_path.read_text(encoding="utf-8"))
    findings = envelope.get("findings") if isinstance(envelope, dict) else None
    if not isinstance(findings, list):
        raise CliUsageError(
            "findings file must contain a 'findings' list at the top level"
        )

    results = apply_remediation(asset_dir, findings)
    applied = sum(1 for r in results if r["status"] == "applied")
    escalated = sum(1 for r in results if r["status"] == "escalated")
    failed = sum(1 for r in results if r["status"] == "failed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    output_dir = (
        Path(args.output_dir).resolve()
        if getattr(args, "output_dir", None)
        else asset_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "completed",
        "asset_dir": str(asset_dir),
        "findings_path": str(findings_path),
        "summary": {
            "applied": applied,
            "escalated": escalated,
            "failed": failed,
            "skipped": skipped,
        },
        "results": results,
    }
    (output_dir / "operator-sk-remediation.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0 if failed == 0 else 1


def cmd_generate_from_template(args: argparse.Namespace) -> int:
    from sk_codegen_lib import (
        list_templates,
        load_template,
        render_template_files,
        resolve_template_params,
    )

    template_id = args.template_id
    templates_dir = Path(__file__).resolve().parent.parent / "templates"
    found = next(
        (
            item
            for item in list_templates(templates_dir)
            if item.get("id") == template_id and "error" not in item
        ),
        None,
    )
    if found is None:
        raise CliUsageError(f"template not found: {template_id}")
    template = load_template(Path(found["path"]))

    overrides: dict[str, Any] = {}
    for raw in args.param or []:
        if "=" not in raw:
            raise CliUsageError(f"--param must be key=value, got {raw!r}")
        k, v = raw.split("=", 1)
        overrides[k.strip()] = v.strip()
    try:
        params = resolve_template_params(template, overrides)
    except ValueError as exc:
        raise CliUsageError(str(exc)) from exc

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise CliUsageError(f"output directory not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = render_template_files(template, params)
    written: list[str] = []
    for rel, body in files:
        target = (output_dir / rel).resolve()
        try:
            target.relative_to(output_dir.resolve())
        except ValueError:
            raise CliUsageError(
                f"template attempted to write outside output dir: {rel}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        written.append(rel)
    manifest = {
        "status": "generated",
        "template_id": template_id,
        "parameters": params,
        "output_dir": str(output_dir),
        "written_files": written,
    }
    (output_dir / "operator-sk-template-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


def cmd_list_templates(args: argparse.Namespace) -> int:
    from sk_codegen_lib import list_templates

    templates_dir = Path(__file__).resolve().parent.parent / "templates"
    items = list_templates(templates_dir)
    print(json.dumps({"templates": items}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    # sk_codegen_lib is imported lazily inside each cmd_ function so that
    # `--help` works even if optional deps (pyyaml) are unavailable.
    import sys as _sys  # noqa

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in _sys.path:
        _sys.path.insert(0, str(script_dir))

    parser = argparse.ArgumentParser(
        description="Top-level CLI for sk-operator-codegen"
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    intake = subparsers.add_parser(
        "intake", help="Scan one operator asset and generate base delivery artifacts"
    )
    intake.add_argument("asset", help="Operator asset directory or file")
    intake.add_argument(
        "--output-dir", default="operator-delivery-output", help="Output directory"
    )
    intake.add_argument(
        "--with-ai", action="store_true", help="Request optional AI-layer hints"
    )
    intake.set_defaults(func=cmd_intake)

    plan = subparsers.add_parser(
        "plan", help="Generate the same base artifacts as an execution plan"
    )
    plan.add_argument("asset", help="Operator asset directory or file")
    plan.add_argument(
        "--output-dir", default="operator-delivery-output", help="Output directory"
    )
    plan.add_argument(
        "--with-ai", action="store_true", help="Request optional AI-layer hints"
    )
    plan.set_defaults(func=cmd_plan)

    sk_conversion = subparsers.add_parser(
        "analyze-sk-conversion",
        help="Analyze source-to-SK conversion readiness without generating source, building, or validating",
    )
    sk_conversion.add_argument("asset", help="Operator asset directory or file")
    sk_conversion.add_argument(
        "--output-dir", default="operator-delivery-output", help="Output directory"
    )
    sk_conversion.add_argument(
        "--support-dir",
        action="append",
        default=[],
        help="Support source directory to copy into SK source, optionally name=/path",
    )
    sk_conversion.add_argument(
        "--include-dir",
        action="append",
        default=[],
        help="External include directory to use during generated CMake builds",
    )
    sk_conversion.add_argument(
        "--ascend-cann-path",
        help="CANN root containing tools/bisheng_compiler/bin/bisheng for Ascend build validation",
    )
    sk_conversion.add_argument(
        "--ascend-arch",
        help="Ascend NPU arch passed as --npu-arch=<value> when Ascend build validation is enabled",
    )
    sk_conversion.add_argument(
        "--ascend-compile-option",
        action="append",
        default=[],
        help="Extra Bisheng compile option for generated Ascend CMake builds",
    )
    sk_conversion.add_argument(
        "--ascend-force-include",
        action="append",
        default=[],
        help="Header file to force include before generated Ascend source compilation",
    )
    sk_conversion.set_defaults(func=cmd_analyze_sk_conversion)

    sk_binding = subparsers.add_parser(
        "adapt-sk-binding-scaffold",
        help="Generate a reviewable SK binding source scaffold from an exact-current conversion analysis",
    )
    sk_binding.add_argument(
        "analysis_output_dir", help="Output directory produced by analyze-sk-conversion"
    )
    sk_binding.set_defaults(func=cmd_adapt_sk_binding_scaffold)

    sk_source = subparsers.add_parser(
        "generate-sk-source-scaffold",
        help="Generate a reviewable SK source scaffold from an exact-current ready conversion analysis",
    )
    sk_source.add_argument(
        "analysis_output_dir", help="Output directory produced by analyze-sk-conversion"
    )
    sk_source.set_defaults(func=cmd_generate_sk_source_scaffold)

    detect_form = subparsers.add_parser(
        "detect-sk-form",
        help="Classify each source file as none / legacy-spk / current-sk-bind / partial / unknown",
    )
    detect_form.add_argument("asset", help="Operator asset directory or .asc/.cpp file")
    detect_form.add_argument(
        "--output-dir", default="operator-delivery-output", help="Output directory"
    )
    detect_form.set_defaults(func=cmd_detect_sk_form)

    scan_assets = subparsers.add_parser(
        "scan-operator-assets",
        help="Scan source assets into fact inventory and explicit layout decision artifacts",
    )
    scan_assets.add_argument("asset", help="Operator asset directory or source file")
    scan_assets.add_argument("--output-dir", required=True, help="Output directory")
    scan_assets.set_defaults(func=cmd_scan_operator_assets)

    analyze_asset = subparsers.add_parser(
        "analyze-operator-asset",
        help="Analyze an operator asset and emit operator-asset-understanding.json",
    )
    analyze_asset.add_argument("asset", help="Operator asset directory or source file")
    analyze_asset.add_argument("--output-dir", required=True, help="Output directory")
    analyze_asset.add_argument(
        "--asset-layout", help="Explicit operator-asset-layout.json to drive analysis"
    )
    analyze_asset.add_argument(
        "--target-chip",
        default="",
        help="Comma/semicolon separated target chips used to resolve per-op arches",
    )
    analyze_asset.set_defaults(func=cmd_analyze_operator_asset)

    normalize_asset = subparsers.add_parser(
        "normalize-operator-asset",
        help="Normalize an analyzed operator asset into per-operator unit directories",
    )
    normalize_asset.add_argument(
        "asset", help="Operator asset directory or source file"
    )
    normalize_asset.add_argument(
        "--understanding", help="Existing operator-asset-understanding.json"
    )
    normalize_asset.add_argument(
        "--asset-layout",
        help="Explicit operator-asset-layout.json to drive normalization",
    )
    normalize_asset.add_argument("--output-dir", required=True, help="Output directory")
    normalize_asset.add_argument(
        "--target-chip",
        default="",
        help="Comma/semicolon separated target chips used to resolve per-op arches",
    )
    normalize_asset.set_defaults(func=cmd_normalize_operator_asset)

    adapt_sk = subparsers.add_parser(
        "adapt-sk-from-global",
        help="Convert supported .asc/.cpp input forms to current SK_BIND source or emit human escalations",
    )
    adapt_sk.add_argument("asset", help="Operator asset directory or .asc/.cpp file")
    adapt_sk.add_argument(
        "--output-dir",
        required=True,
        help="Output directory; operator-sk-adapted/ will be created here",
    )
    adapt_sk.add_argument(
        "--understanding",
        help="operator-asset-understanding.json used to carry per-entry support metadata",
    )
    adapt_sk.add_argument(
        "--target-chip",
        default="",
        help="Comma/semicolon separated target chips used to resolve sparse arch builds",
    )
    adapt_sk.add_argument(
        "--package-name",
        default="op_extension",
        help="Distribution/Python package name for generated pybind project",
    )
    adapt_sk.add_argument(
        "--package-version", default="0.1.0", help="Distribution package version"
    )
    adapt_sk.add_argument(
        "--name-resolution",
        help="name-resolution-report.json used to namespace duplicate public entry names",
    )
    adapt_sk.add_argument(
        "--source-asset",
        help="Original source asset path used to match name-resolution entries",
    )
    adapt_sk.add_argument(
        "--io-contract",
        help=(
            "JSON contract declaring tensor IO semantics per entry. "
            "Schema: {'schema_version':1,'entries':{'entry':{'inputs':[...],'outputs':[...],'pybind_return_tensor':'...'}}}."
        ),
    )
    adapt_sk.add_argument(
        "--mask",
        default="4",
        help="SK_BIND mask: 4=DCCI default; accepts bit combinations 0..7",
    )
    adapt_sk.add_argument(
        "--num-splits", default="4", help="Number of SK split symbols (1..4)"
    )
    adapt_sk.add_argument(
        "--support-dir",
        action="append",
        default=[],
        help="Support source directory to copy into generated csrc, optionally name=/path",
    )
    adapt_sk.add_argument(
        "--with-sys-args",
        choices=["auto", "always", "never"],
        default="auto",
        help="Whether to inject `sk::SkSystemArgs *sysArgs` parameter (auto detects AscendC::GetBlockNum() usage)",
    )
    adapt_sk.set_defaults(func=cmd_adapt_sk_from_global)

    aggregate_sk = subparsers.add_parser(
        "aggregate-sk-adapted",
        help="Aggregate multiple aclgraph-canonical adapted outputs into one package tree",
    )
    aggregate_sk.add_argument(
        "--adapted-output-dir",
        action="append",
        required=True,
        help="Output directory containing operator-sk-adapted.json and operator-sk-adapted/ (repeatable)",
    )
    aggregate_sk.add_argument(
        "--output-dir",
        required=True,
        help="Fresh directory for aggregate operator-sk-adapted/ output",
    )
    aggregate_sk.add_argument(
        "--aggregate-wheel-name",
        default="op_extension",
        help="Distribution package name for the aggregate wheel",
    )
    aggregate_sk.add_argument(
        "--package-version",
        default="0.1.0",
        help="Distribution package version for the aggregate wheel",
    )
    aggregate_sk.set_defaults(func=cmd_aggregate_sk_adapted)

    standalone_compare = subparsers.add_parser(
        "generate-standalone-compare",
        help="Generate a standalone ASC executable project for bind-target baseline vs SK differential validation",
    )
    standalone_compare.add_argument(
        "aggregate_output_dir",
        help="Aggregate output directory containing operator-sk-adapted.json and operator-sk-adapted/",
    )
    standalone_compare.add_argument(
        "--output-dir",
        required=True,
        help="Fresh directory for operator-sk-standalone-verify output",
    )
    standalone_compare.add_argument(
        "--runtime-fixture-dir", help="Directory containing per-op runtime fixtures"
    )
    standalone_compare.add_argument(
        "--target-chip",
        default="",
        help="Target chip recorded in the standalone manifest",
    )
    standalone_compare.add_argument(
        "--npu-arch",
        default="",
        help="ASC --npu-arch value. When omitted, --target-chip must resolve to one source-backed arch.",
    )
    standalone_compare.set_defaults(func=cmd_generate_standalone_compare)

    apply_rem = subparsers.add_parser(
        "apply-remediation",
        help="Apply auto-remediable findings (rename-symbol / remove-line-containing / add-include / replace-pattern) to source files",
    )
    apply_rem.add_argument(
        "asset_dir",
        help="Directory whose files the findings reference (target_file paths are relative to this)",
    )
    apply_rem.add_argument("findings_json", help="Path to a findings envelope JSON")
    apply_rem.add_argument(
        "--output-dir",
        help="Where to write operator-sk-remediation.json (defaults to asset_dir)",
    )
    apply_rem.set_defaults(func=cmd_apply_remediation)

    gen_tmpl = subparsers.add_parser(
        "generate-from-template",
        help="Render a templates/<id>.yaml into a fresh source tree (for kicking off the closed-loop pipeline)",
    )
    gen_tmpl.add_argument(
        "template_id", help="Template id (matches templates/<id>.yaml)"
    )
    gen_tmpl.add_argument(
        "--param",
        action="append",
        help="Template parameter override: key=value (repeatable)",
    )
    gen_tmpl.add_argument(
        "--output-dir", required=True, help="Output directory (must be empty or absent)"
    )
    gen_tmpl.set_defaults(func=cmd_generate_from_template)

    list_tmpl = subparsers.add_parser(
        "list-templates",
        help="List available codegen templates under templates/*.yaml",
    )
    list_tmpl.set_defaults(func=cmd_list_templates)

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
