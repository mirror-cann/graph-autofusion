# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Auto sample-generation library for sk-operator-sample-gen.

Self-contained (no cross-skill imports). Provides the "auto" half of the sample
chain: it synthesises the *closed spec files* a human would otherwise hand-write,
so the existing manual commands (provide-sk-runtime-input-values,
collect-correctness-oracle-spec, run-sk-target-runtime-validation) validate and
canonicalise them unchanged. The orchestrator wires construction (auto) to
validation (manual).

Pieces:
- build_input_values_spec(runtime_input_spec, shape, dtype, fill): produce the
  closed input-values spec dict (schema {"input_values": [...]}).
- ReferenceImplRegistry: load reference_impls/<name>.py modules and look up by
  kernel entry name. Each module exposes REFERENCE_IMPL metadata + compute(inputs).
- build_oracle_spec(input_values_artifact, registry): compatibility path that
  computes expected outputs per input set via numpy reference implementations.
- build_bind_target_oracle_spec(input_values_artifact, runtime_input_spec):
  declare an oracle whose expected values are produced by the bind target's
  baseline chevron path.
- render_runner_script / build_target_runtime_command_spec: emit a runnable sample
  runner plus the command spec consumed by run-sk-target-runtime-validation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Optional

_SCALAR_CTYPE_TO_DTYPE = {
    "uint8_t": "uint8",
    "int8_t": "int8",
    "uint16_t": "uint16",
    "int16_t": "int16",
    "uint32_t": "uint32",
    "int32_t": "int32",
    "uint64_t": "uint64",
    "int64_t": "int64",
    "float": "float32",
    "half": "float16",
}

_TENSOR_TYPE_MARKERS = ("GM_ADDR", "__gm__", "*")


def _is_tensor_param(c_type: str) -> bool:
    return any(marker in c_type for marker in _TENSOR_TYPE_MARKERS)


def _zero_nested(shape: list[int]) -> Any:
    if not shape:
        return 0
    return [_zero_nested(shape[1:]) for _ in range(shape[0])]


def _scalar_dtype(c_type: str) -> str:
    for key, label in _SCALAR_CTYPE_TO_DTYPE.items():
        if key in c_type:
            return label
    return "int32"


def build_input_values_spec(
    runtime_input_spec: dict[str, Any],
    *,
    shape: list[int],
    dtype: str,
    fill: str = "zero",
) -> dict[str, Any]:
    """Synthesise a closed input-values spec from a runtime-input-spec.

    Tensor parameters (GM_ADDR / pointer types) get a zero-filled tensor of the
    declared shape/dtype. Scalar parameters get the element count of `shape`
    (a sane length-style default) with a dtype derived from the C type.
    """
    if fill != "zero":
        raise ValueError(f"unsupported fill mode {fill!r}; only 'zero' is supported")
    element_count = 1
    for dim in shape:
        element_count *= dim

    input_values: list[dict[str, Any]] = []
    for input_spec in runtime_input_spec["input_specs"]:
        parameter_values: list[dict[str, Any]] = []
        for param in input_spec["parameters"]:
            c_type = param["type"]
            if _is_tensor_param(c_type):
                parameter_values.append(
                    {
                        "name": param["name"],
                        "shape": list(shape),
                        "dtype": dtype,
                        "layout": "ND",
                        "value_source": {
                            "kind": "inline_json",
                            "value": _zero_nested(list(shape)),
                        },
                    }
                )
            else:
                parameter_values.append(
                    {
                        "name": param["name"],
                        "shape": [],
                        "dtype": _scalar_dtype(c_type),
                        "layout": "ND",
                        "value_source": {"kind": "inline_json", "value": element_count},
                    }
                )
        input_values.append(
            {"input_set_id": input_spec["id"], "parameter_values": parameter_values}
        )
    return {"input_values": input_values}


# ---------- reference impl registry ----------


class ReferenceImplRegistry:
    """Indexes reference_impls/<name>.py modules by kernel entry name.

    Each module exposes:
        REFERENCE_IMPL = {"entry_name": "add_custom", "comparator": "exact"|"allclose",
                          "tolerance": {"rtol": .., "atol": ..}, "description": "..."}
        def compute(inputs: dict[str, dict]) -> Any   # returns the expected output value
    where inputs maps parameter name -> {"shape", "dtype", "value"}.
    """

    def __init__(self) -> None:
        self._by_entry: dict[str, dict[str, Any]] = {}

    @classmethod
    def load(cls, reference_dir: Path) -> "ReferenceImplRegistry":
        registry = cls()
        if not reference_dir.exists():
            return registry
        for path in sorted(reference_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            spec = importlib.util.spec_from_file_location(
                f"sk_refimpl_{path.stem}", path
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            meta = getattr(module, "REFERENCE_IMPL", None)
            compute = getattr(module, "compute", None)
            if isinstance(meta, dict) and callable(compute):
                entry = meta.get("entry_name") or path.stem
                registry._by_entry[entry] = {"meta": meta, "compute": compute}
        return registry

    def lookup(self, entry_name: str) -> Optional[dict[str, Any]]:
        return self._by_entry.get(entry_name)

    def entry_names(self) -> list[str]:
        return sorted(self._by_entry)


def build_oracle_spec(
    input_values_artifact: dict[str, Any],
    runtime_input_spec: dict[str, Any],
    registry: ReferenceImplRegistry,
) -> tuple[dict[str, Any], list[str]]:
    """Compute expected outputs for each input set, returning (oracle_spec, missing_entries).

    missing_entries lists input_set_ids whose kernel entry has no reference impl
    (the caller turns these into human-escalation findings).
    """
    # Map input_set_id -> entry_name via the runtime-input-spec.
    entry_by_input_set = {}
    for spec in runtime_input_spec["input_specs"]:
        entry_by_input_set[spec["id"]] = spec["entry_name"]

    oracle_specs: list[dict[str, Any]] = []
    missing: list[str] = []
    for input_set in input_values_artifact["input_values"]:
        input_set_id = input_set["input_set_id"]
        entry_name = entry_by_input_set.get(input_set_id, "")
        ref = registry.lookup(entry_name)
        if ref is None:
            missing.append(input_set_id)
            continue
        inputs = {
            pv["name"]: {
                "shape": pv["shape"],
                "dtype": pv["dtype"],
                "value": pv["value_source"]["value"],
            }
            for pv in input_set["parameter_values"]
        }
        expected_value = ref["compute"](inputs)
        meta = ref["meta"]
        comparator = meta.get("comparator", "exact")
        tolerance = meta.get("tolerance", {"rtol": 0, "atol": 0})
        oracle_specs.append(
            {
                "input_set_id": input_set_id,
                "expected_output": {"kind": "inline_json", "value": expected_value},
                "comparator": comparator,
                "tolerance": tolerance,
            }
        )
    return {"oracle_specs": oracle_specs}, missing


def build_bind_target_oracle_spec(
    input_values_artifact: dict[str, Any],
    runtime_input_spec: dict[str, Any],
) -> dict[str, Any]:
    """Declare runner-produced bind-target baseline values as the oracle source."""
    entry_by_input_set = {}
    for spec in runtime_input_spec["input_specs"]:
        entry_by_input_set[spec["id"]] = spec["entry_name"]
    oracle_specs: list[dict[str, Any]] = []
    for input_set in input_values_artifact["input_values"]:
        input_set_id = input_set["input_set_id"]
        entry_name = entry_by_input_set.get(input_set_id, "")
        oracle_specs.append(
            {
                "input_set_id": input_set_id,
                "expected_output": {
                    "kind": "inline_json",
                    "value": {
                        "source": "bind-target-on-wheel",
                        "entry_name": entry_name,
                    },
                },
                "comparator": "allclose",
                "tolerance": {"rtol": 0, "atol": 0},
            }
        )
    return {"oracle_specs": oracle_specs}


# ---------- runner script + command spec ----------

_RUNNER_TEMPLATE = '''\
#!/usr/bin/env python3
"""Auto-generated SK sample runner. Loads the values manifest passed as argv[1],
invokes both the bind-target baseline and SK operator paths when available, and
prints structured stdout JSON with an \"outputs\" object keyed by operator name."""
import importlib
import json
import os
from pathlib import Path
import sys


def _entry_name(input_set):
    explicit = input_set.get("entry_name")
    if explicit:
        return explicit
    input_set_id = input_set.get("input_set_id", "")
    if input_set_id.startswith("entry:"):
        parts = input_set_id.split(":")
        if len(parts) >= 2:
            return parts[1]
    return input_set_id.replace(":", "_")


def _primary_value(input_set):
    if not input_set.get("parameter_values"):
        return []
    for parameter in input_set["parameter_values"]:
        shape = parameter.get("shape", [])
        if shape:
            return parameter["value_source"]["value"]
    return input_set["parameter_values"][0]["value_source"]["value"]


def _load_package():
    try:
        return importlib.import_module("op_extension")
    except Exception as exc:
        return None, f"import_op_extension_failed:{type(exc).__name__}"


def _to_jsonable(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _callable_args(input_set):
    args = []
    try:
        import torch
    except Exception:
        torch = None
    for parameter in input_set["parameter_values"]:
        value = parameter["value_source"]["value"]
        if parameter.get("shape") and torch is not None:
            args.append(torch.tensor(value))
        else:
            args.append(value)
    return args


def _call_pair(op_name, input_set):
    fallback = _primary_value(input_set)
    if os.getenv("SK_OPERATOR_MOCK_NPU") == "1":
        return {
            "output": {"baseline": fallback, "sk": fallback},
            "calls": [f"run_{op_name}:baseline", f"torch.ops.ascendc_ops.{op_name}:sk"],
            "status": {"status": "mock-passed", "reason": "mock_npu_bind_target_route_matched"},
        }
    package, package_error = _load_package()
    if package is None:
        return {
            "output": None,
            "calls": [],
            "status": {"status": "failed", "reason": package_error},
        }
    sk_fn = getattr(package, f"run_{op_name}", None)
    if sk_fn is None:
        return {
            "output": None,
            "calls": [],
            "status": {"status": "failed", "reason": f"baseline_entry_missing:run_{op_name}"},
        }
    args = _callable_args(input_set)
    try:
        baseline = _to_jsonable(sk_fn(*args))
    except Exception as exc:
        return {
            "output": None,
            "calls": [f"run_{op_name}:baseline"],
            "status": {"status": "failed", "reason": f"baseline_call_failed:{type(exc).__name__}"},
        }
    try:
        import torch
        torch_ops = getattr(torch.ops, "ascendc_ops")
        torch_fn = getattr(torch_ops, op_name)
        sk = _to_jsonable(torch_fn(*args))
        sk_call = f"torch.ops.ascendc_ops.{op_name}:sk"
    except Exception as exc:
        return {
            "output": None,
            "calls": [f"run_{op_name}:baseline"],
            "status": {"status": "failed", "reason": f"sk_dispatch_failed:{type(exc).__name__}"},
        }
    return {
        "output": {"baseline": baseline, "sk": sk},
        "calls": [f"run_{op_name}:baseline", sk_call],
        "status": {"status": "passed", "reason": "bind_target_baseline_and_sk_outputs_collected"},
    }


def _overall_status(statuses):
    values = [item.get("status", "failed") for item in statuses.values()]
    if any(value == "failed" for value in values):
        return "failed"
    if values and all(value == "mock-passed" for value in values):
        return "mock-passed"
    if values and all(value == "passed" for value in values):
        return "passed"
    return "mixed"


def main() -> int:
    manifest_path = sys.argv[1]
    with open(manifest_path, encoding="utf-8") as f:
        values = json.load(f)
    outputs = {}
    calls = {}
    statuses = {}
    for input_set in values["input_values"]:
        op_name = _entry_name(input_set)
        result = _call_pair(op_name, input_set)
        if result["output"] is not None:
            outputs[op_name] = result["output"]
        calls[op_name] = result["calls"]
        statuses[op_name] = result["status"]
    status = _overall_status(statuses)
    print(json.dumps({"status": status, "outputs": outputs, "calls": calls, "statuses": statuses}))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def render_runner_script(oracle_source: str = "bind-target-on-wheel") -> str:
    if oracle_source not in {"bind-target-on-wheel", "reference-impls-numpy"}:
        raise ValueError(f"unsupported oracle source {oracle_source!r}")
    return _RUNNER_TEMPLATE


def build_target_runtime_command_spec(
    runner_rel_path: str, values_manifest_rel: str
) -> dict[str, Any]:
    """Build the closed command spec consumed by run-sk-target-runtime-validation.

    The values manifest path is bound as argv[2] (1-based index 2) via the
    values_manifest_argv binding the validator requires.
    """
    return {
        "runtime_command": {
            "argv": ["python3", runner_rel_path, values_manifest_rel],
            "cwd": ".",
            "timeout_seconds": 60,
            "env": {},
            "input_binding": {"kind": "values_manifest_argv", "argv_index": 2},
        }
    }


def build_standalone_runtime_fixture_spec(
    entry_name: str, fixture_path: str, *, available: bool
) -> dict[str, Any]:
    return {
        "backend": "standalone",
        "entry_name": entry_name,
        "fixture": {
            "status": "available" if available else "insufficient",
            "path": fixture_path,
        },
    }


def build_standalone_insufficient_fixture_verdict(
    entry_name: str, reason: str = "runtime fixture not available"
) -> dict[str, Any]:
    return {
        "backend": "standalone",
        "entry_name": entry_name,
        "status": "skipped-insufficient-runtime-spec",
        "reason": reason,
        "outputs": {},
    }


# ---------- single-op and network verification contracts ----------

_PARAMETER_ROLES = {
    "input",
    "output",
    "workspace",
    "tiling",
    "scalar",
    "descriptor",
    "unknown",
}
_COMPARATORS = {"exact", "allclose", "bytewise"}


def _as_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _as_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _as_string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    if not allow_empty and not value:
        raise ValueError(f"{label} must be non-empty")
    return value


def _as_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _normalize_comparator(value: Any, label: str) -> str:
    comparator = _as_string(value, label)
    if comparator not in _COMPARATORS:
        raise ValueError(f"{label} must be one of {sorted(_COMPARATORS)}")
    return comparator


def _normalize_tolerance(value: Any, comparator: str) -> dict[str, int | float] | None:
    if comparator != "allclose":
        return None
    tolerance = _as_object(
        value if value is not None else {"rtol": 0, "atol": 0}, "tolerance"
    )
    normalized: dict[str, int | float] = {}
    for key in ("rtol", "atol"):
        raw = tolerance.get(key, 0)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"tolerance.{key} must be a number")
        normalized[key] = raw
    return normalized


def _runtime_parameter_role(parameter: dict[str, Any]) -> str:
    raw_role = parameter.get("role")
    if raw_role is None and parameter.get("kind") == "scalar":
        return "scalar"
    role = raw_role if raw_role is not None else "unknown"
    role = _as_string(role, f"parameter {parameter.get('name', '<unknown>')} role")
    if role not in _PARAMETER_ROLES:
        raise ValueError(f"invalid parameter role {role!r}")
    return role


def build_single_op_verification_contract(
    runtime_contract: dict[str, Any],
) -> dict[str, Any]:
    """Build the canonical single-op differential verification contract.

    This contract intentionally does not reference a wheel. It records that the
    same explicit input fixture must be launched through the non-SK and SK
    standalone entries, then compared through declared output buffers.
    """
    payload = _as_object(runtime_contract, "runtime contract")
    entry_name = _as_string(payload.get("entry_name"), "runtime contract entry_name")
    raw_parameters = _as_list(payload.get("parameters"), "runtime contract parameters")
    parameters: list[dict[str, Any]] = []
    for index, raw_parameter in enumerate(raw_parameters):
        parameter = _as_object(raw_parameter, "runtime contract parameter")
        name = _as_string(parameter.get("name"), "runtime contract parameter name")
        role = _runtime_parameter_role(parameter)
        compare = _as_bool(parameter.get("compare", False), f"parameter {name} compare")
        normalized = {
            "index": int(parameter.get("index", index)),
            "name": name,
            "kind": _as_string(
                parameter.get("kind", "unknown"), f"parameter {name} kind"
            ),
            "role": role,
            "compare": compare,
            "dtype": parameter.get("dtype"),
            "shape": parameter.get("shape"),
            "bytes": parameter.get("bytes"),
            "source": parameter.get("source", ""),
        }
        parameters.append(normalized)

    comparison = _as_object(
        payload.get("comparison", {}), "runtime contract comparison"
    )
    declared_outputs = []
    for item in _as_list(
        comparison.get("outputs", []), "runtime contract comparison outputs"
    ):
        declared_outputs.append(_as_string(item, "runtime contract comparison output"))
    if not declared_outputs:
        for parameter in parameters:
            if parameter["compare"]:
                declared_outputs.append(parameter["name"])

    by_name = {parameter["name"]: parameter for parameter in parameters}
    unknown_outputs = []
    non_output_roles = []
    for name in declared_outputs:
        if name not in by_name:
            unknown_outputs.append(name)
        elif by_name[name]["role"] not in {"output"}:
            non_output_roles.append(name)
    device_runnable = (
        _as_object(payload.get("fixture", {}), "runtime contract fixture").get(
            "device_runnable"
        )
        is True
    )
    required_user_declarations: list[str] = []
    checks: list[dict[str, Any]] = [
        {
            "name": "single_op_requires_no_wheel",
            "status": "passed",
            "reason": "single_op_uses_standalone_non_sk_and_sk_entries",
            "evidence": [],
        }
    ]
    if not device_runnable:
        required_user_declarations.append("fixture.device_runnable")
        checks.append(
            {
                "name": "single_op_fixture_runnable",
                "status": "blocked",
                "reason": "runtime_fixture_not_device_runnable",
                "evidence": [],
            }
        )
    else:
        checks.append(
            {
                "name": "single_op_fixture_runnable",
                "status": "passed",
                "reason": "runtime_fixture_device_runnable",
                "evidence": [],
            }
        )
    if not declared_outputs:
        required_user_declarations.append("comparison.outputs")
        checks.append(
            {
                "name": "single_op_declared_outputs",
                "status": "blocked",
                "reason": "comparable_outputs_not_declared",
                "evidence": [],
            }
        )
    elif unknown_outputs:
        required_user_declarations.append("comparison.outputs")
        checks.append(
            {
                "name": "single_op_declared_outputs",
                "status": "blocked",
                "reason": "comparison_outputs_unknown",
                "evidence": unknown_outputs,
            }
        )
    elif non_output_roles:
        required_user_declarations.append("parameters.role")
        checks.append(
            {
                "name": "single_op_declared_outputs",
                "status": "blocked",
                "reason": "comparison_outputs_must_have_output_role",
                "evidence": non_output_roles,
            }
        )
    else:
        checks.append(
            {
                "name": "single_op_declared_outputs",
                "status": "passed",
                "reason": "comparable_outputs_declared",
                "evidence": declared_outputs,
            }
        )

    status = (
        "available" if not required_user_declarations else "needs-user-confirmation"
    )
    reason = (
        "single_op_differential_verification_contract_available"
        if status == "available"
        else "single_op_differential_verification_requires_explicit_contract"
    )
    return {
        "schema_version": 1,
        "verification_level": "single-op",
        "backend": "standalone",
        "status": status,
        "reason": reason,
        "entry_name": entry_name,
        "requires_wheel": False,
        "execution": {
            "baseline": "non_sk_entry",
            "actual": "sk_entry",
            "input_policy": "same_explicit_runtime_fixture",
        },
        "parameters": parameters,
        "comparison": {
            "mode": _as_object(
                payload.get("comparison", {}), "runtime contract comparison"
            ).get("mode", "bytewise"),
            "outputs": declared_outputs,
            "tolerance": _as_object(
                payload.get("comparison", {}), "runtime contract comparison"
            ).get("tolerance"),
        },
        "fixture": payload.get("fixture", {}),
        "required_user_declarations": required_user_declarations,
        "checks": checks,
    }


def normalize_network_sample_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Canonicalise a network-level wheel verification contract.

    The network contract is intentionally explicit: topology, comparable final
    outputs, runner adapter, package, and expected SK fusion coverage must all be
    declared by the user or an adapter skill before this skill generates a
    runner.
    """
    raw = _as_object(payload, "network sample contract")
    schema_version = raw.get("schema_version", 1)
    if schema_version != 1:
        raise ValueError("network sample contract schema_version must be 1")
    network_name = _as_string(
        raw.get("network_name"), "network sample contract network_name"
    )
    package = _as_object(raw.get("package"), "network sample contract package")
    package_name = _as_string(
        package.get("name"), "network sample contract package.name"
    )
    runner = _as_object(raw.get("runner"), "network sample contract runner")
    runner_kind = _as_string(runner.get("kind"), "network sample contract runner.kind")
    if runner_kind not in {"python_module", "generated_mock"}:
        raise ValueError(
            "network sample contract runner.kind must be python_module or generated_mock"
        )
    canonical_runner = {"kind": runner_kind}
    if runner_kind == "python_module":
        canonical_runner["module"] = _as_string(
            runner.get("module"), "network sample contract runner.module"
        )
        canonical_runner["callable"] = _as_string(
            runner.get("callable"), "network sample contract runner.callable"
        )

    nodes: list[dict[str, str]] = []
    seen_node_ids: set[str] = set()
    node_ops: list[str] = []
    for raw_node in _as_list(raw.get("nodes"), "network sample contract nodes"):
        node = _as_object(raw_node, "network sample contract node")
        node_id = _as_string(node.get("id"), "network sample contract node.id")
        op = _as_string(node.get("op"), "network sample contract node.op")
        if node_id in seen_node_ids:
            raise ValueError(f"duplicate network node id: {node_id}")
        seen_node_ids.add(node_id)
        node_ops.append(op)
        nodes.append({"id": node_id, "op": op})

    edges: list[dict[str, str]] = []
    for raw_edge in _as_list(raw.get("edges", []), "network sample contract edges"):
        edge = _as_object(raw_edge, "network sample contract edge")
        edges.append(
            {
                "from": _as_string(
                    edge.get("from"), "network sample contract edge.from"
                ),
                "to": _as_string(edge.get("to"), "network sample contract edge.to"),
            }
        )

    inputs: list[dict[str, Any]] = []
    for raw_input in _as_list(raw.get("inputs", []), "network sample contract inputs"):
        item = _as_object(raw_input, "network sample contract input")
        shape = []
        for dim in _as_list(item.get("shape"), "network sample contract input.shape"):
            shape.append(int(dim))
        inputs.append(
            {
                "id": _as_string(item.get("id"), "network sample contract input.id"),
                "dtype": _as_string(
                    item.get("dtype"), "network sample contract input.dtype"
                ),
                "shape": shape,
            }
        )

    outputs: list[dict[str, Any]] = []
    comparable_outputs: list[str] = []
    for raw_output in _as_list(raw.get("outputs"), "network sample contract outputs"):
        item = _as_object(raw_output, "network sample contract output")
        output_id = _as_string(item.get("id"), "network sample contract output.id")
        compare = _as_bool(
            item.get("compare"), "network sample contract output.compare"
        )
        comparator = _normalize_comparator(
            item.get("comparator", "exact"), "network sample contract output.comparator"
        )
        tolerance = _normalize_tolerance(item.get("tolerance"), comparator)
        outputs.append(
            {
                "id": output_id,
                "compare": compare,
                "comparator": comparator,
                "tolerance": tolerance,
            }
        )
        if compare:
            comparable_outputs.append(output_id)

    expected_fusion = _as_object(
        raw.get("expected_fusion"), "network sample contract expected_fusion"
    )
    expected_ops = []
    for item in _as_list(
        expected_fusion.get("ops"), "network sample contract expected_fusion.ops"
    ):
        expected_ops.append(
            _as_string(item, "network sample contract expected_fusion.ops item")
        )
    require_all_fused = _as_bool(
        expected_fusion.get("require_all_fused", True),
        "network sample contract expected_fusion.require_all_fused",
    )
    unknown_expected_ops = sorted(set(expected_ops) - set(node_ops))
    missing_node_ops = (
        sorted(set(node_ops) - set(expected_ops)) if require_all_fused else []
    )

    required_user_declarations: list[str] = []
    checks: list[dict[str, Any]] = []
    if comparable_outputs:
        checks.append(
            {
                "name": "network_comparable_outputs_declared",
                "status": "passed",
                "reason": "network_comparable_outputs_declared",
                "evidence": comparable_outputs,
            }
        )
    else:
        required_user_declarations.append("outputs.compare")
        checks.append(
            {
                "name": "network_comparable_outputs_declared",
                "status": "blocked",
                "reason": "network_comparable_outputs_not_declared",
                "evidence": [],
            }
        )
    if unknown_expected_ops:
        required_user_declarations.append("expected_fusion.ops")
        checks.append(
            {
                "name": "network_expected_fusion_ops_declared",
                "status": "blocked",
                "reason": "expected_fusion_ops_not_in_network_nodes",
                "evidence": unknown_expected_ops,
            }
        )
    elif missing_node_ops:
        required_user_declarations.append("expected_fusion.ops")
        checks.append(
            {
                "name": "network_expected_fusion_ops_declared",
                "status": "blocked",
                "reason": "require_all_fused_missing_network_ops",
                "evidence": missing_node_ops,
            }
        )
    else:
        checks.append(
            {
                "name": "network_expected_fusion_ops_declared",
                "status": "passed",
                "reason": "network_expected_fusion_ops_declared",
                "evidence": expected_ops,
            }
        )
    checks.append(
        {
            "name": "network_requires_wheel",
            "status": "passed",
            "reason": "network_verification_uses_packaged_wheel_surface",
            "evidence": [package_name],
        }
    )

    status = (
        "available" if not required_user_declarations else "needs-user-confirmation"
    )
    return {
        "schema_version": 1,
        "verification_level": "network",
        "status": status,
        "reason": "network_sample_contract_available"
        if status == "available"
        else "network_sample_contract_requires_explicit_user_declarations",
        "network_name": network_name,
        "network_kind": _as_string(
            raw.get("network_kind", network_name),
            "network sample contract network_kind",
        ),
        "package": {
            "name": package_name,
            "wheel": _as_string(
                package.get("wheel", ""),
                "network sample contract package.wheel",
                allow_empty=True,
            ),
            "arch": _as_string(
                package.get("arch", ""),
                "network sample contract package.arch",
                allow_empty=True,
            ),
        },
        "runner": canonical_runner,
        "nodes": nodes,
        "edges": edges,
        "inputs": inputs,
        "outputs": outputs,
        "run": _as_object(raw.get("run", {}), "network sample contract run"),
        "adapter_config": _as_object(
            raw.get("adapter_config", {}), "network sample contract adapter_config"
        ),
        "prepare": _as_list(raw.get("prepare", []), "network sample contract prepare"),
        "expected_fusion": {
            "scope": _as_string(
                expected_fusion.get("scope", ""),
                "network sample contract expected_fusion.scope",
                allow_empty=True,
            ),
            "ops": expected_ops,
            "require_all_fused": require_all_fused,
        },
        "comparison": {
            "baseline": "aclgraph",
            "actual": "aclgraph+sk",
            "outputs": comparable_outputs,
        },
        "requires_wheel": True,
        "required_user_declarations": required_user_declarations,
        "checks": checks,
    }


_NETWORK_RUNNER_TEMPLATE = '''\
#!/usr/bin/env python3
"""Generated network-level SK verification runner.

The runner does not guess operator semantics. In real mode it delegates network
construction and execution to the contract's runner adapter. In mock mode it
only validates the route and stdout schema.
"""
import importlib
import json
import os
from pathlib import Path
import sys


def _mock_result(contract):
    outputs = {item["id"]: [] for item in contract.get("outputs", []) if item.get("compare")}
    return {
        "status": "mock-passed",
        "network_name": contract["network_name"],
        "package": contract["package"]["name"],
        "runs": [
            {"run_mode": "aclgraph", "outputs": outputs},
            {"run_mode": "aclgraph+sk", "outputs": outputs},
        ],
        "fusion": {
            "status": "mocked",
            "expected_scope": contract["expected_fusion"].get("scope", ""),
            "expected_ops": contract["expected_fusion"]["ops"],
            "require_all_fused": contract["expected_fusion"]["require_all_fused"],
        },
    }


def _real_result(contract, contract_path):
    runner = contract["runner"]
    module = importlib.import_module(runner["module"])
    fn = getattr(module, runner["callable"])
    resolved_contract_path = Path(contract_path).resolve()
    context = {
        "contract_path": str(resolved_contract_path),
        "contract_dir": str(resolved_contract_path.parent),
        "cwd": os.getcwd(),
    }
    try:
        result = fn(contract, context)
    except TypeError as exc:
        try:
            result = fn(contract)
        except TypeError:
            raise exc
    if not isinstance(result, dict):
        raise TypeError("network runner adapter must return a dict")
    return result


def main() -> int:
    if len(sys.argv) != 2:
        print(json.dumps({"status": "failed", "reason": "usage: runner <operator-network-sample-contract.json>"}))
        return 2
    contract_path = sys.argv[1]
    with open(contract_path, encoding="utf-8") as f:
        contract = json.load(f)
    try:
        if os.getenv("SK_OPERATOR_MOCK_NPU") == "1" or contract["runner"]["kind"] == "generated_mock":
            result = _mock_result(contract)
        else:
            result = _real_result(contract, contract_path)
    except Exception as exc:
        result = {"status": "failed", "reason": f"network_runner_failed:{type(exc).__name__}"}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def render_network_runner_script() -> str:
    return _NETWORK_RUNNER_TEMPLATE


def build_network_target_runtime_command_spec(
    runner_rel_path: str, contract_rel_path: str
) -> dict[str, Any]:
    return {
        "runtime_command": {
            "argv": ["python3", runner_rel_path, contract_rel_path],
            "cwd": ".",
            "timeout_seconds": 600,
            "env": {},
            "contract_binding": {"kind": "network_contract_argv", "argv_index": 2},
        }
    }


def build_network_fusion_expectation(
    network_contract: dict[str, Any],
) -> dict[str, Any]:
    contract = normalize_network_sample_contract(network_contract)
    return {
        "schema_version": 1,
        "network_name": contract["network_name"],
        "network_contract_path": "operator-network-sample-contract.json",
        "sk_meta_path": "",
        "expected_fusion": contract["expected_fusion"],
        "expected_nodes": contract["nodes"],
        "checks": [
            {
                "name": "expected_fusion_scope_declared",
                "status": "passed" if contract["expected_fusion"]["scope"] else "open",
                "reason": "expected_fusion_scope_declared"
                if contract["expected_fusion"]["scope"]
                else "expected_fusion_scope_optional",
                "evidence": [contract["expected_fusion"]["scope"]]
                if contract["expected_fusion"]["scope"]
                else [],
            },
            {
                "name": "expected_fusion_ops_declared",
                "status": "passed",
                "reason": "expected_fusion_ops_declared",
                "evidence": contract["expected_fusion"]["ops"],
            },
        ],
    }
