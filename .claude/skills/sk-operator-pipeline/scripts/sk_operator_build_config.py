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
import os
from pathlib import Path
from typing import Any


BUILD_CONFIG_SCHEMA_VERSION = "sk.operator.build_config.v1"
RESOLVED_BUILD_CONFIG_SCHEMA_VERSION = "sk.operator.build_config.resolved.v1"
BUILD_CONFIG_FIELDS = {
    "include_dirs",
    "support_dirs",
    "force_includes",
    "compile_options",
    "compile_definitions",
    "link_dirs",
    "link_libraries",
    "link_options",
    "build_env",
    "runtime_env",
    "package_files",
}
BUILD_CONFIG_ENV_FIELDS = {"build_env", "runtime_env"}


class BuildConfigError(ValueError):
    pass


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BuildConfigError(
            f"operator build config is not valid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise BuildConfigError("operator build config must be a JSON object")
    schema_version = payload.get("schema_version")
    if schema_version not in {None, "", BUILD_CONFIG_SCHEMA_VERSION}:
        raise BuildConfigError(
            f"unsupported operator build config schema_version: {schema_version!r}"
        )
    return payload


def _require_string_list(payload: dict[str, Any], field: str) -> list[str]:
    raw_value = payload.get(field, [])
    if raw_value is None or raw_value == "":
        return []
    if not isinstance(raw_value, list):
        raise BuildConfigError(f"operator build config field {field!r} must be a list")
    result: list[str] = []
    for index, item in enumerate(raw_value):
        if not isinstance(item, str) or not item.strip():
            raise BuildConfigError(
                f"operator build config field {field}[{index}] must be a non-empty string"
            )
        result.append(item.strip())
    return result


def _require_env(payload: dict[str, Any], field: str) -> dict[str, str]:
    raw_value = payload.get(field, {})
    if raw_value is None or raw_value == "":
        return {}
    if not isinstance(raw_value, dict):
        raise BuildConfigError(
            f"operator build config field {field!r} must be an object"
        )
    result: dict[str, str] = {}
    for key, value in raw_value.items():
        if not isinstance(key, str) or not key.strip():
            raise BuildConfigError(
                f"operator build config field {field!r} contains an empty env key"
            )
        if not isinstance(value, str):
            raise BuildConfigError(
                f"operator build config field {field}.{key} must be a string"
            )
        result[key.strip()] = value
    return result


def _resolve_path(raw_path: str, base_dir: Path) -> str:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _classify_path(path: str, repo_root: Path, cann_root: Path | None) -> str:
    resolved = Path(path).resolve()
    if cann_root is not None:
        try:
            resolved.relative_to(cann_root)
            return "cann-derived"
        except ValueError:
            pass
    try:
        resolved.relative_to(repo_root)
        return "user-workspace"
    except ValueError:
        return "external-explicit"


def _support_spec_to_record(
    raw_spec: str, base_dir: Path, repo_root: Path, cann_root: Path | None
) -> dict[str, Any]:
    raw_name = ""
    raw_path = raw_spec
    if "=" in raw_spec:
        raw_name, raw_path = raw_spec.split("=", 1)
        raw_name = raw_name.strip()
    resolved_path = _resolve_path(raw_path.strip(), base_dir)
    source_name = raw_name or Path(resolved_path).name
    if not source_name:
        raise BuildConfigError(f"support directory spec has no source name: {raw_spec}")
    return {
        "source_name": source_name,
        "source_path": resolved_path,
        "cli_spec": f"{source_name}={resolved_path}",
        "exists": Path(resolved_path).is_dir(),
        "source": _classify_path(resolved_path, repo_root, cann_root),
    }


def _path_records(
    paths: list[str], repo_root: Path, cann_root: Path | None
) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        resolved = str(Path(path).resolve())
        records.append(
            {
                "path": resolved,
                "exists": Path(resolved).exists(),
                "source": _classify_path(resolved, repo_root, cann_root),
            }
        )
    return records


def _validate_declared_paths(
    *,
    include_dirs: list[str],
    force_includes: list[str],
    link_dirs: list[str],
    package_files: list[str],
    support_dirs: list[dict[str, Any]],
) -> None:
    for path in include_dirs:
        if not Path(path).is_dir():
            raise BuildConfigError(f"operator include directory not found: {path}")
    for path in force_includes:
        if not Path(path).is_file():
            raise BuildConfigError(f"operator force include file not found: {path}")
    for path in link_dirs:
        if not Path(path).is_dir():
            raise BuildConfigError(f"operator link directory not found: {path}")
    for path in package_files:
        if not Path(path).exists():
            raise BuildConfigError(f"operator package file not found: {path}")
    for record in support_dirs:
        source_path = str(record["source_path"])
        if not Path(source_path).is_dir():
            raise BuildConfigError(
                f"operator support directory not found: {source_path}"
            )


def _candidate_cann_roots(
    explicit_target_cann: str, env: dict[str, str] | None
) -> list[tuple[str, Path]]:
    source_env = env if env is not None else os.environ
    candidates: list[tuple[str, Path]] = []
    if explicit_target_cann:
        candidates.append(("target-cann", Path(explicit_target_cann).expanduser()))
    for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        raw = source_env.get(name, "")
        if raw:
            candidates.append((name, Path(raw).expanduser()))
    return candidates


def resolve_cann_root(
    explicit_target_cann: str = "", env: dict[str, str] | None = None
) -> tuple[str, Path | None]:
    for source, path in _candidate_cann_roots(explicit_target_cann, env):
        resolved = path.resolve()
        if resolved.is_dir():
            return source, resolved
    return "", None


def cann_include_candidates(cann_root: Path | None) -> list[dict[str, Any]]:
    if cann_root is None:
        return []
    asc_root = cann_root / "aarch64-linux" / "asc"
    candidates = [
        asc_root / "include",
        asc_root / "include" / "basic_api",
        cann_root / "aarch64-linux" / "ascendc" / "include" / "highlevel_api",
        asc_root / "impl",
        asc_root / "impl" / "basic_api",
        asc_root,
    ]
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        records.append(
            {
                "path": str(candidate.resolve()),
                "exists": candidate.is_dir(),
                "source": "cann-derived",
            }
        )
    return records


def _load_declared_config(
    config_path: Path | None,
) -> tuple[dict[str, Any], Path | None]:
    if config_path is None:
        return {}, None
    resolved = config_path.expanduser().resolve()
    if not resolved.is_file():
        raise BuildConfigError(f"operator build config not found: {resolved}")
    return _load_config(resolved), resolved


def _parse_override(raw_override: str) -> tuple[list[str], Any]:
    if "=" not in raw_override:
        raise BuildConfigError(
            "operator build config override must use FIELD=JSON_VALUE format: "
            f"{raw_override}"
        )
    raw_field, raw_value = raw_override.split("=", 1)
    field = raw_field.strip()
    if not field:
        raise BuildConfigError("operator build config override contains an empty field")
    parts = field.split(".")
    if len(parts) > 2 or any(not part for part in parts):
        raise BuildConfigError(
            f"unsupported operator build config override field: {field}"
        )
    if parts[0] not in BUILD_CONFIG_FIELDS:
        raise BuildConfigError(
            f"unsupported operator build config override field: {field}"
        )
    if len(parts) == 2 and parts[0] not in BUILD_CONFIG_ENV_FIELDS:
        raise BuildConfigError(
            "nested operator build config overrides are only supported for "
            f"build_env/runtime_env: {field}"
        )
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise BuildConfigError(
            f"operator build config override {field!r} is not valid JSON: {exc}"
        ) from exc
    return parts, value


def _apply_overrides(
    payload: dict[str, Any], overrides: list[str] | None
) -> dict[str, Any]:
    result = dict(payload)
    for raw_override in overrides or []:
        parts, value = _parse_override(raw_override)
        if len(parts) == 1:
            result[parts[0]] = value
            continue
        env_name, key = parts
        env_payload = result.get(env_name, {})
        if env_payload is None or env_payload == "":
            env_payload = {}
        if not isinstance(env_payload, dict):
            raise BuildConfigError(
                f"operator build config field {env_name!r} must be an object "
                f"before applying override {env_name}.{key}"
            )
        updated_env = dict(env_payload)
        updated_env[key] = value
        result[env_name] = updated_env
    return result


def _resolved_path_list(
    *,
    payload: dict[str, Any],
    field: str,
    config_base_dir: Path,
) -> list[str]:
    config_paths = [
        _resolve_path(item, config_base_dir)
        for item in _require_string_list(payload, field)
    ]
    return _dedupe(config_paths)


def _resolved_support_dirs(
    *,
    payload: dict[str, Any],
    config_base_dir: Path,
    repo_root: Path,
    cann_root: Path | None,
) -> list[dict[str, Any]]:
    raw_config = _require_string_list(payload, "support_dirs")
    records = [
        _support_spec_to_record(item, config_base_dir, repo_root, cann_root)
        for item in raw_config
    ]
    by_name: dict[str, dict[str, Any]] = {}
    for record in records:
        previous = by_name.get(record["source_name"])
        if previous is not None and previous["source_path"] != record["source_path"]:
            raise BuildConfigError(
                "duplicate support directory name with different paths: "
                f"{record['source_name']}"
            )
        by_name[record["source_name"]] = record
    return list(by_name.values())


def resolve_operator_build_config(
    *,
    repo_root: Path,
    config_path: Path | None = None,
    cli_base_dir: Path | None = None,
    target_cann: str = "",
    overrides: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload, resolved_config_path = _load_declared_config(config_path)
    resolved_cli_base = (cli_base_dir or Path.cwd()).resolve()
    config_base_dir = (
        resolved_config_path.parent if resolved_config_path else resolved_cli_base
    )
    payload = _apply_overrides(payload, overrides)
    resolved_repo_root = repo_root.resolve()
    cann_source, cann_root = resolve_cann_root(target_cann, env)

    include_path_values = _resolved_path_list(
        payload=payload,
        field="include_dirs",
        config_base_dir=config_base_dir,
    )
    support_records = _resolved_support_dirs(
        payload=payload,
        config_base_dir=config_base_dir,
        repo_root=resolved_repo_root,
        cann_root=cann_root,
    )
    force_include_values = _resolved_path_list(
        payload=payload,
        field="force_includes",
        config_base_dir=config_base_dir,
    )
    link_dir_values = _resolved_path_list(
        payload=payload,
        field="link_dirs",
        config_base_dir=config_base_dir,
    )
    package_file_values = _resolved_path_list(
        payload=payload,
        field="package_files",
        config_base_dir=config_base_dir,
    )
    _validate_declared_paths(
        include_dirs=include_path_values,
        force_includes=force_include_values,
        link_dirs=link_dir_values,
        package_files=package_file_values,
        support_dirs=support_records,
    )
    config_build_env = _require_env(payload, "build_env")
    config_runtime_env = _require_env(payload, "runtime_env")

    compile_option_values = _dedupe(_require_string_list(payload, "compile_options"))
    compile_definition_values = _dedupe(
        _require_string_list(payload, "compile_definitions")
    )
    link_library_values = _dedupe(_require_string_list(payload, "link_libraries"))
    link_option_values = _dedupe(_require_string_list(payload, "link_options"))
    cann_candidates = cann_include_candidates(cann_root)
    cann_include_dirs = [item["path"] for item in cann_candidates if item["exists"]]
    all_include_dirs = _dedupe([*cann_include_dirs, *include_path_values])
    path_records = {
        "include_dirs": _path_records(
            include_path_values, resolved_repo_root, cann_root
        ),
        "force_includes": _path_records(
            force_include_values, resolved_repo_root, cann_root
        ),
        "link_dirs": _path_records(link_dir_values, resolved_repo_root, cann_root),
        "package_files": _path_records(
            package_file_values, resolved_repo_root, cann_root
        ),
    }
    external_records = []
    for records in [*path_records.values(), support_records]:
        for record in records:
            if record.get("source") == "external-explicit":
                external_records.append(record)
    return {
        "schema_version": RESOLVED_BUILD_CONFIG_SCHEMA_VERSION,
        "status": "ready",
        "source_config": str(resolved_config_path) if resolved_config_path else "",
        "repo_root": str(resolved_repo_root),
        "external_paths": external_records,
        "cann": {
            "source": cann_source,
            "root": str(cann_root) if cann_root else "",
            "include_candidates": cann_candidates,
            "include_dirs": cann_include_dirs,
        },
        "include_dirs": all_include_dirs,
        "user_include_dirs": include_path_values,
        "support_dirs": support_records,
        "support_dir_specs": [item["cli_spec"] for item in support_records],
        "force_includes": force_include_values,
        "compile_options": compile_option_values,
        "compile_definitions": compile_definition_values,
        "link_dirs": link_dir_values,
        "link_libraries": link_library_values,
        "link_options": link_option_values,
        "build_env": config_build_env,
        "runtime_env": config_runtime_env,
        "package_files": package_file_values,
        "path_records": path_records,
    }
