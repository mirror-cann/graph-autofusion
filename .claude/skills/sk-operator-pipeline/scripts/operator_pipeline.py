#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Top-level CLI for the sk-operator-pipeline built-in skill."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
DEFAULT_SKILLS_ROOT = SKILL_DIR.parent
REPO_ROOT_MARKERS = (".git", "AGENTS.md", "build.sh")
DEFAULT_BUILD_CACHE_REL = Path("build") / "sk-operator-build-cache"

ROUTING_RULES = [
    (
        "sk-network-analysis",
        [
            "融合",
            "性能",
            "hang",
            "coredump",
            "卡死",
            "scope",
            "task",
            "queue",
            "update",
        ],
    ),
    (
        "sk-operator-asset-adapter",
        ["资产", "asset", "layout", "adapter", "接入", "仓库", "repo"],
    ),
    (
        "sk-operator-validate",
        [
            "校验",
            "validate",
            "规范",
            "lint",
            "风险",
            "spec",
            "compat",
            "兼容",
            "cann",
            "sdk",
            "driver",
            "acl",
            "版本",
        ],
    ),
    ("sk-operator-build-package", ["编译", "build", "打包", "wheel", "pybind"]),
    (
        "sk-operator-sample-gen",
        ["样例", "sample", "runtime", "correctness", "verdict", "oracle"],
    ),
    (
        "sk-operator-codegen",
        ["算子", "operator", "codegen", "使能", "scaffold", "intake"],
    ),
]
_ASSET_SKIP_DIRS = {
    "op_verify",
    "run",
    "runs",
    "input",
    "output",
    "outputs",
    "build",
    "build_out",
    "kernel_template",
}
_ASSET_ROOT_MANIFEST_NAMES = (
    "operator-assets.json",
    "sk-operator-assets.json",
)
_ASSET_ROOT_NON_BLOCKING_STATUSES = (
    "packaged",
    "verified",
    "skipped-no-npu",
    "skipped-insufficient-runtime-spec",
    "skipped-by-user",
    "skipped-target-arch",
    "clean",
    "adapted",
    "pybind-generated",
    "analyzed",
)
_PIPELINE_TERMINAL_SUCCESS_STATUSES = ("packaged", "verified")


class CliUsageError(Exception):
    pass


def _emit(message: object = "", *, file=None, end: str = "\n") -> None:
    stream = sys.stdout if file is None else file
    stream.write(f"{message}{end}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _resolve_skills_root(raw: str | os.PathLike[str] | None = None) -> Path:
    return (Path(raw) if raw else DEFAULT_SKILLS_ROOT).expanduser().resolve()


def _find_repo_root(raw: str | os.PathLike[str] | None = None) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if any((candidate / marker).exists() for marker in REPO_ROOT_MARKERS):
            return candidate
    return start


def _default_build_cache_dir(repo_root: Path) -> Path:
    raw = os.environ.get("SK_OPERATOR_BUILD_CACHE_DIR") or os.environ.get("SK_TOOLS_BUILD_CACHE_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return (repo_root / DEFAULT_BUILD_CACHE_REL).resolve()


def _resolve_output_dir(raw: str | os.PathLike[str]) -> Path:
    return Path(raw).expanduser().resolve()


def _skill_script(skills_root: Path, skill_name: str, script_name: str) -> Path:
    return skills_root / skill_name / "scripts" / script_name


def _ensure_script_import(skills_root: Path, skill_name: str) -> Path:
    import sys as _sys

    script_dir = skills_root / skill_name / "scripts"
    if str(script_dir) not in _sys.path:
        _sys.path.insert(0, str(script_dir))
    return script_dir


def _discover_operator_assets(root: Path, *, skills_root: Path) -> list[Path]:
    return [
        Path(item["path"])
        for item in _discover_operator_asset_records(root, skills_root=skills_root)
        if item["status"] == "selected"
    ]


def _discover_operator_asset_records(root: Path, *, skills_root: Path) -> list[dict[str, Any]]:
    _ensure_script_import(skills_root, "sk-operator-codegen")
    from operator_asset_layout import build_inventory, build_layout_from_inventory

    manifest = _load_asset_root_manifest(root)
    include_paths = _manifest_include_paths(root, manifest)
    excluded = _manifest_excluded_paths(root, manifest)
    candidates: list[Path] = []
    skipped: list[dict[str, Any]] = [
        {
            "path": str(path),
            "status": "skipped",
            "reason": reason,
            "human_questions": [],
        }
        for path, reason in excluded.items()
    ]
    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in excluded:
            return
        if resolved in seen:
            return
        for existing in seen:
            try:
                resolved.relative_to(existing)
                return
            except ValueError:
                pass
        inventory = build_inventory(resolved)
        layout = build_layout_from_inventory(inventory)
        if layout.get("status") == "ready" and layout.get("operator_units"):
            seen.add(resolved)
            candidates.append(resolved)
        elif (
            inventory.get("kernel_source_count")
            or inventory.get("kernel_candidate_count")
            or inventory.get("rtc_host_program_count")
        ):
            skipped.append(
                {
                    "path": str(resolved),
                    "status": "skipped",
                    "reason": "asset-layout-required",
                    "human_questions": layout.get("human_questions", []),
                }
            )

    if include_paths:
        for path in include_paths:
            add(path)
    else:
        for path in sorted(root.rglob("*")):
            if not path.is_dir():
                continue
            rel_parts = path.relative_to(root).parts
            if any(part in _ASSET_SKIP_DIRS for part in rel_parts):
                continue
            if path.name in {"op_kernel", "op_host"}:
                continue
            if any(item.is_file() for item in path.iterdir()):
                add(path)
    if not candidates and not skipped:
        add(root)
    records = [
        {
            "path": str(path),
            "status": "selected",
            "reason": "operator-kernel-source-detected",
        }
        for path in candidates
    ]
    return [*records, *skipped]


def _load_asset_root_manifest(root: Path) -> dict[str, Any]:
    for name in _ASSET_ROOT_MANIFEST_NAMES:
        path = root / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CliUsageError(f"invalid asset-root manifest {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise CliUsageError(f"asset-root manifest must be a JSON object: {path}")
        return payload
    return {}


def _manifest_include_paths(root: Path, manifest: dict[str, Any]) -> list[Path]:
    raw = manifest.get("default_include") or manifest.get("include") or []
    if raw in (None, ""):
        return []
    if not isinstance(raw, list):
        raise CliUsageError("asset-root manifest default_include must be a list")
    resolved_root = root.resolve()
    paths: list[Path] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise CliUsageError("asset-root manifest include paths must be non-empty strings")
        path = (root / item).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError as exc:
            raise CliUsageError(f"asset-root manifest include path escapes root: {item}") from exc
        if not path.exists():
            raise CliUsageError(f"asset-root manifest include path not found: {item}")
        paths.append(path)
    return paths


def _manifest_excluded_paths(root: Path, manifest: dict[str, Any]) -> dict[Path, str]:
    raw = manifest.get("excluded") or manifest.get("exclude") or []
    if raw in (None, ""):
        return {}
    if not isinstance(raw, list):
        raise CliUsageError("asset-root manifest excluded must be a list")
    excluded: dict[Path, str] = {}
    for item in raw:
        if isinstance(item, str):
            rel_path = item
            reason = "excluded-by-asset-root-manifest"
        elif isinstance(item, dict):
            rel_path = str(item.get("path") or "")
            reason = str(item.get("reason") or "excluded-by-asset-root-manifest")
        else:
            raise CliUsageError("asset-root manifest excluded entries must be strings or objects")
        if not rel_path.strip():
            raise CliUsageError("asset-root manifest excluded path must be non-empty")
        path = (root / rel_path).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise CliUsageError(f"asset-root manifest excluded path escapes root: {rel_path}") from exc
        excluded[path] = reason
    return excluded


def _safe_slug(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value or "")).strip("._-")
    return slug or "asset"


def _asset_run_name(root: Path | None, asset: Path, used: set[str]) -> str:
    if root is not None:
        try:
            raw = "__".join(asset.relative_to(root).parts)
        except ValueError:
            raw = asset.name
    else:
        raw = asset.name
    name = _safe_slug(raw)
    if name not in used:
        used.add(name)
        return name
    index = 2
    while f"{name}_{index}" in used:
        index += 1
    final_name = f"{name}_{index}"
    used.add(final_name)
    return final_name


def _entry_records_for_assets(asset_paths: list[Path], *, skills_root: Path) -> list[dict[str, Any]]:
    _ensure_script_import(skills_root, "sk-operator-codegen")
    from operator_asset_analyzer import analyze_asset

    records: list[dict[str, Any]] = []
    for asset in asset_paths:
        try:
            manifest = analyze_asset(asset)
            entries = [unit.entry_name for unit in manifest.operator_units]
            status = "analyzed" if entries else "needs-human"
            reason = "" if entries else "no-operator-unit-detected"
        except Exception as exc:
            entries = []
            status = "failed"
            reason = str(exc)
        records.append(
            {
                "path": str(asset),
                "status": status,
                "entries": entries,
                "reason": reason,
            }
        )
    return records


def _build_name_resolution_payload(
    *,
    roots: list[Path],
    asset_paths: list[Path],
    entry_records: list[dict[str, Any]],
    policy: str,
    skills_root: Path,
) -> dict[str, Any]:
    _ensure_script_import(skills_root, "sk-operator-codegen")
    from operator_name_resolution import build_name_resolution

    entry_by_path = {str(record["path"]): record for record in entry_records}
    assets: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for asset in asset_paths:
        root = _root_for_asset(asset, roots)
        assets.append(
            {
                "path": str(asset),
                "namespace": _asset_run_name(root, asset, used_names),
                "entries": entry_by_path.get(str(asset), {}).get("entries", []),
            }
        )
    return build_name_resolution(
        assets=assets,
        root=roots[0] if len(roots) == 1 else None,
        policy=policy,
    )


def _duplicate_entries(entry_records: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_entry: dict[str, list[str]] = {}
    for record in entry_records:
        for entry in record.get("entries", []):
            by_entry.setdefault(str(entry), []).append(str(record["path"]))
    return {entry: paths for entry, paths in by_entry.items() if len(paths) > 1}


def _root_for_asset(asset: Path, roots: list[Path]) -> Path | None:
    for root in roots:
        try:
            asset.relative_to(root)
            return root
        except ValueError:
            pass
    return None


def _write_asset_root_coverage(output_dir: Path, payload: dict[str, Any]) -> None:
    _write_json(output_dir / "asset-root-coverage.json", payload)
    rows = [
        "# Asset Root Coverage",
        "",
        f"- mode: `{payload['mode']}`",
        f"- status: `{payload['status']}`",
        f"- selected assets: `{payload['selected_asset_count']}`",
        f"- skipped assets: `{payload['skipped_asset_count']}`",
        "",
        "| asset | status | entries | output | reason |",
        "|---|---|---|---|---|",
    ]
    for record in payload["assets"]:
        entries = ", ".join(record.get("entries", []))
        rows.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                record.get("path", ""),
                record.get("status", ""),
                entries,
                record.get("output_dir", ""),
                record.get("reason", ""),
            )
        )
    _write_text(output_dir / "asset-root-coverage.md", "\n".join(rows) + "\n")


def _write_asset_root_aggregate_coverage(
    *,
    output_dir: Path,
    roots: list[Path],
    selected_assets: list[Path],
    skipped_records: list[dict[str, Any]],
    entry_records: list[dict[str, Any]],
    state: dict[str, Any],
) -> dict[str, Any]:
    entry_by_path = {record["path"]: record for record in entry_records}
    aggregate_entries = set(state.get("aggregate", {}).get("entries", []))
    records: list[dict[str, Any]] = []
    for asset in selected_assets:
        preflight = entry_by_path.get(str(asset), {})
        entries = [str(entry) for entry in preflight.get("entries", [])]
        status = "selected"
        if entries and aggregate_entries:
            status = "aggregated" if any(entry in aggregate_entries for entry in entries) else "skipped"
        records.append(
            {
                "path": str(asset),
                "status": status,
                "entries": entries,
                "output_dir": "work/stage-work/aggregate",
                "reason": preflight.get("reason", ""),
            }
        )
    for record in skipped_records:
        records.append(
            {
                "path": record["path"],
                "status": "skipped",
                "entries": [],
                "output_dir": "",
                "reason": record.get("reason", ""),
                "human_questions": record.get("human_questions", []),
            }
        )
    failed_count = 0 if state.get("status") in _ASSET_ROOT_NON_BLOCKING_STATUSES else len(selected_assets)
    payload = {
        "schema_version": 1,
        "mode": "asset-root-mode-aggregate",
        "status": "completed" if failed_count == 0 else "failed",
        "roots": [str(root) for root in roots],
        "duplicate_entries": {},
        "duplicate_asset_slugs": {},
        "selected_asset_count": len(selected_assets),
        "skipped_asset_count": len(skipped_records),
        "passed_asset_count": len(selected_assets) if failed_count == 0 else 0,
        "failed_asset_count": failed_count,
        "assets": records,
    }
    _write_asset_root_coverage(output_dir, payload)
    return payload


def _run_asset_root_separately(
    *,
    args: argparse.Namespace,
    roots: list[Path],
    selected_assets: list[Path],
    skipped_records: list[dict[str, Any]],
    entry_records: list[dict[str, Any]],
    duplicate_entries: dict[str, list[str]],
    duplicate_asset_slugs: dict[str, list[str]],
    output_dir: Path,
    orchestrator: Any,
    wheel_mode: str | None,
    build_cache_dir: Path,
) -> tuple[int, dict[str, Any]]:
    from sk_artifact_layout import ArtifactLayout, package_name_for_slug
    from sk_pipeline_orchestrator import PipelineOrchestrator

    layout = ArtifactLayout(output_dir)
    resolved_verify_backend, resolved_wheel_mode = orchestrator.resolve_profile_options(
        profile=args.profile,
        verify_backend=args.verify_backend,
        wheel_mode=wheel_mode,
        no_verify=bool(args.no_verify),
        no_package=bool(args.no_package),
    )
    selected_stages = PipelineOrchestrator.parse_stages(
        args.stages,
        do_package=not args.no_package,
        profile=args.profile,
        wheel_mode=resolved_wheel_mode,
        verify_backend=resolved_verify_backend,
        reuse_wheel=args.reuse_wheel,
    )
    used_names: set[str] = set()
    entry_by_path = {record["path"]: record for record in entry_records}
    records: list[dict[str, Any]] = []
    for asset in selected_assets:
        root = _root_for_asset(asset, roots)
        run_name = _asset_run_name(root, asset, used_names)
        package_name = package_name_for_slug(args.aggregate_wheel_name, run_name)
        try:
            state = orchestrator.run(
                [asset],
                output_dir,
                target_chip=args.target_chip or "",
                target_cann=getattr(args, "target_cann", "") or "",
                max_iterations=int(args.max_iterations),
                do_package=not args.no_package,
                stages=args.stages,
                aggregate_wheel_name=package_name,
                package_version=args.package_version,
                do_verify=not args.no_verify,
                profile=args.profile,
                verify_backend=args.verify_backend,
                wheel_mode=wheel_mode,
                reuse_wheel=args.reuse_wheel,
                build_cache_dir=build_cache_dir,
                jobs=args.jobs,
                run_slug=run_name,
                io_contract=(Path(args.io_contract).resolve() if getattr(args, "io_contract", None) else None),
                allow_structural_toolchain=bool(getattr(args, "allow_structural_toolchain", False)),
                allow_mock_npu=bool(getattr(args, "allow_mock_npu", False)),
            )
            status = state["status"]
            reason = ""
        except Exception as exc:
            status = "failed"
            reason = str(exc)
        preflight = entry_by_path.get(str(asset), {})
        records.append(
            {
                "path": str(asset),
                "status": status,
                "entries": preflight.get("entries", []),
                "output_dir": str(output_dir / "assets" / run_name),
                "reason": reason,
            }
        )
    for record in skipped_records:
        records.append(
            {
                "path": record["path"],
                "status": "skipped",
                "entries": [],
                "output_dir": "",
                "reason": record.get("reason", ""),
            }
        )
        root = _root_for_asset(Path(record["path"]), roots)
        skipped_name = _asset_run_name(root, Path(record["path"]), used_names)
        layout.skipped_asset(
            asset_slug=skipped_name,
            source_path=Path(record["path"]),
            reason=record.get("reason", ""),
        )
    selected_statuses = []
    failed = []
    for record in records:
        if record["status"] == "skipped":
            continue
        selected_statuses.append(record["status"])
        if record["status"] not in _ASSET_ROOT_NON_BLOCKING_STATUSES:
            failed.append(record)
    status = "completed" if not failed else "failed"
    release_success = (
        status == "completed"
        and not skipped_records
        and bool(selected_statuses)
        and all(item in _PIPELINE_TERMINAL_SUCCESS_STATUSES for item in selected_statuses)
    )
    payload = {
        "schema_version": 1,
        "mode": "separate",
        "status": status,
        "release_success": release_success,
        "reason": ("duplicate-entry-names" if duplicate_entries else "asset-root-mode-separate"),
        "duplicate_entries": duplicate_entries,
        "duplicate_asset_slugs": duplicate_asset_slugs,
        "roots": [str(root) for root in roots],
        "selected_asset_count": len(selected_assets),
        "skipped_asset_count": len(skipped_records),
        "passed_asset_count": len(selected_statuses) - len(failed),
        "failed_asset_count": len(failed),
        "assets": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_asset_root_coverage(output_dir, payload)
    assets: list[dict[str, Any]] = []
    for manifest_path in sorted((output_dir / "assets").glob("*/asset-manifest.json")):
        assets.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    artifact_map = layout.build_map(
        status=status,
        mode="separate",
        profile=args.profile,
        selected_stages=selected_stages,
        assets=assets,
        reason=payload["reason"],
    )
    layout.write_lint_report(artifact_map)
    _write_json(
        output_dir / "pipeline-state.json",
        {
            "status": "asset-root-covered" if status == "completed" else "failed",
            "release_success": release_success,
            "asset_root_coverage": payload,
            "artifact_map": "artifact-map.json",
        },
    )
    return (0 if release_success else 1), payload


def _index_section(directory: Path, *, label: str | None = None) -> dict[str, Any]:
    files = sorted(path for path in directory.rglob("*") if path.is_file()) if directory.exists() else []
    return {
        "root": label or str(directory),
        "file_count": len(files),
        "sample_files": [
            (str(path.relative_to(directory)) if directory in path.parents or path.parent == directory else str(path))
            for path in files[:20]
        ],
    }


def _route_query(query: str) -> str:
    lowered = query.lower()
    for skill_name, keywords in ROUTING_RULES:
        if any(keyword.lower() in lowered for keyword in keywords):
            return skill_name
    return "sk-operator-pipeline"


def _render_index_report(index_payload: dict[str, Any]) -> str:
    lines = [
        "# SK Operator Pipeline Index Report",
        "",
        f"- base mode: `{index_payload['execution']['base_mode']}`",
        f"- ai requested: `{index_payload['execution']['ai_requested']}`",
        "",
        "## Indexed Roots",
        "",
    ]
    for section in index_payload["sections"]:
        lines.append(f"- `{section['root']}` -> `{section['file_count']}` files")
    lines.extend(
        [
            "",
            "## Current Note",
            "",
            "- 这是本地规则和文件扫描层的能力索引。",
            "- 后续如果接入 AI，只能在这份基础索引之上补更细的提示和关联。",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_route_report(route_payload: dict[str, Any]) -> str:
    lines = [
        "# SK Operator Pipeline Routing Report",
        "",
        f"- query: `{route_payload['query']}`",
        f"- routed skill: `{route_payload['route']['skill']}`",
        f"- confidence: `{route_payload['route']['confidence']}`",
        f"- ai requested: `{route_payload['execution']['ai_requested']}`",
        "",
        "## Suggested Next Step",
        "",
        f"- use `{route_payload['route']['entry']}`",
    ]
    return "\n".join(lines) + "\n"


def _render_ai_hints(payload: dict[str, Any]) -> str:
    lines = [
        "# SK Operator Pipeline AI Hints",
        "",
        f"- requested: `{payload['execution']['ai_requested']}`",
        "- status: `not-configured`",
        "- 当前仓已经固定了 AI 增强层位置，但还未接入真实模型后端。",
        "",
        "## Manual / Future AI Focus",
        "",
        "- 先基于本地能力索引或路由结果确定目标 skill。",
        "- 再把基础 JSON / Markdown 产物交给后续 AI 层做精细提示。",
    ]
    return "\n".join(lines) + "\n"


def cmd_index(args: argparse.Namespace) -> int:
    skills_root = _resolve_skills_root(args.skills_root)
    output_dir = _resolve_output_dir(args.output_dir)
    index_roots = [Path(item).expanduser().resolve() for item in (args.index_root or [])]
    if not index_roots:
        index_roots = [skills_root]
    payload = {
        "skill": "sk-operator-pipeline",
        "execution": {
            "entry": "operator_pipeline.py",
            "base_mode": True,
            "ai_requested": args.with_ai,
            "ai_status": "not-configured" if args.with_ai else "not-requested",
            "skills_root": str(skills_root),
        },
        "sections": [_index_section(root, label=str(root)) for root in index_roots],
    }
    _write_json(output_dir / "operator-pipeline-index.json", payload)
    _write_text(output_dir / "operator-pipeline-index-report.md", _render_index_report(payload))
    if args.with_ai:
        _write_text(
            output_dir / "operator-pipeline-index-ai-hints.md",
            _render_ai_hints(payload),
        )
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    skills_root = _resolve_skills_root(args.skills_root)
    output_dir = _resolve_output_dir(args.output_dir)
    target = _route_query(args.query)

    def entry(skill_name: str, script_name: str, *extra: str) -> str:
        return " ".join(
            [
                "python3",
                str(_skill_script(skills_root, skill_name, script_name)),
                *extra,
            ]
        )

    entry = {
        "sk-network-analysis": entry("sk-network-analysis", "network_analysis.py", "diagnose", "<run_dir>"),
        "sk-operator-asset-adapter": entry(
            "sk-operator-asset-adapter",
            "operator_asset_adapter.py",
            "adapt-asset",
            "<asset>",
            "--output-dir",
            "<dir>",
        ),
        "sk-operator-codegen": entry("sk-operator-codegen", "operator_codegen.py", "intake", "<asset>"),
        "sk-operator-validate": entry(
            "sk-operator-validate",
            "operator_validate.py",
            "validate-operator",
            "--asset",
            "<asset>",
            "--output-dir",
            "<dir>",
        ),
        "sk-operator-sample-gen": entry(
            "sk-operator-sample-gen",
            "operator_sample_gen.py",
            "collect-runtime-input-spec",
            "<output_dir>",
        ),
        "sk-operator-build-package": entry(
            "sk-operator-build-package",
            "operator_build_package.py",
            "run-sk-build-validation",
            "<output_dir>",
        ),
        "sk-operator-pipeline": entry("sk-operator-pipeline", "operator_pipeline.py", "index"),
    }[target]
    payload = {
        "skill": "sk-operator-pipeline",
        "execution": {
            "entry": "operator_pipeline.py",
            "base_mode": True,
            "ai_requested": args.with_ai,
            "ai_status": "not-configured" if args.with_ai else "not-requested",
            "skills_root": str(skills_root),
        },
        "query": args.query,
        "route": {
            "skill": target,
            "confidence": "rule-based",
            "entry": entry,
        },
    }
    _write_json(output_dir / "operator-pipeline-route.json", payload)
    _write_text(output_dir / "operator-pipeline-route-report.md", _render_route_report(payload))
    if args.with_ai:
        _write_text(
            output_dir / "operator-pipeline-route-ai-hints.md",
            _render_ai_hints(payload),
        )
    return 0


def cmd_run_sk_pipeline(args: argparse.Namespace) -> int:
    import sys as _sys

    if str(SCRIPT_DIR) not in _sys.path:
        _sys.path.insert(0, str(SCRIPT_DIR))
    from sk_pipeline_orchestrator import OrchestratorError, PipelineOrchestrator
    from sk_artifact_layout import prepare_output_dir

    skills_root = _resolve_skills_root(args.skills_root)
    repo_root = _find_repo_root(args.repo_root)
    build_cache_dir = (
        Path(args.build_cache_dir).expanduser().resolve()
        if args.build_cache_dir
        else _default_build_cache_dir(repo_root)
    )
    asset_paths: list[Path] = []
    root_paths: list[Path] = []
    root_skipped_records: list[dict[str, Any]] = []
    for raw_asset in args.assets or []:
        asset_paths.append(Path(raw_asset).resolve())
    for raw_root in args.asset_root or []:
        root = Path(raw_root).resolve()
        if not root.exists() or not root.is_dir():
            raise CliUsageError(f"asset root not found: {root}")
        root_paths.append(root)
        root_records = _discover_operator_asset_records(root, skills_root=skills_root)
        asset_paths.extend(Path(item["path"]) for item in root_records if item["status"] == "selected")
        root_skipped_records.extend(item for item in root_records if item["status"] == "skipped")
    output_dir = _resolve_output_dir(args.output_dir)
    if not asset_paths:
        if root_skipped_records:
            payload = {
                "schema_version": 1,
                "mode": "scan",
                "status": "needs-human",
                "reason": "asset-layout-required",
                "duplicate_entries": {},
                "duplicate_asset_slugs": {},
                "roots": [str(root) for root in root_paths],
                "selected_asset_count": 0,
                "skipped_asset_count": len(root_skipped_records),
                "passed_asset_count": 0,
                "failed_asset_count": 0,
                "assets": [
                    {
                        "path": record["path"],
                        "status": "skipped",
                        "entries": [],
                        "output_dir": "",
                        "reason": record.get("reason", ""),
                        "human_questions": record.get("human_questions", []),
                    }
                    for record in root_skipped_records
                ],
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_asset_root_coverage(output_dir, payload)
            raise CliUsageError(f"asset layout required; inspect {output_dir / 'asset-root-coverage.json'}")
        raise CliUsageError("--asset or --asset-root must provide at least one operator asset")
    for asset in asset_paths:
        if not asset.exists():
            raise CliUsageError(f"asset path not found: {asset}")
    try:
        prepare_output_dir(
            output_dir,
            clean_output=bool(args.clean_output),
            resume_from=Path(args.resume_from) if args.resume_from else None,
        )
    except ValueError as exc:
        raise CliUsageError(str(exc)) from exc

    orchestrator = PipelineOrchestrator(skills_root, _sys.executable, repo_root=repo_root)
    wheel_mode = "never" if args.no_package and args.wheel_mode is None else args.wheel_mode
    only_asset_roots = bool(root_paths) and not (args.assets or [])
    asset_names: dict[str, str] | None = None
    if only_asset_roots:
        entry_records = _entry_records_for_assets(asset_paths, skills_root=skills_root)
        duplicates = _duplicate_entries(entry_records)
        used_asset_names: set[str] = set()
        asset_names = {
            str(asset.resolve()): _asset_run_name(_root_for_asset(asset, root_paths), asset, used_asset_names)
            for asset in asset_paths
        }
        slugs: dict[str, list[str]] = {}
        for asset in asset_paths:
            slugs.setdefault(_safe_slug(asset.name), []).append(str(asset))
        duplicate_slugs = {slug: paths for slug, paths in slugs.items() if len(paths) > 1}
        namespace_duplicates = args.duplicate_entry_policy == "namespace"
        separate_mode = args.asset_root_mode == "separate" or (
            args.asset_root_mode == "auto" and (bool(duplicates) or bool(duplicate_slugs)) and not namespace_duplicates
        )
        if separate_mode:
            rc, payload = _run_asset_root_separately(
                args=args,
                roots=root_paths,
                selected_assets=asset_paths,
                skipped_records=root_skipped_records,
                entry_records=entry_records,
                duplicate_entries=duplicates,
                duplicate_asset_slugs=duplicate_slugs,
                output_dir=output_dir,
                orchestrator=orchestrator,
                wheel_mode=wheel_mode,
                build_cache_dir=build_cache_dir,
            )
            _emit(
                json.dumps(
                    {
                        "status": payload["status"],
                        "release_success": payload.get("release_success", False),
                        "mode": payload["mode"],
                        "asset_count": payload["selected_asset_count"],
                        "skipped_asset_count": payload["skipped_asset_count"],
                        "failed_asset_count": payload["failed_asset_count"],
                    }
                )
            )
            return rc
        aggregate_has_duplicates = bool(duplicates or duplicate_slugs)
        should_explain_namespace_fix = (
            args.asset_root_mode == "aggregate" and aggregate_has_duplicates and not namespace_duplicates
        )
        if should_explain_namespace_fix:
            details = {
                "duplicate_entries": duplicates,
                "duplicate_asset_slugs": duplicate_slugs,
            }
            raise CliUsageError(
                "asset-root aggregate mode cannot continue with duplicate identities: "
                + json.dumps(details, ensure_ascii=False)
            )
        if namespace_duplicates and duplicates:
            from sk_artifact_layout import ArtifactLayout

            layout = ArtifactLayout(output_dir)
            _ensure_script_import(skills_root, "sk-operator-codegen")
            from operator_name_resolution import public_name_resolution_payload

            name_resolution_payload = _build_name_resolution_payload(
                roots=root_paths,
                asset_paths=asset_paths,
                entry_records=entry_records,
                policy="namespace",
                skills_root=skills_root,
            )
            layout.write_name_resolution_reports(public_name_resolution_payload(name_resolution_payload))
            internal_name_resolution = output_dir / "work" / "name-resolution-internal.json"
            layout.write_json(internal_name_resolution, name_resolution_payload)
            args.name_resolution = str(internal_name_resolution)
    try:
        run_slug = "aggregate" if len(asset_paths) > 1 else _safe_slug(asset_paths[0].name)
        state = orchestrator.run(
            asset_paths,
            output_dir,
            target_chip=args.target_chip or "",
            target_cann=getattr(args, "target_cann", "") or "",
            max_iterations=int(args.max_iterations),
            do_package=not args.no_package,
            stages=args.stages,
            aggregate_wheel_name=args.aggregate_wheel_name,
            package_version=args.package_version,
            do_verify=not args.no_verify,
            profile=args.profile,
            verify_backend=args.verify_backend,
            wheel_mode=wheel_mode,
            reuse_wheel=args.reuse_wheel,
            build_cache_dir=build_cache_dir,
            jobs=args.jobs,
            run_slug=run_slug,
            name_resolution=(Path(args.name_resolution).resolve() if getattr(args, "name_resolution", None) else None),
            asset_names=asset_names,
            io_contract=(Path(args.io_contract).resolve() if getattr(args, "io_contract", None) else None),
            allow_structural_toolchain=bool(getattr(args, "allow_structural_toolchain", False)),
            allow_mock_npu=bool(getattr(args, "allow_mock_npu", False)),
        )
    except OrchestratorError as exc:
        raise CliUsageError(str(exc)) from exc
    if only_asset_roots:
        _write_asset_root_aggregate_coverage(
            output_dir=output_dir,
            roots=root_paths,
            selected_assets=asset_paths,
            skipped_records=root_skipped_records,
            entry_records=entry_records,
            state=state,
        )
    summary = {
        "status": state["status"],
        "stages": len(state["stages"]),
        "asset_count": len(asset_paths),
    }
    name_resolution_report = output_dir / "name-resolution-report.json"
    if name_resolution_report.exists():
        try:
            name_resolution_payload = json.loads(name_resolution_report.read_text(encoding="utf-8"))
            summary.update(
                {
                    "renamed_entry_count": name_resolution_payload.get("renamed_entry_count", 0),
                    "duplicate_entry_group_count": len(name_resolution_payload.get("duplicate_entry_groups", {})),
                    "name_resolution_report": "name-resolution-report.json",
                }
            )
        except (OSError, json.JSONDecodeError):
            summary["name_resolution_report"] = "unreadable"
    _emit(json.dumps(summary))
    # 0 only when the pipeline reached a release terminal success.
    return 0 if state["status"] in _PIPELINE_TERMINAL_SUCCESS_STATUSES else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Top-level CLI for sk-operator-pipeline")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    pipeline = subparsers.add_parser(
        "run-sk-pipeline",
        help="Drive the closed-loop SK operator pipeline (detect/adapt -> validate rule packs -> build/package)",
    )
    pipeline.add_argument(
        "--skills-root",
        help="Root directory containing SK skill directories; defaults to this skill's parent directory",
    )
    pipeline.add_argument(
        "--repo-root",
        help="User workspace root for cache/workspace semantics; defaults to cwd marker discovery",
    )
    pipeline.add_argument(
        "--asset",
        dest="assets",
        action="append",
        default=[],
        help="Operator asset directory or source file (repeatable)",
    )
    pipeline.add_argument(
        "--asset-root",
        action="append",
        default=[],
        help="Parent directory whose direct child directories are operator assets",
    )
    pipeline.add_argument(
        "--asset-root-mode",
        choices=["auto", "aggregate", "separate"],
        default="auto",
        help=(
            "How to run discovered --asset-root assets: aggregate one wheel, "
            "separate per asset, or auto-separate when samples collide"
        ),
    )
    pipeline.add_argument(
        "--duplicate-entry-policy",
        choices=["reject", "namespace"],
        default="reject",
        help="How aggregate asset-root runs handle duplicate entry names: reject or namespace colliding entries",
    )
    pipeline.add_argument("--name-resolution", help=argparse.SUPPRESS)
    pipeline.add_argument(
        "--stages",
        help="Comma-separated stage ids to run, e.g. 01,02,05,06 (default follows --profile)",
    )
    pipeline.add_argument(
        "--output-dir",
        default="sk-pipeline-output",
        help="Pipeline working/output directory",
    )
    pipeline.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove an existing output directory before running",
    )
    pipeline.add_argument("--resume-from", help="Resume using an artifact-map.json inside --output-dir")
    pipeline.add_argument(
        "--target-chip",
        default="",
        help="Target chip id(s), comma/semicolon separated; used for sparse arch build resolution",
    )
    pipeline.add_argument("--target-cann", default="", help=argparse.SUPPRESS)
    pipeline.add_argument(
        "--aggregate-wheel-name",
        default="op_extension",
        help="Distribution name for the aggregate wheel",
    )
    pipeline.add_argument(
        "--package-version",
        default="0.1.0",
        help="Distribution package version for the aggregate wheel",
    )
    pipeline.add_argument(
        "--io-contract",
        help=(
            "JSON tensor IO contract forwarded to adapt-sk-from-global; required "
            "when multi-tensor output semantics are ambiguous"
        ),
    )
    pipeline.add_argument(
        "--max-iterations",
        default="5",
        help="Convergence cap on remediation iterations",
    )
    pipeline.add_argument(
        "--no-package",
        action="store_true",
        help="Stop after default validation stages unless --stages explicitly selects build/verify",
    )
    pipeline.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip differential verification after wheel build",
    )
    pipeline.add_argument(
        "--allow-structural-toolchain",
        action="store_true",
        help="Use generated fake cmake/bisheng placeholders for development-only structural checks",
    )
    pipeline.add_argument(
        "--allow-mock-npu",
        action="store_true",
        help="Allow mock NPU validation status for development-only route checks",
    )
    pipeline.add_argument(
        "--profile",
        choices=["fast", "release"],
        default="release",
        help="Pipeline profile: fast skips wheel build; release keeps delivery packaging",
    )
    pipeline.add_argument(
        "--verify-backend",
        choices=["standalone", "wheel", "both", "none"],
        help="Verification backend override",
    )
    pipeline.add_argument(
        "--wheel-mode",
        choices=["never", "cache", "always"],
        help="Wheel build mode override",
    )
    pipeline.add_argument(
        "--reuse-wheel",
        help="Existing aggregate wheel to reuse instead of invoking build-native-wheel",
    )
    pipeline.add_argument(
        "--build-cache-dir",
        default=None,
        help="Directory for standalone and wheel build caches (default: <repo-root>/build/sk-operator-build-cache)",
    )
    pipeline.add_argument(
        "--jobs",
        type=int,
        help="Pipeline worker count and native wheel compile parallelism",
    )
    pipeline.set_defaults(func=cmd_run_sk_pipeline)

    index = subparsers.add_parser("index", help="Build a local SK capability index")
    index.add_argument(
        "--skills-root",
        help="Root directory containing SK skill directories; defaults to this skill's parent directory",
    )
    index.add_argument(
        "--index-root",
        action="append",
        default=[],
        help="Directory to index; repeatable. Defaults to --skills-root only",
    )
    index.add_argument("--output-dir", default="operator-pipeline-output", help="Output directory")
    index.add_argument("--with-ai", action="store_true", help="Request optional AI-layer hints")
    index.set_defaults(func=cmd_index)

    route = subparsers.add_parser("route", help="Route one question to a built-in skill")
    route.add_argument("query", help="Question or task description")
    route.add_argument(
        "--skills-root",
        help="Root directory containing SK skill directories; defaults to this skill's parent directory",
    )
    route.add_argument("--output-dir", default="operator-pipeline-output", help="Output directory")
    route.add_argument("--with-ai", action="store_true", help="Request optional AI-layer hints")
    route.set_defaults(func=cmd_route)

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
