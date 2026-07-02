# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Artifact Layout v2 helpers for SK pipeline runs.

The pipeline still needs a private working tree for compiler inputs, cache
metadata, and debug logs. This module builds the user-facing artifact map on
top of that working tree: stable relative paths, clear deliverables, and a
single place to answer "what should I look at?".
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable


_SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_slug(value: str, fallback: str = "asset") -> str:
    slug = _SAFE_SEGMENT_RE.sub("_", str(value or "").strip()).strip("._-").lower()
    return slug or fallback


def package_name_for_slug(base_name: str, asset_slug: str) -> str:
    base = safe_slug(base_name or "op_extension", "op_extension")
    slug = safe_slug(asset_slug, "asset")
    if slug == "run":
        return base
    return f"{base}_{slug}"


def prepare_output_dir(output_dir: Path, *, clean_output: bool = False, resume_from: Path | None = None) -> None:
    if clean_output and resume_from is not None:
        raise ValueError("--clean-output cannot be combined with --resume-from")
    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    if resume_from is not None:
        resolved_output = output_dir.resolve()
        resolved_resume = resume_from.resolve()
        if not resolved_resume.is_file():
            raise ValueError(f"resume artifact map not found: {resume_from}")
        try:
            resolved_resume.relative_to(resolved_output)
        except ValueError as exc:
            raise ValueError("--resume-from must point inside --output-dir") from exc
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}; use --clean-output or --resume-from")


class ArtifactLayout:
    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    @staticmethod
    def render_markdown(payload: dict[str, Any]) -> str:
        lines = [
            "# SK Pipeline Artifact Map",
            "",
            f"- status: `{payload.get('run', {}).get('status', '')}`",
            f"- mode: `{payload.get('run', {}).get('mode', '')}`",
            f"- profile: `{payload.get('run', {}).get('profile', '')}`",
            "",
            "## Deliverables",
            "",
        ]
        wheels = payload.get("deliverables", {}).get("wheels", [])
        if wheels:
            lines.extend(f"- wheel: `{path}`" for path in wheels)
        else:
            lines.append("- wheel: none")
        lines.extend(["", "## Assets", ""])
        for asset in payload.get("assets", []):
            lines.append(f"- `{asset.get('asset_slug')}`: `{asset.get('status')}` ({len(asset.get('ops', []))} ops)")
            if asset.get("reason"):
                lines.append(f"  - reason: {asset.get('reason')}")
        name_resolution = payload.get("name_resolution") or {}
        if name_resolution:
            lines.extend(["", "## Name Resolution", ""])
            lines.append(f"- policy: `{name_resolution.get('policy', '')}`")
            lines.append(f"- renamed entries: `{name_resolution.get('renamed_entry_count', 0)}`")
            lines.append("- report: `name-resolution-report.md`")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def render_readme(payload: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# SK Pipeline Output",
                "",
                "Start here:",
                "",
                "- `artifact-map.md`: human-readable asset map.",
                "- `artifact-map.json`: machine-readable asset map.",
                "- `name-resolution-report.md`: duplicate entry rename map when present.",
                "- `deliverables/`: files intended for handoff.",
                "- `artifacts/`: canonical pipeline artifacts.",
                "- `assets/`: per-asset and per-stage reports.",
                "- `work/`: internal debug/build workspace.",
                "",
                f"Run status: `{payload.get('run', {}).get('status', '')}`",
                "",
            ]
        )

    @staticmethod
    def lint(payload: dict[str, Any]) -> list[dict[str, str]]:
        issues: list[dict[str, str]] = []
        paths: list[str] = []

        def collect(value: Any) -> None:
            if isinstance(value, str):
                if value.startswith("/"):
                    issues.append(
                        {
                            "severity": "error",
                            "message": f"user-facing absolute path: {value}",
                        }
                    )
                elif value.startswith("external:"):
                    issues.append(
                        {
                            "severity": "warning",
                            "message": f"user-facing external path reference: {value}",
                        }
                    )
                if value:
                    paths.append(value)
            elif isinstance(value, list):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)

        collect(payload)
        wheel_names: dict[str, int] = {}
        for wheel in payload.get("deliverables", {}).get("wheels", []):
            name = Path(wheel).name
            wheel_names[name] = wheel_names.get(name, 0) + 1
        for name, count in wheel_names.items():
            if count > 1:
                issues.append(
                    {
                        "severity": "error",
                        "message": f"duplicate delivery wheel filename: {name}",
                    }
                )
        return issues

    @staticmethod
    def _external_path_ref(path: Path) -> str:
        parts = [part for part in path.as_posix().lstrip("/").split("/") if part]
        if not parts:
            return "external:path"
        legacy_cache_dir_name = "sk-tools" + "-build-cache"
        matched_cache = False
        for cache_dir_name in ("sk-operator-build-cache", legacy_cache_dir_name):
            if cache_dir_name in parts:
                cache_index = parts.index(cache_dir_name)
                parts = parts[cache_index:]
                matched_cache = True
                break
        if not matched_cache and len(parts) > 2:
            parts = parts[-2:]
        return "external:" + "/".join(parts)

    def rel(self, path: Path | str | None) -> str:
        if path is None:
            return ""
        candidate = Path(path)
        if not candidate.is_absolute():
            return candidate.as_posix()
        try:
            return candidate.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return self._external_path_ref(candidate)

    def write_name_resolution_reports(self, payload: dict[str, Any]) -> None:
        self.write_json(self.root / "name-resolution-report.json", payload)
        lines = [
            "# Name Resolution Report",
            "",
            f"- policy: `{payload.get('policy', '')}`",
            f"- duplicate entry groups: `{len(payload.get('duplicate_entry_groups', {}))}`",
            f"- renamed entries: `{payload.get('renamed_entry_count', 0)}`",
            "",
            "| asset namespace | source entry | public entry | reason |",
            "|---|---|---|---|",
        ]
        renamed = [item for item in payload.get("resolutions", []) if item.get("renamed")]
        if renamed:
            for item in renamed:
                lines.append(
                    "| `{}` | `{}` | `{}` | `{}` |".format(
                        item.get("asset_namespace", ""),
                        item.get("source_entry_name", ""),
                        item.get("public_entry_name", ""),
                        item.get("rename_reason", ""),
                    )
                )
        else:
            lines.append("| none | none | none | none |")
        self.write_text(self.root / "name-resolution-report.md", "\n".join(lines) + "\n")

    def stage_report_path(
        self,
        asset_slug: str,
        stage_id: str,
        stage_name: str,
        filename: str = "stage-manifest.json",
    ) -> Path:
        directory_name = stage_name if stage_name.startswith(f"{stage_id}-") else f"{stage_id}-{stage_name}"
        return self.root / "assets" / asset_slug / "stages" / directory_name / filename

    def write_lint_report(self, payload: dict[str, Any]) -> list[dict[str, str]]:
        issues = self.lint(payload)
        status = "passed" if not any(item["severity"] == "error" for item in issues) else "failed"
        self.write_json(
            self.root / "artifact-layout-lint.json",
            {"status": status, "issues": issues},
        )
        return issues

    def ingest_run(
        self,
        *,
        asset_slug: str,
        state: dict[str, Any],
        work_output_dir: Path,
        source_path: Path | None = None,
        status: str | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        stage_root = work_output_dir
        asset_slug = safe_slug(asset_slug)
        asset_manifest_path = self.root / "assets" / asset_slug / "asset-manifest.json"
        source_artifact = ""
        if source_path is not None and source_path.exists():
            source_artifact = self._copy_tree(source_path, self.root / "artifacts" / "sources" / asset_slug)

        sk_source = self._copy_first_existing_tree(
            [
                stage_root / "02-adapt-sk-from-global" / "_aggregate" / "outputs" / "operator-sk-adapted",
                stage_root / "02-adapt-sk-from-global" / asset_slug / "outputs" / "operator-sk-adapted",
                stage_root / "05-generate-pybind-binding" / "outputs" / "operator-sk-adapted",
            ],
            self.root / "artifacts" / "sk-source" / asset_slug,
        )
        pybind_project = self._copy_first_existing_tree(
            [stage_root / "05-generate-pybind-binding" / "outputs" / "operator-sk-adapted"],
            self.root / "artifacts" / "pybind-projects" / asset_slug,
        )
        deliverable_wheels = self._copy_existing_files(
            state.get("wheel", {}).get("wheel_paths", []),
            self.root / "deliverables" / "wheels" / asset_slug,
        )
        if sk_source:
            self._copy_tree(
                self.root / sk_source,
                self.root / "deliverables" / "sk-source" / asset_slug,
            )
        if pybind_project:
            self._copy_tree(
                self.root / pybind_project,
                self.root / "deliverables" / "pybind-projects" / asset_slug,
            )

        stages = self._stage_records(state, asset_slug)
        ops = self._build_ops(
            asset_slug=asset_slug,
            state=state,
            stage_root=stage_root,
            deliverable_wheels=deliverable_wheels,
            source_artifact=source_artifact,
        )
        asset_status = status or state.get("status", "unknown")
        asset_payload = {
            "asset_slug": asset_slug,
            "status": asset_status,
            "reason": reason,
            "source": source_artifact,
            "work_state": self.rel(work_output_dir / "pipeline-state.json"),
            "stages": stages,
            "ops": ops,
            "deliverables": {
                "wheels": deliverable_wheels,
                "sk_source": self.rel(self.root / "deliverables" / "sk-source" / asset_slug) if sk_source else "",
                "pybind_project": self.rel(self.root / "deliverables" / "pybind-projects" / asset_slug)
                if pybind_project
                else "",
            },
        }
        self.write_json(asset_manifest_path, asset_payload)
        for op in ops:
            self.write_json(
                self.root / "assets" / asset_slug / "ops" / f"{safe_slug(op['entry_name'])}.json",
                op,
            )
        return asset_payload

    def skipped_asset(self, *, asset_slug: str, source_path: Path, reason: str) -> dict[str, Any]:
        asset_slug = safe_slug(asset_slug)
        source_artifact = ""
        if source_path.exists():
            source_dest = self.root / "artifacts" / "sources" / asset_slug
            if source_path.is_dir():
                source_artifact = self._copy_tree(source_path, source_dest)
            elif source_path.is_file():
                source_artifact = self._copy_file(source_path, source_dest / source_path.name)
        payload = {
            "asset_slug": asset_slug,
            "status": "skipped",
            "reason": reason,
            "source": source_artifact,
            "stages": {},
            "ops": [],
            "deliverables": {"wheels": [], "sk_source": "", "pybind_project": ""},
        }
        self.write_json(self.root / "assets" / asset_slug / "asset-manifest.json", payload)
        return payload

    def build_map(
        self,
        *,
        status: str,
        mode: str,
        profile: str,
        selected_stages: list[str],
        assets: list[dict[str, Any]],
        skipped_assets: list[dict[str, Any]] | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        all_assets = [*assets, *(skipped_assets or [])]
        deliverable_wheels: list[str] = []
        sk_sources: list[str] = []
        reports: list[str] = []
        for asset in all_assets:
            deliverable_wheels.extend(asset.get("deliverables", {}).get("wheels", []) or [])
            sk_source = asset.get("deliverables", {}).get("sk_source")
            if sk_source:
                sk_sources.append(sk_source)
        payload = {
            "schema_version": 2,
            "run": {
                "status": status,
                "mode": mode,
                "profile": profile,
                "reason": reason,
                "selected_stages": selected_stages,
                "output_dir": ".",
            },
            "assets": all_assets,
            "deliverables": {
                "wheels": deliverable_wheels,
                "sk_source": sk_sources,
                "reports": reports,
            },
        }
        name_resolution_path = self.root / "name-resolution-report.json"
        if name_resolution_path.exists():
            try:
                payload["name_resolution"] = json.loads(name_resolution_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload["name_resolution"] = {
                    "status": "unreadable",
                    "path": "name-resolution-report.json",
                }
        map_path = self.root / "artifact-map.json"
        self.write_json(map_path, payload)
        md_path = self.root / "artifact-map.md"
        self.write_text(md_path, self.render_markdown(payload))
        readme_path = self.root / "README.md"
        self.write_text(readme_path, self.render_readme(payload))
        self.write_json(
            self.root / "run-manifest.json",
            {"artifact_map": "artifact-map.json", "status": status},
        )
        return payload

    def _stage_records(self, state: dict[str, Any], asset_slug: str) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        for record in state.get("stages", []):
            stage_id = str(record.get("stage", ""))
            stage_name = str(record.get("name", stage_id))
            if not stage_id:
                continue
            report_path = self.stage_report_path(asset_slug, stage_id, stage_name)
            payload = {key: self._relativize_value(value) for key, value in record.items()}
            self.write_json(report_path, payload)
            md_path = report_path.with_name("report.md")
            self.write_text(
                md_path,
                "\n".join(
                    [
                        f"# Stage {stage_id} {stage_name}",
                        "",
                        f"- status: `{record.get('status', '')}`",
                        f"- started_at: `{record.get('started_at', '')}`",
                        f"- finished_at: `{record.get('finished_at', '')}`",
                        "",
                    ]
                ),
            )
            records[stage_id] = {
                "status": record.get("status", "unknown"),
                "name": stage_name,
                "report": self.rel(report_path),
                "summary": self.rel(md_path),
            }
        return records

    def _relativize_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.rel(value)
        if isinstance(value, list):
            return [self._relativize_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self._relativize_value(item) for key, item in value.items()}
        return value

    def _copy_first_existing_tree(self, candidates: Iterable[Path], dest: Path) -> str:
        for candidate in candidates:
            if candidate.exists():
                return self._copy_tree(candidate, dest)
        return ""

    def _copy_existing_files(self, sources: Iterable[str | Path], dest_dir: Path) -> list[str]:
        paths: list[str] = []
        for raw in sources:
            source = Path(raw)
            if source.is_file():
                paths.append(self._copy_file(source, dest_dir / source.name))
        return paths

    def _build_ops(
        self,
        *,
        asset_slug: str,
        state: dict[str, Any],
        stage_root: Path,
        deliverable_wheels: list[str],
        source_artifact: str,
    ) -> list[dict[str, Any]]:
        baseline = state.get("baseline", {}).get("per_op", {})
        ops: list[dict[str, Any]] = []
        wheel_for_asset = deliverable_wheels[0] if deliverable_wheels else ""
        for op_name, op_state in sorted(state.get("ops", {}).items()):
            entry_name = str(op_state.get("entry_name") or op_name)
            op_slug = safe_slug(entry_name, op_name)
            baseline_artifacts = baseline.get(entry_name, {}).get("artifacts", {})
            baseline_sos = self._copy_existing_files(
                baseline_artifacts.get("shared_objects", []),
                self.root / "artifacts" / "baseline-so" / asset_slug / op_slug,
            )
            sk_extensions = self._copy_existing_files(
                (stage_root / "05-generate-pybind-binding" / "outputs" / "operator-sk-adapted" / "build").rglob("*.so")
                if (stage_root / "05-generate-pybind-binding" / "outputs" / "operator-sk-adapted" / "build").exists()
                else [],
                self.root / "artifacts" / "sk-extensions" / asset_slug,
            )
            unit_payload = {
                "name": op_name,
                "entry_name": entry_name,
                "source_entry_name": op_state.get("source_entry_name") or op_state.get("entry_name") or entry_name,
                "public_entry_name": op_state.get("public_entry_name") or entry_name,
                "internal_symbol_name": op_state.get("internal_symbol_name") or entry_name,
                "bind_target": op_state.get("bind_target") or op_state.get("source_entry_name") or entry_name,
                "name_resolution": op_state.get("name_resolution", {}),
                "kernel_source": self._relativize_value(op_state.get("kernel_source")),
                "host_source": self._relativize_value(op_state.get("host_source")),
                "supported_arches": op_state.get("supported_arches", []),
                "supported_soc_versions": op_state.get("supported_soc_versions", []),
                "target_resolution": op_state.get("target_resolution", {}),
                "source_origin": source_artifact,
            }
            unit_path = self.root / "artifacts" / "operator-units" / asset_slug / f"{op_slug}.json"
            self.write_json(unit_path, unit_payload)
            ops.append(
                {
                    "name": op_name,
                    "entry_name": entry_name,
                    "source_entry_name": op_state.get("source_entry_name") or entry_name,
                    "public_entry_name": op_state.get("public_entry_name") or entry_name,
                    "internal_symbol_name": op_state.get("internal_symbol_name") or entry_name,
                    "bind_target": op_state.get("bind_target") or op_state.get("source_entry_name") or entry_name,
                    "name_resolution": op_state.get("name_resolution", {}),
                    "status": "completed",
                    "artifacts": {
                        "operator_unit": self.rel(unit_path),
                        "baseline_so": baseline_sos,
                        "sk_extensions": sk_extensions,
                        "wheel": wheel_for_asset,
                    },
                }
            )
        return ops

    def _copy_file(self, source: Path, dest: Path) -> str:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        return self.rel(dest)

    def _copy_tree(self, source: Path, dest: Path) -> str:
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, dest, symlinks=True)
        return self.rel(dest)
