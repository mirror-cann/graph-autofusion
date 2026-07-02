# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Closed-loop SK operator pipeline orchestrator.

Drives the five SK operator skills through one feedback loop:

    detect form -> (adapt if pure non-SK) -> validate core spec rule packs
      -> if blocking findings: apply auto-remediation and re-scan in another pass
      -> if findings need a human: stop and escalate
      -> if clean: build pybind binding + native wheel
            -> build failure -> failure-analyzer turns the log into findings
               that feed the same remediation loop.

Convergence protection: a hard iteration cap, plus a "the exact same set of
blocking finding_ids reappeared with nothing fixed" check.

State is accumulated (never overwritten) in operator-sk-pipeline-state.json so a
human can inspect what happened each round.

Self-contained: invokes each skill via its CLI as a subprocess; reasons over the
unified finding envelopes the skills emit.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from sk_artifact_layout import ArtifactLayout, safe_slug

_AUTO_REMEDIATION_KINDS = frozenset(
    {"remove-line-containing", "rename-symbol", "add-include", "replace-pattern"}
)

_SKILL_SCRIPTS = {
    "asset_adapter": ("sk-operator-asset-adapter", "operator_asset_adapter.py"),
    "codegen": ("sk-operator-codegen", "operator_codegen.py"),
    "validate": ("sk-operator-validate", "operator_validate.py"),
    "build_package": ("sk-operator-build-package", "operator_build_package.py"),
    "sample_gen": ("sk-operator-sample-gen", "operator_sample_gen.py"),
}

_STAGE_DIRS = {
    "00": "00-understand-asset",
    "01": "01-detect-form",
    "02": "02-adapt-sk-from-global",
    "03": "03-validate-spec",
    "04": "04-validate-compat",
    "05": "05-generate-pybind-binding",
    "06": "06-build-and-verify",
}
_STAGE_PREREQS = {
    "00": set(),
    "01": {"00"},
    "02": {"01"},
    "03": {"02"},
    "04": {"02"},
    "05": {"02"},
    "06": {"02"},
}
_STAGE_ORDER = tuple(_STAGE_DIRS)
_VERIFY_BACKENDS = frozenset({"standalone", "wheel", "both", "none"})
_WHEEL_MODES = frozenset({"never", "cache", "always"})
DEFAULT_BUILD_CACHE_DIR = Path("build") / "sk-operator-build-cache"
_SUPPORT_HEADER_SUFFIXES = frozenset({".h", ".hh", ".hpp", ".hxx", ".inc"})
_LOCAL_INCLUDE_SOURCE_SUFFIXES = frozenset(
    {".asc", ".c", ".cc", ".cpp", ".cxx", *_SUPPORT_HEADER_SUFFIXES}
)
_SUPPORT_SKIP_DIRS = frozenset(
    {".git", "__pycache__", "build", "cmake-build-debug", "out", "tests"}
)
_QUOTED_INCLUDE_RE = re.compile(r'(?m)^\s*#\s*include\s+"([^"]+)"')


class OrchestratorError(Exception):
    pass


# ---------- failure analyzer ----------


def analyze_build_log(log_text: str) -> list[dict[str, Any]]:
    """Map a build/runtime failure log to unified findings.

    Returns a list of finding dicts (without the envelope wrapper). The mapping
    table is intentionally small and extensible -- add a rule per common error.
    """
    findings: list[dict[str, Any]] = []
    # bisheng "use of undeclared identifier 'skBlockNum'" -> rename legacy API.
    m = re.search(r"undeclared identifier '([A-Za-z_][A-Za-z0-9_]*)'", log_text)
    if m and m.group(1) in ("skBlockNum",):
        legacy = m.group(1)
        findings.append(
            {
                "finding_id": f"build.undeclared-{legacy}",
                "rule_id": "build.undeclared-legacy-api",
                "severity": "blocker",
                "category": "build",
                "actionable_by": ["codegen.apply-remediation"],
                "remediation_hint": {
                    "kind": "rename-symbol",
                    "old_value": legacy,
                    "new_value": "skNumBlocks",
                },
                "evidence": [m.group(0)],
                "message": f"build failed on undeclared legacy API {legacy!r}; rename to skNumBlocks.",
            }
        )
        return findings
    # Generic build failure -> human.
    findings.append(
        {
            "finding_id": "build.generic-failure",
            "rule_id": "build.compile-error",
            "severity": "blocker",
            "category": "build",
            "actionable_by": ["human"],
            "remediation_hint": {"kind": "human-decision"},
            "evidence": [log_text[-400:]],
            "message": "build failed and no auto-remediation rule matched; human review needed.",
        }
    )
    return findings


def _is_auto_remediable(finding: dict[str, Any]) -> bool:
    if "codegen.apply-remediation" not in finding.get("actionable_by", []):
        return False
    return finding.get("remediation_hint", {}).get("kind") in _AUTO_REMEDIATION_KINDS


# ---------- orchestrator ----------


class PipelineOrchestrator:
    def __init__(
        self,
        skills_root: Path,
        python_exe: str,
        *,
        repo_root: Path | None = None,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        self.skills_root = Path(skills_root).expanduser().resolve()
        self.repo_root = (
            Path(repo_root).expanduser().resolve()
            if repo_root
            else Path.cwd().resolve()
        )
        self.python = python_exe
        self.env = env  # subprocess env (lets tests inject the fake toolchain PATH)
        self._current_target_chip = ""
        self._toolchain_mode = "real"

    @classmethod
    def parse_stages(
        cls,
        stages: str | Sequence[str] | None,
        *,
        do_package: bool,
        profile: str,
        wheel_mode: str,
        verify_backend: str,
        reuse_wheel: str | None,
    ) -> list[str]:
        return cls._parse_stages(
            stages,
            do_package=do_package,
            profile=profile,
            wheel_mode=wheel_mode,
            verify_backend=verify_backend,
            reuse_wheel=reuse_wheel,
        )

    def resolve_profile_options(
        self,
        *,
        profile: str,
        verify_backend: str | None,
        wheel_mode: str | None,
        no_verify: bool,
        no_package: bool,
    ) -> tuple[str, str]:
        return self._resolve_profile_options(
            profile=profile,
            verify_backend=verify_backend,
            wheel_mode=wheel_mode,
            no_verify=no_verify,
            no_package=no_package,
        )

    def run(
        self,
        assets: Path | Sequence[Path],
        output_dir: Path,
        *,
        target_chip: str = "",
        target_cann: str = "",
        max_iterations: int = 5,
        do_package: bool = True,
        stages: str | Sequence[str] | None = None,
        aggregate_wheel_name: str = "op_extension",
        package_version: str = "0.1.0",
        do_verify: bool = True,
        profile: str = "release",
        verify_backend: str | None = None,
        wheel_mode: str | None = None,
        reuse_wheel: str | None = None,
        build_cache_dir: Path | str = DEFAULT_BUILD_CACHE_DIR,
        jobs: int | None = None,
        run_slug: str | None = None,
        emit_layout: bool = True,
        name_resolution: Path | None = None,
        asset_names: dict[str, str] | None = None,
        io_contract: Path | None = None,
        allow_structural_toolchain: bool = False,
        allow_mock_npu: bool = False,
    ) -> dict[str, Any]:
        del max_iterations
        public_output_dir = output_dir
        layout = ArtifactLayout(public_output_dir)
        explicit_run_slug = run_slug is not None
        if run_slug is None:
            raw_assets = [assets] if isinstance(assets, Path) else list(assets)
            if len(raw_assets) == 1:
                raw_asset = Path(raw_assets[0])
                run_slug = raw_asset.stem if raw_asset.is_file() else raw_asset.name
            else:
                run_slug = aggregate_wheel_name
        run_slug = safe_slug(run_slug, "run")
        output_dir = public_output_dir / "work" / "stage-work" / run_slug
        output_dir.mkdir(parents=True, exist_ok=True)
        self._current_target_chip = target_chip
        if allow_mock_npu:
            if self.env is None:
                self.env = os.environ.copy()
            self.env["SK_OPERATOR_MOCK_NPU"] = "1"
        resolved_verify_backend, resolved_wheel_mode = self._resolve_profile_options(
            profile=profile,
            verify_backend=verify_backend,
            wheel_mode=wheel_mode,
            no_verify=not do_verify,
            no_package=not do_package,
        )
        selected_stages = self._parse_stages(
            stages,
            do_package=do_package,
            profile=profile,
            wheel_mode=resolved_wheel_mode,
            verify_backend=resolved_verify_backend,
            reuse_wheel=reuse_wheel,
        )
        ops, asset_understanding = self._understand_and_expand_assets(
            assets, output_dir, asset_names=asset_names
        )
        if explicit_run_slug:
            asset_slug = run_slug
        elif len(asset_understanding.get("assets", [])) == 1:
            asset_slug = safe_slug(
                str(asset_understanding["assets"][0].get("name") or "asset")
            )
        else:
            asset_slug = safe_slug(aggregate_wheel_name, "run")
        source_path = None
        if len(asset_understanding.get("assets", [])) == 1:
            raw_path = asset_understanding["assets"][0].get("path")
            source_path = Path(raw_path) if raw_path else None
        job_env = self.env if self.env is not None else os.environ
        resolved_jobs = (
            jobs
            if jobs and jobs > 0
            else self._default_jobs(len(ops), self._arch_count_from_env(job_env))
        )
        cache_dir = Path(build_cache_dir).expanduser()
        if not cache_dir.is_absolute():
            cache_dir = self.repo_root / cache_dir
        cache_dir = cache_dir.resolve()
        config = {
            "schema_version": 1,
            "assets": asset_understanding["assets"],
            "operator_units": asset_understanding["operator_units"],
            "selected_stages": selected_stages,
            "target_chip": target_chip,
            "target_cann": target_cann,
            "aggregate_wheel_name": aggregate_wheel_name,
            "package_version": package_version,
            "verify": do_verify,
            "profile": profile,
            "verify_backend": resolved_verify_backend,
            "wheel_mode": resolved_wheel_mode,
            "reuse_wheel": reuse_wheel,
            "build_cache_dir": str(cache_dir),
            "jobs": resolved_jobs,
            "name_resolution": str(name_resolution) if name_resolution else "",
            "io_contract": str(io_contract) if io_contract else "",
            "allow_structural_toolchain": allow_structural_toolchain,
            "allow_mock_npu": allow_mock_npu,
        }
        self._write_json(output_dir / "00-pipeline-config.json", config)
        state: dict[str, Any] = {
            "status": "running",
            "output_dir": str(output_dir),
            "target_chip": target_chip,
            "target_cann": target_cann,
            "assets": config["assets"],
            "operator_units": config["operator_units"],
            "selected_stages": selected_stages,
            "profile": profile,
            "verify_backend": resolved_verify_backend,
            "wheel_mode": resolved_wheel_mode,
            "build_cache_dir": str(cache_dir),
            "jobs": resolved_jobs,
            "allow_structural_toolchain": allow_structural_toolchain,
            "allow_mock_npu": allow_mock_npu,
            "public_output_dir": str(public_output_dir),
            "work_output_dir": str(output_dir),
            "asset_slug": asset_slug,
            "stages": [],
            "ops": {
                op["name"]: {
                    "asset": str(op["asset"]),
                    "sk_asset": str(op.get("sk_asset", op["asset"])),
                    "source_asset": str(op.get("source_asset", op["asset"])),
                    "asset_kind": op.get("asset_kind"),
                    "entry_name": op.get("entry_name"),
                    "kernel_source": op.get("kernel_source"),
                    "host_source": op.get("host_source"),
                    "json_spec": op.get("json_spec"),
                    "tiling_headers": op.get("tiling_headers", []),
                    "build_backends": op.get("build_backends", []),
                    "supported_soc_versions": op.get("supported_soc_versions", []),
                    "supported_arches": op.get("supported_arches", []),
                    "target_resolution": op.get("target_resolution", {}),
                    "support_source": op.get("support_source", ""),
                }
                for op in ops
            },
            "iterations": [
                {"iteration_index": 0, "stages": [], "findings": [], "escalations": []}
            ],
        }
        if "00" in selected_stages:
            state["stages"].append(
                self._stage_record(
                    "00",
                    "passed",
                    self._stage_dir(output_dir, "00"),
                    {
                        "asset_count": len(asset_understanding["assets"]),
                        "operator_unit_count": len(ops),
                    },
                )
            )
            state["iterations"][0]["stages"].append("understand-asset")
        self._write_state(output_dir, state)

        findings: list[dict[str, Any]] = []
        adapted_outputs: list[Path] = []

        if "01" in selected_stages:
            stage_started_at = self._utc_now()
            stage_root = self._stage_dir(output_dir, "01")

            def worker(op: dict[str, Any]) -> dict[str, Any]:
                op_root = stage_root / op["name"]
                inputs = op_root / "inputs"
                outputs = op_root / "outputs"
                materialized = self._materialize_input(
                    op.get("sk_asset", op["asset"]), inputs / "asset"
                )
                manifest = self._detect_form(inputs / "asset", outputs)
                return {
                    "op_name": op["name"],
                    "status": "passed",
                    "form": manifest["overall_form"],
                    "outputs": str(outputs),
                    "input_mode": materialized,
                }

            stage_results, parallel = self._run_parallel_ops(ops, resolved_jobs, worker)
            if parallel["failed_ops"]:
                raise OrchestratorError(
                    f"01 failed for: {', '.join(parallel['failed_ops'])}"
                )
            for item in stage_results:
                state["ops"][item["op_name"]]["form"] = item["form"]
                state["ops"][item["op_name"]]["stage01"] = {
                    "outputs": item["outputs"],
                    "input_mode": item["input_mode"],
                }
            state["stages"].append(
                self._stage_record(
                    "01",
                    "passed",
                    stage_root,
                    {"parallel": parallel},
                    started_at=stage_started_at,
                )
            )
            state["iterations"][0]["stages"].append("detect-sk-form")
            self._write_state(output_dir, state)

        if "02" in selected_stages:
            stage_started_at = self._utc_now()
            stage_root = self._stage_dir(output_dir, "02")

            def worker(op: dict[str, Any]) -> dict[str, Any]:
                op_root = stage_root / op["name"]
                inputs = op_root / "inputs"
                outputs = op_root / "outputs"
                self._materialize_input(
                    self._stage_dir(output_dir, "01") / op["name"] / "outputs",
                    inputs / "detect-form",
                )
                stage_asset = Path(op.get("sk_asset", op["asset"]))
                self._materialize_input(stage_asset, inputs / "asset")
                stage_io_contract = self._stage_io_contract(io_contract, inputs)
                support_args: list[str] = []
                source_asset = Path(op.get("source_asset", op["asset"]))
                for (
                    support_name,
                    support_source,
                ) in self._shared_support_dirs_for_stage_asset(
                    stage_asset, source_asset
                ):
                    support_dest = inputs / "support" / support_name
                    self._materialize_input(support_source, support_dest)
                    support_args.append(f"{support_name}={support_dest}")
                manifest = self._adapt(
                    inputs / "asset",
                    outputs,
                    support_args,
                    understanding=Path(op["understanding_outputs"])
                    / "operator-asset-understanding.json",
                    target_chip=target_chip,
                    package_name=aggregate_wheel_name,
                    package_version=package_version,
                    name_resolution=name_resolution,
                    source_asset=Path(op.get("source_asset", op["asset"])),
                    io_contract=stage_io_contract,
                )
                return {
                    "op_name": op["name"],
                    "status": "passed",
                    "outputs": str(outputs),
                    "manifest": manifest,
                    "support_dirs": support_args,
                }

            stage_results, parallel = self._run_parallel_ops(ops, resolved_jobs, worker)
            if parallel["failed_ops"]:
                raise OrchestratorError(
                    f"02 failed for: {', '.join(parallel['failed_ops'])}"
                )
            for item in stage_results:
                manifest = item["manifest"]
                op_name = item["op_name"]
                manifest_entries = []
                for record in manifest.get("per_file", []):
                    for entry in record.get("entries", []):
                        if isinstance(entry, dict):
                            manifest_entries.append(entry)
                if manifest_entries:
                    entry_meta = manifest_entries[0]
                    state["ops"][op_name]["entry_name"] = entry_meta.get(
                        "entry_name"
                    ) or state["ops"][op_name].get("entry_name")
                    state["ops"][op_name]["source_entry_name"] = entry_meta.get(
                        "source_entry_name"
                    ) or state["ops"][op_name].get("source_entry_name")
                    state["ops"][op_name]["public_entry_name"] = entry_meta.get(
                        "public_entry_name"
                    ) or state["ops"][op_name].get("entry_name")
                    state["ops"][op_name]["internal_symbol_name"] = entry_meta.get(
                        "internal_symbol_name"
                    ) or state["ops"][op_name].get("entry_name")
                    state["ops"][op_name]["bind_target"] = entry_meta.get(
                        "bind_target"
                    ) or state["ops"][op_name].get("source_entry_name")
                    state["ops"][op_name]["name_resolution"] = entry_meta.get(
                        "name_resolution", {}
                    )
                state["ops"][op_name]["stage02"] = {
                    "outputs": item["outputs"],
                    "pybind_layout": manifest.get("pybind_layout"),
                    "support_dirs": item.get("support_dirs", []),
                }
                if manifest.get("status") == "needs-human":
                    state["status"] = "needs-human"
                    state["ops"][op_name]["escalations"] = manifest.get(
                        "escalations", []
                    )
                    state["iterations"][0]["escalations"].extend(
                        manifest.get("escalations", [])
                    )
                    self._write_state(output_dir, state)
                    if emit_layout:
                        self._emit_layout(
                            layout,
                            public_output_dir,
                            output_dir,
                            state,
                            asset_slug=asset_slug,
                            source_path=source_path,
                            mode="single",
                            profile=profile,
                            selected_stages=selected_stages,
                        )
                    return state
                if manifest.get("pybind_layout") != "aclgraph-canonical":
                    finding = {
                        "finding_id": "orchestrator.pybind-layout-not-canonical",
                        "severity": "blocker",
                        "message": f"{op_name} did not produce aclgraph-canonical pybind layout",
                    }
                    state["status"] = "needs-human"
                    state["iterations"][0]["escalations"].append(finding)
                    self._write_state(output_dir, state)
                    if emit_layout:
                        self._emit_layout(
                            layout,
                            public_output_dir,
                            output_dir,
                            state,
                            asset_slug=asset_slug,
                            source_path=source_path,
                            mode="single",
                            profile=profile,
                            selected_stages=selected_stages,
                        )
                    return state
                adapted_outputs.append(Path(item["outputs"]))
            aggregate_root = stage_root / "_aggregate"
            aggregate_inputs = aggregate_root / "inputs"
            aggregate_outputs = aggregate_root / "outputs"
            for op in ops:
                self._materialize_input(
                    stage_root / op["name"] / "outputs", aggregate_inputs / op["name"]
                )
            aggregate_manifest = self._aggregate(
                adapted_outputs,
                aggregate_outputs,
                aggregate_wheel_name,
                package_version,
            )
            state["aggregate"] = {
                "stage02_outputs": str(aggregate_outputs),
                "entries": [
                    entry["entry_name"]
                    for entry in self._entries_from_manifest(aggregate_manifest)
                ],
            }
            state["stages"].append(
                self._stage_record(
                    "02",
                    "passed",
                    stage_root,
                    {"aggregate_outputs": str(aggregate_outputs), "parallel": parallel},
                    started_at=stage_started_at,
                )
            )
            state["iterations"][0]["stages"].append("adapt-sk-from-global")
            state["iterations"][0]["pybind_layout"] = "aclgraph-canonical"
            self._write_state(output_dir, state)

        if "03" in selected_stages:
            stage_started_at = self._utc_now()
            stage_root = self._stage_dir(output_dir, "03")

            def worker(op: dict[str, Any]) -> dict[str, Any]:
                op_root = stage_root / op["name"]
                inputs = op_root / "inputs"
                outputs = op_root / "outputs"
                self._materialize_input(
                    self._stage_dir(output_dir, "02") / op["name"] / "outputs",
                    inputs / "adapted-output",
                )
                result = self._scan_spec(
                    inputs / "adapted-output" / "operator-sk-adapted", outputs
                )
                op_findings = result.get("findings", [])
                return {
                    "op_name": op["name"],
                    "status": "passed",
                    "outputs": str(outputs),
                    "findings": op_findings,
                }

            stage_results, parallel = self._run_parallel_ops(ops, resolved_jobs, worker)
            if parallel["failed_ops"]:
                raise OrchestratorError(
                    f"03 failed for: {', '.join(parallel['failed_ops'])}"
                )
            for item in stage_results:
                op_findings = item["findings"]
                findings.extend(op_findings)
                state["ops"][item["op_name"]]["stage03"] = {
                    "outputs": item["outputs"],
                    "findings": len(op_findings),
                }
            state["stages"].append(
                self._stage_record(
                    "03",
                    "passed",
                    stage_root,
                    {"findings": len(findings), "parallel": parallel},
                    started_at=stage_started_at,
                )
            )
            state["iterations"][0]["stages"].append("operator-validate.spec")
            self._write_state(output_dir, state)

        if "04" in selected_stages:
            stage_started_at = self._utc_now()
            stage_root = self._stage_dir(output_dir, "04")
            before = len(findings)

            def worker(op: dict[str, Any]) -> dict[str, Any]:
                op_root = stage_root / op["name"]
                inputs = op_root / "inputs"
                outputs = op_root / "outputs"
                self._materialize_input(
                    self._stage_dir(output_dir, "02") / op["name"] / "outputs",
                    inputs / "adapted-output",
                )
                result = self._scan_compat(
                    inputs / "adapted-output" / "operator-sk-adapted",
                    outputs,
                    target_chip,
                    target_cann,
                )
                op_findings = result.get("findings", [])
                return {
                    "op_name": op["name"],
                    "status": "passed",
                    "outputs": str(outputs),
                    "findings": op_findings,
                }

            stage_results, parallel = self._run_parallel_ops(ops, resolved_jobs, worker)
            if parallel["failed_ops"]:
                raise OrchestratorError(
                    f"04 failed for: {', '.join(parallel['failed_ops'])}"
                )
            for item in stage_results:
                op_findings = item["findings"]
                findings.extend(op_findings)
                state["ops"][item["op_name"]]["stage04"] = {
                    "outputs": item["outputs"],
                    "findings": len(op_findings),
                }
            state["stages"].append(
                self._stage_record(
                    "04",
                    "passed",
                    stage_root,
                    {"findings": len(findings) - before, "parallel": parallel},
                    started_at=stage_started_at,
                )
            )
            state["iterations"][0]["stages"].append("operator-validate.compat")
            self._write_state(output_dir, state)

        blocking = [
            finding for finding in findings if finding.get("severity") == "blocker"
        ]
        if blocking:
            state["status"] = "needs-human"
            state["iterations"][0]["findings"] = findings
            state["iterations"][0]["escalations"] = blocking
            self._write_state(output_dir, state)
            if emit_layout:
                self._emit_layout(
                    layout,
                    public_output_dir,
                    output_dir,
                    state,
                    asset_slug=asset_slug,
                    source_path=source_path,
                    mode="single",
                    profile=profile,
                    selected_stages=selected_stages,
                )
            return state
        state["iterations"][0]["findings"] = findings

        if "05" in selected_stages:
            stage_started_at = self._utc_now()
            stage_root = self._stage_dir(output_dir, "05")
            inputs = stage_root / "inputs"
            outputs = stage_root / "outputs"
            aggregate_outputs = (
                self._stage_dir(output_dir, "02") / "_aggregate" / "outputs"
            )
            self._materialize_input(aggregate_outputs, inputs / "aggregate-output")
            self._copy_contents(aggregate_outputs, outputs)
            binding = self._generate_pybind_binding(outputs)
            state["stages"].append(
                self._stage_record(
                    "05",
                    "passed",
                    stage_root,
                    {"outputs": str(outputs)},
                    started_at=stage_started_at,
                )
            )
            state["iterations"][0]["stages"].append("generate-pybind-binding")
            state["binding"] = {
                "package_name": binding.get("package_name"),
                "kernel_entries": binding.get("kernel_entries", []),
            }
            self._write_state(output_dir, state)

        if "06" in selected_stages:
            stage_started_at = self._utc_now()
            stage_root = self._stage_dir(output_dir, "06")
            inputs = stage_root / "inputs"
            aggregate_outputs = (
                self._stage_dir(output_dir, "02") / "_aggregate" / "outputs"
            )
            self._materialize_input(aggregate_outputs, inputs / "aggregate-output")
            aggregate_manifest = self._read_json(
                aggregate_outputs / "operator-sk-adapted.json"
            )
            entries = self._entries_from_manifest(aggregate_manifest)
            ops_by_entry: dict[str, dict[str, Any]] = {}
            for op in ops:
                for key in (
                    op.get("entry_name"),
                    op.get("public_entry_name"),
                    op.get("source_entry_name"),
                    op.get("bind_target"),
                    op["name"],
                ):
                    if key:
                        ops_by_entry.setdefault(str(key), op)

            baseline_root = stage_root / "baseline"

            def baseline_worker(entry: dict[str, Any]) -> dict[str, Any]:
                entry_name = entry["entry_name"]
                source_entry_name = (
                    entry.get("bind_target")
                    or entry.get("source_entry_name")
                    or entry_name
                )
                op = ops_by_entry.get(entry_name)
                if op is None:
                    op = ops_by_entry.get(str(source_entry_name))
                if op is None and len(ops) == 1:
                    op = ops[0]
                if op is None:
                    raise OrchestratorError(
                        f"no normalized operator unit found for baseline entry {entry_name}"
                    )
                entry_root = baseline_root / entry_name
                baseline_inputs = entry_root / "inputs"
                baseline_outputs = entry_root / "outputs"
                self._materialize_input(Path(op["asset"]), baseline_inputs / "asset")
                manifest = self._build_baseline(
                    baseline_inputs / "asset",
                    baseline_outputs,
                    entry_name=str(source_entry_name),
                    allow_structural_toolchain=allow_structural_toolchain,
                )
                return {
                    "op_name": entry_name,
                    "status": (
                        "passed"
                        if manifest.get("status")
                        in {
                            "passed",
                            "structural-passed",
                            "skipped-real-backend-not-enabled",
                        }
                        else "failed"
                    ),
                    "manifest": manifest,
                    "manifest_path": str(
                        baseline_outputs / "operator-baseline-build.json"
                    ),
                    "outputs": str(baseline_outputs),
                }

            baseline_results, baseline_parallel = self._run_parallel_ops(
                entries, resolved_jobs, baseline_worker
            )
            if baseline_parallel["failed_ops"]:
                raise OrchestratorError(
                    f"baseline build failed for: {', '.join(baseline_parallel['failed_ops'])}"
                )
            baseline_manifests = {
                item["op_name"]: Path(item["manifest_path"])
                for item in baseline_results
            }

            standalone_root = stage_root / "standalone"
            standalone_inputs = standalone_root / "inputs"
            standalone_outputs = standalone_root / "outputs"
            self._materialize_input(
                aggregate_outputs, standalone_inputs / "aggregate-output"
            )
            fixture_statuses = self._copy_runtime_fixtures(
                ops, standalone_inputs / "runtime-fixtures"
            )
            if len(ops) == 1:
                source_fixture_name = ops[0]["name"]
                source_fixture_dir = (
                    standalone_inputs / "runtime-fixtures" / source_fixture_name
                )
                for entry in entries:
                    if (
                        entry["entry_name"] not in fixture_statuses
                        and source_fixture_dir.exists()
                    ):
                        entry_fixture_dir = (
                            standalone_inputs / "runtime-fixtures" / entry["entry_name"]
                        )
                        self._materialize_input(source_fixture_dir, entry_fixture_dir)
                        source_status = next(iter(fixture_statuses.values()))
                        fixture_statuses[entry["entry_name"]] = {
                            **source_status,
                            "path": str(entry_fixture_dir),
                        }
            standalone_manifest = self._generate_standalone_compare(
                standalone_inputs / "aggregate-output",
                standalone_outputs,
                standalone_inputs / "runtime-fixtures",
                target_chip,
            )
            if standalone_manifest.get("status") == "needs-target-arch":
                standalone_build_status = "skipped-target-arch"
                standalone_build = {
                    "status": "skipped-target-arch",
                    "reason": standalone_manifest.get("reason", "needs-target-arch"),
                    "message": standalone_manifest.get("message", ""),
                    "executables": {},
                }
                standalone_cache_key = ""
                cached_standalone = None
                standalone_cache_hit = False
            else:
                self._ensure_structural_toolchain_env(
                    standalone_outputs,
                    allow_structural_toolchain=allow_structural_toolchain,
                )
                standalone_cache_key = self._cache_key_for_standalone(
                    standalone_outputs,
                    target_chip=target_chip,
                    target_cann=target_cann,
                    fixture_dir=standalone_inputs / "runtime-fixtures",
                )
                cached_standalone = self._copy_cached_namespace(
                    cache_dir, "standalone", standalone_cache_key, standalone_outputs
                )
                if cached_standalone:
                    standalone_manifest = self._relocate_standalone_verify_manifest(
                        standalone_outputs
                    )
                    standalone_build = self._relocate_standalone_build_manifest(
                        standalone_outputs
                    )
                    standalone_build_status = standalone_build.get("status", "reused")
                    standalone_cache_hit = True
                else:
                    standalone_build_status, standalone_build, standalone_log = (
                        self._build_standalone_executable(
                            standalone_outputs,
                            target_chip,
                            target_cann,
                            allow_structural_toolchain=allow_structural_toolchain,
                        )
                    )
                    if standalone_build_status not in {"passed", "structural-passed"}:
                        state["status"] = "needs-human"
                        state["iterations"][0]["findings"] = (
                            findings + analyze_build_log(standalone_log)
                        )
                        state["stages"].append(
                            self._stage_record(
                                "06",
                                "failed",
                                stage_root,
                                {
                                    "standalone": {
                                        "status": "failed",
                                        "cache_key": standalone_cache_key,
                                        "cache_hit": False,
                                    }
                                },
                                started_at=stage_started_at,
                            )
                        )
                        self._write_state(output_dir, state)
                        if emit_layout:
                            self._emit_layout(
                                layout,
                                public_output_dir,
                                output_dir,
                                state,
                                asset_slug=asset_slug,
                                source_path=source_path,
                                mode="single",
                                profile=profile,
                                selected_stages=selected_stages,
                            )
                        return state
                    cached_standalone = self._store_cached_namespace(
                        cache_dir,
                        "standalone",
                        standalone_cache_key,
                        standalone_outputs,
                    )
                    standalone_cache_hit = False
            manifest_fixture_statuses = standalone_manifest.get("runtime_fixtures")
            if isinstance(manifest_fixture_statuses, dict):
                fixture_statuses = {
                    entry_name: {
                        **fixture_statuses.get(entry_name, {}),
                        **fixture_status,
                    }
                    for entry_name, fixture_status in manifest_fixture_statuses.items()
                    if isinstance(fixture_status, dict)
                }
            runtime_contracts = self._build_runtime_contracts(
                stage_root / "runtime-contracts",
                entries,
                fixture_statuses=fixture_statuses,
            )
            standalone_precheck_status = None
            if do_verify and self._verify_uses_standalone(resolved_verify_backend):
                standalone_precheck_status = (
                    "skipped-target-arch"
                    if standalone_build_status == "skipped-target-arch"
                    else self._npu_verify_enabled(allow_mock_npu=allow_mock_npu)
                )
            standalone_executable_paths = None
            executables = standalone_build.get("executables")
            if isinstance(executables, dict):
                standalone_executable_paths = {}
                for entry, path in executables.items():
                    standalone_executable_paths[entry] = Path(path)
            standalone_verification = self._run_standalone_verification(
                standalone_outputs / "verify",
                entries,
                fixture_statuses=fixture_statuses,
                verify_enabled=do_verify
                and self._verify_uses_standalone(resolved_verify_backend),
                jobs=resolved_jobs,
                executable_path=(
                    Path(standalone_build.get("executable", ""))
                    if standalone_build.get("executable")
                    else None
                ),
                executable_paths=standalone_executable_paths,
                precheck_status=standalone_precheck_status,
            )

            wheel_summary: dict[str, Any] = {
                "mode": resolved_wheel_mode,
                "status": "skipped",
                "cache_key": None,
                "cache_hit": False,
                "reused_from": None,
                "wheel_paths": [],
                "wheels": [],
            }
            wheel_verification: dict[str, Any] | None = None
            wheel_paths: list[str] = []
            if self._needs_wheel_stage(
                resolved_wheel_mode, resolved_verify_backend, reuse_wheel
            ):
                wheel_root = stage_root / "wheel"
                wheel_inputs = wheel_root / "inputs"
                wheel_outputs = wheel_root / "outputs"
                stage05_outputs = self._stage_dir(output_dir, "05") / "outputs"
                self._materialize_input(stage05_outputs, wheel_inputs / "pybind-output")
                self._copy_contents(stage05_outputs, wheel_outputs)
                wheels_dest = wheel_outputs / "wheels"
                wheels_dest.mkdir(parents=True, exist_ok=True)
                if reuse_wheel:
                    reuse_path = Path(reuse_wheel).resolve()
                    expected_prefix = f"{aggregate_wheel_name}-{package_version}-"
                    if not reuse_path.is_file():
                        raise OrchestratorError(f"reuse wheel not found: {reuse_path}")
                    if (
                        not reuse_path.name.startswith(expected_prefix)
                        or reuse_path.suffix != ".whl"
                    ):
                        raise OrchestratorError(
                            f"reuse wheel does not match {aggregate_wheel_name} {package_version}: {reuse_path.name}"
                        )
                    dest = wheels_dest / reuse_path.name
                    shutil.copy2(reuse_path, dest)
                    wheel_paths = [str(dest)]
                    wheel_summary.update(
                        {
                            "status": "reused",
                            "reused_from": str(reuse_path),
                            "wheel_paths": wheel_paths,
                            "wheels": [dest.name],
                        }
                    )
                else:
                    self._ensure_structural_toolchain_env(
                        wheel_outputs,
                        allow_structural_toolchain=allow_structural_toolchain,
                    )
                    wheel_cache_key = self._cache_key_for_wheel(
                        wheel_outputs,
                        target_chip=target_chip,
                        target_cann=target_cann,
                        package_name=aggregate_wheel_name,
                        package_version=package_version,
                    )
                    wheel_summary["cache_key"] = wheel_cache_key
                    cached_wheel = None
                    if resolved_wheel_mode == "cache":
                        cached_wheel = self._copy_cached_namespace(
                            cache_dir, "wheels", wheel_cache_key, wheel_outputs
                        )
                    if cached_wheel:
                        cached_manifest = self._relocate_wheel_build_manifest(
                            wheel_outputs
                        )
                        cached_status = (
                            cached_manifest.get("status")
                            if isinstance(cached_manifest, dict)
                            else ""
                        )
                        wheel_paths = [
                            str(path)
                            for path in sorted((wheel_outputs / "wheels").glob("*.whl"))
                        ]
                        wheel_summary.update(
                            {
                                "status": (
                                    "structural-reused"
                                    if cached_status == "structural-passed"
                                    else "reused"
                                ),
                                "cache_hit": True,
                                "reused_from": cached_wheel,
                                "wheel_paths": wheel_paths,
                                "wheels": [Path(path).name for path in wheel_paths],
                            }
                        )
                    else:
                        build_status, build_result, build_log = (
                            self._build_native_wheel(
                                wheel_outputs,
                                jobs=resolved_jobs,
                                target_chip=target_chip,
                                allow_structural_toolchain=allow_structural_toolchain,
                            )
                        )
                        wheels_src = (
                            wheel_outputs / "operator-sk-native-wheel-build" / "wheels"
                        )
                        wheel_paths = []
                        for wheel in (
                            sorted(wheels_src.glob("*.whl"))
                            if wheels_src.exists()
                            else []
                        ):
                            dest = wheels_dest / wheel.name
                            shutil.copy2(wheel, dest)
                            wheel_paths.append(str(dest))
                        log_path_text = build_result.get("log_path")
                        log_path = Path(log_path_text) if log_path_text else None
                        if log_path is not None and log_path.is_file():
                            shutil.copy2(log_path, wheel_outputs / "pip-wheel.log")
                        wheel_status = (
                            "structural-built"
                            if build_status == "structural-passed"
                            else "built" if build_status == "passed" else "failed"
                        )
                        wheel_summary.update(
                            {
                                "status": wheel_status,
                                "cache_hit": False,
                                "wheels": build_result.get("wheels", []),
                                "wheel_paths": wheel_paths,
                            }
                        )
                        if build_status not in {"passed", "structural-passed"}:
                            state["status"] = "needs-human"
                            state["iterations"][0]["findings"] = (
                                findings + analyze_build_log(build_log)
                            )
                            state["stages"].append(
                                self._stage_record(
                                    "06",
                                    "failed",
                                    stage_root,
                                    {"wheel": wheel_summary},
                                    started_at=stage_started_at,
                                )
                            )
                            self._write_state(output_dir, state)
                            if emit_layout:
                                self._emit_layout(
                                    layout,
                                    public_output_dir,
                                    output_dir,
                                    state,
                                    asset_slug=asset_slug,
                                    source_path=source_path,
                                    mode="single",
                                    profile=profile,
                                    selected_stages=selected_stages,
                                )
                            return state
                        self._store_cached_namespace(
                            cache_dir, "wheels", wheel_cache_key, wheel_outputs
                        )
                if self._verify_uses_wheel(resolved_verify_backend):
                    if do_verify:
                        verify_status = self._npu_verify_enabled(
                            allow_mock_npu=allow_mock_npu
                        )
                    else:
                        verify_status = "skipped-by-user"
                    wheel_verification = self._run_sample_gen_verification(
                        wheel_outputs / "verify",
                        entries,
                        status=verify_status,
                        wheel_paths=wheel_paths,
                    )
                legacy_outputs = stage_root / "outputs"
                self._remove_existing(legacy_outputs)
                shutil.copytree(wheel_outputs, legacy_outputs, symlinks=True)

            verification = {
                "backend": resolved_verify_backend,
                "standalone": standalone_verification,
                "wheel": wheel_verification,
                "summary": standalone_verification.get("summary"),
            }
            differential = self._build_differential_verdicts(
                stage_root / "differential",
                entries,
                baseline_manifests=baseline_manifests,
                runtime_contracts=runtime_contracts,
                standalone_verification=standalone_verification,
                wheel_verification=wheel_verification,
            )
            state["verification"] = verification
            state["runtime_contracts"] = runtime_contracts
            state["baseline"] = {
                "status": "passed" if not baseline_parallel["failed_ops"] else "failed",
                "per_op": {
                    item["op_name"]: {
                        "status": item["manifest"].get("status"),
                        "correctness": item["manifest"].get("correctness"),
                        "manifest": item["manifest_path"],
                        "artifacts": item["manifest"].get("artifacts", {}),
                    }
                    for item in baseline_results
                },
                "parallel": baseline_parallel,
            }
            state["differential"] = differential
            state["standalone"] = {
                "status": standalone_verification["status"],
                "build_status": (
                    "reused" if standalone_cache_hit else standalone_build_status
                ),
                "verify_status": standalone_verification["status"],
                "cache_key": standalone_cache_key,
                "cache_hit": standalone_cache_hit,
                "reused_from": cached_standalone if standalone_cache_hit else None,
                "manifest": str(
                    standalone_outputs / "operator-sk-standalone-verify.json"
                ),
                "build_manifest": str(
                    standalone_outputs / "operator-sk-standalone-build.json"
                ),
            }
            state["wheel"] = wheel_summary
            legacy_build_status = (
                "structural-passed"
                if wheel_summary.get("status")
                in {"structural-built", "structural-reused"}
                else (
                    "passed"
                    if wheel_summary.get("status") in {"built", "reused"}
                    else wheel_summary.get("status", "skipped")
                )
            )
            state["iterations"][0]["build"] = {
                **wheel_summary,
                "status": legacy_build_status,
            }

            legacy_verify = stage_root / "verify"
            self._remove_existing(legacy_verify)
            source_verify = (
                (stage_root / "wheel" / "outputs" / "verify")
                if wheel_verification
                else (standalone_outputs / "verify")
            )
            shutil.copytree(source_verify, legacy_verify, symlinks=True)
            state["stages"].append(
                self._stage_record(
                    "06",
                    "passed",
                    stage_root,
                    {
                        "standalone": state["standalone"],
                        "baseline": state["baseline"],
                        "wheel": wheel_summary,
                        "differential": {
                            "status": differential["status"],
                            "summary": differential["summary"],
                        },
                        "verification": resolved_verify_backend,
                    },
                    started_at=stage_started_at,
                )
            )
            state["iterations"][0]["stages"].append("build-and-verify")
            self._write_state(output_dir, state)

        if "06" in selected_stages:
            wheel_status = state.get("wheel", {}).get("status")
            standalone_status = (
                state.get("verification", {}).get("standalone", {}).get("status")
            )
            wheel_verification_state = state.get("verification", {}).get("wheel")
            if not isinstance(wheel_verification_state, dict):
                wheel_verification_state = {}
            wheel_verify_status = wheel_verification_state.get("status")
            if (
                wheel_status in {"structural-built", "structural-reused"}
                or standalone_build_status == "structural-passed"
                or allow_structural_toolchain
            ):
                state["status"] = "structural-only"
            elif (
                wheel_verify_status == "mock-passed"
                or standalone_status == "mock-passed"
            ):
                state["status"] = "mock-only"
            elif wheel_status in {"built", "reused"}:
                if resolved_verify_backend == "none" or not do_verify:
                    state["status"] = "packaged"
                elif wheel_verify_status == "passed" or standalone_status == "passed":
                    state["status"] = "verified"
                elif standalone_status in {
                    "skipped-no-npu",
                    "skipped-insufficient-runtime-spec",
                    "skipped-by-user",
                    "skipped-target-arch",
                }:
                    state["status"] = standalone_status
                else:
                    state["status"] = "packaged"
            else:
                if standalone_status in {
                    "skipped-no-npu",
                    "skipped-insufficient-runtime-spec",
                    "skipped-by-user",
                    "skipped-target-arch",
                }:
                    state["status"] = standalone_status
                elif standalone_status == "failed":
                    state["status"] = "needs-human"
                else:
                    state["status"] = "verified"
        elif "05" in selected_stages:
            state["status"] = "pybind-generated"
        elif "02" in selected_stages:
            state["status"] = "adapted"
        else:
            state["status"] = "analyzed"
        if state["status"] == "adapted" and (
            "03" in selected_stages or "04" in selected_stages
        ):
            state["status"] = "clean"
        self._write_state(output_dir, state)
        if emit_layout:
            self._emit_layout(
                layout,
                public_output_dir,
                output_dir,
                state,
                asset_slug=asset_slug,
                source_path=source_path,
                mode="single",
                profile=profile,
                selected_stages=selected_stages,
            )
        return state

    @staticmethod
    def _utc_now() -> str:
        return (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _write_state(output_dir: Path, state: dict[str, Any]) -> None:
        state["updated_at"] = PipelineOrchestrator._utc_now()
        state["final_iteration"] = (
            len(state.get("iterations", [])) - 1 if state.get("iterations") else -1
        )
        (output_dir / "pipeline-state.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _remove_existing(path: Path) -> None:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()

    @staticmethod
    def _existing_asset_manifests(public_output_dir: Path) -> list[dict[str, Any]]:
        assets: list[dict[str, Any]] = []
        for path in sorted(
            (public_output_dir / "assets").glob("*/asset-manifest.json")
        ):
            try:
                assets.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return assets

    @staticmethod
    def _copy_support_headers(source_dir: Path, dest_dir: Path) -> list[str]:
        copied: list[str] = []
        if not source_dir.is_dir():
            return copied
        for source_path in sorted(
            item for item in source_dir.rglob("*") if item.is_file()
        ):
            rel = source_path.relative_to(source_dir)
            if any(part in _SUPPORT_SKIP_DIRS for part in rel.parts[:-1]):
                continue
            if source_path.suffix not in _SUPPORT_HEADER_SUFFIXES:
                continue
            dest_path = dest_dir / rel
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest_path, follow_symlinks=False)
            copied.append(rel.as_posix())
        return copied

    @staticmethod
    def _shared_support_dirs_for_asset(asset: Path) -> list[tuple[str, Path]]:
        asset_root = asset if asset.is_dir() else asset.parent
        if not asset_root.is_dir():
            return []
        support_dirs: dict[str, Path] = {}
        source_suffixes = {".asc", ".cpp"} | set(_SUPPORT_HEADER_SUFFIXES)
        for source in sorted(
            item
            for item in asset_root.rglob("*")
            if item.is_file() and item.suffix in source_suffixes
        ):
            rel = source.relative_to(asset_root)
            if any(part in {"test", "tests"} for part in rel.parts):
                continue
            text = source.read_text(encoding="utf-8", errors="replace")
            for include in _QUOTED_INCLUDE_RE.findall(text):
                include_path = Path(include)
                if include_path.is_absolute() or len(include_path.parts) < 2:
                    continue
                source_name = include_path.parts[0]
                if source_name in {".", ".."}:
                    continue
                candidate = (asset_root.parent / source_name).resolve()
                asset_include = (asset_root / include_path).resolve()
                if asset_include.is_file():
                    continue
                support_rel = Path(*include_path.parts[1:])
                if candidate.is_dir() and (candidate / support_rel).is_file():
                    support_dirs.setdefault(source_name, candidate)
        return sorted(support_dirs.items())

    @staticmethod
    def _stage_dir(output_dir: Path, stage_id: str) -> Path:
        return output_dir / _STAGE_DIRS[stage_id]

    @staticmethod
    def _safe_slug(path: Path) -> str:
        slug = re.sub(
            r"[^A-Za-z0-9_.-]", "_", path.stem if path.is_file() else path.name
        ).strip("._-")
        return slug or "op"

    @staticmethod
    def _default_jobs(op_count: int, arch_count: int = 1) -> int:
        cpu_count = os.cpu_count() or 1
        work_items = max(1, op_count) * max(1, arch_count)
        return max(1, min(work_items, cpu_count))

    @staticmethod
    def _arch_count_from_env(env: Any) -> int:
        raw_arches = env.get("SK_NPU_ARCHS", "")
        if raw_arches:
            arches = [
                arch.strip() for arch in re.split(r"[,;]", raw_arches) if arch.strip()
            ]
            return max(1, len(dict.fromkeys(arches)))
        return 1

    @staticmethod
    def _resolve_profile_options(
        *,
        profile: str,
        verify_backend: str | None,
        wheel_mode: str | None,
        no_verify: bool,
        no_package: bool,
    ) -> tuple[str, str]:
        if profile not in {"fast", "release"}:
            raise OrchestratorError("--profile must be fast or release")
        if no_verify:
            verify_backend = "none"
        if no_package:
            wheel_mode = "never"
            if verify_backend is None:
                verify_backend = "standalone"
        resolved_verify = verify_backend or (
            "standalone" if profile == "fast" else "both"
        )
        resolved_wheel = wheel_mode or ("never" if profile == "fast" else "cache")
        if resolved_verify not in _VERIFY_BACKENDS:
            raise OrchestratorError(
                "--verify-backend must be standalone, wheel, both, or none"
            )
        if resolved_wheel not in _WHEEL_MODES:
            raise OrchestratorError("--wheel-mode must be never, cache, or always")
        if resolved_wheel == "never" and resolved_verify in {"wheel", "both"}:
            resolved_verify = "standalone" if resolved_verify == "both" else "none"
        return resolved_verify, resolved_wheel

    @staticmethod
    def _verify_uses_wheel(verify_backend: str) -> bool:
        return verify_backend in {"wheel", "both"}

    @staticmethod
    def _verify_uses_standalone(verify_backend: str) -> bool:
        return verify_backend in {"standalone", "both"}

    @staticmethod
    def _needs_wheel_stage(
        wheel_mode: str, verify_backend: str, reuse_wheel: str | None
    ) -> bool:
        return (
            bool(reuse_wheel)
            or wheel_mode != "never"
            or PipelineOrchestrator._verify_uses_wheel(verify_backend)
        )

    @staticmethod
    def _parse_stages(
        stages: str | Sequence[str] | None,
        *,
        do_package: bool,
        profile: str,
        wheel_mode: str,
        verify_backend: str,
        reuse_wheel: str | None,
    ) -> list[str]:
        if stages is None:
            if not do_package:
                selected = ["01", "02", "03"]
            elif profile == "fast" or not PipelineOrchestrator._needs_wheel_stage(
                wheel_mode, verify_backend, reuse_wheel
            ):
                selected = ["01", "02", "03", "06"]
            else:
                selected = ["01", "02", "03", "05", "06"]
        elif isinstance(stages, str):
            selected = [
                item.strip().split("-", 1)[0]
                for item in stages.split(",")
                if item.strip()
            ]
        else:
            selected = [
                str(item).strip().split("-", 1)[0]
                for item in stages
                if str(item).strip()
            ]
        if not selected:
            raise OrchestratorError("--stages must select at least one stage")
        if "00" not in selected:
            selected = ["00", *selected]
        unknown = [stage for stage in selected if stage not in _STAGE_DIRS]
        if unknown:
            raise OrchestratorError(f"unknown stage: {unknown[0]}")
        selected_set = set(selected)
        for stage in selected:
            prereqs = set(_STAGE_PREREQS[stage])
            if stage == "06" and PipelineOrchestrator._needs_wheel_stage(
                wheel_mode, verify_backend, reuse_wheel
            ):
                prereqs.add("05")
            for prereq in sorted(prereqs):
                if prereq not in selected_set:
                    raise OrchestratorError(f"{stage} needs {prereq}")
        return [stage for stage in _STAGE_ORDER if stage in selected_set]

    @staticmethod
    def _run_parallel_ops(
        ops: list[dict[str, Any]],
        jobs: int,
        worker: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        started = time.monotonic()
        max_workers = max(1, min(jobs, len(ops) or 1))
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_by_name = {
                executor.submit(worker, op): (op.get("name") or op.get("entry_name"))
                for op in ops
            }
            for future in as_completed(future_by_name):
                op_name = future_by_name[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"op_name": op_name, "status": "failed", "error": str(exc)}
                result.setdefault("op_name", op_name)
                results.append(result)
        results.sort(key=lambda item: item["op_name"])
        failed = [item["op_name"] for item in results if item.get("status") == "failed"]
        passed = [item["op_name"] for item in results if item.get("status") != "failed"]
        failed_details = {
            str(item["op_name"]): str(item.get("error", "failed"))
            for item in results
            if item.get("status") == "failed"
        }
        return results, {
            "jobs": max_workers,
            "failed_ops": failed,
            "failed_details": failed_details,
            "passed_ops": passed,
            "duration_seconds": round(time.monotonic() - started, 3),
        }

    @staticmethod
    def _entries_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
        entries_by_name: dict[str, dict[str, Any]] = {}
        for per_file in manifest.get("per_file", []):
            for entry in per_file.get("entries", []):
                entries_by_name.setdefault(entry["entry_name"], entry)
        return list(entries_by_name.values())

    @staticmethod
    def _zero_value_for_param(param: dict[str, Any]) -> Any:
        c_type = param.get("c_type", "")
        if "GM_ADDR" in c_type or "*" in c_type or "__gm__" in c_type:
            return [0] * 16
        return 16

    @staticmethod
    def _runtime_input_spec_for_entry(entry: dict[str, Any]) -> dict[str, Any]:
        parameters = []
        for index, param in enumerate(entry.get("parameters", [])):
            c_type = param.get("c_type", "GM_ADDR")
            name = param["name"]
            parameters.append(
                {
                    "index": index,
                    "name": name,
                    "type": c_type,
                    "source": f"{c_type} {name}",
                }
            )
        return {
            "input_specs": [
                {
                    "id": f"entry:{entry['entry_name']}:default",
                    "entry_name": entry["entry_name"],
                    "source_file": entry.get(
                        "source_file", f"{entry['entry_name']}.asc"
                    ),
                    "parameters": parameters,
                    "requires_user_input": True,
                }
            ]
        }

    @staticmethod
    def _standalone_runner_stdout(op_name: str, *, status: str) -> dict[str, Any]:
        return {
            "backend": "standalone",
            "status": status,
            "outputs": {},
            "calls": {op_name: []},
        }

    def _skill_path(self, key: str) -> Path:
        skill_name, script_name = _SKILL_SCRIPTS[key]
        return self.skills_root / skill_name / "scripts" / script_name

    def _script_dir(self, skill_name: str) -> Path:
        return self.skills_root / skill_name / "scripts"

    def _run(self, key: str, *argv: str) -> subprocess.CompletedProcess:
        script_path = self._skill_path(key)
        if not script_path.is_file():
            raise OrchestratorError(f"skill script not found for {key}: {script_path}")
        return subprocess.run(
            [self.python, str(script_path), *argv],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=self.env,
        )

    def _ensure_structural_toolchain_env(
        self, output_dir: Path, *, allow_structural_toolchain: bool
    ) -> None:
        env = (self.env or os.environ).copy()
        if env.get("SK_OPERATOR_USE_REAL_TOOLCHAIN") == "1":
            self.env = env
            self._toolchain_mode = "real"
            return
        if not allow_structural_toolchain:
            env.pop("SK_OPERATOR_STRUCTURAL_TOOLCHAIN", None)
            env.pop("SK_OPERATOR_ALLOW_STANDALONE_MOCK_EXECUTABLE", None)
            self.env = env
            self._toolchain_mode = "real"
            return
        if env.get("SK_OPERATOR_STRUCTURAL_TOOLCHAIN") == "1":
            self.env = env
            self._toolchain_mode = "structural"
            return
        tool_dir = output_dir / "operator-sk-structural-toolchain"
        tool_dir.mkdir(parents=True, exist_ok=True)
        cmake = tool_dir / "cmake"
        cmake.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then\n'
            "  printf '%s\\n' 'cmake version 3.26.0-structural'\n"
            "  exit 0\n"
            "fi\n"
            'if [ "$1" = "-S" ]; then\n'
            '  mkdir -p "$4"\n'
            "  printf '%s\\n' 'configured' > \"$4/configured.txt\"\n"
            "  exit 0\n"
            "fi\n"
            'if [ "$1" = "--build" ]; then\n'
            '  mkdir -p "$2"\n'
            "  printf '%s\\n' 'built' > \"$2/built.txt\"\n"
            "  exit 0\n"
            "fi\n"
            "printf '%s\\n' 'structural cmake: unsupported arguments' >&2\n"
            "exit 2\n",
            encoding="utf-8",
        )
        cmake.chmod(0o755)
        bisheng = tool_dir / "bisheng"
        bisheng.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then\n'
            "  printf '%s\\n' 'bisheng structural 1.0'\n"
            "  exit 0\n"
            "fi\n"
            'out=""\n'
            "while [ $# -gt 0 ]; do\n"
            '  case "$1" in\n'
            '    -o) out="$2"; shift 2 ;;\n'
            '    -o*) out="${1#-o}"; shift ;;\n'
            "    *) shift ;;\n"
            "  esac\n"
            "done\n"
            'if [ -n "$out" ]; then\n'
            '  mkdir -p "$(dirname "$out")"\n'
            "  printf '%s\\n' 'STRUCTURAL BISHENG OUTPUT' > \"$out\"\n"
            "fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        bisheng.chmod(0o755)
        env["PATH"] = str(tool_dir) + os.pathsep + env.get("PATH", "")
        env["SK_OPERATOR_STRUCTURAL_TOOLCHAIN"] = "1"
        env["SK_OPERATOR_ALLOW_STANDALONE_MOCK_EXECUTABLE"] = "1"
        self.env = env
        self._toolchain_mode = "structural"

    def _materialize_input(self, source: Path, dest: Path) -> str:
        self._remove_existing(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, dest, symlinks=True)
        elif not dest.suffix:
            dest.mkdir(parents=True, exist_ok=True)
            self._copy_file_asset_with_local_includes(source, dest)
        else:
            shutil.copy2(source, dest)
        return "copy"

    @staticmethod
    def _copy_file_asset_with_local_includes(source: Path, dest_dir: Path) -> None:
        root = source.parent.resolve()
        queue = [source.resolve()]
        copied: set[Path] = set()
        while queue:
            current = queue.pop(0)
            try:
                current.relative_to(root)
            except ValueError:
                continue
            if current in copied or not current.is_file():
                continue
            copied.add(current)
            rel = current.relative_to(root)
            target = dest_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(current, target)
            if current.suffix not in _LOCAL_INCLUDE_SOURCE_SUFFIXES:
                continue
            text = current.read_text(encoding="utf-8", errors="replace")
            for include in _QUOTED_INCLUDE_RE.findall(text):
                include_path = Path(include)
                if include_path.is_absolute() or ".." in include_path.parts:
                    continue
                candidate = (current.parent / include_path).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError:
                    continue
                if candidate.is_file() and candidate not in copied:
                    queue.append(candidate)

    def _copy_contents(self, source_dir: Path, dest_dir: Path) -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        for item in source_dir.iterdir():
            dest = dest_dir / item.name
            self._remove_existing(dest)
            if item.is_dir() and not item.is_symlink():
                shutil.copytree(item, dest, symlinks=True)
            else:
                shutil.copy2(item, dest, follow_symlinks=False)

    def _shared_support_dirs_for_stage_asset(
        self, stage_asset: Path, source_asset: Path | None = None
    ) -> list[tuple[str, Path]]:
        support_dirs: dict[str, Path] = {}
        for asset in [stage_asset, source_asset]:
            if asset is None:
                continue
            for support_name, support_source in self._shared_support_dirs_for_asset(
                asset
            ):
                resolved = support_source.resolve()
                existing = support_dirs.get(support_name)
                if existing is not None and existing.resolve() != resolved:
                    raise OrchestratorError(
                        f"conflicting shared support directory {support_name!r}: {existing} vs {support_source}"
                    )
                support_dirs[support_name] = support_source
        return sorted(support_dirs.items())

    def _normalize_assets(
        self,
        assets: Path | Sequence[Path],
        asset_names: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        raw_assets = [assets] if isinstance(assets, Path) else list(assets)
        if not raw_assets:
            raise OrchestratorError(
                "--asset or --asset-root must provide at least one asset"
            )
        seen_paths: set[Path] = set()
        seen_slugs: set[str] = set()
        normalized: list[dict[str, Any]] = []
        for raw in raw_assets:
            path = Path(raw).resolve()
            if not path.exists():
                raise OrchestratorError(f"asset path not found: {path}")
            if path in seen_paths:
                continue
            seen_paths.add(path)
            slug = safe_slug(
                (asset_names or {}).get(str(path)) or self._safe_slug(path), "op"
            )
            if slug in seen_slugs:
                raise OrchestratorError(f"duplicate op asset name: {slug}")
            seen_slugs.add(slug)
            normalized.append({"name": slug, "asset": path})
        return normalized

    def _understand_and_expand_assets(
        self,
        assets: Path | Sequence[Path],
        output_dir: Path,
        asset_names: dict[str, str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        raw_assets = self._normalize_assets(assets, asset_names=asset_names)
        stage_root = self._stage_dir(output_dir, "00")
        stage_root.mkdir(parents=True, exist_ok=True)
        ops: list[dict[str, Any]] = []
        asset_records: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for raw in raw_assets:
            asset = Path(raw["asset"]).resolve()
            asset_name = str(raw["name"])
            op_root = stage_root / asset_name
            inputs = op_root / "inputs"
            outputs = op_root / "outputs"
            self._materialize_input(asset, inputs / "asset")
            scan_argv = [
                "adapt-asset",
                str(inputs / "asset"),
                "--output-dir",
                str(outputs),
            ]
            if self._current_target_chip:
                scan_argv.extend(["--target-chip", self._current_target_chip])
            scan = self._run("asset_adapter", *scan_argv)
            if scan.returncode not in {0, 2}:
                raise OrchestratorError(
                    f"asset adapter failed for {asset_name}: {scan.stdout}"
                )
            layout_path = outputs / "operator-asset-layout.json"
            if scan.returncode == 2:
                raise OrchestratorError(
                    f"asset adapter requires user input for {asset_name}; inspect "
                    f"{outputs / 'adapter-report.json'} and provide explicit "
                    "contracts"
                )
            analyze_argv = [
                "analyze-operator-asset",
                str(inputs / "asset"),
                "--asset-layout",
                str(layout_path),
                "--output-dir",
                str(outputs),
            ]
            if self._current_target_chip:
                analyze_argv.extend(["--target-chip", self._current_target_chip])
            analyze = self._run("codegen", *analyze_argv)
            if analyze.returncode != 0:
                raise OrchestratorError(
                    f"analyze-operator-asset failed for {asset_name}: {analyze.stdout}"
                )
            normalize_argv = [
                "normalize-operator-asset",
                str(inputs / "asset"),
                "--understanding",
                str(outputs / "operator-asset-understanding.json"),
                "--asset-layout",
                str(layout_path),
                "--output-dir",
                str(outputs),
            ]
            if self._current_target_chip:
                normalize_argv.extend(["--target-chip", self._current_target_chip])
            normalize = self._run("codegen", *normalize_argv)
            if normalize.returncode != 0:
                raise OrchestratorError(
                    f"normalize-operator-asset failed for {asset_name}: {normalize.stdout}"
                )
            understanding = self._read_json(
                outputs / "operator-asset-understanding.json"
            )
            units_index = self._read_json(outputs / "operator-units.json")
            unit_records = []
            for unit in units_index.get("operator_units", []):
                unit_name = str(unit["unit_id"])
                if len(units_index.get("operator_units", [])) == 1:
                    op_name = asset_name
                else:
                    op_name = f"{asset_name}_{unit_name}"
                op_name = re.sub(r"[^A-Za-z0-9_.-]", "_", op_name).strip("._-") or "op"
                if op_name in seen_names:
                    raise OrchestratorError(
                        f"duplicate operator unit name after asset understanding: {op_name}"
                    )
                seen_names.add(op_name)
                op = {
                    "name": op_name,
                    "entry_name": unit.get("entry_name"),
                    "source_entry_name": unit.get("source_entry_name")
                    or unit.get("entry_name"),
                    "public_entry_name": unit.get("public_entry_name")
                    or unit.get("entry_name"),
                    "internal_symbol_name": unit.get("internal_symbol_name")
                    or unit.get("entry_name"),
                    "bind_target": unit.get("bind_target")
                    or unit.get("source_entry_name")
                    or unit.get("entry_name"),
                    "name_resolution": dict(unit.get("name_resolution", {})),
                    "asset": Path(unit["asset"]).resolve(),
                    "sk_asset": Path(unit.get("sk_asset") or unit["asset"]).resolve(),
                    "source_asset": asset,
                    "asset_kind": unit.get("asset_kind")
                    or understanding.get("asset_kind"),
                    "kernel_source": unit.get("kernel_source"),
                    "host_source": unit.get("host_source"),
                    "json_spec": unit.get("json_spec"),
                    "tiling_headers": list(unit.get("tiling_headers", [])),
                    "build_backends": list(unit.get("build_backends", [])),
                    "supported_soc_versions": list(
                        unit.get("supported_soc_versions", [])
                    ),
                    "supported_arches": list(unit.get("supported_arches", [])),
                    "target_resolution": dict(unit.get("target_resolution", {})),
                    "support_source": unit.get("support_source", ""),
                    "understanding_outputs": outputs,
                }
                ops.append(op)
                unit_records.append(
                    {
                        "name": op_name,
                        "entry_name": op["entry_name"],
                        "source_entry_name": op.get("source_entry_name"),
                        "public_entry_name": op.get("public_entry_name"),
                        "internal_symbol_name": op.get("internal_symbol_name"),
                        "bind_target": op.get("bind_target"),
                        "name_resolution": op.get("name_resolution", {}),
                        "asset": str(op["asset"]),
                        "sk_asset": str(op["sk_asset"]),
                        "source_asset": str(asset),
                        "asset_kind": op["asset_kind"],
                        "kernel_source": op.get("kernel_source"),
                        "host_source": op.get("host_source"),
                        "json_spec": op.get("json_spec"),
                        "tiling_headers": op.get("tiling_headers", []),
                        "build_backends": op["build_backends"],
                        "supported_soc_versions": op.get("supported_soc_versions", []),
                        "supported_arches": op.get("supported_arches", []),
                        "target_resolution": op.get("target_resolution", {}),
                        "support_source": op.get("support_source", ""),
                    }
                )
            asset_records.append(
                {
                    "name": asset_name,
                    "path": str(asset),
                    "asset_kind": understanding.get("asset_kind"),
                    "understanding": str(outputs / "operator-asset-understanding.json"),
                    "units": unit_records,
                }
            )
        summary = {
            "schema_version": 1,
            "status": "analyzed",
            "assets": asset_records,
            "operator_units": [
                {
                    "name": op["name"],
                    "entry_name": op.get("entry_name"),
                    "source_entry_name": op.get("source_entry_name"),
                    "public_entry_name": op.get("public_entry_name"),
                    "internal_symbol_name": op.get("internal_symbol_name"),
                    "bind_target": op.get("bind_target"),
                    "name_resolution": op.get("name_resolution", {}),
                    "asset": str(op["asset"]),
                    "sk_asset": str(op["sk_asset"]),
                    "source_asset": str(op["source_asset"]),
                    "asset_kind": op.get("asset_kind"),
                    "kernel_source": op.get("kernel_source"),
                    "host_source": op.get("host_source"),
                    "json_spec": op.get("json_spec"),
                    "tiling_headers": op.get("tiling_headers", []),
                    "build_backends": op.get("build_backends", []),
                    "supported_soc_versions": op.get("supported_soc_versions", []),
                    "supported_arches": op.get("supported_arches", []),
                    "target_resolution": op.get("target_resolution", {}),
                    "support_source": op.get("support_source", ""),
                }
                for op in ops
            ],
        }
        self._write_json(stage_root / "operator-units.json", summary)
        return ops, summary

    def _detect_form(self, source: Path, output_dir: Path) -> dict[str, Any]:
        result = self._run(
            "codegen", "detect-sk-form", str(source), "--output-dir", str(output_dir)
        )
        if result.returncode != 0:
            raise OrchestratorError(f"detect-sk-form failed: {result.stdout}")
        return self._read_json(output_dir / "operator-sk-form-analysis.json")

    def _adapt(
        self,
        asset: Path,
        output_dir: Path,
        support_dirs: list[str] | None = None,
        *,
        understanding: Path | None = None,
        target_chip: str = "",
        package_name: str = "op_extension",
        package_version: str = "0.1.0",
        name_resolution: Path | None = None,
        source_asset: Path | None = None,
        io_contract: Path | None = None,
    ) -> dict[str, Any]:
        argv = ["adapt-sk-from-global", str(asset), "--output-dir", str(output_dir)]
        argv.extend(
            ["--package-name", package_name, "--package-version", package_version]
        )
        if understanding is not None:
            argv.extend(["--understanding", str(understanding)])
        if target_chip:
            argv.extend(["--target-chip", target_chip])
        if name_resolution is not None:
            argv.extend(["--name-resolution", str(name_resolution)])
        if source_asset is not None:
            argv.extend(["--source-asset", str(source_asset)])
        if io_contract is not None:
            argv.extend(["--io-contract", str(io_contract)])
        for support_dir in support_dirs or []:
            argv.extend(["--support-dir", support_dir])
        result = self._run("codegen", *argv)
        if result.returncode != 0:
            raise OrchestratorError(f"adapt-sk-from-global failed: {result.stdout}")
        return self._read_json(output_dir / "operator-sk-adapted.json")

    def _stage_io_contract(self, io_contract: Path | None, inputs: Path) -> Path | None:
        if io_contract is None:
            return None
        dest = inputs / "operator-io-contract.json"
        self._materialize_input(io_contract, dest)
        return dest

    def _scan_spec(self, source: Path, output_dir: Path) -> dict[str, Any]:
        result = self._run(
            "validate",
            "validate-operator",
            "--asset",
            str(source),
            "--output-dir",
            str(output_dir),
            "--rule-pack",
            "spec",
            "--stage",
            "post-adapt",
            "--iteration-index",
            "0",
        )
        if result.returncode != 0:
            raise OrchestratorError(f"validate spec failed: {result.stdout}")
        return self._read_json(output_dir / "operator-validation-findings.json")

    def _scan_compat(
        self, source: Path, output_dir: Path, target_chip: str, target_cann: str
    ) -> dict[str, Any]:
        argv = [
            "validate-operator",
            "--asset",
            str(source),
            "--output-dir",
            str(output_dir),
            "--rule-pack",
            "compat",
            "--target-chip",
            target_chip,
            "--stage",
            "post-adapt",
            "--iteration-index",
            "0",
        ]
        if target_cann:
            argv.extend(["--target-cann", target_cann])
        result = self._run("validate", *argv)
        findings_path = output_dir / "operator-validation-findings.json"
        if result.returncode != 0 and not findings_path.exists():
            raise OrchestratorError(f"validate compat failed: {result.stdout}")
        return self._read_json(findings_path)

    def _aggregate(
        self,
        adapted_outputs: list[Path],
        output_dir: Path,
        package_name: str,
        package_version: str,
    ) -> dict[str, Any]:
        argv: list[str] = ["aggregate-sk-adapted"]
        for adapted_output in adapted_outputs:
            argv.extend(["--adapted-output-dir", str(adapted_output)])
        argv.extend(
            [
                "--output-dir",
                str(output_dir),
                "--aggregate-wheel-name",
                package_name,
                "--package-version",
                package_version,
            ]
        )
        result = self._run("codegen", *argv)
        if result.returncode != 0:
            raise OrchestratorError(f"aggregate-sk-adapted failed: {result.stdout}")
        return self._read_json(output_dir / "operator-sk-adapted.json")

    def _generate_pybind_binding(self, output_dir: Path) -> dict[str, Any]:
        result = self._run("build_package", "generate-pybind-binding", str(output_dir))
        if result.returncode != 0:
            raise OrchestratorError(f"generate-pybind-binding failed: {result.stdout}")
        return self._read_json(output_dir / "operator-sk-pybind-binding.json")

    def _build_native_wheel(
        self,
        output_dir: Path,
        *,
        jobs: int | None = None,
        target_chip: str = "",
        allow_structural_toolchain: bool = False,
    ) -> tuple[str, dict[str, Any], str]:
        self._ensure_structural_toolchain_env(
            output_dir, allow_structural_toolchain=allow_structural_toolchain
        )
        for stale in (
            "operator-sk-native-wheel-build",
            "operator-sk-native-wheel.json",
        ):
            self._remove_existing(output_dir / stale)
        argv = ["build-native-wheel", str(output_dir)]
        if jobs and jobs > 0:
            argv.extend(["--jobs", str(jobs)])
        if target_chip:
            argv.extend(["--target-chip", target_chip])
        result = self._run("build_package", *argv)
        manifest_path = output_dir / "operator-sk-native-wheel.json"
        manifest = self._read_json(manifest_path) if manifest_path.exists() else {}
        log = ""
        if manifest.get("log_path") and Path(manifest["log_path"]).exists():
            log = Path(manifest["log_path"]).read_text(
                encoding="utf-8", errors="replace"
            )
        status = manifest.get("status", "failed")
        return status, manifest, result.stdout + "\n" + log

    def _build_baseline(
        self,
        asset: Path,
        output_dir: Path,
        *,
        entry_name: str,
        allow_structural_toolchain: bool = False,
    ) -> dict[str, Any]:
        use_structural = allow_structural_toolchain
        argv = [
            "build-baseline",
            str(asset),
            "--output-dir",
            str(output_dir),
            "--entry-name",
            entry_name,
            "--backend",
            "baseline_direct_asc",
        ]
        if use_structural:
            argv.append("--structural")
        result = self._run("build_package", *argv)
        manifest_path = output_dir / "operator-baseline-build.json"
        if result.returncode != 0 and not manifest_path.exists():
            raise OrchestratorError(
                f"build-baseline failed for {entry_name}: {result.stdout}"
            )
        return self._read_json(manifest_path)

    def _build_differential_verdicts(
        self,
        output_dir: Path,
        entries: list[dict[str, Any]],
        *,
        baseline_manifests: dict[str, Path],
        runtime_contracts: dict[str, dict[str, Any]],
        standalone_verification: dict[str, Any],
        wheel_verification: dict[str, Any] | None,
    ) -> dict[str, Any]:
        sample_gen_dir = self._script_dir("sk-operator-sample-gen")
        if str(sample_gen_dir) not in sys.path:
            sys.path.insert(0, str(sample_gen_dir))
        import operator_differential_verify

        per_op: dict[str, Any] = {}
        summary_rows = [
            "# Operator Differential Verification",
            "",
            "| op | status | baseline | sk | wheel |",
            "|---|---|---|---|---|",
        ]
        wheel_per_op = (
            (wheel_verification or {}).get("per_op")
            if isinstance(wheel_verification, dict)
            else {}
        )
        if not isinstance(wheel_per_op, dict):
            wheel_per_op = {}
        standalone_per_op = (
            standalone_verification.get("per_op")
            if isinstance(standalone_verification, dict)
            else {}
        )
        if not isinstance(standalone_per_op, dict):
            standalone_per_op = {}
        for entry in entries:
            entry_name = entry["entry_name"]
            verdict = operator_differential_verify.build_differential_verdict(
                output_dir / entry_name,
                entry_name=entry_name,
                baseline_manifest=baseline_manifests.get(entry_name),
                runtime_contract=runtime_contracts.get(entry_name, {}).get("path"),
                sk_verdict=standalone_per_op.get(entry_name),
                wheel_verdict=(
                    wheel_per_op.get(entry_name) if wheel_verification else None
                ),
            )
            per_op[entry_name] = verdict
            summary_rows.append(
                "| `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                    entry_name,
                    verdict["status"],
                    verdict["baseline"]["build_status"],
                    verdict["sk"].get("status"),
                    verdict["wheel"].get("status"),
                )
            )
        statuses = {str(item.get("status")) for item in per_op.values()}
        if "failed" in statuses:
            status = "failed"
        elif not statuses:
            status = "unavailable"
        elif len(statuses) == 1:
            status = next(iter(statuses))
        elif statuses <= {"passed", "skipped-structural-baseline"}:
            status = "skipped-structural-baseline"
        elif all(item.startswith("skipped") for item in statuses):
            status = "skipped"
        else:
            status = "mixed"
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.md"
        summary_path.write_text("\n".join(summary_rows) + "\n", encoding="utf-8")
        manifest = {"status": status, "per_op": per_op, "summary": str(summary_path)}
        self._write_json(output_dir / "operator-differential-summary.json", manifest)
        return manifest

    def _build_runtime_contracts(
        self,
        output_dir: Path,
        entries: list[dict[str, Any]],
        *,
        fixture_statuses: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        sample_gen_dir = self._script_dir("sk-operator-sample-gen")
        if str(sample_gen_dir) not in sys.path:
            sys.path.insert(0, str(sample_gen_dir))
        import operator_runtime_contract

        contracts: dict[str, dict[str, Any]] = {}
        for entry in entries:
            entry_name = entry["entry_name"]
            entry_dir = output_dir / entry_name
            contract = operator_runtime_contract.build_runtime_contract(
                entry_dir,
                entry=entry,
                fixture_status=fixture_statuses.get(
                    entry_name, {"status": "insufficient"}
                ),
            )
            contracts[entry_name] = {
                "status": contract.get("status"),
                "reason": contract.get("reason"),
                "path": str(entry_dir / "operator-runtime-contract.json"),
            }
        self._write_json(
            output_dir / "operator-runtime-contract-summary.json",
            {"schema_version": 1, "contracts": contracts},
        )
        return contracts

    def _cache_lib(self):
        build_pkg_dir = self._script_dir("sk-operator-build-package")
        if str(build_pkg_dir) not in sys.path:
            sys.path.insert(0, str(build_pkg_dir))
        import sk_build_cache_lib

        return sk_build_cache_lib

    def _generate_standalone_compare(
        self,
        aggregate_output_dir: Path,
        output_dir: Path,
        fixture_dir: Path,
        target_chip: str,
    ) -> dict[str, Any]:
        result = self._run(
            "codegen",
            "generate-standalone-compare",
            str(aggregate_output_dir),
            "--output-dir",
            str(output_dir),
            "--runtime-fixture-dir",
            str(fixture_dir),
            "--target-chip",
            target_chip,
        )
        if result.returncode != 0:
            raise OrchestratorError(
                f"generate-standalone-compare failed: {result.stdout}"
            )
        return self._read_json(output_dir / "operator-sk-standalone-verify.json")

    def _build_standalone_executable(
        self,
        output_dir: Path,
        target_chip: str,
        target_cann: str,
        *,
        allow_structural_toolchain: bool = False,
    ) -> tuple[str, dict[str, Any], str]:
        self._ensure_structural_toolchain_env(
            output_dir, allow_structural_toolchain=allow_structural_toolchain
        )
        for stale in (
            "operator-sk-standalone-build.json",
            "build-log.txt",
            "executable-path.txt",
        ):
            self._remove_existing(output_dir / stale)
        argv = [
            "build-standalone-executable",
            str(output_dir),
            "--target-chip",
            target_chip,
        ]
        if target_cann:
            argv.extend(["--target-cann", target_cann])
        result = self._run("build_package", *argv)
        manifest_path = output_dir / "operator-sk-standalone-build.json"
        manifest = self._read_json(manifest_path) if manifest_path.exists() else {}
        log = (
            (output_dir / "build-log.txt").read_text(encoding="utf-8", errors="replace")
            if (output_dir / "build-log.txt").exists()
            else ""
        )
        return manifest.get("status", "failed"), manifest, result.stdout + "\n" + log

    def _cache_key_for_standalone(
        self,
        standalone_output_dir: Path,
        *,
        target_chip: str,
        target_cann: str,
        fixture_dir: Path,
    ) -> str:
        lib = self._cache_lib()
        env = self.env if self.env is not None else os.environ
        return lib.cache_key(
            {
                "kind": "standalone",
                "toolchain_mode": self._toolchain_mode,
                "target_chip": target_chip,
                "target_cann": target_cann,
                "ascend_home_path": (
                    str(Path(env.get("ASCEND_HOME_PATH", "")).resolve())
                    if env.get("ASCEND_HOME_PATH")
                    else ""
                ),
                "cmake_version": lib.command_version("cmake", env=env),
                "compiler_version": lib.command_version("bisheng", env=env),
                "tool_files": lib.hash_paths(
                    [
                        self._skill_path("codegen"),
                        self._script_dir("sk-operator-codegen") / "sk_codegen_lib.py",
                        self._skill_path("build_package"),
                        self._script_dir("sk-operator-build-package")
                        / "sk_build_cache_lib.py",
                    ]
                ),
                "files": lib.hash_paths(
                    [
                        standalone_output_dir
                        / "operator-sk-standalone-verify"
                        / "runtime_compare.asc",
                        *(standalone_output_dir / "operator-sk-standalone-verify").glob(
                            "runtime_compare_*.asc"
                        ),
                        standalone_output_dir
                        / "operator-sk-standalone-verify"
                        / "CMakeLists.txt",
                        *(
                            standalone_output_dir
                            / "operator-sk-standalone-verify"
                            / "csrc"
                        ).rglob("*"),
                        *fixture_dir.rglob("*"),
                    ]
                ),
            }
        )

    def _cache_key_for_wheel(
        self,
        wheel_output_dir: Path,
        *,
        target_chip: str,
        target_cann: str,
        package_name: str,
        package_version: str,
    ) -> str:
        lib = self._cache_lib()
        package_root = wheel_output_dir / "operator-sk-adapted"
        return lib.cache_key(
            {
                "kind": "wheel",
                "toolchain_mode": self._toolchain_mode,
                "python_version": sys.version,
                "package_name": package_name,
                "package_version": package_version,
                "target_chip": target_chip,
                "target_cann": target_cann,
                "tool_files": lib.hash_paths(
                    [
                        self._skill_path("codegen"),
                        self._script_dir("sk-operator-codegen") / "sk_codegen_lib.py",
                        self._skill_path("build_package"),
                        self._script_dir("sk-operator-build-package")
                        / "sk_pybind_lib.py",
                        self._script_dir("sk-operator-build-package")
                        / "sk_build_cache_lib.py",
                    ]
                ),
                "files": lib.hash_paths(
                    [
                        package_root / "setup.py",
                        *(package_root / "op_extension").glob("*.py"),
                        *(package_root / "csrc").rglob("*"),
                    ]
                ),
            }
        )

    def _copy_cached_namespace(
        self, cache_dir: Path, namespace: str, cache_key: str, dest: Path
    ) -> str | None:
        lib = self._cache_lib()
        cached = lib.lookup_cache(cache_dir, namespace, cache_key)
        if cached is None:
            return None
        lib.copy_cached_outputs(cached, dest)
        return str(cached)

    def _store_cached_namespace(
        self, cache_dir: Path, namespace: str, cache_key: str, source: Path
    ) -> str:
        lib = self._cache_lib()
        return str(lib.store_cache(cache_dir, namespace, cache_key, source))

    def _relocate_standalone_build_manifest(self, output_dir: Path) -> dict[str, Any]:
        manifest_path = output_dir / "operator-sk-standalone-build.json"
        manifest = self._read_json(manifest_path)
        source_root = output_dir / "operator-sk-standalone-verify"
        build_dir = source_root / "build"
        executable_targets = (
            manifest.get("executable_targets")
            if isinstance(manifest.get("executable_targets"), dict)
            else {}
        )
        if executable_targets:
            executable_targets = {
                target: str(build_dir / Path(path).name)
                for target, path in executable_targets.items()
            }
        else:
            executable_targets = {"runtime_compare": str(build_dir / "runtime_compare")}
        executables = (
            manifest.get("executables")
            if isinstance(manifest.get("executables"), dict)
            else {}
        )
        if executables:
            executables = {
                entry: str(build_dir / Path(path).name)
                for entry, path in executables.items()
            }
        executable = Path(next(iter(executable_targets.values())))
        log_path = output_dir / "build-log.txt"
        manifest.update(
            {
                "standalone_verify_dir": str(source_root),
                "build_dir": str(build_dir),
                "configure_command": [
                    "cmake",
                    "-S",
                    str(source_root),
                    "-B",
                    str(build_dir),
                ],
                "build_command": ["cmake", "--build", str(build_dir)],
                "executable": str(executable),
                "executable_targets": executable_targets,
                "executables": executables,
                "log_path": str(log_path),
            }
        )
        self._write_json(manifest_path, manifest)
        (output_dir / "executable-path.txt").write_text(
            str(executable) + "\n", encoding="utf-8"
        )
        return manifest

    def _relocate_standalone_verify_manifest(self, output_dir: Path) -> dict[str, Any]:
        manifest_path = output_dir / "operator-sk-standalone-verify.json"
        manifest = self._read_json(manifest_path)
        source_root = output_dir / "operator-sk-standalone-verify"
        inputs_root = output_dir.parent / "inputs"
        manifest.update(
            {
                "aggregate_output_dir": str(inputs_root / "aggregate-output"),
                "standalone_verify_dir": str(source_root),
            }
        )
        runtime_fixtures = manifest.get("runtime_fixtures")
        if isinstance(runtime_fixtures, dict):
            fixture_root = inputs_root / "runtime-fixtures"
            for entry_name, fixture in runtime_fixtures.items():
                if isinstance(fixture, dict) and "path" in fixture:
                    fixture["path"] = str(fixture_root / entry_name)
        self._write_json(manifest_path, manifest)
        return manifest

    def _relocate_wheel_build_manifest(self, output_dir: Path) -> dict[str, Any] | None:
        manifest_path = output_dir / "operator-sk-native-wheel.json"
        if not manifest_path.exists():
            return None
        manifest = self._read_json(manifest_path)
        build_dir = output_dir / "operator-sk-native-wheel-build"
        log_path = build_dir / "pip-wheel.log"
        manifest.update(
            {
                "pybind_source_root": str(output_dir / "operator-sk-adapted"),
                "build_dir": str(build_dir),
                "command": [
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
                    str(output_dir / "operator-sk-native-wheel-build" / "wheels"),
                    str(output_dir / "operator-sk-adapted"),
                ],
                "log_path": str(log_path),
            }
        )
        self._write_json(manifest_path, manifest)
        return manifest

    def _auto_inputs_for_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        params = []
        for param in entry.get("parameters", []):
            value = self._zero_value_for_param(param)
            is_tensor = isinstance(value, list)
            params.append(
                {
                    "name": param["name"],
                    "shape": [len(value)] if is_tensor else [],
                    "dtype": "float16" if is_tensor else "uint32",
                    "layout": "ND",
                    "value_source": {"kind": "inline_json", "value": value},
                }
            )
        return {
            "input_values": [
                {
                    "input_set_id": f"entry:{entry['entry_name']}:default",
                    "entry_name": entry["entry_name"],
                    "parameter_values": params,
                }
            ]
        }

    def _npu_verify_enabled(self, *, allow_mock_npu: bool = False) -> str:
        env = self.env if self.env is not None else os.environ
        if allow_mock_npu and env.get("SK_OPERATOR_MOCK_NPU") == "1":
            return "mock-passed"
        if env.get("SK_OPERATOR_RUN_DEVICE_COMPARE") != "1":
            return "skipped-no-npu"
        if any(
            Path(path).exists()
            for path in ("/dev/davinci_manager", "/dev/davinci0", "/dev/hisi_hdc")
        ):
            return "passed"
        return "skipped-no-npu"

    def _copy_runtime_fixtures(
        self, ops: list[dict[str, Any]], fixture_root: Path
    ) -> dict[str, dict[str, Any]]:
        fixture_root.mkdir(parents=True, exist_ok=True)
        statuses: dict[str, dict[str, Any]] = {}
        for op in ops:
            op_fixture_dir = fixture_root / op["name"]
            op_fixture_dir.mkdir(parents=True, exist_ok=True)
            asset = Path(op["asset"])
            copied: list[str] = []
            asset_root = asset if asset.is_dir() else asset.parent
            test_main = (
                asset / "tests" / "main.cpp"
                if asset.is_dir()
                else asset.parent / "tests" / "main.cpp"
            )
            if test_main.is_file():
                dest = op_fixture_dir / "tests" / "main.cpp"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(test_main, dest)
                copied.append("tests/main.cpp")
            runtime_spec = (
                asset / "operator-sk-runtime-fixture.json"
                if asset.is_dir()
                else asset.parent / "operator-sk-runtime-fixture.json"
            )
            if runtime_spec.is_file():
                shutil.copy2(
                    runtime_spec, op_fixture_dir / "operator-sk-runtime-fixture.json"
                )
                copied.append("operator-sk-runtime-fixture.json")
            for candidate in (
                asset / "interface" if asset.is_dir() else asset.parent / "interface",
                asset.parent / "interface",
            ):
                if candidate.is_dir():
                    dest = op_fixture_dir / "interface"
                    self._remove_existing(dest)
                    shutil.copytree(candidate, dest, symlinks=True)
                    copied.append("interface/")
                    break
            support_root = op_fixture_dir / "asset-support"
            support_files = self._copy_support_headers(asset_root, support_root)
            shared_common = asset_root.parent / "common"
            if (
                shared_common.is_dir()
                and shared_common.resolve() != (asset_root / "common").resolve()
            ):
                support_files.extend(
                    self._copy_support_headers(shared_common, support_root / "common")
                )
            shared_interface = asset_root.parent / "interface"
            if (
                shared_interface.is_dir()
                and shared_interface.resolve() != (asset_root / "interface").resolve()
            ):
                support_files.extend(
                    self._copy_support_headers(
                        shared_interface, support_root / "interface"
                    )
                )
            if support_files:
                copied.append(f"asset-support/{len(support_files)} files")
            status = "available" if copied else "insufficient"
            statuses[op["name"]] = {
                "status": status,
                "path": str(op_fixture_dir),
                "copied": copied,
                "reason": (
                    "runtime fixture available"
                    if copied
                    else "runtime fixture not found"
                ),
            }
        return statuses

    def _run_standalone_verification(
        self,
        verify_dir: Path,
        entries: list[dict[str, Any]],
        *,
        fixture_statuses: dict[str, dict[str, Any]],
        verify_enabled: bool,
        jobs: int,
        executable_path: Path | None,
        executable_paths: dict[str, Path] | None = None,
        precheck_status: str | None = None,
    ) -> dict[str, Any]:
        verify_dir.mkdir(parents=True, exist_ok=True)

        def worker(entry: dict[str, Any]) -> dict[str, Any]:
            op_name = entry["entry_name"]
            op_dir = verify_dir / op_name
            op_dir.mkdir(parents=True, exist_ok=True)
            fixture_status = fixture_statuses.get(op_name, {"status": "insufficient"})
            if precheck_status == "skipped-target-arch":
                runner_stdout = self._standalone_runner_stdout(
                    op_name, status="skipped-target-arch"
                )
                comparison = {"status": "skipped-target-arch", "comparisons": []}
                verdict = {
                    "status": "skipped-target-arch",
                    "reason": "standalone target arch is not resolved",
                }
            elif precheck_status == "mock-passed":
                runner_stdout = self._standalone_runner_stdout(
                    op_name, status="mock-passed"
                )
                comparison = {"status": "mock-passed", "comparisons": []}
                verdict = {
                    "status": "mock-passed",
                    "reason": "mock NPU verification explicitly enabled",
                }
            elif precheck_status == "skipped-no-npu":
                runner_stdout = self._standalone_runner_stdout(
                    op_name, status="skipped-no-npu"
                )
                comparison = {"status": "skipped-no-npu", "comparisons": []}
                verdict = {"status": "skipped-no-npu", "reason": "NPU not available"}
            elif not verify_enabled:
                runner_stdout = self._standalone_runner_stdout(
                    op_name, status="skipped-by-user"
                )
                comparison = {"status": "skipped-by-user", "comparisons": []}
                verdict = {
                    "status": "skipped-by-user",
                    "reason": "verification disabled by --no-verify",
                }
            elif (
                fixture_status.get("status") != "available"
                or fixture_status.get("device_runnable") is False
            ):
                reason = fixture_status.get("reason", "runtime fixture not available")
                runner_stdout = self._standalone_runner_stdout(
                    op_name, status="skipped-insufficient-runtime-spec"
                )
                comparison = {
                    "status": "skipped-insufficient-runtime-spec",
                    "comparisons": [],
                }
                verdict = {
                    "status": "skipped-insufficient-runtime-spec",
                    "reason": reason,
                }
            else:
                selected_executable = (executable_paths or {}).get(
                    op_name
                ) or executable_path
                if selected_executable is None or not selected_executable.exists():
                    raise OrchestratorError(
                        f"standalone executable not found for {op_name}: {selected_executable}"
                    )
                env = (self.env or os.environ).copy()
                completed = subprocess.run(
                    [str(selected_executable)],
                    cwd=str(selected_executable.parent),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    env=env,
                    timeout=60,
                )
                if completed.returncode != 0:
                    runner_stdout = {
                        "backend": "standalone",
                        "status": "failed",
                        "stdout": completed.stdout,
                        "outputs": {},
                        "calls": {op_name: []},
                    }
                    comparison = {"status": "failed", "comparisons": []}
                    verdict = {"status": "failed", "reason": "standalone_runner_failed"}
                    self._write_json(op_dir / "runner-stdout.json", runner_stdout)
                    self._write_json(op_dir / "comparison.json", comparison)
                    self._write_json(op_dir / "verdict.json", verdict)
                    return {
                        "op_name": op_name,
                        "status": verdict["status"],
                        "verdict": verdict,
                    }
                stdout_lines = [
                    line for line in completed.stdout.splitlines() if line.strip()
                ]
                json_payload = stdout_lines[-1] if stdout_lines else completed.stdout
                try:
                    full_stdout = json.loads(json_payload)
                except json.JSONDecodeError as exc:
                    raise OrchestratorError(
                        f"standalone runner emitted invalid JSON for {op_name}: {completed.stdout}"
                    ) from exc
                outputs = full_stdout.get("outputs", {})
                calls = full_stdout.get("calls", {})
                statuses = full_stdout.get("statuses", {})
                entry_status = (
                    statuses.get(op_name, {}) if isinstance(statuses, dict) else {}
                )
                entry_status_value = entry_status.get(
                    "status", full_stdout.get("status", "passed")
                )
                entry_reason = entry_status.get("reason", entry_status_value)
                runner_stdout = {
                    "backend": "standalone",
                    "status": entry_status_value,
                    "outputs": (
                        {op_name: outputs[op_name]} if op_name in outputs else {}
                    ),
                    "calls": {op_name: calls.get(op_name, [])},
                }
                self._write_json(op_dir / "runner-stdout.json", runner_stdout)
                if entry_status_value != "passed":
                    comparison = {"status": entry_status_value, "comparisons": []}
                    verdict = {"status": entry_status_value, "reason": entry_reason}
                elif op_name not in outputs:
                    comparison = {"status": "failed", "comparisons": []}
                    verdict = {
                        "status": "failed",
                        "reason": "standalone_runner_missing_outputs",
                    }
                else:
                    compare = self._run(
                        "sample_gen",
                        "compare-sk-runtime-outputs",
                        str(op_dir),
                        str(op_dir / "runner-stdout.json"),
                    )
                    if compare.returncode != 0:
                        raise OrchestratorError(
                            f"standalone compare-sk-runtime-outputs failed for {op_name}: {compare.stdout}"
                        )
                    comparison = self._read_json(
                        op_dir / "operator-sk-runtime-output-comparison.json"
                    )
                    verdict = {
                        "status": (
                            "passed"
                            if comparison.get("status") == "matched"
                            else "failed"
                        ),
                        "reason": (
                            "differential_outputs_matched"
                            if comparison.get("status") == "matched"
                            else "differential_outputs_mismatched"
                        ),
                    }
            self._write_json(op_dir / "runner-stdout.json", runner_stdout)
            self._write_json(op_dir / "comparison.json", comparison)
            self._write_json(op_dir / "verdict.json", verdict)
            return {"op_name": op_name, "status": verdict["status"], "verdict": verdict}

        results, parallel = self._run_parallel_ops(entries, jobs, worker)
        per_op = {
            item["op_name"]: item["verdict"] for item in results if "verdict" in item
        }
        status_values = {item["status"] for item in results}
        if "failed" in status_values:
            status = "failed"
        elif status_values == {"skipped-target-arch"}:
            status = "skipped-target-arch"
        elif status_values == {"mock-passed"}:
            status = "mock-passed"
        elif status_values == {"skipped-no-npu"}:
            status = "skipped-no-npu"
        elif status_values == {"skipped-insufficient-runtime-spec"}:
            status = "skipped-insufficient-runtime-spec"
        elif status_values == {"skipped-by-user"}:
            status = "skipped-by-user"
        elif status_values <= {"passed"}:
            status = "passed"
        else:
            status = "mixed"
        rows = [
            "# Standalone Differential Verification",
            "",
            "| op | status |",
            "|---|---|",
        ]
        for op_name in sorted(per_op):
            rows.append(f"| `{op_name}` | `{per_op[op_name]['status']}` |")
        (verify_dir / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
        return {
            "status": status,
            "per_op": per_op,
            "summary": str(verify_dir / "summary.md"),
            "parallel": parallel,
        }

    def _run_sample_gen_verification(
        self,
        verify_dir: Path,
        entries: list[dict[str, Any]],
        *,
        status: str,
        wheel_paths: list[str],
    ) -> dict[str, Any]:
        verify_dir.mkdir(parents=True, exist_ok=True)
        summary_rows = [
            "# Differential Verification",
            "",
            "| op | status |",
            "|---|---|",
        ]
        per_op: dict[str, Any] = {}
        for entry in entries:
            op_name = entry["entry_name"]
            op_dir = verify_dir / op_name
            op_dir.mkdir(parents=True, exist_ok=True)
            self._write_json(
                op_dir / "operator-sk-runtime-input-spec.json",
                self._runtime_input_spec_for_entry(entry),
            )
            construct = self._run(
                "sample_gen", "auto-construct-runtime-input-values", str(op_dir)
            )
            if construct.returncode != 0:
                raise OrchestratorError(
                    f"auto-construct-runtime-input-values failed for {op_name}: {construct.stdout}"
                )
            shutil.copy2(
                op_dir / "operator-sk-auto-input-values-spec.json",
                op_dir / "auto-inputs.json",
            )
            runtime_values = op_dir / "operator-sk-runtime-input-values.json"
            if runtime_values.is_file():
                oracle = self._run(
                    "sample_gen", "auto-build-correctness-oracle", str(op_dir)
                )
                if oracle.returncode != 0:
                    raise OrchestratorError(
                        f"auto-build-correctness-oracle failed for {op_name}: {oracle.stdout}"
                    )
                runner = self._run("sample_gen", "generate-runner-script", str(op_dir))
                if runner.returncode != 0:
                    raise OrchestratorError(
                        f"generate-runner-script failed for {op_name}: {runner.stdout}"
                    )
            runner_spec = {
                "oracle_source": "bind-target-on-wheel",
                "entry_name": op_name,
                "functions": {
                    "baseline": f"run_{op_name}",
                    "sk": f"torch.ops.ascendc_ops.{op_name}",
                },
                "sample_gen_command_spec": (
                    self._read_json(
                        op_dir / "operator-sk-target-runtime-command-spec.json"
                    )
                    if (
                        op_dir / "operator-sk-target-runtime-command-spec.json"
                    ).is_file()
                    else {}
                ),
                "wheels": wheel_paths,
            }
            self._write_json(op_dir / "runner-spec.json", runner_spec)
            if status == "skipped-by-user":
                runner_stdout = {"status": "skipped-by-user", "outputs": {}}
                comparison = {"status": "skipped-by-user", "comparisons": []}
                verdict = {
                    "status": "skipped-by-user",
                    "reason": "verification disabled by --no-verify",
                }
            elif status == "skipped-no-npu":
                runner_stdout = {"status": "skipped-no-npu", "outputs": {}}
                comparison = {"status": "skipped-no-npu", "comparisons": []}
                verdict = {"status": "skipped-no-npu", "reason": "NPU not available"}
            elif status == "mock-passed":
                runner_stdout = {"status": "mock-passed", "outputs": {}}
                comparison = {"status": "mock-passed", "comparisons": []}
                verdict = {
                    "status": "mock-passed",
                    "reason": "mock NPU verification explicitly enabled",
                }
            elif not runtime_values.is_file():
                runner_stdout = {
                    "status": "skipped-insufficient-runtime-spec",
                    "outputs": {},
                }
                comparison = {
                    "status": "skipped-insufficient-runtime-spec",
                    "comparisons": [],
                }
                verdict = {
                    "status": "skipped-insufficient-runtime-spec",
                    "reason": "canonical runtime input values not provided",
                    "auto_input_suggestion": "auto-inputs.json",
                }
            else:
                env = (self.env or os.environ).copy()
                completed = subprocess.run(
                    [
                        self.python,
                        str(op_dir / "operator-sample-runner.py"),
                        str(op_dir / "operator-sk-runtime-input-values.json"),
                    ],
                    cwd=str(op_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    env=env,
                    timeout=60,
                )
                if completed.returncode != 0:
                    runner_stdout = {
                        "status": "failed",
                        "stdout": completed.stdout,
                        "outputs": {},
                    }
                    comparison = {"status": "failed", "comparisons": []}
                    verdict = {"status": "failed", "reason": "runner_failed"}
                else:
                    runner_stdout = json.loads(completed.stdout)
                    runner_stdout["status"] = "passed"
                    self._write_json(op_dir / "runner-stdout.json", runner_stdout)
                    compare = self._run(
                        "sample_gen",
                        "compare-sk-runtime-outputs",
                        str(op_dir),
                        str(op_dir / "runner-stdout.json"),
                    )
                    if compare.returncode != 0:
                        raise OrchestratorError(
                            f"compare-sk-runtime-outputs failed for {op_name}: {compare.stdout}"
                        )
                    comparison = self._read_json(
                        op_dir / "operator-sk-runtime-output-comparison.json"
                    )
                    verdict = {
                        "status": (
                            "passed"
                            if comparison.get("status") == "matched"
                            else "failed"
                        ),
                        "reason": (
                            "differential_outputs_matched"
                            if comparison.get("status") == "matched"
                            else "differential_outputs_mismatched"
                        ),
                    }
            self._write_json(op_dir / "runner-stdout.json", runner_stdout)
            self._write_json(op_dir / "comparison.json", comparison)
            self._write_json(op_dir / "verdict.json", verdict)
            per_op[op_name] = verdict
            summary_rows.append(f"| `{op_name}` | `{verdict['status']}` |")
        if status == "skipped-no-npu":
            summary_rows.extend(
                ["", "NPU not available; differential validation skipped."]
            )
        elif status == "skipped-by-user":
            summary_rows.extend(["", "Differential validation skipped by --no-verify."])
        elif status == "mock-passed":
            summary_rows.extend(
                [
                    "",
                    "Mock NPU validation was explicitly requested; numerical correctness was not executed.",
                ]
            )
        (verify_dir / "summary.md").write_text(
            "\n".join(summary_rows) + "\n", encoding="utf-8"
        )
        status_values = {item.get("status") for item in per_op.values()}
        if status_values == {"skipped-insufficient-runtime-spec"}:
            final_status = "skipped-insufficient-runtime-spec"
        else:
            final_status = status
        return {
            "status": final_status,
            "per_op": per_op,
            "summary": str(verify_dir / "summary.md"),
        }

    def _stage_record(
        self,
        stage_id: str,
        status: str,
        path: Path,
        extra: dict[str, Any] | None = None,
        *,
        started_at: str | None = None,
    ) -> dict[str, Any]:
        record = {
            "stage": stage_id,
            "name": _STAGE_DIRS[stage_id],
            "status": status,
            "path": str(path),
            "started_at": started_at or self._utc_now(),
            "finished_at": self._utc_now(),
        }
        if extra:
            record.update(extra)
        return record

    def _emit_layout(
        self,
        layout: ArtifactLayout,
        public_output_dir: Path,
        work_output_dir: Path,
        state: dict[str, Any],
        *,
        asset_slug: str,
        source_path: Path | None,
        mode: str,
        profile: str,
        selected_stages: list[str],
    ) -> dict[str, Any]:
        asset_payload = layout.ingest_run(
            asset_slug=asset_slug,
            state=state,
            work_output_dir=work_output_dir,
            source_path=source_path,
        )
        assets = [
            asset
            for asset in self._existing_asset_manifests(public_output_dir)
            if asset.get("asset_slug") != asset_payload.get("asset_slug")
        ]
        assets.append(asset_payload)
        status = state.get("status", "unknown")
        artifact_map = layout.build_map(
            status=status,
            mode=mode,
            profile=profile,
            selected_stages=selected_stages,
            assets=assets,
        )
        layout.write_lint_report(artifact_map)
        return artifact_map
