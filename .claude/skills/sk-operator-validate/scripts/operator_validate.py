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

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATE_ROOT = SCRIPT_DIR.parent
if str(VALIDATE_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATE_ROOT))

from operator_asset_contracts import (  # noqa: E402
    ContractError,
    contract_finding,
    read_json,
    validate_asset_layout_contract,
    validate_build_context_contract,
    validate_verify_context_contract,
    write_json,
)
from rule_packs.compat_runner import (  # noqa: E402
    CliUsageError as CompatRulePackError,
    list_compat_targets,
    run_compat_rules,
)
from rule_packs.spec_runner import (  # noqa: E402
    CliUsageError as SpecRulePackError,
    list_spec_rules,
    run_spec_rules,
)


SCHEMA_VERSION = 1


class CliUsageError(Exception):
    pass


def _normalize_contract_finding(finding: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(finding)
    normalized["actionable_by"] = ["human"]
    normalized["remediation_hint"] = {"kind": "human-decision"}
    return normalized


def _contract_findings(
    args: argparse.Namespace, *, require_contracts: bool
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    checks = [
        ("contract.asset-layout", args.layout, validate_asset_layout_contract),
        ("contract.build-context", args.build_context, validate_build_context_contract),
        (
            "contract.verify-context",
            args.verify_context,
            validate_verify_context_contract,
        ),
    ]
    for rule_id, raw_path, validator in checks:
        if not raw_path:
            if not require_contracts:
                continue
            severity = "warning" if rule_id == "contract.verify-context" else "blocker"
            findings.append(
                _normalize_contract_finding(
                    contract_finding(
                        rule_id,
                        f"{raw_path or rule_id} not provided",
                        severity=severity,
                    )
                )
            )
            continue
        path = Path(raw_path).resolve()
        try:
            validator(read_json(path), path.parent)
        except (OSError, json.JSONDecodeError, ContractError, ValueError) as exc:
            findings.append(
                _normalize_contract_finding(contract_finding(rule_id, str(exc)))
            )
    return findings


def _spec_findings(
    asset: Path, stage: str, iteration_index: int
) -> list[dict[str, Any]]:
    try:
        payload = run_spec_rules(asset, stage=stage, iteration_index=iteration_index)
    except SpecRulePackError as exc:
        return [
            _normalize_contract_finding(
                contract_finding("validate.spec-pack-error", str(exc))
            )
        ]
    return list(payload.get("findings", []))


def _compat_findings(
    asset: Path,
    target_chip: str,
    target_cann: str,
    stage: str,
    iteration_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        payload = run_compat_rules(
            asset,
            target_chip=target_chip,
            target_cann=target_cann,
            stage=stage,
            iteration_index=iteration_index,
        )
    except CompatRulePackError as exc:
        return [
            _normalize_contract_finding(
                contract_finding("validate.compat-pack-error", str(exc))
            )
        ], {}
    return list(payload.get("findings", [])), dict(payload.get("metadata", {}))


def cmd_validate_operator(args: argparse.Namespace) -> int:
    asset = Path(args.asset).resolve()
    if not asset.exists():
        raise CliUsageError(f"asset path not found: {asset}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    findings: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    findings.extend(
        _contract_findings(
            args, require_contracts=args.rule_pack in {"all", "contract"}
        )
    )
    if args.rule_pack in {"all", "spec"}:
        findings.extend(_spec_findings(asset, args.stage, int(args.iteration_index)))
    if args.rule_pack in {"all", "compat"}:
        compat_findings, compat_metadata = _compat_findings(
            asset,
            args.target_chip,
            getattr(args, "target_cann", "") or "",
            args.stage,
            int(args.iteration_index),
        )
        findings.extend(compat_findings)
        if compat_metadata:
            metadata["compat"] = compat_metadata

    payload = {
        "schema_version": SCHEMA_VERSION,
        "skill_source": "operator-validate",
        "stage": args.stage,
        "iteration_index": int(args.iteration_index),
        "findings": findings,
    }
    if metadata:
        payload["metadata"] = metadata
    write_json(output_dir / "operator-validation-findings.json", payload)
    _write_markdown(output_dir / "operator-validation-report.md", payload)
    return 1 if any(item.get("severity") == "blocker" for item in findings) else 0


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# SK Operator Validation",
        "",
        f"Findings: {len(payload.get('findings', []))}",
        "",
    ]
    if not payload.get("findings"):
        lines.append("No validation findings.")
    compat_metadata = payload.get("metadata", {}).get("compat", {})
    if compat_metadata:
        lines.extend(
            [
                "",
                "## Compat Coverage",
                f"Target chip: {compat_metadata.get('target_chip', '') or '(not provided)'}",
                f"Target CANN: {compat_metadata.get('target_cann', '') or '(not provided)'}",
                f"Verdict: {compat_metadata.get('overall_verdict', '')}",
                f"Verification: {compat_metadata.get('verification', '')}",
                f"Checked rules: {', '.join(compat_metadata.get('checked_rules', [])) or '(none)'}",
                "",
            ]
        )
    for finding in payload.get("findings", []):
        lines.append(f"## {finding.get('rule_id')} ({finding.get('severity')})")
        lines.append(str(finding.get("message", "")))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def cmd_list_rule_pack(args: argparse.Namespace) -> int:
    if args.rule_pack == "spec":
        payload = list_spec_rules()
    elif args.rule_pack == "compat":
        payload = list_compat_targets()
    else:
        raise CliUsageError(f"unsupported listable rule pack: {args.rule_pack}")
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified SK operator validation")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    validate = subparsers.add_parser(
        "validate-operator", help="Validate operator contracts and rule packs"
    )
    validate.add_argument(
        "--asset", required=True, help="Operator source tree to validate"
    )
    validate.add_argument(
        "--output-dir", required=True, help="Where to write validation artifacts"
    )
    validate.add_argument("--layout", default="", help="operator-asset-layout.json")
    validate.add_argument(
        "--build-context", default="", help="operator-build-context.json"
    )
    validate.add_argument(
        "--verify-context", default="", help="operator-verify-context.json"
    )
    validate.add_argument("--target-chip", default="", help="Target chip id")
    validate.add_argument("--target-cann", default="", help=argparse.SUPPRESS)
    validate.add_argument(
        "--rule-pack", choices=["all", "contract", "spec", "compat"], default="all"
    )
    validate.add_argument("--stage", default="post-adapt")
    validate.add_argument("--iteration-index", default="0")
    validate.set_defaults(func=cmd_validate_operator)

    list_rule_pack = subparsers.add_parser(
        "list-rule-pack", help="List bundled validation rule-pack metadata"
    )
    list_rule_pack.add_argument(
        "--rule-pack", choices=["spec", "compat"], required=True
    )
    list_rule_pack.set_defaults(func=cmd_list_rule_pack)

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
