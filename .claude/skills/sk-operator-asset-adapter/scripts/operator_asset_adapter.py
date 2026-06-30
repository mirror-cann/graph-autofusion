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
from pathlib import Path


from operator_asset_contracts import (
    ContractError,
    build_adapter_report,
    build_default_build_context,
    build_default_verify_context,
    read_json,
    validate_adapter_report_contract,
    validate_asset_layout_contract,
    validate_build_context_contract,
    validate_verify_context_contract,
    write_json,
)
from operator_asset_layout import build_inventory, build_layout_from_inventory


class CliUsageError(Exception):
    pass


def cmd_adapt_asset(args: argparse.Namespace) -> int:
    asset = Path(args.asset).resolve()
    if not asset.exists():
        raise CliUsageError(f"asset path not found: {asset}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_inventory(asset)
    layout = build_layout_from_inventory(inventory)
    build_context = build_default_build_context(
        layout,
        inventory,
        target_chips=args.target_chip,
        target_arches=args.target_arch,
    )
    verify_context = build_default_verify_context(layout)
    report = build_adapter_report(
        asset_root=asset,
        inventory=inventory,
        layout=layout,
        build_context=build_context,
        verify_context=verify_context,
    )

    write_json(output_dir / "operator-asset-inventory.json", inventory)
    write_json(output_dir / "operator-asset-layout.json", layout)
    write_json(output_dir / "operator-build-context.json", build_context)
    write_json(output_dir / "operator-verify-context.json", verify_context)
    write_json(output_dir / "adapter-report.json", report)
    (output_dir / "adapter-report.md").write_text(
        _render_report(report), encoding="utf-8"
    )

    return 0 if report.get("readiness", {}).get("can_adapt") else 2


def _render_report(report: dict) -> str:
    readiness = report.get("readiness", {})
    lines = [
        "# Operator Asset Adapter Report",
        "",
        f"Status: **{report.get('status')}**",
        "",
        "## Readiness",
        "",
        f"- layout: {readiness.get('layout')}",
        f"- build: {readiness.get('build')}",
        f"- verify: {readiness.get('verify')}",
        f"- can_adapt: {readiness.get('can_adapt')}",
        f"- can_build: {readiness.get('can_build')}",
        f"- can_claim_equivalence: {readiness.get('can_claim_equivalence')}",
        "",
        "## Human Questions",
        "",
    ]
    questions = report.get("human_questions", [])
    if not questions:
        lines.append("None.")
    for question in questions:
        lines.append(f"- `{question.get('id')}`: {question.get('message')}")
    return "\n".join(lines) + "\n"


def cmd_validate_contracts(args: argparse.Namespace) -> int:
    layout_path = Path(args.layout).resolve()
    build_path = Path(args.build_context).resolve()
    verify_path = Path(args.verify_context).resolve()
    report_path = Path(args.adapter_report).resolve() if args.adapter_report else None

    validate_asset_layout_contract(read_json(layout_path), layout_path.parent)
    validate_build_context_contract(read_json(build_path), build_path.parent)
    validate_verify_context_contract(read_json(verify_path), verify_path.parent)
    if report_path:
        validate_adapter_report_contract(read_json(report_path), report_path.parent)
    print(json.dumps({"status": "valid"}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SK operator asset adapter")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    adapt = subparsers.add_parser(
        "adapt-asset", help="Generate stable SK operator contracts from an asset"
    )
    adapt.add_argument("asset", help="User operator asset directory or source file")
    adapt.add_argument(
        "--output-dir", required=True, help="Where to write adapter contracts"
    )
    adapt.add_argument(
        "--target-chip", default="", help="Comma/semicolon separated target chips"
    )
    adapt.add_argument(
        "--target-arch", default="", help="Comma/semicolon separated target NPU arches"
    )
    adapt.set_defaults(func=cmd_adapt_asset)

    validate = subparsers.add_parser(
        "validate-contracts", help="Validate generated adapter contracts"
    )
    validate.add_argument("--layout", required=True, help="operator-asset-layout.json")
    validate.add_argument(
        "--build-context", required=True, help="operator-build-context.json"
    )
    validate.add_argument(
        "--verify-context", required=True, help="operator-verify-context.json"
    )
    validate.add_argument("--adapter-report", default="", help="adapter-report.json")
    validate.set_defaults(func=cmd_validate_contracts)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(args.func(args))
    except (CliUsageError, ContractError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
