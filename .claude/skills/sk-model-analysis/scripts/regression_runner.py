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

"""Run sk-model-analysis regressions against an existing sample without polluting it."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from sk_library_extractor import find_model_dir


def _emit(message: object = "", *, file=None, end: str = "\n") -> None:
    stream = sys.stdout if file is None else file
    stream.write(f"{message}{end}")


CASES = [
    ("analyze_result_root", "analyze", "."),
    ("analyze_sk_meta", "analyze", "sk_meta"),
    ("analyze_model_dir", "analyze", "__AUTO_MODEL_DIR__"),
    ("hang_result_root", "diagnose-hang-crash", "."),
    ("performance_result_root", "diagnose-performance", "."),
    ("trace_result_root", "trace-nodes", "."),
]


def _assert_outputs(case_name: str, temp_root: Path) -> None:
    reports_dir = temp_root / "reports"
    if case_name.startswith("analyze_"):
        expected = ["run-portal.html"]
    elif case_name.startswith("hang_"):
        expected = ["hang-crash-report.html"]
    elif case_name.startswith("performance_"):
        expected = ["performance-report.html"]
    elif case_name.startswith("update_"):
        expected = ["scope-library.json", "graph-library.json"]
    elif case_name.startswith("trace_"):
        expected = ["node-trace.json", "node-trace_meta.json"]
    else:
        expected = ["run-portal.html"]

    roots = [reports_dir, reports_dir / "data", reports_dir / "views"]
    missing = []
    for name in expected:
        if not any((root / name).exists() for root in roots):
            missing.append(name)
    if missing:
        raise RuntimeError(
            f"{case_name}: missing expected outputs: {', '.join(missing)}"
        )


def run_case(
    sample_root: Path, case_name: str, command_name: str, relative_input: str
) -> None:
    script_path = Path(__file__).resolve().with_name("model_analysis.py")
    with tempfile.TemporaryDirectory(
        prefix="sk-model-analysis-regression-"
    ) as temp_dir:
        temp_root = Path(temp_dir) / sample_root.name
        shutil.copytree(sample_root, temp_root)
        if relative_input == "__AUTO_MODEL_DIR__":
            input_path = find_model_dir(temp_root)
        else:
            input_path = temp_root / relative_input
        command = [sys.executable, str(script_path), command_name, str(input_path)]
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{case_name}: command failed\n{completed.stdout}")
        _assert_outputs(case_name, temp_root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run non-persistent sk-model-analysis regression cases"
    )
    parser.add_argument(
        "sample_root",
        help="Existing result root sample to copy into a temporary workspace",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Optional case name filter; may be specified multiple times",
    )
    args = parser.parse_args()

    sample_root = Path(args.sample_root).resolve()
    if not sample_root.is_dir():
        raise SystemExit(f"sample root not found: {sample_root}")

    selected = CASES
    if args.case:
        case_names = set(args.case)
        selected = [case for case in CASES if case[0] in case_names]
        if not selected:
            raise SystemExit("no matching regression case")

    for case_name, command_name, relative_input in selected:
        run_case(sample_root, case_name, command_name, relative_input)
        _emit(f"[ok] {case_name}")


if __name__ == "__main__":
    main()
