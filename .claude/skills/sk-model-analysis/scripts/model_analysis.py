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

"""Top-level CLI for the sk-model-analysis built-in skill."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


SCRIPT_DIR = Path(__file__).resolve().parent
DIAGNOSE_SCRIPT = SCRIPT_DIR / "diagnose_run.py"


class DiagnoseRunOptions(NamedTuple):
    mode: str
    input_path: str
    with_ai: bool
    jobs: int | None
    no_parallel: bool
    no_cache: bool
    profile: bool


def _run(command: list[str]) -> int:
    completed = subprocess.run(command, check=False)
    return completed.returncode


def _run_diagnose(options: DiagnoseRunOptions) -> int:
    command = [sys.executable, str(DIAGNOSE_SCRIPT), "--mode", options.mode]
    if options.with_ai:
        command.append("--with-ai")
    if options.jobs is not None:
        command.extend(["--jobs", str(options.jobs)])
    if options.no_parallel:
        command.append("--no-parallel")
    if options.no_cache:
        command.append("--no-cache")
    if options.profile:
        command.append("--profile")
    command.append(options.input_path)
    return _run(command)


def cmd_analyze(args: argparse.Namespace) -> int:
    return _run_diagnose(
        DiagnoseRunOptions(
            "full",
            args.input,
            args.with_ai,
            args.jobs,
            args.no_parallel,
            args.no_cache,
            args.profile,
        )
    )


def cmd_diagnose_hang_crash(args: argparse.Namespace) -> int:
    return _run_diagnose(
        DiagnoseRunOptions(
            "hang",
            args.input,
            args.with_ai,
            args.jobs,
            args.no_parallel,
            args.no_cache,
            args.profile,
        )
    )


def cmd_diagnose_performance(args: argparse.Namespace) -> int:
    return _run_diagnose(
        DiagnoseRunOptions(
            "performance",
            args.input,
            args.with_ai,
            args.jobs,
            args.no_parallel,
            args.no_cache,
            args.profile,
        )
    )


def cmd_trace_nodes(args: argparse.Namespace) -> int:
    return _run_diagnose(
        DiagnoseRunOptions(
            "trace",
            args.input,
            args.with_ai,
            args.jobs,
            args.no_parallel,
            args.no_cache,
            args.profile,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Top-level CLI for sk-model-analysis")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    analyze = subparsers.add_parser(
        "analyze", help="Analyze one user-provided result directory"
    )
    analyze.add_argument(
        "input", help="Result root, model asset directory, or model directory"
    )
    analyze.add_argument(
        "--with-ai", action="store_true", help="Request optional AI-layer hints"
    )
    analyze.add_argument(
        "--jobs", type=int, help="Override worker count for multi-model execution"
    )
    analyze.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable multi-process execution even in multi-model mode",
    )
    analyze.add_argument(
        "--no-cache", action="store_true", help="Disable parser cache reads"
    )
    analyze.add_argument(
        "--profile", action="store_true", help="Write diagnose timing profile"
    )
    analyze.set_defaults(func=cmd_analyze)

    hang = subparsers.add_parser(
        "diagnose-hang-crash", help="Generate hang and coredump oriented reports"
    )
    hang.add_argument(
        "input", help="Result root, model asset directory, or model directory"
    )
    hang.add_argument(
        "--with-ai", action="store_true", help="Request optional AI-layer hints"
    )
    hang.add_argument(
        "--jobs", type=int, help="Override worker count for multi-model execution"
    )
    hang.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable multi-process execution even in multi-model mode",
    )
    hang.add_argument(
        "--no-cache", action="store_true", help="Disable parser cache reads"
    )
    hang.add_argument(
        "--profile", action="store_true", help="Write diagnose timing profile"
    )
    hang.set_defaults(func=cmd_diagnose_hang_crash)

    perf = subparsers.add_parser(
        "diagnose-performance", help="Generate performance oriented reports"
    )
    perf.add_argument(
        "input", help="Result root, model asset directory, or model directory"
    )
    perf.add_argument(
        "--with-ai", action="store_true", help="Request optional AI-layer hints"
    )
    perf.add_argument(
        "--jobs", type=int, help="Override worker count for multi-model execution"
    )
    perf.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable multi-process execution even in multi-model mode",
    )
    perf.add_argument(
        "--no-cache", action="store_true", help="Disable parser cache reads"
    )
    perf.add_argument(
        "--profile", action="store_true", help="Write diagnose timing profile"
    )
    perf.set_defaults(func=cmd_diagnose_performance)

    trace = subparsers.add_parser(
        "trace-nodes", help="Generate node tracing artifacts only"
    )
    trace.add_argument(
        "input", help="Result root, model asset directory, or model directory"
    )
    trace.add_argument(
        "--with-ai", action="store_true", help="Request optional AI-layer hints"
    )
    trace.add_argument(
        "--jobs", type=int, help="Override worker count for multi-model execution"
    )
    trace.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable multi-process execution even in multi-model mode",
    )
    trace.add_argument(
        "--no-cache", action="store_true", help="Disable parser cache reads"
    )
    trace.add_argument(
        "--profile", action="store_true", help="Write diagnose timing profile"
    )
    trace.set_defaults(func=cmd_trace_nodes)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
