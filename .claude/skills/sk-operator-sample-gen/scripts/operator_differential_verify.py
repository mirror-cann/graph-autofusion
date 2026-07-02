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
from pathlib import Path
from typing import Any


def build_differential_verdict(
    output_dir: Path,
    *,
    entry_name: str,
    baseline_manifest: Path | None,
    runtime_contract: str | None = None,
    sk_verdict: dict[str, Any] | None = None,
    wheel_verdict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_status = "unavailable"
    baseline_correctness = "not-available"
    if baseline_manifest is not None and baseline_manifest.is_file():
        baseline_payload = json.loads(baseline_manifest.read_text(encoding="utf-8"))
        baseline_status = str(baseline_payload.get("status", "unknown"))
        baseline_correctness = str(baseline_payload.get("correctness", "not-available"))
    sk_status = str((sk_verdict or {}).get("status", "unavailable"))
    wheel_status = (
        str((wheel_verdict or {}).get("status", "unavailable")) if wheel_verdict is not None else "not-requested"
    )
    statuses = {baseline_status, sk_status}
    if wheel_verdict is not None:
        statuses.add(wheel_status)
    if "failed" in statuses:
        status = "failed"
    elif baseline_status == "structural-passed" or baseline_correctness == "not-executed":
        status = "skipped-structural-baseline"
    elif baseline_status in {"unavailable", "skipped-real-backend-not-enabled"}:
        status = "skipped-baseline-unavailable"
    elif "mock-passed" in statuses:
        status = "mock-passed"
    elif sk_status.startswith("skipped"):
        status = sk_status
    elif wheel_status.startswith("skipped"):
        status = wheel_status
    elif statuses <= {"passed"}:
        status = "passed"
    else:
        status = "mixed"
    verdict = {
        "schema_version": 1,
        "status": status,
        "entry_name": entry_name,
        "runtime_contract": runtime_contract or "",
        "baseline": {
            "build_manifest": str(baseline_manifest) if baseline_manifest is not None else "",
            "build_status": baseline_status,
            "correctness": baseline_correctness,
        },
        "sk": sk_verdict or {"status": "unavailable"},
        "wheel": wheel_verdict or {"status": "not-requested"},
        "comparisons": [],
    }
    (output_dir / "operator-differential-verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return verdict
