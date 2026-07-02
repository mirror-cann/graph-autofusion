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


def build_baseline(
    asset: Path,
    output_dir: Path,
    *,
    backend: str,
    entry_name: str,
    structural: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "outputs" / "baseline"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    status = "structural-passed" if structural else "skipped-real-backend-not-enabled"
    so_path = artifacts_dir / f"{entry_name}.so"
    runner_path = artifacts_dir / f"run_{entry_name}"
    log_path = output_dir / "baseline-build.log"
    if structural:
        so_path.write_text("STRUCTURAL BASELINE OUTPUT\n", encoding="utf-8")
        runner_path.write_text("#!/bin/sh\nprintf '%s\\n' structural-baseline\n", encoding="utf-8")
        runner_path.chmod(0o755)
        log_path.write_text(
            "structural baseline build; numerical correctness not executed\n",
            encoding="utf-8",
        )
    else:
        log_path.write_text("real baseline backend not enabled in this environment\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": status,
        "backend": backend,
        "entry_name": entry_name,
        "asset": str(asset),
        "structural": structural,
        "artifacts": {
            "shared_objects": [str(so_path)] if so_path.exists() else [],
            "runner": str(runner_path) if runner_path.exists() else "",
            "metadata": [],
        },
        "commands": [],
        "log_path": str(log_path),
        "correctness": "not-executed" if structural else "not-available",
    }
    (output_dir / "operator-baseline-build.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
