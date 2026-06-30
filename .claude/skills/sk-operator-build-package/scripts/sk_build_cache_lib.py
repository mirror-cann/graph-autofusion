# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Shared build cache helpers for standalone executable and wheel artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_paths(paths: Iterable[Path]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for path in sorted({Path(item) for item in paths}, key=lambda item: str(item)):
        if not path.exists() or not path.is_file():
            continue
        items.append({"path": path.name, "sha256": _hash_file(path)})
    return items


def command_version(command: str, *, env: dict[str, str] | None = None) -> str:
    try:
        completed = subprocess.run(
            [command, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=10,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    first_line = (completed.stdout or "").splitlines()
    return first_line[0] if first_line else f"returncode={completed.returncode}"


def cache_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def lookup_cache(cache_dir: Path, namespace: str, key: str) -> Path | None:
    candidate = cache_dir / namespace / key
    if not candidate.exists() or not candidate.is_dir():
        return None
    return candidate


def copy_cached_outputs(cached_dir: Path, dest_dir: Path) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(cached_dir, dest_dir, symlinks=True)


def store_cache(cache_dir: Path, namespace: str, key: str, source_dir: Path) -> Path:
    target = cache_dir / namespace / key
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(source_dir, tmp, symlinks=True)
    if target.exists():
        shutil.rmtree(target)
    tmp.replace(target)
    return target
