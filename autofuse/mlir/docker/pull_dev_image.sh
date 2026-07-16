#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash mlir/docker/pull_dev_image.sh <remote-image-ref>

Environment:
  AF_MLIR_REMOTE_IMAGE       Remote image ref, used when no positional ref is provided
  AF_MLIR_DEV_IMAGE          Optional local image tag after pull. Defaults to the remote ref
  AF_MLIR_REGISTRY           Optional registry host for docker login. Defaults to inferred host
  AF_MLIR_REGISTRY_USER      Optional registry login user
  AF_MLIR_REGISTRY_TOKEN     Optional registry login token/password, passed via --password-stdin
  AF_MLIR_PULL_LOGOUT        Set to 1 to docker logout after pulling, default 0

  SWR_IMAGE, SWR_REGISTRY, SWR_USER, and SWR_TOKEN are also accepted for
  compatibility with the Ascend-MLIR real-NPU image workflow.

Examples:
  export AF_MLIR_REMOTE_IMAGE=swr.cn-east-2.myhuaweicloud.com/<namespace>/autofuse-mlir-dev:llvm21-aarch64
  bash mlir/docker/pull_dev_image.sh

  AF_MLIR_DEV_IMAGE=autofuse-mlir-dev:llvm21-aarch64 \
    bash mlir/docker/pull_dev_image.sh "${AF_MLIR_REMOTE_IMAGE}"
EOF
}

infer_registry() {
    local image_ref="$1"
    local first_part="${image_ref%%/*}"
    if [[ "${image_ref}" != */* ]]; then
        echo "docker.io"
    elif [[ "${first_part}" == *.* || "${first_part}" == *:* || "${first_part}" == "localhost" ]]; then
        echo "${first_part}"
    else
        echo "docker.io"
    fi
}

remote_image="${1:-${AF_MLIR_REMOTE_IMAGE:-${SWR_IMAGE:-}}}"
local_image="${AF_MLIR_DEV_IMAGE:-${remote_image}}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ -z "${remote_image}" ]]; then
    usage >&2
    exit 2
fi

if [[ -z "${local_image}" ]]; then
    echo "ERROR: local image tag is empty" >&2
    exit 2
fi

if [[ "${remote_image}" != */* && "${remote_image}" == autofuse-mlir* ]]; then
    if docker image inspect "${remote_image}" >/dev/null 2>&1; then
        echo "ERROR: ${remote_image} is a local image tag, not a remote image ref." >&2
        echo "It already exists locally. Skip pull and run:" >&2
        echo "  export AF_MLIR_DEV_IMAGE=${remote_image}" >&2
    else
        echo "ERROR: ${remote_image} is not a full remote image ref." >&2
        echo "Use a registry-qualified image, for example:" >&2
        echo "  swr.cn-east-2.myhuaweicloud.com/<namespace>/autofuse-mlir-dev:llvm21-aarch64" >&2
    fi
    exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is required to pull the image" >&2
    exit 1
fi

registry="${AF_MLIR_REGISTRY:-${SWR_REGISTRY:-$(infer_registry "${remote_image}")}}"
registry_user="${AF_MLIR_REGISTRY_USER:-${SWR_USER:-}}"
registry_token="${AF_MLIR_REGISTRY_TOKEN:-${SWR_TOKEN:-}}"

if [[ -n "${registry_user}" || -n "${registry_token}" ]]; then
    if [[ -z "${registry_user}" || -z "${registry_token}" ]]; then
        echo "ERROR: AF_MLIR_REGISTRY_USER and AF_MLIR_REGISTRY_TOKEN must be set together" >&2
        exit 2
    fi
    printf '%s' "${registry_token}" | docker login -u "${registry_user}" --password-stdin "${registry}"
fi

docker pull "${remote_image}"

if [[ "${local_image}" != "${remote_image}" ]]; then
    docker tag "${remote_image}" "${local_image}"
fi

docker image inspect "${local_image}" \
    --format 'arch={{.Architecture}} os={{.Os}} id={{.Id}} size={{.Size}}'

if [[ "${AF_MLIR_PULL_LOGOUT:-0}" == "1" && -n "${registry_user}" ]]; then
    docker logout "${registry}"
fi

echo "Pulled image: ${remote_image}"
echo "Use image: ${local_image}"
echo "Set AF_MLIR_DEV_IMAGE=${local_image} before entering the container."
