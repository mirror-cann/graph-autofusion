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
  bash mlir/docker/build_cann_image.sh <cann-toolkit-runfile-url>

Environment:
  AF_MLIR_CANN_BASE_IMAGE     Base dev+LLVM image, default AF_MLIR_DEV_IMAGE or autofuse-mlir-dev:<arch>
  AF_MLIR_CANN_IMAGE          Output image, default autofuse-mlir-dev-cann:<arch>
  AF_CANN_PACKAGE_URL         CANN Toolkit .run URL, used when no positional URL is provided
  AF_CANN_TOOLKIT_SHA256      Optional sha256 for the Toolkit .run
  AF_CANN_OPS_PACKAGE_URL     Optional CANN ops .run URL
  AF_CANN_OPS_SHA256         Optional sha256 for the ops .run
  AF_CANN_3RD_LIB_URL         Optional prebuilt third_party tar archive URL
  AF_CANN_3RD_LIB_SHA256      Optional sha256 for the third_party archive
  AF_CANN_3RD_LIB_PATH        Path in image, default /opt/autofuse/third_party
  AF_CANN_INSTALL_PATH        Install path in image, default /usr/local/Ascend
  AF_CANN_TOOLKIT_INSTALL_ARGS
                              Toolkit installer args, default "--install --quiet"
  AF_CANN_OPS_INSTALL_ARGS    Ops installer args, default "--install --quiet"
  AF_CANN_ARCH_DIR            Toolkit arch dir, default inferred from host arch
EOF
}

check_sha256() {
    local expected="$1"
    local file="$2"
    if command -v sha256sum >/dev/null 2>&1; then
        echo "${expected}  ${file}" | sha256sum -c -
    elif command -v shasum >/dev/null 2>&1; then
        echo "${expected}  ${file}" | shasum -a 256 -c -
    else
        echo "ERROR: neither sha256sum nor shasum is available for checksum verification" >&2
        exit 2
    fi
}

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
host_arch="$(uname -m)"
image="${AF_MLIR_CANN_IMAGE:-autofuse-mlir-dev-cann:${host_arch}}"
base="${AF_MLIR_CANN_BASE_IMAGE:-${AF_MLIR_DEV_IMAGE:-autofuse-mlir-dev:${host_arch}}}"
cann_url="${1:-${AF_CANN_PACKAGE_URL:-}}"
cann_ops_url="${AF_CANN_OPS_PACKAGE_URL:-}"
cann_3rd_lib_url="${AF_CANN_3RD_LIB_URL:-}"
cann_install_path="${AF_CANN_INSTALL_PATH:-/usr/local/Ascend}"
cann_toolkit_install_args="${AF_CANN_TOOLKIT_INSTALL_ARGS:---install --quiet}"
cann_ops_install_args="${AF_CANN_OPS_INSTALL_ARGS:---install --quiet}"
cann_3rd_lib_path="${AF_CANN_3RD_LIB_PATH:-/opt/autofuse/third_party}"

if [ -z "${cann_url}" ]; then
    usage >&2
    exit 2
fi

case "${host_arch}" in
    arm64 | aarch64)
        default_cann_arch_dir=aarch64-linux
        ;;
    x86_64 | amd64)
        default_cann_arch_dir=x86_64-linux
        ;;
    *)
        echo "ERROR: unsupported host arch for default CANN arch dir: ${host_arch}" >&2
        echo "Set AF_CANN_ARCH_DIR explicitly." >&2
        exit 2
        ;;
esac
cann_arch_dir="${AF_CANN_ARCH_DIR:-${default_cann_arch_dir}}"

url_path="${cann_url%%\?*}"
runfile_name="$(basename "${url_path}")"
if [[ "${runfile_name}" != *.run ]]; then
    echo "ERROR: CANN package URL must resolve to a .run file: ${cann_url}" >&2
    exit 2
fi

tmpdir="$(mktemp -d)"
cleanup() {
    rm -rf "${tmpdir}"
}
trap cleanup EXIT

download_artifact() {
    local source="$1"
    local output="$2"
    case "${source}" in
        http://* | https://* | file://*)
            curl -fL --retry 3 --retry-delay 2 -o "${output}" "${source}"
            ;;
        *)
            if [ ! -f "${source}" ]; then
                echo "ERROR: artifact is not a URL or local file: ${source}" >&2
                exit 2
            fi
            cp "${source}" "${output}"
            ;;
    esac
}

mkdir -p "${tmpdir}/cann"
download_artifact "${cann_url}" "${tmpdir}/cann/toolkit.run"

if [ -n "${AF_CANN_TOOLKIT_SHA256:-}" ]; then
    check_sha256 "${AF_CANN_TOOLKIT_SHA256}" "${tmpdir}/cann/toolkit.run"
fi

if [ -n "${cann_ops_url}" ]; then
    ops_url_path="${cann_ops_url%%\?*}"
    ops_runfile_name="$(basename "${ops_url_path}")"
    if [[ "${ops_runfile_name}" != *.run ]]; then
        echo "ERROR: CANN ops package URL must resolve to a .run file: ${cann_ops_url}" >&2
        exit 2
    fi
    download_artifact "${cann_ops_url}" "${tmpdir}/cann/ops.run"
    if [ -n "${AF_CANN_OPS_SHA256:-}" ]; then
        check_sha256 "${AF_CANN_OPS_SHA256}" "${tmpdir}/cann/ops.run"
    fi
fi

if [ -n "${cann_3rd_lib_url}" ]; then
    third_party_url_path="${cann_3rd_lib_url%%\?*}"
    third_party_archive_name="$(basename "${third_party_url_path}")"
    case "${third_party_archive_name}" in
        *.tar | *.tar.gz | *.tgz)
            ;;
        *)
            echo "ERROR: AF_CANN_3RD_LIB_URL must resolve to a tar archive: ${cann_3rd_lib_url}" >&2
            exit 2
            ;;
    esac
    case "${third_party_archive_name}" in
        *.tar.gz | *.tgz)
            third_party_target="${tmpdir}/cann/third_party.tar.gz"
            ;;
        *)
            third_party_target="${tmpdir}/cann/third_party.tar"
            ;;
    esac
    download_artifact "${cann_3rd_lib_url}" "${third_party_target}"
    if [ -n "${AF_CANN_3RD_LIB_SHA256:-}" ]; then
        check_sha256 "${AF_CANN_3RD_LIB_SHA256}" "${third_party_target}"
    fi
fi

case "${cann_url}" in
    http://* | https://* | file://*)
        cann_source_for_log="${cann_url}"
        ;;
    *)
        cann_source_for_log="${runfile_name}"
        ;;
esac
echo "Building ${image} from ${base} with Toolkit package ${cann_source_for_log}" >&2

DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}" docker build \
    --build-arg BASE_IMAGE="${base}" \
    --build-arg CANN_INSTALL_PATH="${cann_install_path}" \
    --build-arg CANN_TOOLKIT_INSTALL_ARGS="${cann_toolkit_install_args}" \
    --build-arg CANN_OPS_INSTALL_ARGS="${cann_ops_install_args}" \
    --build-arg CANN_ARCH_DIR="${cann_arch_dir}" \
    --build-arg CANN_3RD_LIB_PATH="${cann_3rd_lib_path}" \
    -t "${image}" \
    -f "${repo_root}/autofuse/mlir/docker/Dockerfile.cann" \
    "${tmpdir}"

echo "${image}"
