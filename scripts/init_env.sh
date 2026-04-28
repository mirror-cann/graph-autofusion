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

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CANN_VERSION="9.0.0"
CHIP_TYPE="910b"
INSTALL_OPS=true
INSTALL_PATH="${INSTALL_PATH:-/usr/local/Ascend}"

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_command() { command -v "$1" &>/dev/null; }

get_arch() {
    local arch=$(uname -m)
    case $arch in
        x86_64) echo "x86_64" ;;
        aarch64) echo "aarch64" ;;
        *) log_error "Unsupported architecture: $arch"; return 1 ;;
    esac
}

get_cann_url() {
    local base_url="https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master"
    local latest_dir=$(wget -q --no-check-certificate -O - "$base_url/" | \
                        grep -Eo '>[0-9]{17,}/<' | \
                        tr -d '></' | \
                        sort -r | \
                        head -n 1)
    if [[ -z "$latest_dir" ]]; then
        log_error "Get cann url failed."
    fi
    log_info "CANN timestamp: ${latest_dir}" >&2
    echo "${base_url}/${latest_dir}"
}

install_cann() {
    log_info "Start CANN installation..."

    local arch=$(get_arch)
    local cann_url=$(get_cann_url)
    local download_dir="/tmp/cann_install"
    mkdir -p "$download_dir"
    cd "$download_dir"

    local toolkit_pkg="Ascend-cann-toolkit_${CANN_VERSION}_linux-${arch}.run"
    local ops_pkg="Ascend-cann-${CHIP_TYPE}-ops_${CANN_VERSION}_linux-${arch}.run"
    local toolkit_url="${cann_url}/${toolkit_pkg}"
    local ops_url="${cann_url}/${ops_pkg}"

    if [ ! -f "$toolkit_pkg" ]; then
        log_info "Downloading CANN toolkit (${arch})..."
        log_info "URL: $toolkit_url"
        wget -q --show-progress --no-check-certificate -O "$toolkit_pkg" "$toolkit_url" || {
            log_error "Failed to download toolkit"
            return 1
        }
    fi

    if [ ! -f "$ops_pkg" ] && [ "$INSTALL_OPS" = true ]; then
        log_info "Downloading CANN ops (${CHIP_TYPE}, ${arch})..."
        log_info "URL: $ops_url"
        wget -q --show-progress --no-check-certificate -O "$ops_pkg" "$ops_url" || {
            log_error "Failed to download ops"
            return 1
        }
    fi

    log_info "Installing CANN toolkit..."
    chmod +x "$toolkit_pkg"
    ./$toolkit_pkg --full --quiet --install-path="$INSTALL_PATH" || {
        log_error "Failed to install toolkit"
        return 1
    }

    if [ "$INSTALL_OPS" = true ]; then
        log_info "Installing CANN ops..."
        chmod +x "$ops_pkg"
        ./$ops_pkg --install --quiet --install-path="$INSTALL_PATH" || {
            log_warn "Ops installation may have issues (this can be normal on non-NPU systems)"
        }
    fi

    log_info "Cleaning up..."
    rm -f "$toolkit_pkg" "$ops_pkg"
    cd - > /dev/null

    log_info "CANN installation completed"
}

install_system_deps() {
    log_info "Checking system dependencies..."

    if check_command apt-get; then
        apt-get update 2>/dev/null
        local pkgs=""
        check_command curl || pkgs="$pkgs curl"
        check_command wget || pkgs="$pkgs wget"
        check_command git || pkgs="$pkgs git"
        check_command cmake || pkgs="$pkgs cmake"
        check_command make || pkgs="$pkgs make"
        check_command g++ || pkgs="$pkgs g++"
        check_command ccache || pkgs="$pkgs ccache"
        check_command autoconf || pkgs="$pkgs autoconf"
        check_command automake || pkgs="$pkgs automake"
        check_command libtoolize || pkgs="$pkgs libtool"
        check_command gperf || pkgs="$pkgs gperf"
        check_command sshd || check_command ssh || pkgs="$pkgs openssh-server"
        # Python dev packages needed for compiling Python extensions (coverage, etc.)
        dpkg -l | grep -q "^ii  python3-dev " 2>/dev/null || pkgs="$pkgs python3-dev"

        if [ -n "$pkgs" ]; then
            log_info "Installing:$pkgs"
            apt-get install -y $pkgs 2>/dev/null || log_warn "Some packages may have failed"
        fi
    elif check_command yum; then
        local pkgs=""
        check_command curl || pkgs="$pkgs curl"
        check_command wget || pkgs="$pkgs wget"
        check_command git || pkgs="$pkgs git"
        check_command cmake || pkgs="$pkgs cmake"
        check_command make || pkgs="$pkgs make"
        check_command g++ || pkgs="$pkgs gcc-c++"
        check_command autoconf || pkgs="$pkgs autoconf"
        check_command automake || pkgs="$pkgs automake"
        check_command libtoolize || pkgs="$pkgs libtool"
        check_command gperf || pkgs="$pkgs gperf"
        check_command sshd || check_command ssh || pkgs="$pkgs openssh-server"
        # Python dev packages needed for compiling Python extensions (coverage, etc.)
        rpm -q python3-devel &>/dev/null || pkgs="$pkgs python3-devel"

        if [ -n "$pkgs" ]; then
            log_info "Installing:$pkgs"
            yum install -y $pkgs 2>/dev/null || log_warn "Some packages may have failed"
        fi
    fi

    log_info "cmake: $(cmake --version 2>&1 | head -1)"
    log_info "g++: $(g++ --version 2>&1 | head -1)"
    check_command ccache && log_info "ccache: available" || log_warn "ccache: not installed"
    check_command autoconf && log_info "autoconf: $(autoconf --version 2>&1 | head -1)" || log_warn "autoconf: not installed"
    check_command automake && log_info "automake: $(automake --version 2>&1 | head -1)" || log_warn "automake: not installed"
    check_command libtoolize && log_info "libtool: $(libtoolize --version 2>&1 | head -1)" || log_warn "libtool: not installed"
    check_command gperf && log_info "gperf: $(gperf --version 2>&1 | head -1)" || log_warn "gperf: not installed"
    check_command sshd && log_info "sshd: available" || log_warn "sshd: not installed"

    # Fix libz.so.1 issue
    log_info "Checking and fixing libz.so.1."
    if check_command apt-get; then
        apt-get install --reinstall zlib1g -y >/dev/null 2>&1 || log_warn "Failed to reinstall zlib1g"
    elif check_command yum; then
        yum reinstall zlib -y >/dev/null 2>&1 || log_warn "Failed to reinstall zlib"
    fi

    # Start SSH service if available
    log_info "Checking SSH service status..."
    if check_command sshd; then
        # Check if SSH is already running
        if ! pgrep -x "sshd" > /dev/null; then
            log_info "Starting SSH service..."
            if check_command service; then
                service ssh start 2>/dev/null || service sshd start 2>/dev/null || log_warn "Failed to start SSH service"
            elif check_command systemctl; then
                systemctl start ssh 2>/dev/null || systemctl start sshd 2>/dev/null || log_warn "Failed to start SSH service"
            elif [ -f /etc/init.d/ssh ]; then
                /etc/init.d/ssh start 2>/dev/null || log_warn "Failed to start SSH service"
            elif [ -f /etc/init.d/sshd ]; then
                /etc/init.d/sshd start 2>/dev/null || log_warn "Failed to start SSH service"
            else
                # Direct start sshd
                sshd 2>/dev/null || log_warn "Failed to start SSH service"
            fi
        else
            log_info "SSH service is already running"
        fi
    fi
}

install_python_deps() {
    log_info "Checking Python dependencies..."

    local python="python3"
    check_command python3 || python="python"

    log_info "Python: $($python --version 2>&1)"

    # Configure pip to use faster mirror (Tsinghua University mirror)
    local pip_index_url="https://pypi.tuna.tsinghua.edu.cn/simple"
    local pip_trusted_host="pypi.tuna.tsinghua.edu.cn"
    log_info "Using pip mirror: $pip_index_url"

    local required=("pytest>=9.0.1" "coverage>=7.10.0" "pytest-cov>=7.0.0" "pybind11>=2.13.0")
    local to_install=()

    for pkg in "${required[@]}"; do
        local name=$(echo "$pkg" | sed 's/>=.*//')
        $python -c "import $name" 2>/dev/null || to_install+=("$pkg")
    done

    if [ ${#to_install[@]} -ne 0 ]; then
        log_info "Installing Python packages: ${to_install[*]}"
        $python -m pip install --upgrade pip --index-url "$pip_index_url" --trusted-host "$pip_trusted_host" 2>/dev/null || true

        local failed=()
        for pkg in "${to_install[@]}"; do
            echo "  Installing $pkg..."
            if $python -m pip install --default-timeout=180 "$pkg" --index-url "$pip_index_url" --trusted-host "$pip_trusted_host"; then
                echo "  ✓ $pkg installed"
            else
                echo "  ✗ $pkg failed"
                failed+=("$pkg")
            fi
        done

        if [ ${#failed[@]} -ne 0 ]; then
            log_warn "Failed to install: ${failed[*]}, try again or install manually."
        else
            log_info "All Python packages installed successfully"
        fi
    else
        log_info "All Python dependencies satisfied"
    fi

    check_command pytest && log_info "pytest: $(pytest --version 2>&1 | head -1)" || true
    check_command coverage && log_info "coverage: $(coverage --version 2>&1 | head -1)" || true

    # Install super_kernel Python dependencies
    local super_kernel_reqs="../super_kernel/requirements-dev.txt"
    if [ -f "$super_kernel_reqs" ]; then
        log_info "Installing super_kernel Python dependencies from requirements-dev.txt..."
        # Filter out harmless dependency warnings but preserve actual errors
        local install_output
        install_output=$($python -m pip install -r "$super_kernel_reqs" --index-url "$pip_index_url" --trusted-host "$pip_trusted_host" 2>&1)
        local install_status=$?

        # Show output, filtering only the specific harmless warnings
        echo "$install_output" | grep -v "dependency resolver does not currently take into account" || true

        # Check installation result
        if [ $install_status -eq 0 ] && $python -c "import numpy, scipy, pytest" 2>/dev/null; then
            log_info "super_kernel dependencies installed successfully"
        else
            log_warn "Failed to install super_kernel dependencies, try manually: $python -m pip install -r $super_kernel_reqs"
        fi
    else
        log_warn "super_kernel requirements file not found: $super_kernel_reqs"
    fi
}

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "development environment setup script"
    echo ""
    echo "Options:"
    echo "  --chip-type TYPE         Chip type: 910b, 950, A3 (default: 910b)"
    echo "  --install-path PATH      Installation path (default: ${INSTALL_PATH})"
    echo "  --skip-cann              Skip CANN installation (default: install CANN)"
    echo "  --skip-ops               Skip CANN ops installation (default: install ops)"
    echo "  --help                   Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                       # Install with defaults"
    echo "  $0 --skip-cann           # Skip CANN, install only deps"
    echo "  $0 --skip-ops            # Skip ops, install only toolkit"
    echo "  $0 --chip-type 950       # Use 950 ops package"
    echo ""
}

main() {
    local INSTALL_CANN=true

    while [[ $# -gt 0 ]]; do
        case $1 in
            --chip-type)
                CHIP_TYPE="$2"
                shift 2
                ;;
            --install-path)
                INSTALL_PATH="$2"
                shift 2
                ;;
            --skip-cann)
                INSTALL_CANN=false
                shift
                ;;
            --skip-ops)
                INSTALL_OPS=false
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_warn "Unknown option: $1"
                shift
                ;;
        esac
    done

    echo ""
    echo "=================================="
    echo "  Development Environment Setup"
    echo "=================================="
    echo ""
    echo "Configuration:"
    echo "  CANN Version:  ${CANN_VERSION}"
    echo "  Chip Type:     ${CHIP_TYPE}"
    echo "  Install cann:  ${INSTALL_CANN}"
    echo "  Install ops:   ${INSTALL_OPS}"
    echo "  Install Path:  ${INSTALL_PATH}"
    echo ""

    local work_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$work_dir"
    log_info "Working directory: $work_dir"

    install_system_deps

    if [ "$INSTALL_CANN" = false ]; then
        log_info "Skipping CANN installation (--skip-cann)"
    else
        install_cann
    fi

    install_python_deps

    echo ""
    log_info "=========================================="
    log_info "  Development environment ready!"
    log_info "=========================================="
    echo ""
    if [ "$INSTALL_CANN" = true ]; then        
        echo "Next step:"
        echo "  source ${INSTALL_PATH}/cann/set_env.sh     # Set the environment variables"
        echo ""
    fi
}

main "$@"
