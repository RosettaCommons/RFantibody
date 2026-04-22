#!/usr/bin/env bash
#
# nvidia-docker-install.sh — set up Docker + NVIDIA GPU passthrough on Ubuntu.
#
# What this installs (default):
#   1. Docker Engine (docker-ce)        — the daemon that supports --gpus all
#   2. NVIDIA Container Toolkit         — bridge between Docker and the NVIDIA driver
#   3. /etc/docker/daemon.json wired up via `nvidia-ctk runtime configure`
#   4. Switches `docker context` to `default` so the CLI talks to the engine
#      (Docker Desktop, if present, ignores /etc/docker/daemon.json and won't
#      pass the GPU through.)
#
# Optional:
#   --with-desktop   Also install Docker Desktop (GUI). Only useful on a
#                    graphical workstation. Pulls a ~600 MB .deb. Not needed
#                    for headless / server / CI use of RFantibody.
#   --check          Verify an existing install without changing anything.
#
# Prereqs:
#   - Ubuntu (24.04 noble tested; 22.04 jammy should also work)
#   - x86_64
#   - NVIDIA driver installed and `nvidia-smi` works on the host
#
# Idempotent: safe to re-run.

set -euo pipefail

WITH_DESKTOP=0
CHECK_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --with-desktop) WITH_DESKTOP=1 ;;
        --check)        CHECK_ONLY=1 ;;
        -h|--help)
            sed -n '2,25p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { printf '\033[1;34m[+]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

DOCKER_KEYRING=/etc/apt/keyrings/docker.asc
DOCKER_LIST=/etc/apt/sources.list.d/docker.list
NVIDIA_KEYRING=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
NVIDIA_LIST=/etc/apt/sources.list.d/nvidia-container-toolkit.list
DESKTOP_DEB_URL=https://desktop.docker.com/linux/main/amd64/docker-desktop-amd64.deb
DESKTOP_DEB_PATH=/tmp/docker-desktop-amd64.deb

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

check_preconditions() {
    log "Checking preconditions..."

    command -v nvidia-smi >/dev/null 2>&1 \
        || die "nvidia-smi not found. Install the NVIDIA driver first."

    [[ -f /etc/os-release ]] || die "/etc/os-release missing — cannot detect distro."
    . /etc/os-release
    [[ "${ID:-}" == "ubuntu" || "${ID_LIKE:-}" == *ubuntu* || "${ID_LIKE:-}" == *debian* ]] \
        || die "This script targets Ubuntu/Debian. Detected: ${ID:-unknown}."

    UBUNTU_CODENAME="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"
    [[ -n "$UBUNTU_CODENAME" ]] || die "Cannot determine Ubuntu codename from /etc/os-release."

    [[ "$(uname -m)" == "x86_64" ]] || die "x86_64 only (detected $(uname -m))."

    if [[ "$WITH_DESKTOP" == 1 ]]; then
        [[ -e /dev/kvm ]] || die "Docker Desktop needs KVM virtualization (/dev/kvm missing)."
        command -v gnome-terminal >/dev/null 2>&1 \
            || warn "gnome-terminal missing — Docker Desktop expects it even on non-GNOME desktops."
    fi

    if [[ $EUID -ne 0 ]] && ! sudo -n true 2>/dev/null; then
        log "This script needs sudo. You may be prompted for your password."
    fi

    log "Preconditions OK."
}

# ---------------------------------------------------------------------------
# Shared: Docker apt repo (used by both docker-ce and docker-desktop)
# ---------------------------------------------------------------------------

setup_docker_apt_repo() {
    if [[ -f "$DOCKER_KEYRING" && -f "$DOCKER_LIST" ]] \
        && apt-cache policy docker-ce-cli 2>/dev/null | grep -q 'download.docker.com'; then
        log "Docker apt repo already configured — skipping."
        return
    fi

    log "Adding Docker's apt repository..."
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o "$DOCKER_KEYRING"
    sudo chmod a+r "$DOCKER_KEYRING"

    echo "deb [arch=$(dpkg --print-architecture) signed-by=$DOCKER_KEYRING] https://download.docker.com/linux/ubuntu $UBUNTU_CODENAME stable" \
        | sudo tee "$DOCKER_LIST" >/dev/null

    sudo apt-get update
    log "Docker apt repo added."
}

# ---------------------------------------------------------------------------
# 1. Docker Engine
# ---------------------------------------------------------------------------

install_docker_engine() {
    if dpkg -s docker-ce >/dev/null 2>&1; then
        log "Docker Engine (docker-ce) already installed — skipping."
    else
        log "Installing Docker Engine..."
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
            docker-buildx-plugin docker-compose-plugin
    fi

    log "Enabling and starting docker.service"
    sudo systemctl enable --now docker
}

# ---------------------------------------------------------------------------
# 2. Docker Desktop (optional)
# ---------------------------------------------------------------------------

install_docker_desktop() {
    if dpkg -l docker-desktop 2>/dev/null | grep -q '^ii'; then
        local v; v="$(dpkg-query -W -f='${Version}' docker-desktop 2>/dev/null || echo unknown)"
        log "Docker Desktop already installed (version ${v}) — skipping."
    else
        log "Downloading Docker Desktop .deb..."
        curl -fL --progress-bar -o "$DESKTOP_DEB_PATH" "$DESKTOP_DEB_URL"
        log "Installing Docker Desktop..."
        sudo apt-get install -y "$DESKTOP_DEB_PATH"
    fi

    log "Enabling docker-desktop user service..."
    systemctl --user enable docker-desktop.service >/dev/null 2>&1 \
        || warn "Could not enable docker-desktop user service (OK in non-graphical sessions)."
    loginctl show-user "$USER" 2>/dev/null | grep -q 'Linger=yes' \
        || sudo loginctl enable-linger "$USER" || true
}

# ---------------------------------------------------------------------------
# 3. docker group membership
# ---------------------------------------------------------------------------

add_user_to_docker_group() {
    if groups "$USER" | grep -qw docker; then
        NEEDS_RELOGIN=0
    else
        log "Adding $USER to the docker group (effective on next login)"
        sudo usermod -aG docker "$USER"
        NEEDS_RELOGIN=1
    fi
}

# ---------------------------------------------------------------------------
# 4. NVIDIA Container Toolkit
# ---------------------------------------------------------------------------

install_nvidia_container_toolkit() {
    if dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
        log "NVIDIA Container Toolkit already installed — skipping."
    else
        log "Adding NVIDIA Container Toolkit apt repo..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | sudo gpg --dearmor -o "$NVIDIA_KEYRING"

        curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
            | sed "s#deb https://#deb [signed-by=$NVIDIA_KEYRING] https://#g" \
            | sudo tee "$NVIDIA_LIST" >/dev/null

        log "Installing nvidia-container-toolkit..."
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
    fi

    log "Configuring Docker to use the NVIDIA runtime"
    sudo nvidia-ctk runtime configure --runtime=docker

    log "Restarting docker.service"
    sudo systemctl restart docker
}

# ---------------------------------------------------------------------------
# 5. Switch active Docker context to the engine
# ---------------------------------------------------------------------------

switch_to_engine_context() {
    local current
    current="$(docker context show 2>/dev/null || echo default)"
    if [[ "$current" != "default" ]]; then
        log "Switching docker context from '$current' to 'default' (Docker Engine)"
        docker context use default
    else
        log "docker context already 'default'"
    fi
}

# ---------------------------------------------------------------------------
# 6. Verify
# ---------------------------------------------------------------------------

verify_gpu_passthrough() {
    log "Verifying GPU passthrough"
    local cmd=(docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi)
    if [[ "${NEEDS_RELOGIN:-0}" == "1" ]]; then
        log "  (using sudo because docker group membership hasn't reloaded yet)"
        sudo "${cmd[@]}"
    else
        "${cmd[@]}"
    fi
}

# ---------------------------------------------------------------------------
# --check mode
# ---------------------------------------------------------------------------

run_check() {
    local failures=0

    log "docker CLI:              $(command -v docker >/dev/null 2>&1 && docker --version || echo NOT INSTALLED)"

    if dpkg -s docker-ce >/dev/null 2>&1; then
        log "docker-ce:               installed"
    else
        warn "docker-ce:               NOT installed"; failures=$((failures+1))
    fi

    if dpkg -l docker-desktop 2>/dev/null | grep -q '^ii'; then
        log "docker-desktop:          installed"
    else
        log "docker-desktop:          not installed (only needed for --with-desktop)"
    fi

    if dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
        log "nvidia-container-toolkit: installed"
    else
        warn "nvidia-container-toolkit: NOT installed"; failures=$((failures+1))
    fi

    if systemctl is-active docker >/dev/null 2>&1; then
        log "docker.service:          active"
    else
        warn "docker.service:          inactive"; failures=$((failures+1))
    fi

    log "docker context:          $(docker context show 2>/dev/null || echo unknown)"

    if docker info >/dev/null 2>&1; then
        log "daemon reachable:        yes"
    else
        warn "daemon reachable:        no"; failures=$((failures+1))
    fi

    if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
        log "GPU passthrough:         working"
    else
        warn "GPU passthrough:         failing"; failures=$((failures+1))
    fi

    if [[ $failures -eq 0 ]]; then
        log "All checks passed."
    else
        warn "$failures check(s) failed."
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if [[ "$CHECK_ONLY" == 1 ]]; then
    run_check
    exit $?
fi

check_preconditions
setup_docker_apt_repo
install_docker_engine
[[ "$WITH_DESKTOP" == 1 ]] && install_docker_desktop
add_user_to_docker_group
install_nvidia_container_toolkit
switch_to_engine_context
verify_gpu_passthrough

echo
log "Done. Docker Engine + NVIDIA Container Toolkit are wired up."
[[ "$WITH_DESKTOP" == 1 ]] && log "Docker Desktop is also installed; launch it from your app menu."
[[ "${NEEDS_RELOGIN:-0}" == "1" ]] \
    && warn "Log out and back in (or run \`newgrp docker\`) to use docker without sudo."
log "Re-run with --check anytime to verify."
log "To use Docker Desktop instead of the engine: docker context use desktop-linux"
