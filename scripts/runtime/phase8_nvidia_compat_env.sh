#!/usr/bin/env bash
# Per-process NVIDIA userspace compatibility for Phase-8 queue scripts.
#
# This file is sourced by the queue and supervisor.  It never changes host
# driver state.  It only prepends an explicitly selected, user-owned driver
# bundle to LD_LIBRARY_PATH after checking the loaded kernel version and the
# two libraries required by CUDA/NCCL.

PHASE8_NVIDIA_EXPECTED_VERSION="${PHASE8_NVIDIA_EXPECTED_VERSION:-595.71.05}"
PHASE8_NVIDIA_COMPAT_DIR="${PHASE8_NVIDIA_COMPAT_DIR:-/home/kojiek/.local/opt/nvidia-595.71.05/NVIDIA-Linux-x86_64-595.71.05}"
# Hashes of the exact pair validated on this host.  A same-named or merely
# same-version replacement is not accepted without an intentional code change.
PHASE8_NVIDIA_EXPECTED_LIBCUDA_SHA256="76e0d9678d41cf6b6ae71d18549d88a963d3d434f08108baa12457eb53ca88d6"
PHASE8_NVIDIA_EXPECTED_LIBNVML_SHA256="9eb4358b7fea76556657670a6ae6b0017eaa4256b56c421a36626bf8c2b5f3f5"
PHASE8_NVIDIA_COMPAT_ACTIVE="${PHASE8_NVIDIA_COMPAT_ACTIVE:-false}"
PHASE8_NVIDIA_COMPAT_DIR_REAL="${PHASE8_NVIDIA_COMPAT_DIR_REAL:-}"
PHASE8_NVIDIA_COMPAT_ERROR=""

phase8_nvidia_kernel_version() {
    local version_file=/proc/driver/nvidia/version
    [[ -r "$version_file" ]] || {
        echo "kernel version file is unreadable: $version_file" >&2
        return 1
    }
    awk '{
        for (i = 1; i <= NF; i++) {
            if ($i ~ /^[0-9]+\.[0-9]+\.[0-9]+$/) {
                print $i
                exit
            }
        }
    }' "$version_file"
}

phase8_nvidia_real_library() {
    local dir="$1" name="$2" path real
    path="$dir/$name"
    [[ -e "$path" ]] || {
        echo "missing $name in $dir" >&2
        return 1
    }
    real=$(readlink -e "$path") || {
        echo "cannot resolve $path" >&2
        return 1
    }
    [[ -f "$real" && -r "$real" ]] || {
        echo "library is not a readable regular file: $real" >&2
        return 1
    }
    case "$real" in
        "$dir"/*) ;;
        *)
            echo "$path resolves outside compat directory: $real" >&2
            return 1
            ;;
    esac
    printf '%s\n' "$real"
}

phase8_nvidia_validate_library() {
    local label="$1" path="$2" expected_hash="$3" actual_hash
    readelf -h "$path" >/dev/null 2>&1 || {
        echo "$label is not a readable ELF: $path" >&2
        return 1
    }
    # Do not use grep -q here: with pipefail, early grep exit can turn a
    # valid match into a false SIGPIPE failure from strings.
    strings "$path" 2>/dev/null | grep -F "$PHASE8_NVIDIA_EXPECTED_VERSION" >/dev/null || {
        echo "$label does not contain version $PHASE8_NVIDIA_EXPECTED_VERSION: $path" >&2
        return 1
    }
    actual_hash=$(sha256sum "$path" | awk '{print $1}')
    [[ "$actual_hash" == "$expected_hash" ]] || {
        echo "$label sha256 mismatch: got=$actual_hash expected=$expected_hash" >&2
        return 1
    }
}

phase8_nvidia_compat_preflight() {
    local kernel_version dir_real cuda_real nvml_real
    PHASE8_NVIDIA_COMPAT_ERROR=""

    kernel_version=$(phase8_nvidia_kernel_version 2>&1) || {
        PHASE8_NVIDIA_COMPAT_ERROR="$kernel_version"
        return 1
    }
    if [[ "$kernel_version" != "$PHASE8_NVIDIA_EXPECTED_VERSION" ]]; then
        PHASE8_NVIDIA_COMPAT_ERROR="loaded kernel=$kernel_version expected=$PHASE8_NVIDIA_EXPECTED_VERSION"
        return 1
    fi

    dir_real=$(readlink -e "$PHASE8_NVIDIA_COMPAT_DIR" 2>/dev/null) || {
        PHASE8_NVIDIA_COMPAT_ERROR="compat directory is unavailable: $PHASE8_NVIDIA_COMPAT_DIR"
        return 1
    }
    [[ -d "$dir_real" && -x "$dir_real" ]] || {
        PHASE8_NVIDIA_COMPAT_ERROR="compat directory is not searchable: $dir_real"
        return 1
    }
    cuda_real=$(phase8_nvidia_real_library "$dir_real" libcuda.so.1 2>&1) || {
        PHASE8_NVIDIA_COMPAT_ERROR="$cuda_real"
        return 1
    }
    nvml_real=$(phase8_nvidia_real_library "$dir_real" libnvidia-ml.so.1 2>&1) || {
        PHASE8_NVIDIA_COMPAT_ERROR="$nvml_real"
        return 1
    }
    phase8_nvidia_validate_library libcuda.so.1 "$cuda_real" \
        "$PHASE8_NVIDIA_EXPECTED_LIBCUDA_SHA256" || {
        PHASE8_NVIDIA_COMPAT_ERROR="$(phase8_nvidia_validate_library libcuda.so.1 "$cuda_real" "$PHASE8_NVIDIA_EXPECTED_LIBCUDA_SHA256" 2>&1)"
        return 1
    }
    phase8_nvidia_validate_library libnvidia-ml.so.1 "$nvml_real" \
        "$PHASE8_NVIDIA_EXPECTED_LIBNVML_SHA256" || {
        PHASE8_NVIDIA_COMPAT_ERROR="$(phase8_nvidia_validate_library libnvidia-ml.so.1 "$nvml_real" "$PHASE8_NVIDIA_EXPECTED_LIBNVML_SHA256" 2>&1)"
        return 1
    }

    PHASE8_NVIDIA_COMPAT_DIR_REAL="$dir_real"
    printf 'kernel=%s compat=%s libcuda=%s libnvidia-ml=%s\n' \
        "$kernel_version" "$dir_real" "$cuda_real" "$nvml_real"
}

phase8_nvidia_compat_apply() {
    if ! phase8_nvidia_compat_preflight; then
        PHASE8_NVIDIA_COMPAT_ACTIVE=false
        printf '%s\n' "$PHASE8_NVIDIA_COMPAT_ERROR" >&2
        return 1
    fi
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$PHASE8_NVIDIA_COMPAT_DIR_REAL:"*) ;;
        *)
            export LD_LIBRARY_PATH="$PHASE8_NVIDIA_COMPAT_DIR_REAL${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            ;;
    esac
    export PHASE8_NVIDIA_COMPAT_ACTIVE=true
}

phase8_nvidia_compat_functional_preflight() {
    [[ "${PHASE8_NVIDIA_FUNCTIONAL_PREFLIGHT:-false}" == true ]] || {
        echo "functional probe disabled (set PHASE8_NVIDIA_FUNCTIONAL_PREFLIGHT=true only on a clear GPU)"
        return 0
    }
    [[ "$PHASE8_NVIDIA_COMPAT_ACTIVE" == true ]] || {
        echo "compat environment is not active" >&2
        return 1
    }
    python - <<'PY'
import os
import torch
import torch.distributed as dist

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
device = torch.device("cuda:0")
value = torch.ones(1, device=device)
if float(value.item()) != 1.0:
    raise SystemExit("CUDA tensor functional check failed")
if not dist.is_available():
    raise SystemExit("torch.distributed is unavailable")
store = f"/tmp/phase8_nvidia_compat_preflight_{os.getpid()}"
try:
    dist.init_process_group("nccl", init_method=f"file://{store}", rank=0, world_size=1)
    dist.all_reduce(value)
    if float(value.item()) != 1.0:
        raise SystemExit("NCCL all_reduce functional check failed")
finally:
    if dist.is_initialized():
        dist.destroy_process_group()
    try:
        os.unlink(store)
    except FileNotFoundError:
        pass
print("functional=passed cuda_tensor=passed nccl_all_reduce=passed")
PY
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    phase8_nvidia_compat_apply
    phase8_nvidia_compat_functional_preflight
fi
