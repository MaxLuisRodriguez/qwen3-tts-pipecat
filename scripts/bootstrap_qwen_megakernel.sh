#!/usr/bin/env bash
# Non-interactive, idempotent bootstrap for this monorepo's kernel stack.

set -euo pipefail
IFS=$'\n\t'

on_error() {
  local line="$1"
  local cmd="$2"
  echo "[ERROR] bootstrap failed at line ${line}: ${cmd}" >&2
}
trap 'on_error "${LINENO}" "${BASH_COMMAND}"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
KERNEL_DIR="${REPO_ROOT}/kernel"
ENV_FILE="${REPO_ROOT}/.env.qwen_megakernel"
ENV_TEMPLATE="${REPO_ROOT}/.env.qwen_megakernel.template"
SMOKE_LOG="${REPO_ROOT}/smoke_test.log"
EXPECTED_MODEL_SHA256="f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"

log() {
  echo "[bootstrap] $*"
}

fail() {
  echo "[bootstrap] $*" >&2
  exit 1
}

retry() {
  local attempts="${RETRY_ATTEMPTS:-5}"
  local delay=2
  local i
  for ((i = 1; i <= attempts; i++)); do
    if "$@"; then
      return 0
    fi
    if [[ "${i}" -eq "${attempts}" ]]; then
      break
    fi
    log "retry ${i}/${attempts} failed; sleeping ${delay}s..."
    sleep "${delay}"
    delay=$((delay * 2))
  done
  return 1
}

version_ge() {
  local got="$1"
  local need="$2"
  [[ "$(printf '%s\n%s\n' "${need}" "${got}" | sort -V | head -n1)" == "${need}" ]]
}

ensure_env_template() {
  if [[ -f "${ENV_TEMPLATE}" ]]; then
    return
  fi

  cat >"${ENV_TEMPLATE}" <<'EOF'
# Copy this file to .env.qwen_megakernel and edit values as needed.

# Optional Hugging Face token (leave blank for public model access).
HF_TOKEN=

# Cache paths. Relative paths are resolved from repo root.
HF_HOME=.hf_home
HF_HUB_CACHE=.hf_home/hub
HF_XET_HIGH_PERFORMANCE=1
HF_HUB_OFFLINE=0

# Model source for qwen_megakernel.
# For local paths with one slash, prefer a ./ prefix (example: ./weights/Qwen3-0.6B).
QWEN_MEGAKERNEL_MODEL_NAME=kernel/weights/Qwen3-0.6B

# Build cache + parallelism.
TORCH_EXTENSIONS_DIR=.torch_extensions
MAX_JOBS=8

# Compile-time decode tuning (defaults from this repo's build.py).
LDG_NUM_BLOCKS=128
LDG_BLOCK_SIZE=512
LDG_LM_NUM_BLOCKS=1280
LDG_LM_BLOCK_SIZE=384
LDG_LM_ROWS_PER_WARP=2
LDG_ATTN_BLOCKS=8
LDG_PREFETCH_QK=0
LDG_PREFETCH_THREAD_STRIDE=10
LDG_PREFETCH_DOWN=1
LDG_PREFETCH_ELEM_STRIDE=1
LDG_PREFETCH_BLOCK_STRIDE=1
LDG_PREFETCH_GATE=1
LDG_PREFETCH_UP=1
EOF
}

resolve_path() {
  local value="$1"
  if [[ -z "${value}" ]]; then
    echo ""
    return
  fi
  if [[ "${value}" = /* ]]; then
    echo "${value}"
  else
    echo "${REPO_ROOT}/${value}"
  fi
}

resolve_model_source() {
  local value="$1"
  if [[ -z "${value}" ]]; then
    echo "${KERNEL_DIR}/weights/Qwen3-0.6B"
    return
  fi
  if [[ "${value}" = /* ]]; then
    echo "${value}"
    return
  fi
  # Treat "org/model" as a Hugging Face model id (not a local path).
  if [[ "${value}" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
    echo "${value}"
    return
  fi
  echo "${REPO_ROOT}/${value}"
}

load_env() {
  ensure_env_template
  if [[ ! -f "${ENV_FILE}" ]]; then
    cp "${ENV_TEMPLATE}" "${ENV_FILE}"
    fail "Created ${ENV_FILE}. Fill it in and rerun this script."
  fi

  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a

  export HF_HOME
  HF_HOME="$(resolve_path "${HF_HOME:-.hf_home}")"
  export HF_HUB_CACHE
  HF_HUB_CACHE="$(resolve_path "${HF_HUB_CACHE:-.hf_home/hub}")"
  export TORCH_EXTENSIONS_DIR
  TORCH_EXTENSIONS_DIR="$(resolve_path "${TORCH_EXTENSIONS_DIR:-.torch_extensions}")"
  export QWEN_MEGAKERNEL_MODEL_NAME
  QWEN_MEGAKERNEL_MODEL_NAME="$(resolve_model_source "${QWEN_MEGAKERNEL_MODEL_NAME:-kernel/weights/Qwen3-0.6B}")"
  export MAX_JOBS
  MAX_JOBS="${MAX_JOBS:-8}"

  mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TORCH_EXTENSIONS_DIR}"
}

assert_repo_layout() {
  [[ -d "${KERNEL_DIR}" ]] || fail "Missing kernel directory at ${KERNEL_DIR}"
  [[ -f "${KERNEL_DIR}/qwen_megakernel/model.py" ]] || fail "Missing kernel/qwen_megakernel/model.py"
  [[ -f "${KERNEL_DIR}/qwen_megakernel/bench.py" ]] || fail "Missing kernel/qwen_megakernel/bench.py"
}

check_host_prereqs() {
  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "CUDA not supported on macOS; SSH to Ubuntu+RTX 5090" >&2
    exit 2
  fi

  local cmd
  for cmd in git curl python3 nvidia-smi nvcc gcc g++ sha256sum; do
    command -v "${cmd}" >/dev/null 2>&1 || fail "Required command not found: ${cmd}"
  done

  nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi failed; verify NVIDIA driver install."

  local nvcc_release
  nvcc_release="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)"
  [[ -n "${nvcc_release}" ]] || fail "Could not parse CUDA version from nvcc --version."
  version_ge "${nvcc_release}" "12.8" || fail "CUDA toolkit must be >= 12.8 (found ${nvcc_release})."
}

verify_optimization_symbols() {
  local kernel_cu="${KERNEL_DIR}/csrc/kernel.cu"
  local bindings_cpp="${KERNEL_DIR}/csrc/torch_bindings.cpp"

  grep -q "prefetch.global.L2" "${kernel_cu}" || fail "Missing prefetch.global.L2 in kernel.cu"
  grep -q "ldg_lm_head_fused" "${kernel_cu}" || fail "Missing ldg_lm_head_fused in kernel.cu"
  grep -q "launch_ldg_decode_direct" "${bindings_cpp}" || fail "Missing launch_ldg_decode_direct in torch_bindings.cpp"
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  local uv_root="${REPO_ROOT}/.local/uv"
  mkdir -p "${uv_root}"
  log "uv not found; installing into ${uv_root}"
  retry env UV_UNMANAGED_INSTALL="${uv_root}" sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh' || fail "Failed to install uv."
  export PATH="${uv_root}:${PATH}"
  command -v uv >/dev/null 2>&1 || fail "uv install finished but uv command is still unavailable."
}

setup_python_env() {
  cd "${KERNEL_DIR}"

  if [[ ! -d ".venv" ]]; then
    retry uv venv .venv || fail "Failed to create virtual environment with uv."
  fi

  # shellcheck disable=SC1091
  source .venv/bin/activate

  retry python -m pip install --upgrade pip || fail "Failed to upgrade pip."
  retry pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128 || fail "Failed to install torch cu128 wheel."
  retry pip install "transformers>=4.51.0" triton accelerate ninja huggingface_hub || fail "Failed to install Python dependencies."
  retry pip install -r "${REPO_ROOT}/services/llm_megakernel/requirements.txt" || fail "Failed to install llm service dependencies."
  retry pip install -r "${REPO_ROOT}/services/tts_qwen3/requirements.txt" || fail "Failed to install tts service dependencies."
  retry pip install -r "${REPO_ROOT}/pipecat_demo/requirements.txt" || fail "Failed to install pipecat dependencies."
}

verify_torch_gpu() {
  cd "${KERNEL_DIR}"
  # shellcheck disable=SC1091
  source .venv/bin/activate

  python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False; check driver/toolkit installation.")

cap = torch.cuda.get_device_capability(0)
if cap != (12, 0):
    sys.exit(f"Expected compute capability (12, 0) for RTX 5090, got {cap}.")

arch = [a.lower() for a in torch.cuda.get_arch_list()]
if not any("sm_120" in a for a in arch):
    sys.exit(
        "Torch arch list missing sm_120. Install torch==2.7.0 from cu128 wheels."
    )

print(f"torch={torch.__version__}")
print(f"device={torch.cuda.get_device_name(0)}")
print(f"capability={cap}")
print(f"arch_list={arch}")
PY
}

download_and_verify_weights() {
  cd "${KERNEL_DIR}"
  # shellcheck disable=SC1091
  source .venv/bin/activate

  local weights_dir="${KERNEL_DIR}/weights/Qwen3-0.6B"
  mkdir -p "${weights_dir}"

  log "Downloading Qwen/Qwen3-0.6B weights into ${weights_dir}"
  retry env WEIGHTS_DIR="${weights_dir}" python - <<'PY' || fail "Failed to download model weights."
import os
from huggingface_hub import snapshot_download

token = os.getenv("HF_TOKEN") or None
snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    local_dir=os.environ["WEIGHTS_DIR"],
    token=token,
    resume_download=True,
)
PY

  local weights_file="${weights_dir}/model.safetensors"
  [[ -f "${weights_file}" ]] || fail "Expected ${weights_file} after download."
  local actual_sha
  actual_sha="$(sha256sum "${weights_file}" | awk '{print $1}')"
  if [[ "${actual_sha}" != "${EXPECTED_MODEL_SHA256}" ]]; then
    fail "model.safetensors sha256 mismatch: expected ${EXPECTED_MODEL_SHA256}, got ${actual_sha}"
  fi
}

run_smoke_test() {
  cd "${KERNEL_DIR}"
  # shellcheck disable=SC1091
  source .venv/bin/activate

  log "JIT-building extension via import"
  python -c "import qwen_megakernel" || fail "Failed to import qwen_megakernel."

  log "Running smoke test: python -m qwen_megakernel.bench"
  python -m qwen_megakernel.bench | tee "${SMOKE_LOG}" || fail "Smoke test failed."

  log "Smoke test completed. Last 60 lines:"
  tail -n 60 "${SMOKE_LOG}"
}

main() {
  log "Repo root: ${REPO_ROOT}"
  assert_repo_layout
  load_env
  check_host_prereqs
  verify_optimization_symbols
  ensure_uv
  setup_python_env
  verify_torch_gpu
  download_and_verify_weights
  run_smoke_test
  log "Bootstrap finished successfully."
}

main "$@"
