#!/usr/bin/env bash
# Submit with one of:
#   mkdir -p .cache
#   sbatch --partition=gpu       --time=10:00:00 scripts/run_full_test_suite.sbatch
#   sbatch --partition=gpu-short --time=00:30:00 scripts/run_full_test_suite.sbatch
#   sbatch --partition=gpu-medium --time=2-00:00:00 scripts/run_full_test_suite.sbatch
#   sbatch --partition=gpu-long  --time=7-00:00:00 scripts/run_full_test_suite.sbatch
#   sbatch --partition=PATHgpu   --time=<cluster_limit> scripts/run_full_test_suite.sbatch
#
# Known partition notes:
#   gpu        : 10 hours, res-hpc-gpu[01-02]
#   gpu-short  : 30 minutes, common short GPU queue
#   gpu-medium : 2 days, res-hpc-gpu[06-08]
#   gpu-long   : 7 days, res-hpc-gpu[11-12]
#   PATHgpu    : pathology GPU partition, cluster-specific limit
#
# The script intentionally does not hard-code --partition or --time so the
# caller can choose the queue that matches the intended test depth. All test
# artifacts and caches remain below this repository's ``.cache/`` directory.
#SBATCH --job-name=pathforge-tests
#SBATCH --output=.cache/slurm-pathforge-tests-%j.out
#SBATCH --error=.cache/slurm-pathforge-tests-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="${PATHFORGE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${PROJECT_ROOT}" ]]; then
  PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fi
PROJECT_ROOT="$(cd -- "${PROJECT_ROOT}" && pwd)"
RUN_ROOT="${PROJECT_ROOT}/.cache/full-test-suite"
LOG_DIR="${RUN_ROOT}/runs/${SLURM_JOB_ID:-manual}"
SMOKE_CACHE_DEFAULT="${RUN_ROOT}/download-cache"
SMOKE_REPORT_DIR="${LOG_DIR}/smoke_report"
COMBINED_SMOKE_REPORT_DIR="${LOG_DIR}/combined_smoke_report"

# Optional override:
#   PATHFORGE_ENABLE_CUDA=1 enables the cu128 extra during uv sync.
PATHFORGE_ENABLE_CUDA="${PATHFORGE_ENABLE_CUDA:-0}"
export PATHFORGE_SMOKE_CACHE="${SMOKE_CACHE_DEFAULT}"
export UV_CACHE_DIR="${PROJECT_ROOT}/.cache/uv"
export XDG_CACHE_HOME="${LOG_DIR}/xdg-cache"
export HF_HOME="${LOG_DIR}/huggingface"
export PIP_CACHE_DIR="${LOG_DIR}/pip-cache"
export TORCH_HOME="${LOG_DIR}/torch"
export TMPDIR="${LOG_DIR}/tmp"
export PYTHONPYCACHEPREFIX="${LOG_DIR}/pycache"

mkdir -p "${LOG_DIR}" "${PATHFORGE_SMOKE_CACHE}" "${UV_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" "${HF_HOME}" "${PIP_CACHE_DIR}" "${TORCH_HOME}" \
  "${TMPDIR}" "${PYTHONPYCACHEPREFIX}"
cd "${PROJECT_ROOT}"

ALLOWED_PARTITIONS=("gpu" "gpu-short" "gpu-medium" "gpu-long" "PATHgpu")
CURRENT_PARTITION="${SLURM_JOB_PARTITION:-unknown}"

partition_allowed=0
for partition_name in "${ALLOWED_PARTITIONS[@]}"; do
  if [[ "${CURRENT_PARTITION}" == "${partition_name}" ]]; then
    partition_allowed=1
    break
  fi
done

if [[ "${partition_allowed}" -ne 1 ]]; then
  echo "Unsupported partition: ${CURRENT_PARTITION}" >&2
  echo "Use one of: ${ALLOWED_PARTITIONS[*]}" >&2
  exit 2
fi

echo "[$(date --iso-8601=seconds)] Starting PathForge full test suite"
echo "Project root: ${PROJECT_ROOT}"
echo "Log directory: ${LOG_DIR}"
echo "Smoke cache: ${PATHFORGE_SMOKE_CACHE}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Partition: ${CURRENT_PARTITION}"
echo "CUDA extra enabled: ${PATHFORGE_ENABLE_CUDA}"

run_pytest() {
  local name="$1"
  shift
  uv run python -m pytest -q "$@" \
    --basetemp "${LOG_DIR}/pytest-${name}" \
    -o "cache_dir=${LOG_DIR}/pytest-cache-${name}" \
    2>&1 | tee "${LOG_DIR}/${name}.log"
}

UV_SYNC_CMD=(
  uv sync
  --extra mil-backends
  --extra hf
  --extra dev
)

if [[ "${PATHFORGE_ENABLE_CUDA}" == "1" ]]; then
  UV_SYNC_CMD+=(--extra cu128)
fi

echo "[$(date --iso-8601=seconds)] Syncing environment"
"${UV_SYNC_CMD[@]}" 2>&1 | tee "${LOG_DIR}/uv-sync.log"

echo "[$(date --iso-8601=seconds)] Running unit tests"
run_pytest unit tests/unit

echo "[$(date --iso-8601=seconds)] Running interface tests"
run_pytest interface tests/interface

echo "[$(date --iso-8601=seconds)] Running integration tests"
run_pytest integration tests/integration

echo "[$(date --iso-8601=seconds)] Running full smoke suite"
export PATHFORGE_SMOKE_REPORT_DIR="${SMOKE_REPORT_DIR}"
run_pytest smoke -m smoke tests/smoke
echo "Smoke report directory: ${SMOKE_REPORT_DIR}"

echo "[$(date --iso-8601=seconds)] Running combined repository test pass"
export PATHFORGE_SMOKE_REPORT_DIR="${COMBINED_SMOKE_REPORT_DIR}"
run_pytest combined tests/unit tests/interface tests/integration tests/smoke
echo "Combined smoke report directory: ${COMBINED_SMOKE_REPORT_DIR}"

echo "[$(date --iso-8601=seconds)] Full PathForge test suite completed successfully"
