#!/bin/bash
# Submit with one of:
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
# caller can choose the queue that matches the intended test depth.
#SBATCH --job-name=pathbench-tests
#SBATCH --output=slurm-pathbench-tests-%j.out
#SBATCH --error=slurm-pathbench-tests-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="/exports/path-cutane-lymfomen-hpc/siemen/PathForge"
LOG_DIR="${PROJECT_ROOT}/test_logs/${SLURM_JOB_ID:-manual}"
SMOKE_CACHE_DEFAULT="/exports/path-cutane-lymfomen-hpc/siemen/cache/tmp/pathbench_smoke"
SMOKE_REPORT_DIR="${LOG_DIR}/smoke_report"
COMBINED_SMOKE_REPORT_DIR="${LOG_DIR}/combined_smoke_report"

# Optional overrides:
#   SBATCH_PARTITION is informational only; set partition when submitting.
#   PATHFORGE_ENABLE_CUDA=1 enables the cu128 extra during uv sync.
#   PATHFORGE_SMOKE_CACHE overrides the shared smoke cache location.
PATHFORGE_ENABLE_CUDA="${PATHFORGE_ENABLE_CUDA:-0}"
export PATHFORGE_SMOKE_CACHE="${PATHFORGE_SMOKE_CACHE:-${SMOKE_CACHE_DEFAULT}}"

mkdir -p "${LOG_DIR}" "${PATHFORGE_SMOKE_CACHE}"
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

echo "[$(date --iso-8601=seconds)] Starting PathBench full test suite"
echo "Project root: ${PROJECT_ROOT}"
echo "Log directory: ${LOG_DIR}"
echo "Smoke cache: ${PATHFORGE_SMOKE_CACHE}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Partition: ${CURRENT_PARTITION}"
echo "CUDA extra enabled: ${PATHFORGE_ENABLE_CUDA}"

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
uv run python -m pytest -q tests/unit 2>&1 | tee "${LOG_DIR}/unit.log"

echo "[$(date --iso-8601=seconds)] Running interface tests"
uv run python -m pytest -q tests/interface 2>&1 | tee "${LOG_DIR}/interface.log"

echo "[$(date --iso-8601=seconds)] Running integration tests"
uv run python -m pytest -q tests/integration 2>&1 | tee "${LOG_DIR}/integration.log"

echo "[$(date --iso-8601=seconds)] Running full smoke suite"
export PATHFORGE_SMOKE_REPORT_DIR="${SMOKE_REPORT_DIR}"
uv run python -m pytest -m smoke -q tests/smoke 2>&1 | tee "${LOG_DIR}/smoke.log"
echo "Smoke report directory: ${SMOKE_REPORT_DIR}"

echo "[$(date --iso-8601=seconds)] Running combined repository test pass"
export PATHFORGE_SMOKE_REPORT_DIR="${COMBINED_SMOKE_REPORT_DIR}"
uv run python -m pytest -q tests/unit tests/interface tests/integration tests/smoke 2>&1 | tee "${LOG_DIR}/combined.log"
echo "Combined smoke report directory: ${COMBINED_SMOKE_REPORT_DIR}"

echo "[$(date --iso-8601=seconds)] Full PathBench test suite completed successfully"
