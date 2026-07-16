#!/usr/bin/env bash
# Submit with:
#   sbatch --partition=PATHgpu --time=01:00:00 scripts/run_distributed_smoke.sbatch
#SBATCH --job-name=pathforge-dist-smoke
#SBATCH --output=slurm-distributed-smoke-%j.out
#SBATCH --error=slurm-distributed-smoke-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="/exports/path-cutane-lymfomen-hpc/siemen/PathForge"
LOG_DIR="${PROJECT_ROOT}/test_logs/distributed-${SLURM_JOB_ID:-manual}"
export PATHFORGE_SMOKE_CACHE="${PATHFORGE_SMOKE_CACHE:-/exports/path-cutane-lymfomen-hpc/siemen/cache/tmp/pathbench_smoke}"
export PATHFORGE_SMOKE_REPORT_DIR="${LOG_DIR}/report"

mkdir -p "${LOG_DIR}" "${PATHFORGE_SMOKE_CACHE}"
cd "${PROJECT_ROOT}"

echo "[$(date --iso-8601=seconds)] Starting PathForge distributed smoke test"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Log directory: ${LOG_DIR}"

uv sync --extra mil-backends --extra hf --extra dev --extra distributed \
  2>&1 | tee "${LOG_DIR}/uv-sync.log"

uv run python -m pytest -q \
  tests/unit/test_distributed_execution.py \
  tests/unit/test_policy_optimization.py \
  tests/smoke/test_optimize_cli.py \
  2>&1 | tee "${LOG_DIR}/pytest.log"

echo "[$(date --iso-8601=seconds)] Distributed smoke test completed successfully"
