#!/usr/bin/env bash
# Submit with, for example:
#   sbatch --partition=PATHgpu --time=02:00:00 scripts/run_slide_retrieval_smoke.sbatch
#
# Optional overrides:
#   PATHFORGE_SMOKE_CACHE=/shared/pathforge-smoke-cache
#   PATHFORGE_SMOKE_RUN_DIR=/path/to/output-dir
#
# The default cache and report directories are repository-local. Their absolute
# paths are important: the smoke fixture links source WSIs into its workspace.
# A cache outside the repository is preferable when several jobs reuse it.
#SBATCH --job-name=pathforge-slide-retrieval-smoke
#SBATCH --output=slurm-pathforge-slide-retrieval-smoke-%j.out
#SBATCH --error=slurm-pathforge-slide-retrieval-smoke-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${PATHFORGE_SMOKE_RUN_DIR:-${PROJECT_ROOT}/pathforge_slide_retrieval_smoke_${SLURM_JOB_ID:-manual}}"
CACHE_DIR="${PATHFORGE_SMOKE_CACHE:-${RUN_DIR}/cache}"
REPORT_DIR="${RUN_DIR}/report"

mkdir -p "${CACHE_DIR}" "${REPORT_DIR}"
export PATHFORGE_SMOKE_CACHE="$(cd "${CACHE_DIR}" && pwd)"
export PATHFORGE_SMOKE_REPORT_DIR="$(cd "${REPORT_DIR}" && pwd)"

cd "${PROJECT_ROOT}"

echo "[$(date --iso-8601=seconds)] Starting slide-retrieval smoke suite"
echo "Project root: ${PROJECT_ROOT}"
echo "Smoke cache: ${PATHFORGE_SMOKE_CACHE}"
echo "Report directory: ${PATHFORGE_SMOKE_REPORT_DIR}"

uv run python -m pytest \
  tests/smoke/test_slide_retrieval_benchmark_smoke.py \
  tests/smoke/test_slide_retrieval_e2e_smoke.py \
  tests/smoke/test_slide_retrieval_visualization_smoke.py \
  -m smoke -vv --tb=short 2>&1 | tee "${RUN_DIR}/pytest.log"

echo "[$(date --iso-8601=seconds)] Slide-retrieval smoke suite completed"
echo "Artifacts: ${REPORT_DIR}"
