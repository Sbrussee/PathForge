#!/usr/bin/env bash
# Submit with, for example:
#   mkdir -p test_logs cache/tmp/pathbench_smoke
#   sbatch --partition=PATHgpu --time=04:00:00 scripts/run_slide_retrieval_smoke.sbatch
#
# Every cache, temporary workspace, report, and Slurm log is stored below this
# repository's dynamically resolved project root. Do not override them with $HOME.
# Set ``PATHFORGE_TEST_SCOPE=smoke`` to run only the three GPU smoke tests.
#SBATCH --job-name=pathforge-slide-retrieval-smoke
#SBATCH --output=test_logs/slurm-pathforge-slide-retrieval-smoke-%j.out
#SBATCH --error=test_logs/slurm-pathforge-slide-retrieval-smoke-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

# Slurm executes a spool-file copy, so BASH_SOURCE points into /var/spool.
# SLURM_SUBMIT_DIR preserves the PathForge directory from which sbatch ran.
PROJECT_ROOT="${PATHFORGE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${PROJECT_ROOT}" ]]; then
  PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fi
PROJECT_ROOT="$(cd -- "${PROJECT_ROOT}" && pwd)"
RUN_DIR="${PROJECT_ROOT}/test_logs/${SLURM_JOB_ID:-manual}"
CACHE_ROOT="${PROJECT_ROOT}/cache"
CACHE_DIR="${CACHE_ROOT}/tmp/pathbench_smoke"
REPORT_DIR="${RUN_DIR}/report"
TMP_DIR="${RUN_DIR}/tmp"
PYTEST_CACHE_DIR="${RUN_DIR}/pytest-cache"

mkdir -p "${CACHE_DIR}" "${REPORT_DIR}" "${TMP_DIR}" "${PYTEST_CACHE_DIR}"
export PATHFORGE_SMOKE_CACHE="$(cd "${CACHE_DIR}" && pwd)"
export PATHFORGE_SMOKE_REPORT_DIR="$(cd "${REPORT_DIR}" && pwd)"
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
export XDG_CACHE_HOME="${CACHE_ROOT}/slide-retrieval-smoke/${SLURM_JOB_ID:-manual}/xdg-cache"
export HF_HOME="${CACHE_ROOT}/slide-retrieval-smoke/${SLURM_JOB_ID:-manual}/huggingface"
export PIP_CACHE_DIR="${CACHE_ROOT}/slide-retrieval-smoke/${SLURM_JOB_ID:-manual}/pip-cache"
export TORCH_HOME="${CACHE_ROOT}/slide-retrieval-smoke/${SLURM_JOB_ID:-manual}/torch"
export TMPDIR="${TMP_DIR}"
export PYTHONPYCACHEPREFIX="${CACHE_ROOT}/slide-retrieval-smoke/${SLURM_JOB_ID:-manual}/pycache"
mkdir -p "${UV_CACHE_DIR}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${PIP_CACHE_DIR}" "${TORCH_HOME}" "${PYTHONPYCACHEPREFIX}"

cd "${PROJECT_ROOT}"

TEST_SCOPE="${PATHFORGE_TEST_SCOPE:-retrieval}"
SMOKE_TESTS=(
  tests/smoke/test_slide_retrieval_benchmark_smoke.py
  tests/smoke/test_slide_retrieval_e2e_smoke.py
  tests/smoke/test_slide_retrieval_visualization_smoke.py
)
RETRIEVAL_TESTS=(
  tests/integration/test_slide_retrieval.py
  tests/unit/test_evaluation_config.py
  tests/unit/test_evaluation_metrics.py
  tests/unit/test_evaluation_orchestrator.py
  tests/unit/test_evaluation_registry.py
  tests/unit/test_hshr_features_representation.py
  tests/unit/test_retrieval_*.py
  tests/unit/test_sdm_features_representation_strategy.py
  tests/unit/test_sish_*.py
  tests/unit/test_slide_retrieval_*.py
  tests/unit/test_splice_representation_strategies.py
  tests/unit/test_yottixel_search_strategy.py
  "${SMOKE_TESTS[@]}"
)

case "${TEST_SCOPE}" in
  retrieval) TESTS=("${RETRIEVAL_TESTS[@]}") ;;
  smoke) TESTS=("${SMOKE_TESTS[@]}") ;;
  *)
    echo "PATHFORGE_TEST_SCOPE must be 'retrieval' or 'smoke', got: ${TEST_SCOPE}" >&2
    exit 2
    ;;
esac

echo "[$(date --iso-8601=seconds)] Starting slide-retrieval ${TEST_SCOPE} suite"
echo "Project root: ${PROJECT_ROOT}"
echo "Smoke cache: ${PATHFORGE_SMOKE_CACHE}"
echo "Report directory: ${PATHFORGE_SMOKE_REPORT_DIR}"
echo "Run directory: ${RUN_DIR}"

uv run python -m pytest \
  "${TESTS[@]}" \
  -vv --tb=short \
  --basetemp "${RUN_DIR}/pytest-tmp" \
  -o "cache_dir=${PYTEST_CACHE_DIR}" \
  2>&1 | tee "${RUN_DIR}/pytest.log"

echo "[$(date --iso-8601=seconds)] Slide-retrieval ${TEST_SCOPE} suite completed"
echo "Artifacts: ${REPORT_DIR}"
