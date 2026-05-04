# Testing Guide

This directory contains two main test layers:

- `tests/unit`: fast unit and contract tests
- `tests/interface`: architecture and interface-boundary tests
- `tests/smoke`: broader end-to-end sanity checks, including the Hugging Face-backed smoke suite

All commands below assume you are running from:

```bash
cd /exports/path-cutane-lymfomen-hpc/siemen/PathBench_2.0
```

## Install Test Dependencies

For the standard development environment:

```bash
uv sync --extra lazyslide --extra mil-backends --extra hf --extra dev
```

If you need CUDA 12.8 PyTorch builds on a GPU node:

```bash
uv sync --extra lazyslide --extra mil-backends --extra hf --extra dev --extra cu128
```

## Run Unit Tests

Run the full unit test suite:

```bash
uv run pytest tests/unit -q
```

Run a single unit test module:

```bash
uv run pytest tests/unit/test_config_validation.py -q
```

## Run Interface Tests

Run the full interface test suite:

```bash
uv run pytest tests/interface -q
```

Run one interface test module:

```bash
uv run pytest tests/interface/test_dependency_boundaries.py -q
```

## Run Smoke Tests

Run the full smoke suite:

```bash
export PATHBENCH_SMOKE_CACHE=/path/to/shared/cache/pathbench_smoke
uv run pytest -m smoke tests/smoke -q
```

The `PATHBENCH_SMOKE_CACHE` variable is recommended for compute nodes so the
Hugging Face sample data is downloaded once and reused across runs.

Run only the Hugging Face-backed realistic smoke workflows:

```bash
export PATHBENCH_SMOKE_CACHE=/path/to/shared/cache/pathbench_smoke
uv run pytest -q \
  tests/smoke/test_hf_feature_workflows.py \
  tests/smoke/test_hf_mil_benchmarking.py \
  tests/smoke/test_hf_survival_optuna_inference.py
```

Run only the legacy lightweight smoke tests:

```bash
uv run pytest -q \
  tests/smoke/test_feature_extract_cli.py \
  tests/smoke/test_feature_extraction_smoke.py \
  tests/smoke/test_benchmark_cli.py
```

## Run Everything

Run the full repository test suite:

```bash
uv run pytest -q
```

Run unit plus smoke explicitly:

```bash
uv run pytest -q tests/unit tests/smoke
```

Run unit, interface, and smoke explicitly:

```bash
uv run pytest -q tests/unit tests/interface tests/smoke
```

## Quality Checks

Run the standard repository checks before merging:

```bash
uv run ruff check . --fix
uv run ruff format .
uv run ruff check .
uv run pytest -q
```

If you only changed smoke tests and want a narrower pass first:

```bash
uv run ruff check tests/smoke
uv run ruff format --check tests/smoke
uv run pytest -q tests/smoke
```

## Useful Targeted Commands

Run one specific smoke test function:

```bash
uv run pytest tests/smoke/test_hf_mil_benchmarking.py -k binary -q
```

Run one specific unit test function:

```bash
uv run pytest tests/unit/test_inference_heatmaps.py -k persists_h5 -q
```

Show verbose output for debugging:

```bash
uv run pytest tests/smoke/test_hf_survival_optuna_inference.py -vv -s
```

## Notes

- The realistic smoke suite depends on optional packages such as `torch`,
  `pytorch-lightning`, `lazyslide`, `timm`, `huggingface_hub`, `anndata`, and
  `optuna`.
- The smoke fixtures reuse extracted H5 artifacts and prepared MIL bags within
  a test session to avoid duplicate computation.
- Additional smoke-specific notes are documented in
  [tests/smoke/README.md](/exports/path-cutane-lymfomen-hpc/siemen/PathBench_2.0/tests/smoke/README.md).
