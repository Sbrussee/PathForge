# Smoke Suite

This directory contains the realistic Hugging Face-backed smoke suite for PathForge 2.0.

Run the full suite with:

```bash
pytest -m smoke tests/smoke
```

What it covers:

- tile-level feature extraction on a few small WSIs from `RendeiroLab/LazySlide-data`
- slide-level feature aggregation reusing the extracted H5 tile artifacts
- binary MIL benchmarking on extracted WSI bags
- multiclass MIL benchmarking on extracted WSI bags
- continuous survival MIL smoke on `TCGA_READ_subset_TITAN.h5ad`
- discrete survival MIL smoke on `TCGA_READ_subset_TITAN.h5ad`
- binary Optuna smoke on extracted WSI bags
- inference CLI heatmap generation on a trained MIL model

Efficiency rules implemented by the suite:

- Hugging Face assets are cached under `PATHFORGE_SMOKE_CACHE` when set, otherwise `~/.cache/pathforge_smoke`.
- WSI extraction runs once per test session and downstream tests reuse the produced H5 bags.
- TCGA READ survival features are converted once per test session into one-instance MIL bags for reuse.
- Each major session fixture or smoke test writes a timing and memory JSON sidecar next to its temporary artifacts.

Notes:

- The suite skips cleanly when optional runtime dependencies such as `lazyslide`, `torch`, `timm`, or `anndata` are unavailable.
- The survival smoke currently exercises PathForge trainer/model/loss contracts using precomputed slide-level features from the sample dataset as one-instance MIL bags. This keeps the test lightweight while still covering survival task plumbing end to end.
- When `PATHFORGE_SMOKE_REPORT_DIR` is set, each smoke step mirrors its metrics JSON into `steps/` under that directory and the session writes aggregate `smoke_summary.json` and `smoke_summary.md` reports containing intermediate and final artifact paths for investigation.
