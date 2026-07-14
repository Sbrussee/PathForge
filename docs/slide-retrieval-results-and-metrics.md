# Slide Retrieval Results And Metrics

This page explains where slide-retrieval benchmark outputs are stored in a PathForge project, how to find the files for one specific benchmark combination, and how the saved metrics are structured.

## Overview

Slide-retrieval outputs are written in two places:

1. The **project folder** stores run outputs and evaluation summaries.
2. The **dataset `artifacts_dir`** stores cached retrieval representations in HDF5 files.

If you want to inspect the outcome of one benchmark run, start in the **project folder**.

## Where Results Are Stored

Assume your config contains:

```yaml
experiment:
  project_root: /path/to/my_project
```

Then slide-retrieval outputs are written under:

```text
/path/to/my_project/eval_slide_retrieval/
```

The full layout for one combination is:

```text
<project_root>/
  eval_slide_retrieval/
    <tiling_id>_<feature_name>/
      <retrieval_representation>/
        vis_retrieval_representation/
        <search_strategy>/
          run_<run_hash>/
            manifest.json
            query_results.xlsx
            evaluation_metrics.json
            query_results.csv
            vis_retrieval_results/
              <query_slide_id>.png
```

## How A Combination Maps To A Folder

The path is built from the benchmark combination:

- `tiling_id = "{tile_px}px_{tile_mpp}mpp"`
- `feature_name = feature_extraction` or `"{feature_extraction}_{color_norm}"`
- `retrieval_representation = benchmark_parameters.retrieval_representation`
- `search_strategy = benchmark_parameters.search_strategy`

Example:

```yaml
benchmark_parameters:
  tile_px: [256]
  tile_mpp: [0.5]
  feature_extraction: [uni2]
  color_norm: [null]
  retrieval_representation:
    - yottixel-features:
        perc_selected: 1.0
  search_strategy:
    - yottixel
```

This combination will be stored under:

```text
<project_root>/eval_slide_retrieval/256px_0.5mpp_uni2/yottixel-features/yottixel/
```

Inside that folder, each concrete run gets its own directory:

```text
run_<run_hash>/
```

The `run_hash` is an 8-character hash derived from the saved manifest content, so different settings produce different run directories. In older notes this may be referred to as `run_{jhash}`; it is the same concrete run folder concept.

## Files Inside One Run Folder

A current run folder can contain:

```text
run_<run_hash>/
  manifest.json
  query_results.xlsx
  evaluation_metrics.json
  vis_retrieval_results/
    <query_slide_id>.png
```

The first two files are written by the benchmark run. `evaluation_metrics.json` is added by the evaluation workflow. `vis_retrieval_results/` is added only when the retrieval-results visualization is requested. Legacy or test runs may contain `query_results.csv` instead of, or next to, `query_results.xlsx`; current benchmark runs write `query_results.xlsx`, and the evaluator/visualizer can still read the legacy CSV form.

### `manifest.json`

This file stores the run configuration and a few summary counters. Typical fields are:

- `tiling_id`
- `aggregation_level`
- `feature_extraction`
- `slide_representation`
- `slide_representation_params`
- `search_method`
- `search_params`
- `representation_id`
- `exclusion_level`
- `num_queries`
- `num_reference_items`
- `top_k_saved`

Use this file when you want to verify exactly which settings created the run.

### `query_results.xlsx`

This file stores the ranked retrieval results for every query sample.

Each row is one query, with columns like:

```text
query_sample_id
rank_1_sample_id
rank_1_score
rank_2_sample_id
rank_2_score
...
rank_k_sample_id
rank_k_score
```

Meaning:

- `query_sample_id`: the sample used as the query
- `rank_n_sample_id`: the retrieved sample at rank `n`
- `rank_n_score`: the similarity or ranking score returned by the search strategy for that hit

So if you want to inspect the retrieval list for one sample, find its row in `query_results.xlsx`.

### `evaluation_metrics.json`

After running the evaluation workflow, PathForge writes:

```text
<run_dir>/evaluation_metrics.json
```

This file contains:

- run metadata
- the discovered combination
- the manifest
- the computed metrics

### Visualization Folders

Visualization outputs are written as flattened `vis_<visualization_name>` folders.

Run-level retrieval result visualizations are stored inside the concrete run folder:

```text
<run_dir>/vis_retrieval_results/
```

This folder contains one PNG per selected query slide:

```text
<run_dir>/vis_retrieval_results/<query_slide_id>.png
```

Each image shows the query slide together with the top retrieved hits rendered from `query_results.xlsx` or legacy `query_results.csv`. If a visualization subset file is configured, only those query slides are rendered.

Retrieval representation visualizations are stored at the representation root because
they do not belong to one search run:

```text
<project_root>/eval_slide_retrieval/<tiling_id>_<feature_name>/<retrieval_representation>/vis_retrieval_representation/
```

This folder contains one PNG per selected slide:

```text
<project_root>/eval_slide_retrieval/<tiling_id>_<feature_name>/<retrieval_representation>/vis_retrieval_representation/<slide_id>.png
```

At the top level it looks like:

```json
{
  "task": "slide_retrieval",
  "run_dir": "...",
  "label_column": "category",
  "aggregation_level": "slide",
  "combo_cfg": { "example": "..." },
  "manifest": { "example": "..." },
  "metrics": {
    "hit_at_5": { "example": "..." },
    "precision_at_5": { "example": "..." }
  }
}
```

## How Metrics Are Stored Inside `evaluation_metrics.json`

Each metric is stored as a payload under `metrics.<metric_name>`.

For most retrieval metrics, the payload contains:

```json
{
  "k": 5,
  "macro": 0.81,
  "micro": 0.79,
  "per_label": {
    "class_a": 0.85,
    "class_b": 0.77
  },
  "counts": {
    "num_queries": 120,
    "num_evaluable_queries": 120,
    "num_labels": 6
  },
  "counts_per_label": {
    "class_a": 40,
    "class_b": 20
  }
}
```

Meaning:

- `k`: the cutoff used for the metric
- `macro`: the unweighted mean over labels/classes
- `micro`: the mean weighted by the number of queries per label
- `per_label`: the metric value for each label separately
- `counts.num_queries`: total number of evaluated queries
- `counts.num_evaluable_queries`: number of queries that could actually be evaluated
- `counts.num_labels`: number of labels present
- `counts_per_label`: number of queries for each label

## Available Slide-Retrieval Metrics

These metrics are currently implemented for slide retrieval:

- `hit_at_k`
- `precision_at_k`
- `map_at_k`
- `ndcg_at_k`
- `mmv_at_k`
- `macro_f1_at_k`

In config files you request concrete values such as:

```yaml
evaluation:
  label_column: category
  metrics:
    - hit_at_5
    - precision_at_5
    - map_at_5
    - ndcg_at_5
    - mmv_at_5
    - macro_f1_at_5
```

## What The Metrics Mean

### `hit_at_k`

Checks whether at least one of the top-`k` retrieved samples has the same label as the query.

- High value means: the correct class appears somewhere in the top `k`
- Low value means: the correct class is often missing from the top `k`

### `precision_at_k`

Measures how many of the top-`k` retrieved samples have the same label as the query.

For query $q$, let $r_i(q)$ be 1 when the result at rank $i$ has the same
label as the query and 0 otherwise. Then:

$$
\operatorname{Precision@k}(q) = \frac{1}{k}\sum_{i=1}^{k} r_i(q)
$$

- High value means: a large fraction of the top `k` hits are relevant
- Low value means: the retrieved list contains many irrelevant samples

### `map_at_k`

Mean Average Precision at `k`. This rewards relevant hits appearing early in the ranking.

- High value means: relevant items are not only retrieved, but retrieved near the top
- Low value means: relevant items are late in the ranking or absent

### `ndcg_at_k`

Normalized Discounted Cumulative Gain at `k` using binary relevance.

- High value means: the ranking is close to the ideal ordering
- Low value means: relevant hits appear too late or too infrequently

Compared with `map_at_k`, `ndcg_at_k` also explicitly compares the observed ranking against the best possible ranking given the available reference pool.

### `mmv_at_k`

Majority-match vote at `k`. The labels of the top-`k` hits are used as votes, and the majority label is compared with the query label.

- High value means: the top `k` neighbors mostly agree on the correct class
- Low value means: the neighborhood is mixed or dominated by the wrong class

This behaves more like a `k`-nearest-neighbor classification score than a pure ranking score.

### `macro_f1_at_k`

Computes a label prediction from the top-`k` hits, then evaluates macro F1 across labels.

- High value means: the system predicts labels well across all classes
- Low value means: one or more classes are poorly recovered

This is especially useful when class balance matters, because macro F1 gives equal weight to each label.

## Macro vs Micro

Most retrieval metrics report both `macro` and `micro`. If $M_c$ is the
metric for label $c$, $n_c$ is its number of evaluable queries, and $C$ is the
set of labels, the reported aggregates are:

$$
M_{\mathrm{macro}} = \frac{1}{|C|}\sum_{c \in C} M_c,
\qquad
M_{\mathrm{micro}} = \frac{\sum_{c \in C} n_c M_c}{\sum_{c \in C} n_c}
$$

- `macro`: average over labels, so rare labels count equally with common labels
- `micro`: weighted by the number of queries per label, so common labels have more influence

If your dataset is imbalanced, it is useful to look at both:

- use `macro` to understand balanced per-class performance
- use `micro` to understand overall query-weighted performance

## Finding The Results For One Specific Combination

To find one combination:

1. Start in `<project_root>/eval_slide_retrieval/`
2. Find the folder matching `<tiling_id>_<feature_name>`
3. Open the subfolder matching the retrieval representation
4. Open the subfolder matching the search strategy
5. Open the relevant `run_<run_hash>` folder
6. Check:
   - `manifest.json` for the exact settings
   - `query_results.xlsx` for ranked outputs
   - `evaluation_metrics.json` for computed metrics
   - `vis_retrieval_results/` for run-level retrieval visualizations

If you have multiple `run_*` folders for the same combination, compare their `manifest.json` files first.

## Cached Retrieval Representations

The final run results live in the project folder, but PathForge also caches retrieval representations in each dataset's `artifacts_dir`.

Those cached files are stored at:

```text
<artifacts_dir>/slide_retrieval/<aggregation_level>/<sample_id>.h5
```

These HDF5 files are not the final benchmark result files. They are intermediate cached retrieval representations used to avoid recomputing representations for every run.

## Notes

- Ranked retrieval results are currently written as `query_results.xlsx`.
- Evaluation metrics are currently written as `evaluation_metrics.json`.
- There is a helper in the codebase to flatten metrics to CSV, but the default evaluation writer currently writes JSON.
