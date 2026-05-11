# PathBench-MIL

PathBench-MIL is a modular benchmarking framework for multiple instance
learning (MIL) in computational pathology. It supports WSI feature extraction,
H5 artifact generation, tile overview reports, MIL benchmarking, hyperparameter
optimization, optional TorchMIL backends, optional metrics backends, and
explainability hooks.

The repository follows the Clean Architecture contract described in
[design.md](design.md): policies and trainers resolve implementations through
PathBench interfaces and registries, while concrete third-party packages live in
adapter modules.

![Design](design.png)

## What PathBench-MIL Does

PathBench-MIL is organized around a config-driven workflow:

1. Read a YAML config.
2. Create an experiment folder with copied annotations and metadata.
3. Build WSI datasets from annotation rows.
4. Generate tiling and feature-extraction combinations.
5. Write per-slide `.h5` artifacts with coordinates, tiling specs, optional tile
   overview images, and extracted features.
6. Train or evaluate MIL models through registry-selected trainers and losses.
7. Run benchmark grids or Optuna optimization using the same registry boundary.

Primary functionality:

- **Feature extraction:** tile WSIs, segment tissue, extract tile features, and
  persist row-aligned H5 artifacts.
- **Single-slide feature extraction:** process one WSI from a larger configured
  dataset, useful for SLURM array jobs.
- **Tile reports:** render PDF reports from stored `tiles_overview` payloads.
- **Benchmarking:** evaluate MIL model/loss combinations from config grids.
- **Optimization:** run Optuna studies over model and training choices.
- **Inference:** lightweight inference CLI placeholder for checkpoint prediction
  outputs.
- **Backends:** native PathBench MIL models or optional TorchMIL models through
  one generic adapter.
- **Metrics/loss adapters:** optional TorchMetrics and TorchSurv integrations.
- **Explainability:** optional heatmap adapter for per-instance MIL scores.

## Installation

**Recommended install** — includes all runtime dependencies (Lazyslide feature
extraction backend, TorchMIL, TorchMetrics, TorchSurv):

```bash
uv sync --extra lazyslide --extra mil-backends
```

For GPU (CUDA 12.8) builds, add `--extra cu128`:

```bash
uv sync --extra lazyslide --extra mil-backends --extra cu128
```

Development install (adds pytest):

```bash
uv sync --extra lazyslide --extra mil-backends --extra dev
```

Individual extras:

| Extra | Installs |
|---|---|
| `lazyslide` | `lazyslide`, `wsidata`, `timm`, `geopandas`, `anndata` |
| `mil-backends` | `torchmil`, `torchmetrics`, `torchsurv` |
| `cu128` | CUDA 12.8 PyTorch builds (via the `pytorch-cu128` index) |
| `gnn` | `torch-geometric` |
| `hf` | `huggingface_hub`, `typer` |
| `dev` | `pytest`, `pytest-cov` |

`mil-backends` installs:

- `torchmil`
- `torchmetrics`
- `torchsurv`

These packages are optional. Native PathBench workflows must remain import-safe
and runnable without them.

## Command Line Entry Points

The package declares these console scripts in `pyproject.toml`:

```bash
pathbench-benchmark --config config.yaml
pathbench-optimize --config config.yaml
pathbench-features --config config.yaml
pathbench-infer --model_path checkpoint.ckpt --input features.pt --output predictions.json
```

The modules can also be called directly:

```bash
python -m pathbench.cli.feature_extraction --config config.yaml --log-level INFO
python -m pathbench.cli.feature_extraction_slide --config config.yaml --dataset DatasetA --input /path/to/slide.svs
python -m pathbench.cli.tiles_report --config config.yaml --log-level INFO
python -m pathbench.cli.benchmark --config config.yaml
python -m pathbench.cli.optimize --config config.yaml
python -m pathbench.cli.inference --model_path checkpoint.ckpt --input features.pt --output predictions.json
```

Use module commands during development when validating CLI changes; they make it
obvious which implementation file is being executed.

## Data Layout

### Annotation CSV

The experiment copies `experiment.annotation_file` into the experiment root as
`annotations.csv`. WSI datasets expect at least these columns:

```csv
dataset,slide,patient,category
TrainingSet,SLIDE_001,PATIENT_001,case
TrainingSet,SLIDE_002,PATIENT_002,control
```

Optional column:

- `fallback_mpp`: positive floating-point microns-per-pixel fallback used when a
  WSI backend cannot read valid base MPP metadata.

Rules:

- `dataset` must match one entry in `datasets[].name`.
- `slide` is matched against files in `datasets[].slides_dir` using
  `{slide}.*`.
- Supported WSI suffixes include `.svs`, `.ndpi`, `.tiff`, `.tif`, and `.mrxs`.
- `patient` and `category` are preserved in WSI metadata and downstream
  grouping.

### Dataset Directories

Each dataset entry points to slide inputs and artifact outputs:

```yaml
datasets:
  - name: TrainingSet
    slides_dir: /data/slides/train
    artifacts_dir: /data/pathbench_artifacts/train
    tissue_annotations_dir: null
    used_for: training
```

`artifacts_dir` is created if needed. Each slide writes one H5 file:

```text
artifacts_dir/{slide_id}.h5
```

## TCGA-Tools Datasets

PathBench can call the `tcga-tools` package to check whether requested datasets
exist in TCGA or TCIA, download metadata first, select the configured task
column, and download image data only when it is missing.

```yaml
datasets:
  - source: gdc
    dataset_names: ["TCGA-LUSC", "TCGA-LUAD"]
    annotation_column: diagnoses.0.vital_status
    metadata_table: clinical_csv
    annotations: ["clinical"]
    datatype: ["wsi"]
    used_for: ["training", "testing"]
```

This allows users to:

- specify TCGA or TCIA datasets directly in the PathBench config
- let PathBench validate those dataset names through `tcga-tools`
- generate a PathBench annotation CSV automatically under `datasets/`
- split one downloaded dataset across multiple roles when `used_for` contains more than one role

To find which columns exist for a dataset, use `tcga-tools` to do a metadata-only
download first and inspect the generated CSV files such as `files_metadata.csv`,
`clinical.csv`, `molecular_index.csv`, or `diagnosis.csv`. The chosen column name
then becomes `annotation_column` in the PathBench config.

## Configuration Reference

Minimal feature extraction config:

```yaml
experiment:
  project_name: example_features
  annotation_file: /data/annotations.csv
  project_root: /data/pathbench_projects
  mode: feature_extraction
  task: null
  report: true
  mixed_precision: true
  num_workers: 8

slide_processing:
  backend: lazyslide
  save_tiles: false
  segmentation_method: otsu
  qc_filters: []

datasets:
  - name: TrainingSet
    slides_dir: /data/slides/train
    artifacts_dir: /data/artifacts/train
    tissue_annotations_dir: null
    used_for: training

benchmark_parameters:
  tile_px: [256]
  tile_mpp: [0.5]
  feature_extraction: [resnet18]
  mil: []

weights_dir: ./pretrained_weights
hf_key: null
```

Top-level sections:

- `experiment`: project lifecycle, task, mode, reporting, and workers.
- `slide_processing`: WSI backend and tissue/tiling behavior.
- `datasets`: slide directories, artifact directories, and dataset roles.
- `benchmark_parameters`: grids for tiling, feature extractors, MIL models,
  losses, activations, and optimizers.
- `mil`: training loop, backend selection, model kwargs, and MIL hyperparameters.
- `metrics`: metric backend selection.
- `explainability`: heatmap backend selection.
- `optimization`: Optuna study settings.

Supported `experiment.mode` values:

- `feature_extraction`
- `benchmark`
- `optimization`

Supported `experiment.task` values:

- `classification`
- `regression`
- `survival`
- `survival_discrete`

`experiment.task` may be omitted only for `feature_extraction` mode.

## Feature Extraction

Feature extraction creates WSI H5 artifacts. It uses:

- `benchmark_parameters.tile_px`
- `benchmark_parameters.tile_mpp`
- `benchmark_parameters.feature_extraction`
- `slide_processing.backend`
- `slide_processing.segmentation_method`
- `experiment.report`

Run all configured datasets and combinations:

```bash
python -m pathbench.cli.feature_extraction --config features.yaml --log-level INFO
```

The policy builds combinations over:

```text
feature_extraction x tile_px x tile_mpp
```

For each slide and combination, PathBench:

1. Validates base MPP.
2. Reuses existing valid coordinates and tiling specs when possible.
3. Segments or loads tissue polygons.
4. Extracts tile coordinates.
5. Writes `coords` and `tiling_spec` to H5.
6. Optionally writes `tiles_overview` when `experiment.report: true`.
7. Extracts tile features.
8. Writes feature matrices row-aligned with coordinates.

Coordinates are stored as `int32` arrays shaped `(N, 5)`:

```text
[x_level0, y_level0, read_w, read_h, level]
```

Feature matrices are stored as floating arrays shaped `(N, D)`, where rows align
exactly with `coords`.

## Single-Slide Feature Extraction

Use this for cluster jobs where each task processes one slide:

```bash
python -m pathbench.cli.feature_extraction_slide \
  --config features.yaml \
  --dataset TrainingSet \
  --input /data/slides/train/SLIDE_001.svs \
  --log-level INFO
```

Requirements:

- `--dataset` must match one configured dataset name.
- `--input` must exist.
- The source annotation CSV must contain exactly one row matching the selected
  dataset and slide stem.

The CLI rewrites the project annotations for that invocation to a single row and
then runs all configured feature extraction combinations for the selected slide.
When `SLURM_JOB_ID` is present, the project name is suffixed with the job id to
avoid collisions between array jobs.

## Tile Overview Reports

If `experiment.report: true`, feature extraction writes `tiles_overview` image
bytes into the slide H5 files. Generate PDF reports after extraction with:

```bash
python -m pathbench.cli.tiles_report --config features.yaml --log-level INFO
```

The report CLI derives bag ids from all configured `tile_px` and `tile_mpp`
combinations. A bag id has this format:

```text
{tile_px}px_{tile_mpp:g}mpp
```

Example:

```text
256px_0.5mpp
```

The report CLI skips dataset/bag combinations where no overview exists yet and
returns a non-zero exit code only when unexpected report generation failures
occur.

## H5 Artifact Contract

PathBench writes one H5 artifact per slide. The layout is backend-agnostic and
row-aligned:

- coordinates: `(N, 5)` `int32`
- tiling spec: JSON-compatible metadata
- features: `(N, D)` floating matrix
- tile overview: compressed image bytes as a one-dimensional `uint8` payload

Invariants:

- Coordinates and features share row order.
- Tiling specs include `tile_px`, `tile_mpp`, `stride_px`, and
  `coord_space="level0"`.
- Feature extraction can skip recomputation when valid rows already exist.
- Reports are optional and should not affect feature matrix row alignment.

## MIL Backends

PathBench supports two MIL backend modes:

- `native`: use PathBench model classes registered directly in `MODELS`.
- `torchmil`: use one generic TorchMIL adapter registered under the PathBench
  model key `torchmil`.

TorchMIL, TorchMetrics, and TorchSurv are optional integrations. They are not
required to import PathBench or to run native workflows. Package-specific imports
are confined to:

- `src/pathbench/adapters/...`
- `src/pathbench/utils/optional/...`

Trainer, policy, config, and domain code select implementations through
configuration and registries. They do not call `torchmil`, `torchmetrics`, or
`torchsurv` directly.

### Native Backend

Use `native` when you want existing PathBench models and no optional MIL backend
dependency.

```yaml
experiment:
  project_name: native_benchmark
  annotation_file: /data/annotations.csv
  mode: benchmark
  task: classification

mil:
  backend: native
  batch_size: 1
  epochs: 20

metrics:
  classification_backend: native

benchmark_parameters:
  feature_extraction: [resnet18]
  mil: [AttentionMIL]
  loss: [CrossEntropyLoss]
```

Native datasets may return legacy tuple batches:

```python
bag, target = dataset[index]
```

where `bag` is a finite floating tensor shaped `[N, D]` for one slide bag, and
`target` is the task label.

### TorchMIL Backend

Use `torchmil` when you want TorchMIL models while keeping PathBench's trainer,
policy, dataset, and registry contracts.

```yaml
experiment:
  project_name: torchmil_benchmark
  annotation_file: /data/annotations.csv
  mode: benchmark
  task: classification

mil:
  backend: torchmil
  torchmil_model: ABMIL
  torchmil_model_kwargs:
    in_shape: [1024]
    out_shape: 2
  use_torchmil_collate: true
  batch_size: 4
  epochs: 20

metrics:
  classification_backend: torchmetrics

benchmark_parameters:
  feature_extraction: [resnet18]
  mil: [torchmil]
  loss: [CrossEntropyLoss]
```

Important rules:

- `benchmark_parameters.mil` contains the PathBench registry key `torchmil`.
- `mil.torchmil_model` contains the TorchMIL model class name.
- `mil.torchmil_model_kwargs` are forwarded to the TorchMIL constructor.
- `mil.use_torchmil_collate: true` enables padded dict batches compatible with
  TorchMIL semantics.
- The generic `TorchMILBackendModel` is the only PathBench model adapter for
  TorchMIL models.

If `torchmil` is selected but unavailable, config validation raises:

```text
MIL backend 'torchmil' selected, but 'torchmil' is not installed. Install torchmil or set mil.backend='native'.
```

## Canonical MIL Bag Batch

The TorchMIL integration introduces a canonical batch schema shared by adapters:

```python
batch = {
    "X": features,       # float tensor [B, N, D]
    "Y": labels,         # labels [B] or survival target dict
    "mask": mask,        # optional bool tensor [B, N], true = real instance
    "coords": coords,    # optional tensor [B, N, 2]
    "adj": adj,          # optional tensor [B, N, N]
    "y_inst": y_inst,    # optional instance labels [B, N]
}
```

Shape and value contracts:

- `X` is floating point, finite, and shaped `[N, D]` for a single bag or
  `[B, N, D]` for a batch.
- `mask` is boolean or integer binary and shaped `[B, N]`.
- `coords` is shaped `[B, N, 2]` and stores x/y instance coordinates.
- `adj` is shaped `[B, N, N]`; avoid this for large WSI bags unless the selected
  model requires graph structure.
- Padded instances are zero-filled and marked `false` in `mask`.

Legacy datasets returning `(bag, target)` remain supported. The TorchMIL collate
adapter converts them into canonical dictionaries only when the TorchMIL backend
path is selected.

## Benchmarking

Benchmark mode evaluates combinations from `benchmark_parameters`.

Run:

```bash
python -m pathbench.cli.benchmark --config benchmark.yaml
```

Minimal native benchmark:

```yaml
experiment:
  project_name: native_benchmark
  annotation_file: /data/annotations.csv
  mode: benchmark
  task: classification

mil:
  backend: native
  lr: 0.0001
  weight_decay: 0.00001
  batch_size: 1
  epochs: 20

metrics:
  classification_backend: native

benchmark_parameters:
  feature_extraction: [resnet18]
  mil: [AttentionMIL]
  loss: [CrossEntropyLoss]
  activation_function: [ReLU]
  optimizer: [Adam]
```

For a TorchMIL benchmark, every run resolves:

1. `MODELS.get("torchmil")`
2. `TorchMILBackendModel(...)`
3. `mil.torchmil_model`, for example `ABMIL`
4. `mil.torchmil_model_kwargs`, forwarded to the TorchMIL constructor
5. `LightningTrainer`, which accepts canonical dict batches or legacy tuples

This keeps TorchMIL as one backend plugin. Benchmarking policies still interact
with PathBench registries and trainer/model interfaces; they do not import or
call TorchMIL directly.

To compare native and TorchMIL backends, use separate config files. Native
models and TorchMIL models may not share constructor kwargs, so separate configs
keep model construction explicit and reproducible.

## Optimization

Optimization mode runs Optuna studies while preserving the same registry
boundary as benchmarking.

Run:

```bash
python -m pathbench.cli.optimize --config optimize.yaml
```

Example:

```yaml
experiment:
  project_name: torchmil_optimization
  annotation_file: /data/annotations.csv
  mode: optimization
  task: classification

mil:
  backend: torchmil
  torchmil_model: ABMIL
  torchmil_model_kwargs:
    in_shape: [1024]
    out_shape: 2
  batch_size: 4

optimization:
  study_name: torchmil_abmil_search
  objective_metric: val_loss
  objective_mode: min
  sampler: TPESampler
  pruner: HyperbandPruner
  trials: 50

benchmark_parameters:
  feature_extraction: [resnet18]
  mil: [torchmil]
  loss: [CrossEntropyLoss]
```

TorchMIL integration affects optimization in these places:

- Search spaces may include `model = "torchmil"` as a PathBench registry key.
- Search spaces may include `mil.torchmil_model`, for example `ABMIL`, `DSMIL`,
  or another installed TorchMIL class.
- Trial parameters may update `mil.torchmil_model_kwargs`, such as hidden
  dimensions or dropout, if supported by the selected TorchMIL model.
- Objective metrics can be native, TorchMetrics-backed, or TorchSurv-backed,
  selected by config.

The optimization policy should remain package-agnostic: it selects registry keys
and config values, not concrete TorchMIL classes.

## Metrics

Classification metric backend:

```yaml
metrics:
  classification_backend: torchmetrics
```

The default implementation key is `torchmetrics`. It is optional and resolved
through the classification metrics registry. If selected but unavailable,
validation raises:

```text
Classification metrics backend requires 'torchmetrics'. Install torchmetrics or choose another classification metrics backend.
```

Native workflows can opt out:

```yaml
metrics:
  classification_backend: native
```

Continuous survival backend:

```yaml
metrics:
  survival_continuous_backend: torchsurv
```

If `torchsurv` is selected but unavailable, validation raises:

```text
Continuous survival backend requires 'torchsurv'. Install torchsurv or choose another survival backend.
```

## Survival Tasks

Continuous survival support is explicit:

```yaml
experiment:
  task: survival

mil:
  backend: torchmil
  torchmil_model: SomeSurvivalCapableModel
  torchmil_model_kwargs:
    in_shape: [1024]
    out_shape: 1

metrics:
  survival_continuous_backend: torchsurv
```

PathBench expects continuous survival outputs to normalize to risk or log-hazard
tensors shaped `[B]` or `[B, 1]`. Targets should follow the existing survival
loss contract:

```python
target = {
    "time": time,    # float tensor [B]
    "event": event,  # binary tensor [B], one = observed event, zero = censored
}
```

Discrete survival outputs must be shaped `[B, T]`, where `T` is the number of
time bins. Unsupported model/task combinations should be blocked during config
or model construction rather than failing inside a training step.

## Explainability And Heatmaps

The TorchMIL heatmap explainer is optional:

```yaml
explainability:
  heatmap_backend: torchmil
```

It consumes per-instance scores plus coordinates:

```python
payload = {
    "coords": coords,                  # tensor [N, 2]
    "instance_scores": scores,         # tensor [N]
    "mask": optional_mask,             # optional tensor [N]
}
```

The output is a `HeatMap` object containing coordinates and normalized finite
scores in `[0, 1]`. Prediction heatmaps should be stored in a dedicated H5
prediction namespace rather than overloading existing tile overview datasets.

## Inference

The inference CLI currently provides a small stable surface for checkpoint-style
prediction workflows:

```bash
python -m pathbench.cli.inference \
  --model_path checkpoint.ckpt \
  --input /data/artifacts/SLIDE_001.h5 \
  --output predictions.json
```

The current implementation writes a JSON prediction payload. It can also attach
an inference heatmap to a slide H5 artifact when per-instance scores are
available from a backend model.

TorchMIL heatmap inference example:

```bash
python -m pathbench.cli.inference \
  --model_path /models/abmil.ckpt \
  --input /data/artifacts/SLIDE_001.h5 \
  --output /data/predictions/SLIDE_001.json \
  --heatmap-backend torchmil \
  --bag-id 256px_0.5mpp \
  --scores /data/predictions/SLIDE_001_attention.npy \
  --heatmap-name abmil_attention \
  --heatmap-output /data/predictions/SLIDE_001_heatmap.json
```

Inputs:

- `--input`: slide H5 artifact. When `--coords` is omitted, PathBench reads
  `bags/{bag_id}/coords` and uses the first two columns as level-0 x/y
  coordinates.
- `--scores`: `.npy`, `.npz`, or `.json` vector shaped `[N]` containing
  per-instance attention, attribution, or instance score values.
- `--coords`: optional `.npy`, `.npz`, or `.json` matrix shaped `[N, 2]`. Use
  this when scores do not align with H5 bag coordinates.
- `--mask`: optional `.npy`, `.npz`, or `.json` boolean/binary vector shaped
  `[N]`; false entries are removed before persistence.
- `--heatmap-backend`: use `torchmil` to resolve the `torchmil_heatmap`
  explainer through the `EXPLAINERS` registry.
- `--heatmap-name`: H5 namespace for this prediction heatmap.
- `--heatmap-output`: optional JSON sidecar for downstream tools that do not
  read H5.

Output H5 namespace:

```text
bags/{bag_id}/predictions/heatmaps/{heatmap_name}/coords
bags/{bag_id}/predictions/heatmaps/{heatmap_name}/scores
bags/{bag_id}/predictions/heatmaps/{heatmap_name}/metadata
```

Persisted heatmap contracts:

- `coords`: floating array shaped `(N, 2)`.
- `scores`: `float32` array shaped `(N,)`, finite and normalized to `[0, 1]`.
- `metadata`: JSON with backend, explainer key, model path, score path, optional
  coordinate path, optional mask path, score range, and coordinate space.

This path still follows Clean Architecture: inference resolves the heatmap
implementation through `EXPLAINERS`; TorchMIL-specific behavior remains in
`pathbench.adapters.torchmil.heatmap_explainer`.

## Registries And Extensibility

PathBench uses registries as the plugin backbone:

- `MODELS`
- `LOSSES`
- `TRAINERS`
- `TASKS`
- `EXPLAINERS`
- `FEATURE_EXTRACTORS`
- `SLIDE_PROCESSORS`
- `CLASSIFICATION_METRICS`
- `SURVIVAL_METRICS`
- `SURVIVAL_LOSSES`

Register new implementations by importing a module that calls the relevant
registry decorator or explicit registration function. Keep concrete package
logic in adapter/infrastructure modules and expose it through PathBench
interfaces.

Example native model registration:

```python
from pathbench.core.models.mil_base import MILModelBase
from pathbench.utils.registries import MODELS


@MODELS.register("MyMIL")
class MyMIL(MILModelBase):
    ...
```

Optional backends should be registered conditionally through dynamic registry
population so missing packages do not break imports.

## Clean Architecture Guarantees

The integration is intentionally interface-first:

- Domain/core contracts remain stable: `MILModelBase`, `TrainerBase`,
  `ExplainerBase`, and the bag schema.
- Optional package guards live under `pathbench.utils.optional`.
- TorchMIL model/collate/output/heatmap code lives under
  `pathbench.adapters.torchmil`.
- TorchMetrics and TorchSurv code lives under `pathbench.adapters.metrics`.
- Dynamic registry population conditionally registers optional implementations
  only when packages are installed.
- Trainer code accepts canonical batches but does not import `torchmil`,
  `torchmetrics`, or `torchsurv`.
- Policies continue to use `MODELS`, `LOSSES`, `TRAINERS`, and other
  registries.

Architecture tests enforce that direct optional-package imports stay confined to
adapter and optional-guard modules.

## Troubleshooting

`MIL backend 'torchmil' selected, but 'torchmil' is not installed.`

: Install `.[mil-backends]`, install `torchmil`, or set `mil.backend: native`.

`Classification metrics backend requires 'torchmetrics'.`

: Install `torchmetrics` or set `metrics.classification_backend: native`.

`Continuous survival backend requires 'torchsurv'.`

: Install `torchsurv` or choose another survival backend.

`Feature extractor '<name>' is not registered.`

: Ensure the extractor package is installed and dynamic registries are populated
  before config validation. For Lazyslide/timm extractors, install
  `.[lazyslide]`.

`cfg.experiment.project_root must be an absolute path.`

: Use an absolute path such as `/data/pathbench_projects`. If omitted,
  PathBench writes under the repository-level `experiments/` directory.

No slides are found for a dataset.

: Check that annotation `dataset` values match `datasets[].name`, that
  `slides_dir` exists, and that slide filenames use `{slide_id}.svs` or another
  supported WSI suffix.

## Testing

Run focused tests for the backend integration and documentation:

```bash
uv run pytest -q \
  tests/unit/test_torchmil_optional.py \
  tests/unit/test_bag_schema_collate.py \
  tests/unit/test_torchmil_task_output.py \
  tests/unit/test_lightning_batch_unpack.py \
  tests/unit/test_torchmil_architecture.py \
  tests/unit/test_torchmil_docs.py \
  tests/unit/test_config_validation.py
```

Run the standard repository checks before merging:

```bash
uv run ruff check . --fix
uv run ruff format .
uv run ruff check .
uv run pytest -q
```

For CI, use at least two profiles:

- Base profile without `torchmil`, `torchmetrics`, or `torchsurv`: verifies that
  imports, native configs, and missing-backend errors behave correctly.
- Optional-backend profile with `.[mil-backends]`: verifies TorchMIL
  construction, TorchMIL collation, TorchMetrics classification metrics,
  TorchSurv survival losses/metrics, and heatmap explanation.
