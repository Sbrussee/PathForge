# Slide retrieval

PathForge slide retrieval ranks reference slides for each query slide. It
reuses the patch features produced by the normal WSI pipeline; no retrieval
model is trained.

## Setup

Give every dataset one retrieval role:

- ``reference``: included in the search database only.
- ``query``: searched against the database only.
- ``query_reference``: included in both sets, which is useful for leave-one-out
  evaluation.

The ``category`` annotation column is optional for searching, but is required
to calculate label-based retrieval metrics. Set ``exclusion_level`` to prevent
trivial matches in a shared query/reference pool. ``patient`` is the default
and is usually the appropriate choice for clinical data.

The benchmark prepares missing feature artifacts when it runs. Pre-running
feature extraction is therefore optional, but is useful when the same feature
bags will be used for several retrieval experiments. In either case, the
``tile_px``, ``tile_mpp``, and feature extractor selected for retrieval must
identify the feature artifacts that should be used.

## Minimal baseline

The following configuration uses the RGB Yottixel representation and Yottixel
search, the closest general-purpose starting point to the original Yottixel
workflow.

```yaml
experiment:
  project_name: retrieval_baseline
  annotation_file: /data/annotations.csv
  mode: benchmark
  task: slide_retrieval
  aggregation_level: slide

datasets:
  - name: Cohort
    slides_dir: /data/slides
    artifacts_dir: /data/artifacts
    used_for: query_reference

benchmark_parameters:
  tile_px: [256]
  tile_mpp: [0.5]
  feature_extraction: [uni]
  retrieval_representation: [yottixel-rgb]
  search_strategy: [yottixel]

slide_retrieval:
  exclusion_level: patient
```

Run the benchmark with:

```bash
pathforge-benchmark --config retrieval.yaml
```

For a large collection, materialize representations before the benchmark so
they can be reused by several search configurations:

```bash
pathforge retrieval representations --config retrieval.yaml
```

See :doc:`data_preparation` for annotation requirements and
:doc:`slide-retrieval-results-and-metrics` for result files and metric output.

## How a retrieval run works

Each benchmark combination moves through four stages:

1. **Feature extraction** creates the patch feature bags when the requested
   artifacts are missing, or reuses compatible artifacts when they exist.
2. A **representation strategy** selects or transforms the patch feature rows
   that will stand for a slide. The result is cached in the dataset artifact
   store.
3. A **search strategy** builds a database from reference representations and
   prepares each query representation. It applies the configured self-retrieval
   exclusion (``none``, ``slide``, ``case``, or ``patient``) and any
   method-specific candidate filter.
4. The search strategy scores the remaining candidates and returns the ranked
   reference slides.

Representation and search are configured separately. A cached representation
can be reused with another compatible search strategy, but compatibility only
means the data shape is accepted; it does not mean the methods make the same
scientific assumptions.

## Choosing a method

| Starting point | Representation | Search | Selection and ranking approach |
| --- | --- | --- | --- |
| General baseline | ``yottixel-rgb`` | ``yottixel`` | Choose a colour/spatial mosaic, then compare barcode sets. |
| Feature-only Yottixel variant | ``yottixel-features`` | ``yottixel`` | Choose the mosaic from foundation-model features instead of colour. |
| Patch-similarity retrieval | Any patch-vector representation | ``retccl`` | Filter patch matches by cosine similarity and aggregate accepted matches to slide scores. |
| SISH-style retrieval | ``sish_rgb`` | ``sish`` | Select a SISH-style mosaic, index VQ-VAE keys, and verify candidates with Hamming distance. |
| Alternative patch selectors | ``splice-features``, ``splice-rgb``, ``sdm-features``, or ``hshr-features`` | ``yottixel`` or ``retccl`` | Select a smaller or more diverse patch set before the chosen search method runs. |

Start with one representation/search pair. Add combinations to the benchmark
grid only when comparing a clear selection or search hypothesis; changing both
at once makes metric differences difficult to interpret.

## Representations and search methods

### Yottixel RGB and Yottixel search

``yottixel-rgb`` first groups patches by colour and then spatially selects
representatives from those groups. It uses colour to select patch indices, but
the resulting retrieval representation contains the configured
foundation-model feature rows. ``yottixel`` converts the selected rows into
barcodes and ranks slides with the Yottixel median-of-minimum XOR distance.

The upstream code calls its three channel means an “RGB histogram.” PathForge
names that upstream-compatible descriptor ``mean_rgb`` and uses it by default.
It also offers ``histogram_rgb``, a true 256-bin histogram per RGB channel, as
an experimental alternative for the first colour-clustering stage:

```yaml
retrieval_representation:
  yottixel-rgb:
    colour_descriptor: mean_rgb  # upstream-compatible default
    # colour_descriptor: histogram_rgb  # experimental selection variant
```

``yottixel-features`` keeps the same selection and search framework but uses
foundation-model features, rather than colour, for its first selection stage.
It is useful for testing whether the feature-space mosaic is preferable to the
colour-driven baseline.

### RETCCL search

``retccl`` accepts any patch-vector representation. It flattens the reference
database to patch level, finds similar database patches for each query patch,
filters matches below ``cosine_threshold``, and aggregates accepted matches
back to ranked slides. Its score is the average accepted similarity after the
method's weighting and pruning steps.

The default cosine threshold is ``0.7``. Similarity distributions depend on the
feature extractor, so choose and report this setting using a representative
labelled subset rather than treating it as universally calibrated.

### SISH RGB and SISH search

``sish_rgb`` uses a 768-value ``histogram_rgb`` descriptor (256 bins for each
RGB channel) for colour grouping. It then applies SISH white/LBP trash
filtering, spatially selects retained patches, and returns the corresponding
foundation-model feature rows. ``sish`` constructs a VQ-VAE-derived integer
key for each selected patch, uses predecessor/successor index traversal to find
candidates, and filters those candidates by Hamming distance before aggregating
them to slide rankings.

This preserves the central SISH indexing and search workflow. PathForge differs
from the original method in one important respect: its Hamming companion is
packed from the selected foundation-model feature row, whereas upstream SISH
uses DenseNet-121 features. The default ``hamming_thr: 128`` comes from the
upstream implementation, but must be calibrated for the chosen feature
extractor. PathForge SISH is therefore an adaptation, not a paper-level
reproduction.

The released SISH VQ-VAE receives a canonical ``1024 x 1024`` crop at ``0.5
mpp``, centred on each selected source patch. This permits arbitrary
feature-extraction tile geometry while retaining the VQ-VAE field of view;
slides coarser than ``0.5 mpp`` are rejected rather than upsampled. The crop
contract and model identities are stored with the descriptor cache.

SISH requires the upstream VQ-VAE checkpoint and semantic codebook;
``sish_rgb`` also requires its trash classifier. By default PathForge looks in
``model_weights/slide_retrieval``. Configure a different location when needed:

```yaml
slide_retrieval:
  weights_dir: /shared/model_weights/slide_retrieval
```

The directory must contain:

```text
sish_vqvae_checkpoint.pth
sish_codebook.pt
sish_trash_classifier.pkl
```

Precompute VQ-VAE descriptors when several runs will reuse them:

```bash
pathforge retrieval sish-vqvae \
  --config retrieval.yaml \
  --dataset ReferenceSet \
  --slide-id SLIDE_001
```

The normal retrieval run also creates missing descriptors on demand.

### Alternative patch selectors

These strategies change which foundation-model feature rows are retained; they
do not change the feature extractor itself.

- ``hshr-features`` clusters patch feature rows and retains the patch nearest
  each cluster centre. This is a small, close port of the original selection
  logic.
- ``sdm-features`` groups patches by their distance from the global feature
  centroid and selects one reproducible representative from each group. The
  original paper has no official code release, so this is a PathForge reference
  implementation of the published selection idea.
- ``splice-features`` and ``splice-rgb`` use the same SPLICE selection logic,
  driven respectively by foundation-model features or mean RGB values. Their
  output remains the selected foundation-model feature rows.

Use these selectors to test a specific patch-selection hypothesis, then pair
them with ``yottixel`` or ``retccl`` according to whether barcode-set distance
or filtered patch similarity is the intended ranking mechanism.

## Descriptor and representation caches

``mean_rgb`` and ``histogram_rgb`` are separate row-aligned descriptor caches;
neither replaces nor invalidates the other. When they are missing, PathForge
creates them from the source WSI. ``histogram_rgb`` records raw colour
histograms and does not apply trash filtering; SISH filtering is deliberately
performed during ``sish_rgb`` representation creation.

Retrieval representations and SISH VQ-VAE descriptors are also cached. Their
cache metadata records the relevant strategy parameters and source identities,
so an incompatible entry is recomputed instead of silently reused. See
:doc:`slide_retrieval_h5_structure` for the on-disk layout.
