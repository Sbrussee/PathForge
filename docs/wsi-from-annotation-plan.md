# Plan: shared annotation-based WSI construction

Status: implemented and checked with pytest in the repository's `.venv`.
The focused WSI/processor suite passed 72 tests. Full-suite job 25733172
found four integration failures caused by assuming legacy samples expose
`annotations_df`; that compatibility issue is now fixed and covered by a
regression test. The planner and slide-retrieval integration suite passed
39 tests after the fix. Six other full-suite failures were reproduced on the
parent commit. The precomputed SISH smoke fixture now uses canonical `BagSample`
members with category and patient metadata, and its smoke test passes. The
embedding smoke comparison now allows float32 reduction rounding; it passes
against the saved failed-run artifact (maximum difference 3.58e-7). Its origin
relative to the parent commit remains unclassified because that baseline run
was blocked by network access. A complete smoke rerun is still required.
The smoke batch script accepts `PATHFORGE_TEST_SCOPE=all-smoke` and uses the
repository's `.venv`. The full-suite script now records actual failure exit
codes. Shell syntax and failure-status handling were checked. The new Ruff
import-order finding was fixed; existing findings remain elsewhere.

The branch was fast-forwarded to remote commit `8033920` (SISH quality extraction)
after the user fetched it. Local fixes were preserved, including import formatting
in the overlapping SISH regression test. The combined RGB/SISH, planner, retrieval
integration, and precomputed SISH smoke checks passed all 77 tests after the update.

## Problem and scope

PathForge's processor already selects native MPP before `wsi.fallback_mpp`.
Five retrieval construction sites omit the fallback entirely. Dataset loading,
the single-slide feature CLI, and visualization independently interpret annotation
fields, giving different defaults and validation.

This plan targets `PathForge/src/pathforge`. The sibling PathBench2.0 tree is a
separate implementation and is outside this change. Existing direct `WSI(...)`
callers must remain supported.

## Proposed constructor contract

Add `WSI.from_annotation(row, *, slide_path, artifact_path)` to the existing
dataclass in `core/datasets/wsi_dataset.py`.

- Accept a pandas Series or a mapping representing one annotation row.
- Require a non-null, non-blank slide ID; convert valid IDs to strings. Do not
  infer identity from the path: staged files may have different filenames.
- Default a missing/null/blank patient to the slide ID and category to `""`.
  Convert present values to strings. Keep these rules in one place.
- Accept positive finite scalar fallback MPP values, including numeric strings.
  Missing, null, blank, malformed, zero, negative, infinite, boolean, and
  non-scalar values become `None`. Log invalid supplied values with slide context;
  ordinary missing values need no warning. Preserve the existing tolerant policy
  rather than making invalid optional MPP fatal during construction.
- Store a detached dictionary copy of the complete row in `annotations`.
  Keep original values there; typed fields hold normalized values. Processing
  must read typed fields and never re-parse `annotations`.
- Use the supplied paths, converted to `Path`, without discovering or opening
  files. An annotated `wsi_path` must not override an explicit staged path.
- Give `annotations` its own empty dictionary default. Preserve the order and
  meaning of every existing positional parameter, including `_obj`; introduce
  the field as keyword-only or append it after existing fields. Do not introduce
  validation in `__post_init__` that would change direct-constructor behavior.

## Impact map

Paths below are relative to `src/pathforge/`.

| Area | Location | Required change |
| --- | --- | --- |
| WSI and dataset | `core/datasets/wsi_dataset.py`: `WSI`, `WSIDataset._build_samples` | Add constructor and annotations field; replace inline field/fallback parsing. Preserve existing source-file discovery. |
| Single-slide features | `cli/features_slide.py`: `_parse_fallback_mpp`, `_build_single_slide_wsi` | Remove duplicate parser and sample lookup; construct from the selected row and explicit input path. |
| Mean RGB | `slide_retrieval/representation_strategies/mean_rgb.py`: sample resolver and `_create_slide_patch_mean_rgb` | Resolve the matching row and pass a fully populated WSI into pixel computation. |
| RGB histograms | `slide_retrieval/representation_strategies/histogram_rgb.py`: sample resolver and `_create_slide_histograms` | Use the same row resolver and WSI constructor. |
| SISH RGB filtering | `slide_retrieval/representation_strategies/strategies/sish_rgb.py`: `_retained_rows` | Carry the member slide's annotation context into trash filtering. |
| SISH full-slide VQ-VAE descriptors | `slide_retrieval/search_strategies/strategies/sish/sish_vqvae_descriptors.py`: sample resolver and `_create_slide_patch_sish_vqvae_latent` | Carry row/WSI context into canonical crop reads. |
| SISH selected-patch encoding | `slide_retrieval/search_strategies/strategies/sish/sish_precompute.py`: `_SelectedPatchSpec`, `_build_selected_patch_specs`, `_encode_selected_patch_specs` | Resolve context while the retrieval sample's dataset is available; retain it through grouping and encoding. |
| Visualization | `slide_retrieval/visualization/service.py`: asset resolution and WSI creation near lines 452, 521, 584 | Construct from asset annotation metadata for thumbnails, tiles, and overlays; remove duplicate fallback interpretation. |
| Standalone retrieval CLIs | `cli/retrieval_mean_rgb.py`, `cli/retrieval_histogram_rgb.py`, `cli/retrieval_sish_vqvae.py` | Verify dataset identity and explicit input/artifact paths reach the shared resolver. Public CLI options can stay the same. |
| Retrieval orchestration | `slide_retrieval/representation_workflow.py`, `search_strategies/strategies/sish/sish_search.py` | Audit propagation of dataset identity and active annotation context; adjust only where required. |
| Processor | `core/slide_processing/lazyslide/processor.py` | Retain slide opening and native-MPP-first selection. Add regression coverage; annotation reading does not belong here. |

The five current omissions are mean RGB, histograms, SISH filtering, full-slide
VQ-VAE encoding, and selected-patch VQ-VAE encoding. Visualization already forwards
fallback MPP, but has separate parsing and three WSI construction sites.

## Implementation sequence

1. Implement and test the constructor contract, then migrate `WSIDataset` and
   the single-slide feature builder. Update helper fixtures that currently omit
   the slide key even though production annotation rows contain it.
2. Add a shared retrieval annotation resolver in a neutral retrieval module,
   rather than adding more cross-strategy dependencies on private mean-RGB
   helpers. Reuse existing dataset/column helpers where practical. Select by
   `(dataset, slide ID)` and require exactly one matching row. Report missing
   rows or duplicates with both identifiers; never silently choose the first.
3. Load/index annotations once per operation, outside patch loops. Prefer the
   active experiment annotation frame when available; standalone CLIs can load
   `config.experiment.annotation_file`. Pass context explicitly rather than
   storing a global cache. Thread the context through retrieval orchestration
   where needed, preserving the experiment's annotation snapshot.
4. Keep source-path resolution separate from row selection. Explicit staged
   paths take precedence; retain current artifact-path overrides. Build the WSI
   with the selected row and resolved paths, then pass it to pixel-reading
   helpers. Missing stored descriptors trigger this work; cached descriptor
   reads should still avoid opening slides or requiring annotation lookup.
5. Migrate both RGB paths, SISH filtering, and both SISH encoding paths. For
   selected-patch SISH, retain dataset identity or the constructed WSI through
   patch specs and grouping; resolving only by slide ID at encoding time loses
   necessary context.
6. Migrate visualization. Its current asset lookup filters by slide ID and takes
   `iloc[0]`; use dataset context when available and report ambiguity when it is
   absent. Reuse the copied row for WSI construction and displayed metadata.
   Preserve visualization from cached artifacts when source slides are absent.
7. Run focused tests, relevant retrieval smoke tests, and Ruff. Inspect imports
   to verify annotation I/O stays outside the processor and the WSI constructor.

## Regression coverage

- Constructor: valid numeric/string fallback; absent key, `None`, pandas null,
  NaN, empty string, malformed string, zero, negative, positive/negative infinity,
  boolean, and sequence values. Verify default patient/category and invalid IDs.
- Compatibility: existing positional and keyword `WSI(...)` calls, optional
  fallback and `_obj`, independent annotations defaults, detached row copy, and
  typed-field authority after annotation metadata is edited.
- Selection: identical slide IDs in different datasets select different rows;
  zero or multiple matches fail clearly. Grouped patient/case samples resolve
  each physical member's own annotation rather than the group's metadata.
- Staging: an explicit source and artifact path survive construction even when
  the annotation names another source path or regular slides are unavailable.
- RGB processing: run mean RGB and histograms from annotation lookup through
  actual processor MPP normalization with a mocked slide reader whose native
  MPP is missing. Verify successful pixel reads and persisted descriptors. A
  processor stub that merely records `fallback_mpp` is insufficient by itself.
- Processor: native MPP takes priority, fallback supplies missing MPP, and
  missing native MPP plus invalid/missing fallback raises the expected error.
- SISH: check fallback propagation in filtering and both encoding routes, with
  a staged source; mock models to avoid downloading weights.
- Visualization and caching: annotation-based construction in all three source
  read paths; existing descriptors/thumbnails remain usable without source reads.

Extend existing tests in `tests/unit/test_wsi_dataset.py`,
`test_features_slide_cli.py`, `test_slide_retrieval_mean_rgb.py`,
`test_lazyslide_fallback_mpp.py`, `test_lazyslide_reader.py`,
`test_sish_rgb_representation.py`, `test_sish_precompute.py`, and
`test_sish_precompute_descriptors.py`. Add focused constructor and histogram
tests; inspect visualization renderer/smoke fixtures for service coverage.
Run the affected CLI tests and relevant SISH/visualization smoke tests as well.

## Acceptance criteria

All annotation-based source-slide reads use `WSI.from_annotation`; each receives
the annotation row for the correct dataset and physical slide. Slides without
native MPP work in RGB and SISH when a valid annotation fallback exists. Direct
construction remains compatible, explicit staged paths remain effective, and
processors continue to operate solely on WSI objects without annotation I/O.

This is a construction/data-flow change. Descriptor schemas, identifiers,
algorithm choices, and CLI flags require no planned changes. Existing cached
descriptors will not automatically be recomputed when annotation values change.
