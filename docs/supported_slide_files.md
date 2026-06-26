# Supported slide files and `slides_dir` layout

This document describes which slide file layouts are supported by the `WSIDataset` and how slide data must be provided inside `slides_dir`.

The goal is to keep slide discovery deterministic and avoid ambiguous or heuristic matching rules.

## Overview

For each row in the annotation file, the value in the `slide` column is treated as the `slide_id`.

A slide is considered valid only if it is provided in one of the two supported layouts below.

## Supported layouts

### 1. Single-file slide directly in `slides_dir`

A slide may be stored as a single file directly inside `slides_dir`.

The filename must follow this rule:

`<slide_id>.<extension>`

Examples:
- `slides_dir/T12-00123.svs`
- `slides_dir/T12-00124.ndpi`
- `slides_dir/T12-00125.tiff`

Valid examples for `slide_id = T12-00123`:
- `T12-00123.svs`
- `T12-00123.ndpi`
- `T12-00123.mrxs`

The file stem must be exactly equal to `slide_id`.
The supported direct-file suffixes are `.svs`, `.ndpi`, `.tiff`, `.tif`, and `.mrxs`.

### 2. Multi-file DICOM slide in its own folder

A multi-file DICOM slide must be stored in a folder named exactly like the `slide_id`.

The folder must contain one or more `.dcm` files directly inside it.

Example:
- `slides_dir/T12-00126/000001.dcm`
- `slides_dir/T12-00126/000002.dcm`
- `slides_dir/T12-00126/000003.dcm`

For this layout:
- the folder name must be exactly equal to `slide_id`
- the `.dcm` files must be directly inside that folder
- nested subfolders are not supported

When such a folder is found, the dataset resolves the slide by selecting a single `.dcm` file from that folder as the anchor path for the backend.

## Matching rules

Slide discovery is strict and deterministic.

For a given `slide_id`, the dataset supports only:

1. an exact direct-file match:  
   `<slides_dir>/<slide_id>.<extension>`

2. an exact DICOM-folder match:  
   `<slides_dir>/<slide_id>/*.dcm`

No fuzzy matching, prefix matching, or partial matching is supported.

## Invalid layouts

The following layouts are considered invalid.

### Invalid direct filenames

These are not supported because the file stem does not exactly match `slide_id`:
- `T12-00123.2.dcm`
- `T12-00123_extra.dcm`
- `T12-00123-scan1.dcm`

### Invalid DICOM folder structures

These are not supported:
- `slides_dir/T12-00123/nested/000001.dcm`
- `slides_dir/some_other_folder/000001.dcm`
- `slides_dir/T12-00123/` when the folder contains no `.dcm` files

## Ambiguous cases

The following situations are considered ambiguous and should be rejected.

### 1. Both a direct file and a DICOM folder exist for the same `slide_id`

Example:
- `slides_dir/T12-00123.dcm`
- `slides_dir/T12-00123/000001.dcm`
- `slides_dir/T12-00123/000002.dcm`

This is deterministic in the current implementation: PathForge prefers the direct
file match and only falls back to the DICOM folder when no direct file exists.

### 2. Multiple exact direct-file matches exist for the same `slide_id`

Example:
- `slides_dir/T12-00123.dcm`
- `slides_dir/T12-00123.svs`

This is also ambiguous and should not be allowed.

## Recommended conventions

### For non-DICOM or single-file DICOM slides

Store the slide as a single file directly in `slides_dir`:

`slides_dir/<slide_id>.<extension>`

### For multi-file DICOM slides

Store the slide in a folder named exactly like `slide_id`:

- `slides_dir/<slide_id>/file1.dcm`
- `slides_dir/<slide_id>/file2.dcm`
- `slides_dir/<slide_id>/...`

Prefer one folder per slide and do not mix files from different slides in the same directory.

## Summary

The dataset supports exactly two slide layouts:

1. direct file:  
   `<slide_id>.<extension>`

2. DICOM folder:  
   `<slide_id>/*.dcm`

Anything else is considered unsupported.

This keeps slide resolution explicit, predictable, and easy to validate.

## Explicit path override

If the annotation CSV includes a `wsi_path` column, PathForge first tries that
path for the row before resolving the slide through `slides_dir`.
