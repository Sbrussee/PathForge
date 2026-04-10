- Shared reference docs are stored in `docs/` (except this `AGENTS.md` file).
- Code must adhere to the design standards set in `docs/design.md`
- File formatting, typing and shapes should adhere to the standards set in `docs/design.md`

- Docstrings should be implemented as follows:
    - Clarify the expected inputs and outputs with typing and shapes
    - Clarify the semantic goal of the functions / classes
    - Show example usage of the function / class
    - Use inline comments for clarity in the implementations.

- Code formatting should adhere to ruff standards. Use ruff to ensure code style and for deduplication.
- Before implementing any code, first search for already implemented modules that can be used to solve the problem (modular design).

For each function in the codebase we require:
- A unit test testing regular (expected use)
- Unit tests for edge cases

For pipelines (policies) inside the framework we will need a smoke test:
- Load sample data (utils/test_samples.py)
- Put these through the entire pipeline
- For the pipelines, measure time metrics / memory metrics.

- Use pytest as the testing framework. Ensure calculations in tests inside the testing suite are not conducted double and are thus re-used to ensure efficiency.

- Programmatically assess whether the dependency structure (Interfaces/Drivers -> Adapters -> Applications/Use Cases -> Domain) holds and where it fails.

- Identifier naming and builders must match the canonical helpers:
    - `tiling_id`: tiling-only identifier built with `build_tiling_id(combo_cfg)` in `core/experiments/combo_ids.py`; format is `"{tile_px}px_{tile_mpp:g}mpp"` (example: `256px_0.5mpp`).
    - `feature_name`: feature-storage name built with `build_feature_name(combo_cfg)` in `core/experiments/combo_ids.py`; format is `"{feature_extraction}"` or `"{feature_extraction}_{color_norm}"` when `color_norm` is set.
    - `bag_id`: full bag identifier built with `build_bag_id(combo_cfg)` in `core/experiments/combo_ids.py`; format is `"{tiling_id}__{feature_name}"`.
    - `representation_id`: retrieval representation identifier built with `build_retrieval_representation_id(feature_extraction, retrieval_representation, params)` in `slide_retrieval/representation_strategies/storage.py`; format is `"{feature_extraction}__{retrieval_representation}__{params_hash16}"`.
    - `entry_id`: retrieval entry identifier built with `build_retrieval_representation_entry_id(slide_ids)` in `slide_retrieval/representation_strategies/storage.py`; format is `"members_{sha1_16}"`.
    - `tile_id`: retrieval I/O field name for the tiling key in H5 retrieval paths; pass the `tiling_id` value when calling retrieval layout/I/O helpers.
