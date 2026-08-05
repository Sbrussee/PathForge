# Feature extractor registry plan

## Goal

Allow users to register PathForge feature extractors while keeping the existing
unqualified `feature_extraction` configuration values. The configured slide
processor has priority when it provides a feature extractor with the same
name.

## Agreed model

`FEATURE_EXTRACTORS` is the single global registry for PathForge-native
extractor classes only:

```text
extractor name -> FeatureExtractorBase class
```

LazySlide, timm, and future processor-native model names must not be inserted
into `FEATURE_EXTRACTORS`.

Each slide processor supplies its own built-in feature-extractor names. The
full set used for a run is composed transiently for the selected processor:

```text
available names = processor-native names union FEATURE_EXTRACTORS names
```

This union is not a global registry. It is only used to validate and resolve
the configuration for the selected processor.

## Resolution rule

For an unqualified extractor name:

1. If the selected processor provides the name natively, use that processor
   implementation.
2. Otherwise, if `FEATURE_EXTRACTORS` contains the name, use the registered
   PathForge extractor.
3. Otherwise, reject the configuration as unavailable for that processor.

If both sources provide the same name, processor-native wins. Emit a clear log
message stating that the registered PathForge extractor was not selected.

## Implementation slice 1: registry cleanup and validation

1. Keep the public `FEATURE_EXTRACTORS` name, but ensure it stores only
   `FeatureExtractorBase` classes.
2. Remove the timm and LazySlide loops in `populate_dynamic_registries()` that
   register factories or strings into `FEATURE_EXTRACTORS`.
3. Keep `timm_model_names()` and `lazyslide_model_names()` as discovery
   helpers. They remain useful, but they no longer mutate the global feature
   extractor registry.
4. Add a processor capability method such as
   `native_feature_extractor_names() -> set[str]` to `SlideProcessorBase`.
   Its default implementation returns an empty set.
5. Implement that method for `LazySlideProcessor` using the existing LazySlide
   and timm discovery helpers.
6. Add a helper that loads the selected processor and returns the transient
   union of processor-native and PathForge names.
7. Make feature-extractor config validation use that helper after
   `slide_processing.backend` is known. No YAML/config-schema change is
   required.

Expected validation behavior after this slice:

- `backend: lazyslide` accepts LazySlide names, timm names supported by
  LazySlide, and registered PathForge names.
- A processor with no built-in names accepts only registered PathForge names.
- A processor with neither built-in nor PathForge names accepts no extractor
  name.

## Implementation slice 2: execution routing

Add processor-side resolution for the same precedence rule:

- A processor-native name is passed to the processor in the format it expects
  (for LazySlide, the model-name string).
- A PathForge name is looked up in `FEATURE_EXTRACTORS`, instantiated, and
  wrapped by a processor-specific adapter before feature extraction.

The validation helper and execution resolver must use the same precedence
rule.

## Tests

- `FEATURE_EXTRACTORS` contains only PathForge extractor classes.
- LazySlide discovery makes both its built-in and timm names valid for the
  LazySlide processor.
- A PathForge name is valid for a processor that supports PathForge wrapping.
- A backend-native/PathForge collision chooses the processor-native name and
  produces the expected log message.
- A processor with no built-in names rejects a timm/LazySlide name unless an
  appropriate PathForge extractor is registered.
