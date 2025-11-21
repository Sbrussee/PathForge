Minimal deps by default; heavy features hidden behind optional extras.

One‑way dependencies across layers to prevent global churn.

Registries + ABCs for extension without touching core wiring.

Config‑first; every CLI builds from a Config instance.

Fail fast via assertions and smoke tests.

Keep cross‑module calls unidirectional by import discipline and CI check.

Prefer composition (e.g., LazySlide wraps WSIData) over inheritance for I/O backends.

Put code that changes together into the same module; move it when change patterns emerge.

Keep abstract method surfaces tiny; add tests for expected shapes/dtypes.

Treat CLI tools as thin shells around Experiment.run(), so batch/cluster runs are trivial.

Every public class/function gets a docstring with shape/dtype expectations.

Assertions at boundaries (e.g., non‑empty bag) and type hints everywhere.

Keep heavy objects out of global scope; build them in factories using Config.