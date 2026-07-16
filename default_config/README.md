# PathForge default configurations

These self-contained templates use placeholder paths under `/path/to/...`.
Update the annotation, slide, artifact, project, and Optuna storage paths before
running them.

Foundation-model names follow LazySlide's runtime identifiers:

- `h-optimus-1` = HOptimus1
- `uni2` = UNI2
- `virchow2` = Virchow2
- `gpfm` = GenBioPathFM

The benchmark templates evaluate the requested 224-pixel tiles at 20× and 10×
(`tile_mpp: [0.5, 1.0]`), with Macenko and no stain normalization. MIL tasks use
ABMIL, DSMIL, and TransMIL. Slide retrieval does not train a prediction model,
so losses and MIL architectures do not apply to that task.

ABMIL, DSMIL, and TransMIL are adapter-backed models. Their usable task/output
combinations depend on the installed TorchMIL or MIL-Lab version and compatible
constructor kwargs. The templates satisfy PathForge's configuration schema,
but users must confirm that the selected upstream implementation emits the
output shape required by regression or the chosen survival formulation.

Files prefixed with `benchmark_` enumerate task-supported fixed pipeline grids.
Files prefixed with `optimize_` use Optuna ranges plus categorical pipeline
choices. The parallel templates show generated SLURM workflows; parallel optimization needs
a shared PostgreSQL database and must not use SQLite across nodes.

The current Optuna policy is MIL-training-specific. The
`optimize_slide_retrieval.yaml` file documents the intended retrieval search
space, but it is not executable until PathForge gains a retrieval objective
adapter. Use `benchmark_slide_retrieval.yaml` for supported retrieval grid
search today.

Create and submit a distributed plan with:

```bash
pathforge execution plan --config default_config/parallel_benchmark.yaml --output work/benchmark
bash work/benchmark/slurm/submit.sh
```

Validate any edited template with:

```bash
python -c "from pathforge.config.config import Config; Config.from_yaml('CONFIG.yaml')"
```
