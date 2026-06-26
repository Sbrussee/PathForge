from __future__ import annotations

from pathlib import Path

import typer

from . import benchmark_run, evaluate_run, features_run, features_slide
from . import infer_run, report_tiles
from . import retrieval_mean_rgb, retrieval_representations, retrieval_sish_vqvae, visualize_run

app = typer.Typer(
    help="PathBench workflows.",
    no_args_is_help=True,
)

features_app = typer.Typer(help="Feature extraction workflows.", no_args_is_help=True)
retrieval_app = typer.Typer(help="Slide retrieval workflows.", no_args_is_help=True)
benchmark_app = typer.Typer(help="Benchmark workflows.", no_args_is_help=True)
evaluate_app = typer.Typer(help="Evaluation workflows.", no_args_is_help=True)
visualize_app = typer.Typer(help="Visualization workflows.", no_args_is_help=True)
report_app = typer.Typer(help="Reporting workflows.", no_args_is_help=True)
infer_app = typer.Typer(help="Inference workflows.", no_args_is_help=True)
optimize_app = typer.Typer(help="Optimization workflows.", no_args_is_help=True)

features_app.command("run")(features_run.run_command)
features_app.command("slide")(features_slide.run_command)

retrieval_app.command("representations")(retrieval_representations.run_command)
retrieval_app.command("mean-rgb")(retrieval_mean_rgb.run_command)
retrieval_app.command("sish-vqvae")(retrieval_sish_vqvae.run_command)

benchmark_app.command("run")(benchmark_run.run_command)
evaluate_app.command("run")(evaluate_run.run_command)
visualize_app.command("run")(visualize_run.run_command)
report_app.command("tiles")(report_tiles.run_command)
infer_app.command("run")(infer_run.run_command)


def _run_optimize_command(
    config: Path = typer.Option(..., "--config", help="Path to YAML config."),
) -> None:
    from . import optimize_run

    optimize_run.run_command(config=config)


optimize_app.command("run")(_run_optimize_command)

app.add_typer(features_app, name="features")
app.add_typer(retrieval_app, name="retrieval")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(visualize_app, name="visualize")
app.add_typer(report_app, name="report")
app.add_typer(infer_app, name="infer")
app.add_typer(optimize_app, name="optimize")


def main() -> None:
    """Entry point that launches the PathBench Typer application."""
    app()


if __name__ == "__main__":
    main()
