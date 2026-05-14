from __future__ import annotations

import json
from pathlib import Path

from tests.smoke._smoke_dataset import (
    attach_smoke_outputs,
    capture_smoke_metrics,
    write_smoke_report,
)


def test_capture_smoke_metrics_writes_local_and_aggregated_reports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_dir = tmp_path / "report"
    monkeypatch.setenv("PATHBENCH_SMOKE_REPORT_DIR", str(report_dir))

    intermediate_path = tmp_path / "intermediate.txt"
    final_path = tmp_path / "final.txt"
    intermediate_path.write_text("mid", encoding="utf-8")
    final_path.write_text("done", encoding="utf-8")

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="demo_step",
        metadata={"kind": "demo"},
    ) as payload:
        attach_smoke_outputs(
            payload,
            step_name="demo_step",
            intermediate={"intermediate": intermediate_path},
            final={"final": final_path},
        )

    step_json = tmp_path / "metrics" / "demo_step.metrics.json"
    mirrored_json = report_dir / "steps" / "demo_step.metrics.json"
    assert step_json.exists()
    assert mirrored_json.exists()

    mirrored_payload = json.loads(mirrored_json.read_text(encoding="utf-8"))
    assert mirrored_payload["intermediate_outputs"]["intermediate"]["exists"] is True
    assert mirrored_payload["final_outputs"]["final"]["exists"] is True
    assert (
        Path(mirrored_payload["intermediate_outputs"]["intermediate"]["path"]).parent
        == report_dir / "artifacts" / "demo_step" / "intermediate" / "intermediate"
    )
    assert (
        Path(mirrored_payload["final_outputs"]["final"]["path"]).parent
        == report_dir / "artifacts" / "demo_step" / "final" / "final"
    )
    assert mirrored_payload["final_outputs"]["final"]["source_path"] == str(final_path)


def test_write_smoke_report_aggregates_step_metrics(tmp_path: Path) -> None:
    report_dir = tmp_path / "smoke_report"
    steps_dir = report_dir / "steps"
    steps_dir.mkdir(parents=True)
    step_payload = {
        "step_name": "first_step",
        "elapsed_seconds": 1.25,
        "ru_maxrss_mb": 64.0,
        "final_outputs": {
            "artifact": {
                "path": str(tmp_path / "artifact.bin"),
                "exists": False,
                "kind": "missing",
            }
        },
    }
    (steps_dir / "first_step.metrics.json").write_text(
        json.dumps(step_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    output_paths = write_smoke_report(report_dir)

    json_payload = json.loads(output_paths["json"].read_text(encoding="utf-8"))
    markdown_text = output_paths["markdown"].read_text(encoding="utf-8")
    assert json_payload["num_steps"] == 1
    assert json_payload["step_names"] == ["first_step"]
    assert "first_step" in markdown_text
