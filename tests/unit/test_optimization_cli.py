#TODO: Add minimal tests for optimization CLI functionalityimport sys

from pathbench.cli import optimize


def test_optimize_cli_invokes_experiment(monkeypatch, capsys, tmp_path) -> None:
    """
    Ensure the optimization CLI loads config and runs the experiment wrapper.
    """
    captured = {}

    class DummyConfig:
        pass

    class DummyExperiment:
        def __init__(self, cfg):
            captured["cfg"] = cfg

        def run(self):
            captured["ran"] = True
            return {"status": "ok"}

    def fake_from_yaml(path):
        captured["config_path"] = path
        return DummyConfig()

    monkeypatch.setattr(optimize.Config, "from_yaml", staticmethod(fake_from_yaml))
    monkeypatch.setattr(optimize, "OptimizationExperiment", DummyExperiment)
    monkeypatch.setattr(sys, "argv", ["optimize", "--config", str(tmp_path / "config.yaml")])

    optimize.main()
    stdout = capsys.readouterr().out

    assert captured.get("ran") is True
    assert "status" in stdout