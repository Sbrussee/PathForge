from __future__ import annotations
import json
import optuna
from ..config.config import Config


def build_study(cfg: Config) -> optuna.Study:
    return optuna.create_study(direction="maximize")


def suggest_params(trial: optuna.Trial, search_space_path: str) -> dict:
    space = json.load(open(search_space_path))
    params = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif t == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
        elif t == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown type {t}")
    return params