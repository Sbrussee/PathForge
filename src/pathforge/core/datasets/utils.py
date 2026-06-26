from __future__ import annotations


def group_datasets_by_use(
    bag_datasets: list,
) -> dict[str, list]:
    """Group datasets by their configured usage."""
    datasets_by_use: dict[str, list] = {}

    for dataset in bag_datasets:
        use = str(dataset.ds_cfg.used_for)
        datasets_by_use.setdefault(use, []).append(dataset)

    return datasets_by_use
