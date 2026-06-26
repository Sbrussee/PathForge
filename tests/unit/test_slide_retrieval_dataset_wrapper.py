from __future__ import annotations

from pathlib import Path

from pathforge.core.datasets.bag_dataset import (
    BagSample,
    SlideRetrievalBagDataset,
)


class _FakeBagDataset:
    def __init__(self) -> None:
        self.num_bags = 2
        self._samples = [
            BagSample(
                sample_id="sample-0",
                slide_ids=["slide-0"],
                artifact_paths=[Path("/tmp/slide-0.h5")],
                category="tumor",
            ),
            BagSample(
                sample_id="sample-1",
                slide_ids=["slide-1"],
                artifact_paths=[Path("/tmp/slide-1.h5")],
                category="normal",
            ),
        ]

    def get_sample(self, index: int) -> BagSample:
        return self._samples[index]

    def load_bag(self, index: int) -> str:
        return f"bag-{index}"


def _make_dataset(
    *,
    sample_loader,
) -> SlideRetrievalBagDataset:
    base_dataset = _FakeBagDataset()
    dataset = object.__new__(SlideRetrievalBagDataset)
    dataset.samples = list(base_dataset._samples)
    dataset.sample_loader = sample_loader
    dataset._mode = "artifact"  # required by get_sample()
    return dataset


def test_slide_retrieval_wrapper_delegates_loading_per_sample() -> None:
    seen: dict[str, object] = {}

    def _load_sample(*, index: int, sample: BagSample, base_dataset: SlideRetrievalBagDataset) -> dict[str, object]:
        seen["index"] = index
        seen["sample_id"] = sample.sample_id
        seen["base_dataset"] = base_dataset
        return {"bag": f"bag-{index}", "sample_id": sample.sample_id}

    dataset = _make_dataset(sample_loader=_load_sample)

    item = dataset[1]

    assert len(dataset) == 2
    assert item.index == 1
    assert item.sample.sample_id == "sample-1"
    assert item.inputs == {
        "bag": "bag-1",
        "sample_id": "sample-1",
    }
    assert seen["index"] == 1
    assert seen["sample_id"] == "sample-1"
    assert seen["base_dataset"] is dataset


def test_slide_retrieval_wrapper_get_sample_passthrough() -> None:
    dataset = _make_dataset(sample_loader=lambda **_: {})

    sample = dataset.get_sample(0)

    assert sample.sample_id == "sample-0"
    assert sample.slide_ids == ["slide-0"]


def test_slide_retrieval_wrapper_clear_sample_loader_disables_getitem() -> None:
    dataset = _make_dataset(sample_loader=lambda **_: {"bag": "bag-0"})

    dataset.clear_sample_loader()

    try:
        dataset[0]
    except RuntimeError as exc:
        assert "requires a bound sample_loader" in str(exc)
    else:
        raise AssertionError("Expected access without a bound sample_loader to fail.")
