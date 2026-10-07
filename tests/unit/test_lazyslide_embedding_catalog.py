"""Embedding-only discovery must not load model weights or accept other tasks."""

from enum import Enum
from types import ModuleType
import sys

import pytest

from pathforge.core.slide_processing.lazyslide import catalog


class _Task(Enum):
    vision = "vision"
    multimodal = "multimodal"
    slide_encoder = "slide_encoder"


@pytest.fixture
def embedding_registry(monkeypatch):
    """Expose patch, shared, slide-only and unsupported classes without weights."""
    class Patch:
        task = _Task.vision

        def __init__(self):
            pytest.fail("Metadata resolution must not construct models")

        def encode_image(self, images):
            return images

    class Shared(Patch):
        task = [_Task.multimodal, _Task.slide_encoder]
        vision_encoder = "shared"

        def encode_slide(self, embeddings, coords=None, **kwargs):
            return {"embeddings": embeddings}

    class Slide:
        task = _Task.slide_encoder
        vision_encoder = "patch"

        def encode_slide(self, embeddings, coords=None, **kwargs):
            return {"embeddings": embeddings}

    registry = {"patch": Patch, "shared": Shared, "alias": Shared, "slide_only": Slide}
    for task in (
        "segmentation", "tile_prediction", "feature_prediction", "cv_feature",
        "style_transfer", "image_generation",
    ):
        # Even an unrelated model with an image method must be excluded by task.
        registry[task] = type(task, (Patch,), {"task": task})
    module = ModuleType("lazyslide_models")
    module.MODEL_REGISTRY = registry
    monkeypatch.setitem(sys.modules, "lazyslide_models", module)
    catalog.lazyslide_model_names.cache_clear()
    catalog.timm_model_names.cache_clear()
    yield registry
    catalog.lazyslide_model_names.cache_clear()
    catalog.timm_model_names.cache_clear()


def test_catalog_only_advertises_executable_embedding_requests(embedding_registry):
    assert catalog.lazyslide_model_names() == {
        "patch", "shared", "alias", "shared-slide", "alias-slide", "slide_only-slide",
    }


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("patch", ("patch", None)),
        ("shared", ("shared", None)),
        ("alias", ("alias", None)),
        ("shared-slide", ("shared", "shared")),
        ("alias-slide", ("shared", "alias")),
        ("slide_only-slide", ("patch", "slide_only")),
    ],
)
def test_resolve_embedding_roles(embedding_registry, name, expected):
    assert catalog.resolve_lazyslide_embedding(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "missing", "missing-slide", "patch-slide", "slide_only", "segmentation",
        "tile_prediction", "feature_prediction", "cv_feature", "style_transfer",
        "image_generation", "shared-slide-slide",
    ],
)
def test_resolver_rejects_unsupported_requests(embedding_registry, name):
    with pytest.raises(ValueError):
        catalog.resolve_lazyslide_embedding(name)


@pytest.mark.parametrize("patch_name", [None, "missing", "segmentation", "slide_only"])
def test_slide_encoder_requires_a_valid_patch_encoder(embedding_registry, patch_name):
    embedding_registry["slide_only"].vision_encoder = patch_name
    with pytest.raises(ValueError, match="patch encoder"):
        catalog.resolve_lazyslide_embedding("slide_only-slide")
    assert "slide_only-slide" not in catalog.lazyslide_model_names()


def test_declared_tasks_require_corresponding_encoding_methods(embedding_registry):
    embedding_registry["no_image"] = type("NoImage", (), {"task": _Task.vision})
    embedding_registry["no_slide"] = type(
        "NoSlide", (), {"task": _Task.slide_encoder, "vision_encoder": "patch"}
    )
    assert "no_image" not in catalog.lazyslide_model_names()
    assert "no_slide-slide" not in catalog.lazyslide_model_names()


def test_timm_catalog_does_not_restore_disallowed_registry_collisions(
    embedding_registry, monkeypatch,
):
    module = ModuleType("timm")
    module.list_models = lambda: ["resnet18", "segmentation", "patch", "resnet18-slide"]
    monkeypatch.setitem(sys.modules, "timm", module)
    assert catalog.timm_model_names() == {"resnet18"}


@pytest.mark.parametrize("task", [_Task.vision, [_Task.multimodal], "vision", None])
def test_patch_capability_handles_task_forms(embedding_registry, task):
    model_class = type("Example", (embedding_registry["patch"],), {"task": task})
    assert catalog._supports_patch_embeddings(model_class) is (task is not None)


def test_model_listing_shapes_are_normalized():
    from types import SimpleNamespace

    assert catalog._normalize_model_names(
        ["a", {"key": "b"}, SimpleNamespace(key="c")]
    ) == {"a", "b", "c"}
    assert catalog._normalize_model_names(None) == set()


@pytest.mark.parametrize(
    ("name", "level", "expected"),
    [
        ("shared", "patch", ("shared", None)),
        ("shared", "slide", ("shared", "shared")),
        ("slide_only", "slide", ("patch", "slide_only")),
    ],
)
def test_explicit_output_level_controls_backend_resolution(embedding_registry, name, level, expected):
    """Backend capability resolution obeys the policy request without a suffix."""
    assert catalog.resolve_lazyslide_embedding(name, output_level=level) == expected


def test_explicit_slide_request_rejects_patch_only_model(embedding_registry):
    """A backend cannot silently substitute patch output for requested slide output."""
    with pytest.raises(ValueError, match="does not support slide embeddings"):
        catalog.resolve_lazyslide_embedding("patch", output_level="slide")
