from types import SimpleNamespace

from pathforge.config.config import SlideProcessingConfig
from pathforge.core.slide_processing.lazyslide import LazySlideProcessor


def test_slide_processing_reader_defaults_to_auto():
    config = SlideProcessingConfig()
    assert config.reader == "auto"
    assert config.reader_fallbacks == []


def test_lazyslide_passes_explicit_reader_to_wsidata(monkeypatch):
    opened = object()
    calls = []

    def fake_open_wsi(path, *, reader):
        calls.append((path, reader))
        return opened

    monkeypatch.setattr(
        "pathforge.core.slide_processing.lazyslide.processor.open_wsi", fake_open_wsi
    )
    wsi = SimpleNamespace(path="slide.svs", _obj=None)

    LazySlideProcessor(reader="cucim").load_wsi(wsi)

    assert calls == [("slide.svs", "cucim")]
    assert wsi._obj is opened


def test_lazyslide_auto_reader_is_forwarded_as_none(monkeypatch):
    calls = []

    def fake_open_wsi(path, *, reader):
        calls.append((path, reader))
        return object()

    monkeypatch.setattr(
        "pathforge.core.slide_processing.lazyslide.processor.open_wsi", fake_open_wsi
    )
    wsi = SimpleNamespace(path="slide.svs", _obj=None)

    LazySlideProcessor().load_wsi(wsi)

    assert calls == [("slide.svs", None)]


def test_lazyslide_uses_configured_fallback(monkeypatch, caplog):
    calls = []
    opened = object()

    def fake_open_wsi(path, *, reader):
        calls.append((path, reader))
        if reader == "cucim":
            raise RuntimeError("unsupported compression")
        return opened

    monkeypatch.setattr(
        "pathforge.core.slide_processing.lazyslide.processor.open_wsi", fake_open_wsi
    )
    wsi = SimpleNamespace(path="slide.svs", _obj=None)

    LazySlideProcessor(
        reader="cucim", reader_fallbacks=["openslide"]
    ).load_wsi(wsi)

    assert calls == [("slide.svs", "cucim"), ("slide.svs", "openslide")]
    assert wsi._obj is opened
    assert "fallback reader 'openslide'" in caplog.text
