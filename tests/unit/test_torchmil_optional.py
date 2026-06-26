import pytest

from pathforge.utils.optional import torchmil as optional_torchmil


def test_require_torchmil_raises_clear_error_when_absent(monkeypatch):
    monkeypatch.setattr(optional_torchmil, "is_torchmil_available", lambda: False)

    with pytest.raises(RuntimeError, match="Install torchmil or set mil.backend='native'"):
        optional_torchmil.require_torchmil("MIL backend 'torchmil'")


def test_require_torchmetrics_raises_clear_error_when_absent(monkeypatch):
    monkeypatch.setattr(optional_torchmil, "is_torchmetrics_available", lambda: False)

    with pytest.raises(RuntimeError, match="Classification metrics backend requires 'torchmetrics'"):
        optional_torchmil.require_torchmetrics()


def test_require_torchsurv_raises_clear_error_when_absent(monkeypatch):
    monkeypatch.setattr(optional_torchmil, "is_torchsurv_available", lambda: False)

    with pytest.raises(RuntimeError, match="Continuous survival backend requires 'torchsurv'"):
        optional_torchmil.require_torchsurv()
