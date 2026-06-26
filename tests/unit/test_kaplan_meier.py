"""Unit tests for the Kaplan-Meier helper functions in training.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from pathforge.training.metrics import _kaplan_meier_curve


def test_kaplan_meier_basic_decreasing():
    time = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    event = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    times, survival = _kaplan_meier_curve(time, event)
    # survival must be non-increasing
    assert np.all(np.diff(survival) <= 0.0)
    assert times[0] == 0.0
    assert survival[0] == 1.0
    assert survival[-1] < 1.0


def test_kaplan_meier_no_events_returns_flat():
    time = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    event = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    times, survival = _kaplan_meier_curve(time, event)
    # no events → only the initial point
    assert len(times) == 1
    assert times[0] == 0.0
    assert survival[0] == 1.0


def test_kaplan_meier_single_event():
    time = np.array([5.0, 10.0, 10.0], dtype=np.float32)
    event = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    times, survival = _kaplan_meier_curve(time, event)
    assert len(times) == 2  # t=0 and t=5
    assert survival[1] == pytest.approx(2.0 / 3.0, rel=1e-5)


def test_kaplan_meier_tied_event_times():
    time = np.array([2.0, 2.0, 4.0], dtype=np.float32)
    event = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    times, survival = _kaplan_meier_curve(time, event)
    # two events at t=2, all 3 at risk → survival = 1 * (1 - 2/3) = 1/3
    assert survival[-1] == pytest.approx(1.0 / 3.0, rel=1e-5)
    assert 2.0 in times


def test_kaplan_meier_output_types():
    time = np.array([1.0, 2.0], dtype=np.float32)
    event = np.array([1.0, 0.0], dtype=np.float32)
    times, survival = _kaplan_meier_curve(time, event)
    assert times.dtype == np.float32
    assert survival.dtype == np.float32
