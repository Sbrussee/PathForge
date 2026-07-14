"""Tests for the integer index used by the SISH retrieval strategy."""

from __future__ import annotations

import pytest

from pathforge.slide_retrieval.search_strategies.strategies.sish.sish_veb import VEB


def test_veb_finds_members_and_neighbours() -> None:
    """Inserted keys should support membership and nearest-neighbour lookups."""
    tree = VEB(16)
    for value in (2, 4, 9, 15):
        tree.insert(value)

    assert tree.member(4)
    assert not tree.member(5)
    assert tree.successor(4) == 9
    assert tree.predecessor(9) == 4


def test_veb_handles_boundaries_and_duplicate_inserts() -> None:
    """Boundary queries and duplicate keys should remain stable."""
    tree = VEB(4)
    tree.insert(0)
    tree.insert(0)
    tree.insert(3)

    assert tree.predecessor(0) is None
    assert tree.successor(3) is None
    assert tree.min == 0
    assert tree.max == 3


def test_veb_rejects_negative_universe_size() -> None:
    """A negative universe cannot define a valid integer index."""
    with pytest.raises(ValueError, match="non-negative"):
        VEB(-1)
