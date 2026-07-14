"""Integer predecessor/successor index used by the SISH search backend.

The van Emde Boas implementation is adapted from
https://github.com/erikwaing/VEBTree.
"""

from __future__ import annotations

import math


class VEB:
    """Store integer keys in a van Emde Boas tree.

    Args:
        universe_size: Exclusive upper bound for keys that will be inserted.

    Example:
        >>> tree = VEB(16)
        >>> tree.insert(4)
        >>> tree.insert(9)
        >>> tree.successor(4)
        9
    """

    def __init__(self, universe_size: int) -> None:
        if universe_size < 0:
            raise ValueError(f"universe_size must be non-negative, got {universe_size}")
        self.u = 2
        while self.u < universe_size:
            self.u *= self.u
        self.min: int | None = None
        self.max: int | None = None
        if universe_size > 2:
            self.clusters: list[VEB | None] = [None for _ in range(self.high(self.u))]
            self.summary: VEB | None = None

    def high(self, value: int) -> int:
        """Return the cluster index containing ``value``."""
        return int(math.floor(value / math.sqrt(self.u)))

    def low(self, value: int) -> int:
        """Return the offset of ``value`` within its cluster."""
        return int(value % math.ceil(math.sqrt(self.u)))

    def index(self, high: int, low: int) -> int:
        """Combine a cluster index and offset into one key."""
        return int(high * math.floor(math.sqrt(self.u)) + low)

    def member(self, value: int) -> bool:
        """Return whether ``value`` is present in the tree."""
        if value == self.min or value == self.max:
            return True
        if self.u <= 2:
            return False
        cluster = self.clusters[self.high(value)]
        return cluster is not None and cluster.member(self.low(value))

    def successor(self, value: int) -> int | None:
        """Return the smallest stored key greater than ``value``."""
        if self.u <= 2:
            return 1 if value == 0 and self.max == 1 else None
        if self.min is not None and value < self.min:
            return self.min

        high = self.high(value)
        low = self.low(value)
        cluster = self.clusters[high]
        max_low = cluster.max if cluster is not None else None
        if max_low is not None and low < max_low:
            offset = cluster.successor(low)
            return None if offset is None else self.index(high, offset)

        successor_cluster = (
            self.summary.successor(high) if self.summary is not None else None
        )
        if successor_cluster is None:
            return None
        successor = self.clusters[successor_cluster]
        return (
            None
            if successor is None or successor.min is None
            else self.index(successor_cluster, successor.min)
        )

    def predecessor(self, value: int) -> int | None:
        """Return the largest stored key smaller than ``value``."""
        if self.u <= 2:
            return 0 if value == 1 and self.min == 0 else None
        if self.max is not None and value > self.max:
            return self.max

        high = self.high(value)
        low = self.low(value)
        cluster = self.clusters[high]
        min_low = cluster.min if cluster is not None else None
        if min_low is not None and low > min_low:
            offset = cluster.predecessor(low)
            return None if offset is None else self.index(high, offset)

        predecessor_cluster = (
            self.summary.predecessor(high) if self.summary is not None else None
        )
        if predecessor_cluster is None:
            return self.min if self.min is not None and value > self.min else None
        predecessor = self.clusters[predecessor_cluster]
        return (
            None
            if predecessor is None or predecessor.max is None
            else self.index(predecessor_cluster, predecessor.max)
        )

    def _insert_empty(self, value: int) -> None:
        """Insert ``value`` into an empty tree or cluster."""
        self.min = value
        self.max = value

    def insert(self, value: int) -> None:
        """Insert ``value`` into the tree."""
        if self.min is None:
            self._insert_empty(value)
            return

        if value < self.min:
            self.min, value = value, self.min
        if self.u > 2:
            high = self.high(value)
            if self.clusters[high] is None:
                self.clusters[high] = VEB(self.high(self.u))
            if self.summary is None:
                self.summary = VEB(self.high(self.u))
            cluster = self.clusters[high]
            if cluster.min is None:
                self.summary.insert(high)
                cluster._insert_empty(self.low(value))
            else:
                cluster.insert(self.low(value))
        if self.max is None or value > self.max:
            self.max = value
