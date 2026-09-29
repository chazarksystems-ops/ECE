"""Spatial hash: snapshot counts, exclusive scan, then scatter.

Catalog bug class: scanning live atomic counters that are later reused
as write cursors. The reference keeps three distinct arrays:

* counts     — particles per bin (frozen snapshot)
* offsets    — exclusive prefix sum of counts
* ranges     — (start, end) = (offsets[b], offsets[b] + counts[b])
* cursor     — separate write head used only during scatter
"""

from __future__ import annotations

import numpy as np


def bin_index(
    pos: np.ndarray,
    cell: float,
    grid: tuple[int, int],
    world: np.ndarray | None = None,
) -> np.ndarray:
    """Return 2D bin indices, optionally partitioning the world evenly."""
    pos = np.asarray(pos, dtype=np.float64)
    gx, gy = grid
    if world is None:
        widths = np.array([cell, cell], dtype=np.float64)
    else:
        world = np.asarray(world, dtype=np.float64)
        if world.shape != (2,) or np.any(world <= 0):
            raise ValueError("world must contain two positive lengths")
        widths = world / np.asarray(grid, dtype=np.float64)
    ix = np.floor(pos[:, 0] / widths[0]).astype(np.int64) % gx
    iy = np.floor(pos[:, 1] / widths[1]).astype(np.int64) % gy
    return iy * gx + ix


def exclusive_scan(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    offsets = np.empty_like(counts)
    offsets[0] = 0
    if counts.size > 1:
        offsets[1:] = np.cumsum(counts[:-1])
    return offsets


def snapshot_and_scatter(
    pos: np.ndarray,
    cell: float,
    grid: tuple[int, int],
    world: np.ndarray | None = None,
) -> dict:
    """Return frozen hash tables. Never mutates counts after the snapshot."""
    n = len(pos)
    gx, gy = grid
    nbin = gx * gy
    idx = bin_index(pos, cell, grid, world)
    counts = np.bincount(idx, minlength=nbin).astype(np.int64)
    snapshot = counts.copy()
    offsets = exclusive_scan(snapshot)
    ranges = np.stack([offsets, offsets + snapshot], axis=1)
    sorted_indices = np.empty(n, dtype=np.int64)
    cursor = offsets.copy()  # write head, distinct from snapshot
    for i in range(n):
        b = int(idx[i])
        slot = int(cursor[b])
        sorted_indices[slot] = i
        cursor[b] += 1
    # Invariant: cursor finished at range ends; snapshot unchanged.
    if not np.array_equal(cursor, ranges[:, 1]):
        raise RuntimeError("scatter cursor did not land on range ends")
    if not np.array_equal(counts, snapshot):
        raise RuntimeError("counts mutated after snapshot")
    return {
        "counts": snapshot,
        "offsets": offsets,
        "ranges": ranges,
        "sorted_indices": sorted_indices,
        "bin_of": idx,
    }
