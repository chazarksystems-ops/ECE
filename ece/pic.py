"""Periodic particle-in-cell deposit and sample reference operations."""

from __future__ import annotations

import numpy as np


def _particle_grid_coordinates(pos: np.ndarray, world: np.ndarray, shape: tuple[int, int]):
    pos = np.asarray(pos, dtype=np.float64)
    world = np.asarray(world, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    if world.shape != (2,) or np.any(world <= 0.0):
        raise ValueError("world must contain two positive lengths")
    if len(shape) != 2 or any(size < 1 for size in shape):
        raise ValueError("grid shape must contain two positive dimensions")
    wrapped = np.mod(pos, world)
    gx = wrapped[:, 0] * (shape[1] / world[0])
    gy = wrapped[:, 1] * (shape[0] / world[1])
    x0 = np.floor(gx).astype(np.int64)
    y0 = np.floor(gy).astype(np.int64)
    return x0, y0, gx - x0, gy - y0


def deposit_particles(
    pos: np.ndarray,
    types: np.ndarray,
    world: np.ndarray,
    shape: tuple[int, int],
    channels: int,
    masses: np.ndarray | None = None,
) -> np.ndarray:
    """Bilinearly deposit particle mass into a channel-first periodic field."""
    pos = np.asarray(pos, dtype=np.float64)
    types = np.asarray(types, dtype=np.int64)
    if types.shape != (len(pos),):
        raise ValueError("types must contain one channel index per particle")
    if channels < 1 or np.any(types < 0) or np.any(types >= channels):
        raise ValueError("particle types must be valid field channel indices")
    if masses is None:
        masses = np.ones(len(pos), dtype=np.float64)
    else:
        masses = np.asarray(masses, dtype=np.float64)
    if masses.shape != (len(pos),) or not np.isfinite(masses).all() or np.any(masses < 0.0):
        raise ValueError("masses must be finite, non-negative, and match particle count")

    x0, y0, fx, fy = _particle_grid_coordinates(pos, world, shape)
    height, width = shape
    rho = np.zeros((channels, height, width), dtype=np.float64)
    for dy, wy in ((0, 1.0 - fy), (1, fy)):
        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            np.add.at(
                rho,
                (types, (y0 + dy) % height, (x0 + dx) % width),
                masses * wx * wy,
            )
    return rho


def sample_fields(fields: np.ndarray, pos: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Bilinearly sample every channel at particle positions on a torus."""
    fields = np.asarray(fields, dtype=np.float64)
    if fields.ndim != 3:
        raise ValueError("fields must have shape (channels, height, width)")
    channels, height, width = fields.shape
    if channels < 1 or height < 1 or width < 1:
        raise ValueError("fields must have non-empty dimensions")
    x0, y0, fx, fy = _particle_grid_coordinates(pos, world, (height, width))
    sampled = np.zeros((len(x0), channels), dtype=np.float64)
    for dy, wy in ((0, 1.0 - fy), (1, fy)):
        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            sampled += fields[:, (y0 + dy) % height, (x0 + dx) % width].T * (wx * wy)[:, None]
    return sampled