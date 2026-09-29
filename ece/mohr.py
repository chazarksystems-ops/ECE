"""Mohr Particle Life tent — corrected force scaling.

Catalog error: several snippets multiply the unit vector by ``r_max``
after evaluating the dimensionless tent ``F(r, a)``. That makes peak
force depend on world scale and breaks matrix portability.

Correct acceleration of particle i:

    a_i = sum_{j != i} F(r_ij, A[c_i, c_j]) * hat(x_j - x_i)

where ``r_ij = ||x_j - x_i||_wrap / r_max`` and F is Mohr's tent.
A single optional global gain ``g`` may be applied; default g = 1.
"""

from __future__ import annotations

import numpy as np


def mohr_force_scalar(r: np.ndarray, a: np.ndarray, beta: float) -> np.ndarray:
    """Dimensionless Mohr tent.

    Parameters
    ----------
    r : array
        Distance normalized by r_max. Shape broadcastable with ``a``.
    a : array
        Matrix entry A[c_i, c_j] in [-1, 1].
    beta : float
        Close-range wall cutoff in (0, 1). Typical 0.3.
    """
    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    r = np.asarray(r, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    out = np.zeros_like(r, dtype=np.float64)
    wall = r < beta
    mid = (r >= beta) & (r <= 1.0)
    out[wall] = r[wall] / beta - 1.0
    # Peak of the tent sits at r = (1 + beta) / 2.
    out[mid] = a[mid] * (1.0 - np.abs(2.0 * r[mid] - 1.0 - beta) / (1.0 - beta))
    return out


def wrapped_delta(a: np.ndarray, b: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Minimum-image vector a - b on a torus of size ``world``."""
    d = a - b
    d -= world * np.round(d / world)
    return d


def mohr_accelerations(
    pos: np.ndarray,
    types: np.ndarray,
    matrix: np.ndarray,
    r_max: float,
    beta: float,
    world: np.ndarray,
    gain: float = 1.0,
) -> np.ndarray:
    """O(N^2) reference accelerations. Use only for tests and small N.

    ``gain`` is the only dimensionful scale. Do not multiply by r_max.
    """
    pos = np.asarray(pos, dtype=np.float64)
    types = np.asarray(types, dtype=np.int32)
    matrix = np.asarray(matrix, dtype=np.float64)
    world = np.asarray(world, dtype=np.float64)
    n, dim = pos.shape
    accel = np.zeros_like(pos)
    if n == 0:
        return accel
    for i in range(n):
        delta = wrapped_delta(pos, pos[i], world)  # (N, dim), includes i
        dist = np.linalg.norm(delta, axis=1)
        dist[i] = np.inf
        r = dist / r_max
        a = matrix[types[i], types]
        f = mohr_force_scalar(r, a, beta)
        # Safe unit vector: skip self (dist=inf) and coincidences.
        finite = np.isfinite(dist) & (dist > 1e-15)
        unit = np.zeros_like(delta)
        unit[finite] = delta[finite] / dist[finite, None]
        accel[i] = gain * np.sum(f[:, None] * unit, axis=0)
    return accel
