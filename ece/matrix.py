"""Species-matrix generators."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def make_matrix(
    mode: str,
    k: int,
    rng: np.random.Generator,
    lo: float = -1.0,
    hi: float = 1.0,
    path: str | None = None,
) -> np.ndarray:
    if k < 1:
        raise ValueError("k must be >= 1")
    if mode == "random":
        return rng.uniform(lo, hi, size=(k, k))
    if mode == "identity":
        m = np.full((k, k), -0.4, dtype=np.float64)
        np.fill_diagonal(m, 1.0)
        return m
    if mode == "chase":
        m = np.zeros((k, k), dtype=np.float64)
        for i in range(k):
            m[i, (i + 1) % k] = 0.8
            m[i, i] = -0.2
            m[(i + 1) % k, i] = -0.6
        return m
    if mode == "path":
        if not path:
            raise ValueError("matrix mode 'path' requires matrix_path")
        raw = np.loadtxt(path, dtype=np.float64)
        raw = np.atleast_2d(raw)
        if raw.shape != (k, k):
            raise ValueError(f"matrix file shape {raw.shape} != ({k}, {k})")
        return raw
    raise ValueError(f"unknown matrix mode {mode!r}")


def parse_matrix_values(rows: list[list[str]], size: int) -> np.ndarray:
    """Parse editor text into a ``size`` x ``size`` matrix with entries in [-1, 1]."""
    try:
        matrix = np.asarray([[float(value) for value in row] for row in rows])
    except ValueError as exc:
        raise ValueError("matrix entries must be numbers") from exc
    if matrix.shape != (size, size):
        raise ValueError(f"matrix must be {size} by {size}")
    if not np.isfinite(matrix).all() or np.any(matrix < -1.0) or np.any(matrix > 1.0):
        raise ValueError("matrix entries must be finite and between -1 and 1")
    return matrix


def save_matrix(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, fmt="%.6f")
