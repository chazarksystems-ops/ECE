"""CPU Mohr stepper used as the M1 host reference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bins import snapshot_and_scatter, validate_hash_grid
from .config import SimConfig
from .integrate import symplectic_euler
from .mohr import mohr_accelerations, mohr_force_scalar, wrapped_delta


@dataclass
class MohrState:
    pos: np.ndarray
    vel: np.ndarray
    types: np.ndarray
    frame: int = 0


def seed_state(cfg: SimConfig) -> MohrState:
    n = int(cfg.mohr.get("particles", 0))
    rng = np.random.default_rng(cfg.seed + 17)
    dim = cfg.world.size
    pos = rng.random((n, dim)) * cfg.world
    vel = np.zeros((n, dim), dtype=np.float64)
    types = rng.integers(0, cfg.species_count, size=n, dtype=np.int32)
    return MohrState(pos=pos, vel=vel, types=types)


def step(state: MohrState, cfg: SimConfig, use_bins: bool = True) -> MohrState:
    r_max = float(cfg.mohr["r_max"])
    beta = float(cfg.mohr["beta"])
    gain = float(cfg.mohr.get("gain", 1.0))
    lam = float(cfg.mohr.get("lambda", 0.0))
    if use_bins and cfg.world.size == 2 and cfg.hash:
        accel = _binned_accel(state, cfg, r_max, beta, gain)
    else:
        accel = mohr_accelerations(
            state.pos, state.types, cfg.matrix, r_max, beta, cfg.world, gain
        )
    pos, vel = symplectic_euler(state.pos, state.vel, accel, cfg.dt, lam, cfg.world)
    return MohrState(pos=pos, vel=vel, types=state.types, frame=state.frame + 1)


_PAIR_CHUNK = 4096


def _binned_accel(state: MohrState, cfg: SimConfig, r_max, beta, gain) -> np.ndarray:
    grid = tuple(int(x) for x in cfg.hash["grid"])
    neighborhood = int(cfg.hash.get("neighborhood", 3))
    radius = validate_hash_grid(grid, neighborhood, cfg.world, r_max)
    tables = snapshot_and_scatter(state.pos, float(cfg.hash["cell"]), grid, cfg.world)
    pos = state.pos
    types = np.asarray(state.types, dtype=np.int64)
    world = cfg.world
    n = len(pos)
    accel = np.zeros_like(pos)
    if n == 0:
        return accel
    gx, gy = grid
    ranges = tables["ranges"]
    order = tables["sorted_indices"]
    bin_of = tables["bin_of"]
    bx = bin_of % gx
    by = bin_of // gx
    # Same nested (dy, dx) walk order as the GPU kernels. The grid check above
    # guarantees these are distinct bins, so no particle is visited twice.
    offsets = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]
    for start in range(0, n, _PAIR_CHUNK):
        owners = np.arange(start, min(start + _PAIR_CHUNK, n))
        neighbor_bins = np.stack(
            [((by[owners] + dy) % gy) * gx + (bx[owners] + dx) % gx for dy, dx in offsets],
            axis=1,
        ).ravel()
        lo = ranges[neighbor_bins, 0]
        counts = ranges[neighbor_bins, 1] - lo
        total = int(counts.sum())
        if total == 0:
            continue
        first = np.repeat(np.cumsum(counts) - counts, counts)
        slots = np.repeat(lo, counts) + (np.arange(total) - first)
        i = np.repeat(np.repeat(owners, len(offsets)), counts)
        j = order[slots]
        delta = wrapped_delta(pos[j], pos[i], world)
        dist = np.linalg.norm(delta, axis=1)
        keep = (j != i) & (dist >= 1e-15) & (dist <= r_max)
        i, j, delta, dist = i[keep], j[keep], delta[keep], dist[keep]
        f = mohr_force_scalar(dist / r_max, cfg.matrix[types[i], types[j]], beta)
        contrib = (gain * f / dist)[:, None] * delta
        local = i - start
        for axis in range(2):
            accel[owners, axis] = np.bincount(local, weights=contrib[:, axis], minlength=len(owners))
    return accel


def run(cfg: SimConfig, frames: int, use_bins: bool = True) -> MohrState:
    st = seed_state(cfg)
    for _ in range(frames):
        st = step(st, cfg, use_bins=use_bins)
    return st
