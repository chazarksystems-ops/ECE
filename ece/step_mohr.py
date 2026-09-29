"""CPU Mohr stepper used as the M1 host reference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bins import snapshot_and_scatter
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


def _binned_accel(state: MohrState, cfg: SimConfig, r_max, beta, gain) -> np.ndarray:
    cell = float(cfg.hash["cell"])
    grid = tuple(int(x) for x in cfg.hash["grid"])
    tables = snapshot_and_scatter(state.pos, cell, grid, cfg.world)
    n = len(state.pos)
    accel = np.zeros_like(state.pos)
    gx, gy = grid
    ranges = tables["ranges"]
    order = tables["sorted_indices"]
    pos = state.pos
    types = state.types
    world = cfg.world
    bin_widths = world / np.asarray(grid, dtype=np.float64)
    neighborhood_radius = int(cfg.hash.get("neighborhood", 3)) // 2
    for i in range(n):
        x = pos[i]
        ti = int(types[i])
        bx = int(np.floor(x[0] / bin_widths[0]) % gx)
        by = int(np.floor(x[1] / bin_widths[1]) % gy)
        acc = np.zeros(2)
        for dy in range(-neighborhood_radius, neighborhood_radius + 1):
            for dx in range(-neighborhood_radius, neighborhood_radius + 1):
                nx = (bx + dx) % gx
                ny = (by + dy) % gy
                b = ny * gx + nx
                lo, hi = ranges[b]
                for s in range(lo, hi):
                    j = int(order[s])
                    if j == i:
                        continue
                    delta = wrapped_delta(pos[j], x, world)
                    dist = float(np.linalg.norm(delta))
                    if dist < 1e-15:
                        continue
                    r = dist / r_max
                    if r > 1.0:
                        continue
                    a = cfg.matrix[ti, int(types[j])]
                    f = float(mohr_force_scalar(np.array([r]), np.array([a]), beta)[0])
                    acc += gain * f * (delta / dist)
        accel[i] = acc
    return accel


def run(cfg: SimConfig, frames: int, use_bins: bool = True) -> MohrState:
    st = seed_state(cfg)
    for _ in range(frames):
        st = step(st, cfg, use_bins=use_bins)
    return st
