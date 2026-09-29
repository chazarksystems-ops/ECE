"""CPU Field Life runner backed by the MaCE reference kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimConfig
from .mace import mace_step, ring_kernel_2d, ring_kernel_3d


@dataclass
class FieldLifeState:
    rho: np.ndarray
    frame: int = 0


def seed_state(cfg: SimConfig) -> FieldLifeState:
    if "field_life" not in cfg.rules:
        raise ValueError("config rules must include 'field_life'")
    dims = tuple(int(size) for size in cfg.field_cfg.get("dims", ()))
    if len(dims) not in (2, 3) or any(size < 2 for size in dims):
        raise ValueError("Field Life requires two or three dimensions of size at least 2")
    channels = int(cfg.field_cfg.get("channels", cfg.species_count))
    if channels != cfg.species_count:
        raise ValueError("field.channels must equal species.count")
    rng = np.random.default_rng(cfg.seed + 29)
    rho = rng.random((channels, *dims), dtype=np.float64)
    return FieldLifeState(rho=rho)


def step(state: FieldLifeState, cfg: SimConfig) -> FieldLifeState:
    if cfg.field_cfg.get("kernel", "ring") != "ring":
        raise ValueError("CPU Field Life currently supports the ring kernel")
    if state.rho.ndim == 3:
        kernel = ring_kernel_2d()
    else:
        kernel = ring_kernel_3d()
    rho = mace_step(
        state.rho,
        cfg.matrix,
        kernel,
        strength=float(cfg.field_cfg.get("strength", 1.0)),
        crowding_lambda=float(cfg.field_cfg.get("crowding_lambda", 0.0)),
        transport_beta=float(cfg.field_cfg.get("transport_beta", 1.0)),
    )
    return FieldLifeState(rho=rho, frame=state.frame + 1)


def run(cfg: SimConfig, frames: int) -> FieldLifeState:
    if frames < 0:
        raise ValueError("frames must be non-negative")
    state = seed_state(cfg)
    for _ in range(frames):
        state = step(state, cfg)
    return state