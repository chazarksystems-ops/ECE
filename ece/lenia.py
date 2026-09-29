"""CPU Lenia growth rule on periodic scalar fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimConfig
from .mace import convolve_periodic


@dataclass
class LeniaState:
    rho: np.ndarray
    frame: int = 0


def ring_kernel_2d(radius: int) -> np.ndarray:
    if radius < 2:
        raise ValueError("kernel radius must be at least 2")
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    radial = np.sqrt(xx * xx + yy * yy) / radius
    kernel = np.exp(-0.5 * ((radial - 0.5) / 0.15) ** 2)
    kernel[radial > 1.0] = 0.0
    return kernel / kernel.sum()


def lenia_step(rho: np.ndarray, kernel: np.ndarray, mu: float, sigma: float, dt: float) -> np.ndarray:
    if sigma <= 0.0 or dt <= 0.0:
        raise ValueError("sigma and dt must be positive")
    potential = convolve_periodic(rho, kernel)
    growth = 2.0 * np.exp(-0.5 * ((potential - mu) / sigma) ** 2) - 1.0
    return np.clip(np.asarray(rho, dtype=np.float64) + dt * growth, 0.0, 1.0)


def seed_state(cfg: SimConfig) -> LeniaState:
    if "lenia" not in cfg.rules:
        raise ValueError("config rules must include 'lenia'")
    dims = tuple(int(size) for size in cfg.field_cfg.get("dims", ()))
    if len(dims) != 2 or any(size < 2 for size in dims):
        raise ValueError("Lenia currently requires two dimensions of size at least 2")
    if int(cfg.field_cfg.get("channels", 1)) != 1:
        raise ValueError("Lenia currently requires exactly one field channel")
    height, width = dims
    radius = max(2, min(dims) // 8)
    seed_radius = max(2.0, min(dims) * 0.0625)
    yy, xx = np.ogrid[:height, :width]
    distance = np.sqrt((yy - (height - 1) / 2.0) ** 2 + (xx - (width - 1) / 2.0) ** 2)
    rho = np.exp(-0.5 * ((distance - seed_radius) / (seed_radius * 0.22)) ** 2)
    return LeniaState(rho=rho[np.newaxis, ...], frame=0)


def step(state: LeniaState, cfg: SimConfig) -> LeniaState:
    if cfg.field_cfg.get("kernel", "ring") != "ring":
        raise ValueError("CPU Lenia currently supports the ring kernel")
    dims = state.rho.shape[1:]
    kernel = ring_kernel_2d(max(2, min(dims) // 8))
    rho = lenia_step(
        state.rho,
        kernel,
        mu=float(cfg.lenia["mu"]),
        sigma=float(cfg.lenia["sigma"]),
        dt=float(cfg.lenia.get("dt", cfg.dt)),
    )
    return LeniaState(rho=rho, frame=state.frame + 1)


def run(cfg: SimConfig, frames: int) -> LeniaState:
    if frames < 0:
        raise ValueError("frames must be non-negative")
    state = seed_state(cfg)
    for _ in range(frames):
        state = step(state, cfg)
    return state