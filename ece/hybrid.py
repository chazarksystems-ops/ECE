"""Hybrid Mohr and Field Life frame scheduler using PIC exchange."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimConfig
from .field_life import FieldLifeState, step as step_field_life
from .pic import deposit_particles, sample_fields
from .step_mohr import MohrState, seed_state as seed_mohr_state, step as step_mohr


@dataclass
class HybridState:
    particles: MohrState
    rho: np.ndarray
    sampled: np.ndarray

    @property
    def frame(self) -> int:
        return self.particles.frame


def seed_state(cfg: SimConfig) -> HybridState:
    if set(cfg.rules) != {"mohr", "field_life"}:
        raise ValueError("hybrid runner requires both mohr and field_life rules")
    particles = seed_mohr_state(cfg)
    dims = tuple(int(size) for size in cfg.field_cfg.get("dims", ()))
    channels = int(cfg.field_cfg.get("channels", cfg.species_count))
    if len(dims) != 2 or channels != cfg.species_count:
        raise ValueError("hybrid runner requires a 2D field with one channel per species")
    rho = np.zeros((channels, *dims), dtype=np.float64)
    sampled = np.zeros((len(particles.pos), channels), dtype=np.float64)
    return HybridState(particles=particles, rho=rho, sampled=sampled)


def step(
    state: HybridState,
    cfg: SimConfig,
    use_cuda: bool = False,
    use_wgpu: bool = False,
) -> HybridState:
    if use_cuda and use_wgpu:
        raise ValueError("choose only one GPU backend")
    if use_cuda:
        from .cuda_pic import deposit_particles_cuda, sample_fields_cuda
        from .cuda_field_life import step_cuda as step_field_life_cuda
        from .cuda_mohr import step_cuda as step_mohr_cuda

        particles = step_mohr_cuda(state.particles, cfg)
        deposit = deposit_particles_cuda
        sample = sample_fields_cuda
    elif use_wgpu:
        from .wgpu_field_life import step_wgpu as step_field_life_wgpu
        from .wgpu_mohr import step_wgpu as step_mohr_wgpu
        from .wgpu_pic import deposit_particles_wgpu, sample_fields_wgpu

        particles = step_mohr_wgpu(state.particles, cfg)
        deposit = deposit_particles_wgpu
        sample = sample_fields_wgpu
    else:
        particles = step_mohr(state.particles, cfg)
        deposit = deposit_particles
        sample = sample_fields

    dims = tuple(int(size) for size in cfg.field_cfg["dims"])
    channels = int(cfg.field_cfg["channels"])
    deposited = deposit(
        particles.pos,
        particles.types,
        cfg.world,
        dims,
        channels,
    )
    field_state = FieldLifeState(rho=deposited, frame=particles.frame - 1)
    if use_cuda:
        field_state = step_field_life_cuda(field_state, cfg)
    elif use_wgpu:
        field_state = step_field_life_wgpu(field_state, cfg)
    else:
        field_state = step_field_life(field_state, cfg)
    sampled = sample(field_state.rho, particles.pos, cfg.world)
    return HybridState(particles=particles, rho=field_state.rho, sampled=sampled)


def run(
    cfg: SimConfig,
    frames: int,
    use_cuda: bool = False,
    use_wgpu: bool = False,
) -> HybridState:
    if frames < 0:
        raise ValueError("frames must be non-negative")
    if use_cuda and use_wgpu:
        raise ValueError("choose only one GPU backend")
    state = seed_state(cfg)
    for _ in range(frames):
        state = step(state, cfg, use_cuda=use_cuda, use_wgpu=use_wgpu)
    return state