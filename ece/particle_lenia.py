"""Energy-gradient Particle Lenia adapted to ECE's periodic 2D world."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import SimConfig
from .mohr import wrapped_delta


@dataclass
class ParticleLeniaState:
    pos: np.ndarray
    vel: np.ndarray
    types: np.ndarray
    frame: int = 0


def shell_kernel_weight(mu: float, sigma: float, dimensions: int = 2) -> float:
    if mu <= 0.0 or sigma <= 0.0 or dimensions not in (2, 3):
        raise ValueError("kernel scales must be positive and dimension must be 2 or 3")
    radii = np.linspace(0.0, mu + 8.0 * sigma, 32769, dtype=np.float64)
    profile = np.exp(-((radii - mu) / sigma) ** 2) * radii ** (dimensions - 1)
    integral = np.sum((profile[:-1] + profile[1:]) * 0.5 * np.diff(radii))
    surface_area = 2.0 * math.pi ** (dimensions / 2.0) / math.gamma(dimensions / 2.0)
    return float(1.0 / (surface_area * integral))


def particle_lenia_energy(
    query: np.ndarray,
    positions: np.ndarray,
    world: np.ndarray,
    params: dict,
) -> float:
    delta = wrapped_delta(np.asarray(query, dtype=np.float64), positions, world)
    distance = np.linalg.norm(delta, axis=1)
    kernel_mu = float(params["kernel_mu"])
    kernel_sigma = float(params["kernel_sigma"])
    density = shell_kernel_weight(kernel_mu, kernel_sigma) * np.exp(
        -((distance - kernel_mu) / kernel_sigma) ** 2
    ).sum()
    growth_mu = float(params["growth_mu"])
    growth_sigma = float(params["growth_sigma"])
    growth = math.exp(-((density - growth_mu) / growth_sigma) ** 2)
    repulsive_distance = distance[distance > 1e-12]
    repulsion = np.maximum(1.0 - repulsive_distance, 0.0)
    return float(0.5 * float(params["c_rep"]) * np.sum(repulsion**2) - growth)


def particle_lenia_velocity(pos: np.ndarray, world: np.ndarray, params: dict) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64)
    world = np.asarray(world, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 2 or world.shape != (2,):
        raise ValueError("Particle Lenia positions and world must be 2D")
    kernel_mu = float(params["kernel_mu"])
    kernel_sigma = float(params["kernel_sigma"])
    growth_mu = float(params["growth_mu"])
    growth_sigma = float(params["growth_sigma"])
    c_rep = float(params["c_rep"])
    weight = shell_kernel_weight(kernel_mu, kernel_sigma)
    velocities = np.zeros_like(pos)

    for i, position in enumerate(pos):
        delta = wrapped_delta(position, pos, world)
        distance = np.linalg.norm(delta, axis=1)
        kernel = weight * np.exp(-((distance - kernel_mu) / kernel_sigma) ** 2)
        density = float(kernel.sum())
        kernel_derivative = -2.0 * (distance - kernel_mu) / (kernel_sigma**2) * kernel
        nonzero = distance > 1e-12
        direction = np.zeros_like(delta)
        direction[nonzero] = delta[nonzero] / distance[nonzero, None]
        density_gradient = np.sum(kernel_derivative[:, None] * direction, axis=0)

        growth = math.exp(-((density - growth_mu) / growth_sigma) ** 2)
        growth_derivative = -2.0 * (density - growth_mu) / (growth_sigma**2) * growth
        repulsive = nonzero & (distance < 1.0)
        repulsion_scale = np.zeros_like(distance)
        repulsion_scale[repulsive] = c_rep * (1.0 - distance[repulsive]) / distance[repulsive]
        repulsion_velocity = np.sum(repulsion_scale[:, None] * delta, axis=0)
        velocities[i] = repulsion_velocity + growth_derivative * density_gradient

    return velocities


def seed_state(cfg: SimConfig) -> ParticleLeniaState:
    if cfg.rules != ["particle_lenia"]:
        raise ValueError("config rules must be ['particle_lenia']")
    count = int(cfg.particle_lenia["particles"])
    rng = np.random.default_rng(cfg.seed + 53)
    pos = rng.random((count, 2)) * cfg.world
    return ParticleLeniaState(
        pos=pos,
        vel=np.zeros_like(pos),
        types=np.zeros(count, dtype=np.int32),
    )


def step(state: ParticleLeniaState, cfg: SimConfig) -> ParticleLeniaState:
    velocity = particle_lenia_velocity(state.pos, cfg.world, cfg.particle_lenia)
    dt = float(cfg.particle_lenia.get("dt", cfg.dt))
    positions = np.mod(state.pos + dt * velocity, cfg.world)
    return ParticleLeniaState(
        pos=positions,
        vel=velocity,
        types=state.types,
        frame=state.frame + 1,
    )


def run(cfg: SimConfig, frames: int) -> ParticleLeniaState:
    if frames < 0:
        raise ValueError("frames must be non-negative")
    state = seed_state(cfg)
    for _ in range(frames):
        state = step(state, cfg)
    return state