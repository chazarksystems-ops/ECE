"""CUDA Particle Lenia using the author energy-gradient rule."""

from __future__ import annotations

import math

import numpy as np

from .config import SimConfig
from .cuda_mohr import cuda_available
from .particle_lenia import ParticleLeniaState, seed_state, shell_kernel_weight


_KERNEL = None


def _get_kernel():
    global _KERNEL
    if _KERNEL is None:
        from numba import cuda

        @cuda.jit
        def particle_lenia_step(pos, next_pos, velocity, world, kernel_mu, kernel_sigma,
                                growth_mu, growth_sigma, c_rep, kernel_weight, dt):
            i = cuda.grid(1)
            if i >= pos.shape[0]:
                return
            density = 0.0
            grad_density_x = 0.0
            grad_density_y = 0.0
            repulsion_x = 0.0
            repulsion_y = 0.0
            for j in range(pos.shape[0]):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dx -= world[0] * math.floor(dx / world[0] + 0.5)
                dy -= world[1] * math.floor(dy / world[1] + 0.5)
                distance = math.sqrt(dx * dx + dy * dy)
                radial = (distance - kernel_mu) / kernel_sigma
                kernel_value = kernel_weight * math.exp(-(radial * radial))
                density += kernel_value
                if distance > 1e-12:
                    kernel_derivative = -2.0 * (distance - kernel_mu) / (kernel_sigma * kernel_sigma) * kernel_value
                    grad_density_x += kernel_derivative * dx / distance
                    grad_density_y += kernel_derivative * dy / distance
                    if distance < 1.0:
                        repulsion_scale = c_rep * (1.0 - distance) / distance
                        repulsion_x += repulsion_scale * dx
                        repulsion_y += repulsion_scale * dy
            growth = math.exp(-((density - growth_mu) / growth_sigma) ** 2)
            growth_derivative = -2.0 * (density - growth_mu) / (growth_sigma * growth_sigma) * growth
            vx = repulsion_x + growth_derivative * grad_density_x
            vy = repulsion_y + growth_derivative * grad_density_y
            x = pos[i, 0] + dt * vx
            y = pos[i, 1] + dt * vy
            x -= world[0] * math.floor(x / world[0])
            y -= world[1] * math.floor(y / world[1])
            next_pos[i, 0] = x
            next_pos[i, 1] = y
            velocity[i, 0] = vx
            velocity[i, 1] = vy

        _KERNEL = particle_lenia_step
    return _KERNEL


def run_cuda(cfg: SimConfig, frames: int) -> ParticleLeniaState:
    if not cuda_available():
        raise RuntimeError("CUDA is unavailable; install the cuda extra and check the driver")
    if frames < 0:
        raise ValueError("frames must be non-negative")
    if cfg.rules != ["particle_lenia"] or cfg.world.size != 2:
        raise ValueError("CUDA Particle Lenia currently supports the 2D single-species rule")
    from numba import cuda

    state = seed_state(cfg)
    count = len(state.pos)
    d_pos = cuda.to_device(np.ascontiguousarray(state.pos, dtype=np.float32))
    d_next = cuda.device_array((count, 2), dtype=np.float32)
    d_velocity = cuda.device_array((count, 2), dtype=np.float32)
    d_world = cuda.to_device(np.ascontiguousarray(cfg.world, dtype=np.float32))
    params = cfg.particle_lenia
    kernel_weight = shell_kernel_weight(float(params["kernel_mu"]), float(params["kernel_sigma"]))
    threads = 128
    blocks = max((count + threads - 1) // threads, 1)
    kernel = _get_kernel()
    for _ in range(frames):
        kernel[blocks, threads](
            d_pos,
            d_next,
            d_velocity,
            d_world,
            np.float32(params["kernel_mu"]),
            np.float32(params["kernel_sigma"]),
            np.float32(params["growth_mu"]),
            np.float32(params["growth_sigma"]),
            np.float32(params["c_rep"]),
            np.float32(kernel_weight),
            np.float32(params.get("dt", cfg.dt)),
        )
        d_pos, d_next = d_next, d_pos
    state.pos = d_pos.copy_to_host().astype(np.float64)
    if frames:
        state.vel = d_velocity.copy_to_host().astype(np.float64)
    state.frame = frames
    return state