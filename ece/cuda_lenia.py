"""Optional CUDA implementation of the 2D Lenia growth update."""

from __future__ import annotations

import math

import numpy as np

from .config import SimConfig
from .cuda_mohr import cuda_available
from .lenia import LeniaState, seed_state
from .lenia import ring_kernel_2d as lenia_ring_kernel_2d


_CUDA_LENIA_KERNEL = None


def _get_kernel():
    global _CUDA_LENIA_KERNEL
    if _CUDA_LENIA_KERNEL is None:
        from numba import cuda

        @cuda.jit
        def lenia_kernel(rho, kernel, output, height, width, radius, mu, sigma, dt):
            index = cuda.grid(1)
            if index >= height * width:
                return
            x = index % width
            y = index // width
            potential = 0.0
            for ky in range(2 * radius + 1):
                yy = (y + ky - radius) % height
                for kx in range(2 * radius + 1):
                    xx = (x + kx - radius) % width
                    potential += kernel[ky, kx] * rho[0, yy, xx]
            growth = 2.0 * math.exp(-0.5 * ((potential - mu) / sigma) ** 2) - 1.0
            value = rho[0, y, x] + dt * growth
            if value < 0.0:
                value = 0.0
            elif value > 1.0:
                value = 1.0
            output[0, y, x] = value

        _CUDA_LENIA_KERNEL = lenia_kernel
    return _CUDA_LENIA_KERNEL


def lenia_step_cuda(rho: np.ndarray, kernel: np.ndarray, mu: float, sigma: float, dt: float) -> np.ndarray:
    if not cuda_available():
        raise RuntimeError("CUDA is unavailable; install the cuda extra and check the driver")
    from numba import cuda

    rho = np.ascontiguousarray(rho, dtype=np.float32)
    kernel = np.ascontiguousarray(kernel, dtype=np.float32)
    if rho.ndim != 3 or rho.shape[0] != 1:
        raise ValueError("CUDA Lenia currently requires one channel of 2D field data")
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or kernel.shape[0] % 2 != 1:
        raise ValueError("kernel must be a square 2D stencil with odd width")
    channels, height, width = rho.shape
    radius = kernel.shape[0] // 2
    if height < kernel.shape[0] or width < kernel.shape[1]:
        raise ValueError("kernel dimensions must not exceed field dimensions")
    if sigma <= 0.0 or dt <= 0.0:
        raise ValueError("sigma and dt must be positive")

    d_rho = cuda.to_device(rho)
    d_kernel = cuda.to_device(kernel)
    d_output = cuda.device_array(rho.shape, dtype=np.float32)
    threads = 128
    blocks = (height * width + threads - 1) // threads
    _get_kernel()[blocks, threads](
        d_rho, d_kernel, d_output, height, width, radius,
        np.float32(mu), np.float32(sigma), np.float32(dt),
    )
    return d_output.copy_to_host().astype(np.float64)


def step_cuda(state: LeniaState, cfg: SimConfig) -> LeniaState:
    if cfg.field_cfg.get("kernel", "ring") != "ring":
        raise ValueError("CUDA Lenia currently supports the ring kernel")
    radius = max(2, min(state.rho.shape[1:]) // 8)
    rho = lenia_step_cuda(
        state.rho,
        lenia_ring_kernel_2d(radius),
        float(cfg.lenia["mu"]),
        float(cfg.lenia["sigma"]),
        float(cfg.lenia.get("dt", cfg.dt)),
    )
    return LeniaState(rho=rho, frame=state.frame + 1)


def run_cuda(cfg: SimConfig, frames: int) -> LeniaState:
    if frames < 0:
        raise ValueError("frames must be non-negative")
    state = seed_state(cfg)
    for _ in range(frames):
        state = step_cuda(state, cfg)
    return state