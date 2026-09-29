"""Optional CUDA implementation of the 2D Field Life update."""

from __future__ import annotations

import math

import numpy as np

from .config import SimConfig
from .field_life import FieldLifeState, seed_state
from .mace import ring_kernel_2d


_CUDA_FIELD_KERNELS = None


def _get_kernels():
    global _CUDA_FIELD_KERNELS
    if _CUDA_FIELD_KERNELS is None:
        from numba import cuda

        @cuda.jit
        def affinity_kernel(rho, matrix, kernel, affinity, channels, height, width, radius, strength, crowding):
            index = cuda.grid(1)
            total = channels * height * width
            if index >= total:
                return
            x = index % width
            y = (index // width) % height
            channel = index // (height * width)
            mixed = 0.0
            for source in range(channels):
                convolution = 0.0
                for ky in range(2 * radius + 1):
                    yy = (y + ky - radius) % height
                    for kx in range(2 * radius + 1):
                        xx = (x + kx - radius) % width
                        convolution += kernel[ky, kx] * rho[source, yy, xx]
                mixed += matrix[channel, source] * convolution
            affinity[channel, y, x] = strength * mixed - crowding * rho[channel, y, x]

        @cuda.jit
        def maximum_kernel(affinity, maxima, channels, height, width):
            index = cuda.grid(1)
            total = channels * height * width
            if index >= total:
                return
            x = index % width
            y = (index // width) % height
            channel = index // (height * width)
            maximum = -math.inf
            for dy in range(-1, 2):
                yy = (y + dy) % height
                for dx in range(-1, 2):
                    xx = (x + dx) % width
                    value = affinity[channel, yy, xx]
                    if value > maximum:
                        maximum = value
            maxima[channel, y, x] = maximum

        @cuda.jit
        def denominator_kernel(affinity, maxima, denominators, channels, height, width, beta):
            index = cuda.grid(1)
            total = channels * height * width
            if index >= total:
                return
            x = index % width
            y = (index // width) % height
            channel = index // (height * width)
            maximum = maxima[channel, y, x]
            denominator = 0.0
            for dy in range(-1, 2):
                yy = (y + dy) % height
                for dx in range(-1, 2):
                    xx = (x + dx) % width
                    denominator += math.exp(beta * (affinity[channel, yy, xx] - maximum))
            denominators[channel, y, x] = denominator

        @cuda.jit
        def gather_kernel(rho, affinity, maxima, denominators, output, channels, height, width, beta):
            index = cuda.grid(1)
            total = channels * height * width
            if index >= total:
                return
            x = index % width
            y = (index // width) % height
            channel = index // (height * width)
            share = 0.0
            for dy in range(-1, 2):
                yy = (y + dy) % height
                for dx in range(-1, 2):
                    xx = (x + dx) % width
                    origin_maximum = maxima[channel, yy, xx]
                    transition = math.exp(beta * (affinity[channel, y, x] - origin_maximum))
                    share += rho[channel, yy, xx] * transition / denominators[channel, yy, xx]
            output[channel, y, x] = share

        _CUDA_FIELD_KERNELS = (affinity_kernel, maximum_kernel, denominator_kernel, gather_kernel)
    return _CUDA_FIELD_KERNELS


def mace_step_cuda(
    rho: np.ndarray,
    matrix: np.ndarray,
    kernel: np.ndarray,
    strength: float,
    crowding_lambda: float,
    transport_beta: float,
) -> np.ndarray:
    """Run a single channel-first 2D Field Life update on CUDA."""
    from .cuda_mohr import cuda_available

    if not cuda_available():
        raise RuntimeError("CUDA is unavailable; install the cuda extra and check the driver")
    from numba import cuda

    rho = np.ascontiguousarray(rho, dtype=np.float32)
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    kernel = np.ascontiguousarray(kernel, dtype=np.float32)
    if rho.ndim != 3 or matrix.shape != (rho.shape[0], rho.shape[0]):
        raise ValueError("rho must be (channels, height, width) and matrix must be square by channel")
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or kernel.shape[0] % 2 != 1:
        raise ValueError("kernel must be a square 2D stencil with odd width")
    channels, height, width = rho.shape
    radius = kernel.shape[0] // 2
    if height < kernel.shape[0] or width < kernel.shape[1]:
        raise ValueError("kernel dimensions must not exceed field dimensions")

    d_rho = cuda.to_device(rho)
    d_matrix = cuda.to_device(matrix)
    d_kernel = cuda.to_device(kernel)
    d_affinity = cuda.device_array(rho.shape, dtype=np.float32)
    d_maxima = cuda.device_array(rho.shape, dtype=np.float32)
    d_denominators = cuda.device_array(rho.shape, dtype=np.float32)
    d_output = cuda.device_array(rho.shape, dtype=np.float32)
    total = channels * height * width
    threads = 128
    blocks = (total + threads - 1) // threads
    affinity_kernel, maximum_kernel, denominator_kernel, gather_kernel = _get_kernels()
    affinity_kernel[blocks, threads](
        d_rho, d_matrix, d_kernel, d_affinity, channels, height, width, radius,
        np.float32(strength), np.float32(crowding_lambda),
    )
    maximum_kernel[blocks, threads](d_affinity, d_maxima, channels, height, width)
    denominator_kernel[blocks, threads](
        d_affinity, d_maxima, d_denominators, channels, height, width, np.float32(transport_beta)
    )
    gather_kernel[blocks, threads](
        d_rho, d_affinity, d_maxima, d_denominators, d_output, channels, height, width,
        np.float32(transport_beta),
    )
    return d_output.copy_to_host().astype(np.float64)


def step_cuda(state: FieldLifeState, cfg: SimConfig) -> FieldLifeState:
    if cfg.field_cfg.get("kernel", "ring") != "ring":
        raise ValueError("CUDA Field Life currently supports the ring kernel")
    kernel = ring_kernel_2d()
    rho = mace_step_cuda(
        state.rho,
        cfg.matrix,
        kernel,
        float(cfg.field_cfg.get("strength", 1.0)),
        float(cfg.field_cfg.get("crowding_lambda", 0.0)),
        float(cfg.field_cfg.get("transport_beta", 1.0)),
    )
    return FieldLifeState(rho=rho, frame=state.frame + 1)


def run_cuda(cfg: SimConfig, frames: int) -> FieldLifeState:
    if frames < 0:
        raise ValueError("frames must be non-negative")
    state = seed_state(cfg)
    for _ in range(frames):
        state = step_cuda(state, cfg)
    return state