"""Deterministic CUDA PIC exchange using sorted cell contributions."""

from __future__ import annotations

import numpy as np

from .cuda_mohr import cuda_available
from .pic import _particle_grid_coordinates


_CUDA_PIC_KERNELS = None


def _get_kernels():
    global _CUDA_PIC_KERNELS
    if _CUDA_PIC_KERNELS is None:
        from numba import cuda

        @cuda.jit
        def reduce_deposits(values, offsets, counts, output):
            cell = cuda.grid(1)
            if cell >= output.size:
                return
            total = 0.0
            for item in range(offsets[cell], offsets[cell] + counts[cell]):
                total += values[item]
            output[cell] = total

        @cuda.jit
        def sample_kernel(fields, x0, y0, fx, fy, world_shape, sampled):
            particle = cuda.grid(1)
            if particle >= sampled.shape[0]:
                return
            height = fields.shape[1]
            width = fields.shape[2]
            ix = x0[particle]
            iy = y0[particle]
            frac_x = fx[particle]
            frac_y = fy[particle]
            for channel in range(fields.shape[0]):
                value = 0.0
                for dy in range(2):
                    wy = frac_y if dy == 1 else 1.0 - frac_y
                    yy = (iy + dy) % height
                    for dx in range(2):
                        wx = frac_x if dx == 1 else 1.0 - frac_x
                        xx = (ix + dx) % width
                        value += fields[channel, yy, xx] * wx * wy
                sampled[particle, channel] = value

        _CUDA_PIC_KERNELS = (reduce_deposits, sample_kernel)
    return _CUDA_PIC_KERNELS


def deposit_particles_cuda(
    pos: np.ndarray,
    types: np.ndarray,
    world: np.ndarray,
    shape: tuple[int, int],
    channels: int,
    masses: np.ndarray | None = None,
) -> np.ndarray:
    """Deposit particle mass with stable host sorting and device segmented sums."""
    if not cuda_available():
        raise RuntimeError("CUDA is unavailable; install the cuda extra and check the driver")
    from numba import cuda

    from .bins import exclusive_scan

    pos = np.asarray(pos, dtype=np.float64)
    types = np.asarray(types, dtype=np.int64)
    if types.shape != (len(pos),) or channels < 1:
        raise ValueError("types must match particles and channels must be positive")
    if np.any(types < 0) or np.any(types >= channels):
        raise ValueError("particle types must be valid field channel indices")
    if masses is None:
        masses = np.ones(len(pos), dtype=np.float64)
    else:
        masses = np.asarray(masses, dtype=np.float64)
    if masses.shape != (len(pos),) or not np.isfinite(masses).all() or np.any(masses < 0.0):
        raise ValueError("masses must be finite, non-negative, and match particle count")

    height, width = shape
    x0, y0, fx, fy = _particle_grid_coordinates(pos, world, shape)
    target_parts = []
    value_parts = []
    for dy, wy in ((0, 1.0 - fy), (1, fy)):
        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            target_parts.append(types * height * width + ((y0 + dy) % height) * width + (x0 + dx) % width)
            value_parts.append(masses * wx * wy)
    targets = np.concatenate(target_parts)
    values = np.concatenate(value_parts)
    order = np.argsort(targets, kind="stable")
    values = np.ascontiguousarray(values[order], dtype=np.float32)
    total_cells = channels * height * width
    counts = np.bincount(targets, minlength=total_cells).astype(np.int32)
    offsets = exclusive_scan(counts).astype(np.int32)

    d_values = cuda.to_device(values)
    d_offsets = cuda.to_device(offsets)
    d_counts = cuda.to_device(counts)
    d_output = cuda.device_array(total_cells, dtype=np.float32)
    threads = 128
    reduce_deposits, _ = _get_kernels()
    reduce_deposits[(total_cells + threads - 1) // threads, threads](
        d_values, d_offsets, d_counts, d_output
    )
    return d_output.copy_to_host().reshape(channels, height, width).astype(np.float64)


def sample_fields_cuda(fields: np.ndarray, pos: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Bilinearly sample a channel-first periodic field on CUDA."""
    if not cuda_available():
        raise RuntimeError("CUDA is unavailable; install the cuda extra and check the driver")
    from numba import cuda

    fields = np.ascontiguousarray(fields, dtype=np.float32)
    pos = np.asarray(pos, dtype=np.float64)
    world = np.asarray(world, dtype=np.float64)
    if fields.ndim != 3:
        raise ValueError("fields must have shape (channels, height, width)")
    x0, y0, fx, fy = _particle_grid_coordinates(pos, world, fields.shape[1:])
    if len(pos) == 0:
        return np.zeros((0, fields.shape[0]), dtype=np.float64)
    d_fields = cuda.to_device(fields)
    d_x0 = cuda.to_device(x0.astype(np.int32))
    d_y0 = cuda.to_device(y0.astype(np.int32))
    d_fx = cuda.to_device(fx.astype(np.float32))
    d_fy = cuda.to_device(fy.astype(np.float32))
    d_sampled = cuda.device_array((len(pos), fields.shape[0]), dtype=np.float32)
    threads = 128
    _, sample_kernel = _get_kernels()
    sample_kernel[(len(pos) + threads - 1) // threads, threads](
        d_fields, d_x0, d_y0, d_fx, d_fy, fields.shape, d_sampled
    )
    return d_sampled.copy_to_host().astype(np.float64)