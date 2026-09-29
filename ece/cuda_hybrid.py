"""Resident CUDA hybrid scheduler with stable host-side PIC planning."""

from __future__ import annotations

import numpy as np

from .config import SimConfig
from .cuda_field_life import _get_kernels as _get_field_kernels
from .cuda_mohr import (
    _get_cuda_binned_kernel,
    _get_cuda_resident_kernels,
    cuda_available,
)
from .cupy_field_life import cupy_available
from .cuda_pic import _get_kernels as _get_pic_kernels
from .field_life import FieldLifeState
from .hybrid import HybridState, seed_state
from .step_mohr import MohrState


def _offsets_from_counts(cp, counts):
    offsets = cp.empty_like(counts)
    offsets[0] = 0
    if counts.size > 1:
        offsets[1:] = cp.cumsum(counts[:-1], dtype=counts.dtype)
    return offsets


def _stable_hash_plan(cp, pos, bin_widths, grid):
    gx, gy = grid
    bx = cp.floor(pos[:, 0] / bin_widths[0]).astype(cp.int32) % gx
    by = cp.floor(pos[:, 1] / bin_widths[1]).astype(cp.int32) % gy
    bin_ids = by * gx + bx
    order = cp.argsort(bin_ids, kind="stable")
    counts = cp.bincount(bin_ids, minlength=gx * gy).astype(cp.int64)
    offsets = _offsets_from_counts(cp, counts)
    ranges = cp.stack((offsets, offsets + counts), axis=1)
    return ranges, order


def _stable_deposit_plan(cp, pos, types, world, shape, channels):
    height, width = shape
    gx = cp.remainder(pos[:, 0], world[0]) * (width / world[0])
    gy = cp.remainder(pos[:, 1], world[1]) * (height / world[1])
    x0 = cp.floor(gx).astype(cp.int32)
    y0 = cp.floor(gy).astype(cp.int32)
    fx = gx - x0
    fy = gy - y0
    target_parts = []
    value_parts = []
    for dy, wy in ((0, 1.0 - fy), (1, fy)):
        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            target_parts.append(
                types * height * width
                + ((y0 + dy) % height) * width
                + (x0 + dx) % width
            )
            value_parts.append(wx * wy)
    targets = cp.concatenate(target_parts)
    values = cp.concatenate(value_parts)
    order = cp.argsort(targets, kind="stable")
    sorted_values = values[order].astype(cp.float32)
    total_cells = channels * height * width
    counts = cp.bincount(targets, minlength=total_cells).astype(cp.int32)
    offsets = _offsets_from_counts(cp, counts)
    return sorted_values, offsets, counts


def run_cuda_resident(cfg: SimConfig, frames: int) -> HybridState:
    """Keep hybrid state and stable hash/PIC planning on CUDA across frames."""
    if not cuda_available():
        raise RuntimeError("CUDA is unavailable; install the cuda extra and check the driver")
    if not cupy_available():
        raise RuntimeError("device-side stable hybrid planning requires the cuda-fft extra")
    if frames < 0:
        raise ValueError("frames must be non-negative")
    initial = seed_state(cfg)
    dims = tuple(int(size) for size in cfg.field_cfg["dims"])
    if len(dims) != 2 or cfg.world.size != 2:
        raise ValueError("resident CUDA hybrid currently requires a 2D world and field")
    if cfg.field_cfg.get("kernel", "ring") != "ring":
        raise ValueError("resident CUDA hybrid currently supports the ring kernel")

    from numba import cuda
    import cupy as cp

    from .mace import ring_kernel_2d

    positions_host = np.ascontiguousarray(initial.particles.pos, dtype=np.float64)
    types_host = np.ascontiguousarray(initial.particles.types, dtype=np.int32)
    particle_count = len(positions_host)
    channels = int(cfg.field_cfg["channels"])
    height, width = dims
    cells = channels * height * width
    grid = tuple(int(size) for size in cfg.hash["grid"])
    grid_x, grid_y = grid
    neighborhood = int(cfg.hash.get("neighborhood", 3))
    if neighborhood not in (3, 5):
        raise ValueError("hash neighborhood must be 3 or 5")
    radius = neighborhood // 2
    bin_widths = np.asarray(cfg.world, dtype=np.float32) / np.asarray(grid, dtype=np.float32)

    d_pos = cuda.to_device(np.ascontiguousarray(positions_host, dtype=np.float32))
    d_vel = cuda.to_device(np.ascontiguousarray(initial.particles.vel, dtype=np.float32))
    d_types = cuda.to_device(types_host)
    d_matrix = cuda.to_device(np.ascontiguousarray(cfg.matrix, dtype=np.float32))
    d_world = cuda.to_device(np.ascontiguousarray(cfg.world, dtype=np.float32))
    d_widths = cuda.to_device(bin_widths)
    d_accel = cuda.device_array((particle_count, 2), dtype=np.float32)
    d_next_pos = cuda.device_array((particle_count, 2), dtype=np.float32)
    d_next_vel = cuda.device_array((particle_count, 2), dtype=np.float32)
    d_rho = cuda.device_array((channels, height, width), dtype=np.float32)
    d_affinity = cuda.device_array_like(d_rho)
    d_maxima = cuda.device_array_like(d_rho)
    d_denominators = cuda.device_array_like(d_rho)
    d_field_output = cuda.device_array_like(d_rho)
    d_field_matrix = cuda.to_device(np.ascontiguousarray(cfg.matrix, dtype=np.float32))
    d_field_kernel = cuda.to_device(np.ascontiguousarray(ring_kernel_2d(), dtype=np.float32))

    force_kernel = _get_cuda_binned_kernel()
    _, _, _, _, _, integrate = _get_cuda_resident_kernels()
    reduce_deposits, _ = _get_pic_kernels()
    affinity_kernel, maximum_kernel, denominator_kernel, gather_kernel = _get_field_kernels()
    threads = 128
    particle_blocks = max((particle_count + threads - 1) // threads, 1)
    cell_blocks = (cells + threads - 1) // threads
    bin_blocks = (grid_x * grid_y + threads - 1) // threads
    field_total = channels * height * width
    field_blocks = (field_total + threads - 1) // threads
    dt = np.float32(cfg.dt)
    decay = np.float32(np.exp(-float(cfg.mohr.get("lambda", 0.0)) * cfg.dt))
    r_max = np.float32(cfg.mohr["r_max"])
    beta = np.float32(cfg.mohr["beta"])
    gain = np.float32(cfg.mohr.get("gain", 1.0))

    world_gpu = cp.asarray(d_world)
    widths_gpu = cp.asarray(d_widths)
    types_gpu = cp.asarray(d_types)
    positions_gpu = cp.asarray(d_pos)
    for _ in range(frames):
        ranges_gpu, order_gpu = _stable_hash_plan(cp, positions_gpu, widths_gpu, grid)
        force_kernel[particle_blocks, threads](
            d_pos, d_types, d_matrix, d_world, d_widths, ranges_gpu, order_gpu,
            grid_x, grid_y, radius, r_max, beta, gain, d_accel,
        )
        integrate[particle_blocks, threads](
            d_pos, d_vel, d_accel, d_next_pos, d_next_vel, d_world, dt, decay
        )
        d_pos, d_next_pos = d_next_pos, d_pos
        d_vel, d_next_vel = d_next_vel, d_vel
        positions_gpu = cp.asarray(d_pos)
        values_gpu, offsets_gpu, counts_gpu = _stable_deposit_plan(
            cp, positions_gpu, types_gpu, world_gpu, dims, channels
        )
        reduce_deposits[cell_blocks, threads](
            values_gpu, offsets_gpu, counts_gpu, d_rho.reshape(cells)
        )

        affinity_kernel[field_blocks, threads](
            d_rho, d_field_matrix, d_field_kernel, d_affinity, channels, height, width, 3,
            np.float32(cfg.field_cfg.get("strength", 1.0)),
            np.float32(cfg.field_cfg.get("crowding_lambda", 0.0)),
        )
        maximum_kernel[field_blocks, threads](d_affinity, d_maxima, channels, height, width)
        denominator_kernel[field_blocks, threads](
            d_affinity, d_maxima, d_denominators, channels, height, width,
            np.float32(cfg.field_cfg.get("transport_beta", 1.0)),
        )
        gather_kernel[field_blocks, threads](
            d_rho, d_affinity, d_maxima, d_denominators, d_field_output, channels, height, width,
            np.float32(cfg.field_cfg.get("transport_beta", 1.0)),
        )

    positions_host = d_pos.copy_to_host().astype(np.float64)
    particles = MohrState(
        pos=positions_host,
        vel=d_vel.copy_to_host().astype(np.float64),
        types=types_host,
        frame=frames,
    )
    if frames:
        rho = d_field_output.copy_to_host().astype(np.float64)
        positions_gpu = cp.asarray(d_pos)
        field_gpu = cp.asarray(d_field_output)
        sample_x = cp.remainder(positions_gpu[:, 0], world_gpu[0]) * (width / world_gpu[0])
        sample_y = cp.remainder(positions_gpu[:, 1], world_gpu[1]) * (height / world_gpu[1])
        x0 = cp.floor(sample_x).astype(cp.int32)
        y0 = cp.floor(sample_y).astype(cp.int32)
        fx = sample_x - x0
        fy = sample_y - y0
        sampled_gpu = cp.zeros((particle_count, channels), dtype=cp.float32)
        for dy, wy in ((0, 1.0 - fy), (1, fy)):
            for dx, wx in ((0, 1.0 - fx), (1, fx)):
                sampled_gpu += field_gpu[:, (y0 + dy) % height, (x0 + dx) % width].T * (wx * wy)[:, None]
        sampled = cp.asnumpy(sampled_gpu).astype(np.float64)
    else:
        rho = initial.rho
        sampled = initial.sampled
    return HybridState(particles=particles, rho=rho, sampled=sampled)