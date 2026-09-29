"""Optional CUDA acceleration for the corrected Mohr force kernel."""

from __future__ import annotations

import math

import numpy as np

from .integrate import symplectic_euler
from .config import SimConfig
from .step_mohr import MohrState


_CUDA_KERNEL = None
_CUDA_BINNED_KERNEL = None
_CUDA_RESIDENT_KERNELS = None


def _get_cuda_kernel():
    global _CUDA_KERNEL
    if _CUDA_KERNEL is None:
        from numba import cuda

        @cuda.jit
        def force_kernel(pos, types, matrix, world, r_max, beta, gain, accel):
            i = cuda.grid(1)
            if i >= pos.shape[0]:
                return
            ax = 0.0
            ay = 0.0
            for j in range(pos.shape[0]):
                if j == i:
                    continue
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dx -= world[0] * math.floor(dx / world[0] + 0.5)
                dy -= world[1] * math.floor(dy / world[1] + 0.5)
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 1e-15 or dist > r_max:
                    continue
                r = dist / r_max
                a = matrix[types[i], types[j]]
                if r < beta:
                    force = r / beta - 1.0
                else:
                    force = a * (1.0 - abs(2.0 * r - 1.0 - beta) / (1.0 - beta))
                scale = gain * force / dist
                ax += scale * dx
                ay += scale * dy
            accel[i, 0] = ax
            accel[i, 1] = ay

        _CUDA_KERNEL = force_kernel
    return _CUDA_KERNEL


def _get_cuda_binned_kernel():
    global _CUDA_BINNED_KERNEL
    if _CUDA_BINNED_KERNEL is None:
        from numba import cuda

        @cuda.jit
        def force_kernel_binned(
            pos,
            types,
            matrix,
            world,
            bin_widths,
            ranges,
            sorted_indices,
            grid_x,
            grid_y,
            neighborhood_radius,
            r_max,
            beta,
            gain,
            accel,
        ):
            i = cuda.grid(1)
            if i >= pos.shape[0]:
                return
            bx = int(math.floor(pos[i, 0] / bin_widths[0])) % grid_x
            by = int(math.floor(pos[i, 1] / bin_widths[1])) % grid_y
            ax = 0.0
            ay = 0.0
            for dy in range(-neighborhood_radius, neighborhood_radius + 1):
                for dx in range(-neighborhood_radius, neighborhood_radius + 1):
                    nx = (bx + dx) % grid_x
                    ny = (by + dy) % grid_y
                    bin_id = ny * grid_x + nx
                    for slot in range(ranges[bin_id, 0], ranges[bin_id, 1]):
                        j = sorted_indices[slot]
                        if j == i:
                            continue
                        delta_x = pos[j, 0] - pos[i, 0]
                        delta_y = pos[j, 1] - pos[i, 1]
                        delta_x -= world[0] * math.floor(delta_x / world[0] + 0.5)
                        delta_y -= world[1] * math.floor(delta_y / world[1] + 0.5)
                        dist = math.sqrt(delta_x * delta_x + delta_y * delta_y)
                        if dist < 1e-15 or dist > r_max:
                            continue
                        r = dist / r_max
                        a = matrix[types[i], types[j]]
                        if r < beta:
                            force = r / beta - 1.0
                        else:
                            force = a * (1.0 - abs(2.0 * r - 1.0 - beta) / (1.0 - beta))
                        scale = gain * force / dist
                        ax += scale * delta_x
                        ay += scale * delta_y
            accel[i, 0] = ax
            accel[i, 1] = ay

        _CUDA_BINNED_KERNEL = force_kernel_binned
    return _CUDA_BINNED_KERNEL


def cuda_available() -> bool:
    try:
        from numba import cuda
        return cuda.is_available()
    except Exception:
        return False


def mohr_accelerations_cuda(
    pos: np.ndarray,
    types: np.ndarray,
    matrix: np.ndarray,
    r_max: float,
    beta: float,
    world: np.ndarray,
    gain: float = 1.0,
) -> np.ndarray:
    """Compute dense 2D Mohr accelerations on CUDA and return float64 host data."""
    if not cuda_available():
        raise RuntimeError("CUDA is unavailable; install the cuda extra and check the driver")
    pos = np.asarray(pos, dtype=np.float32)
    types = np.asarray(types, dtype=np.int32)
    matrix = np.asarray(matrix, dtype=np.float32)
    world = np.asarray(world, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("CUDA Mohr currently requires positions with shape (N, 2)")
    if types.shape != (len(pos),) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("types and matrix must have compatible shapes")
    if np.any(types < 0) or np.any(types >= matrix.shape[0]):
        raise ValueError("types must index rows and columns of matrix")
    if world.shape != (2,) or np.any(world <= 0.0):
        raise ValueError("world must contain two positive lengths")
    if r_max <= 0.0 or not 0.0 < beta < 1.0:
        raise ValueError("r_max must be positive and beta must be in (0, 1)")
    if len(pos) == 0:
        return np.zeros_like(pos, dtype=np.float64)

    from numba import cuda

    d_pos = cuda.to_device(np.ascontiguousarray(pos))
    d_types = cuda.to_device(np.ascontiguousarray(types))
    d_matrix = cuda.to_device(np.ascontiguousarray(matrix))
    d_world = cuda.to_device(np.ascontiguousarray(world))
    d_accel = cuda.device_array(pos.shape, dtype=np.float32)
    threads = 128
    _get_cuda_kernel()[(len(pos) + threads - 1) // threads, threads](
        d_pos, d_types, d_matrix, d_world, np.float32(r_max), np.float32(beta), np.float32(gain), d_accel
    )
    return d_accel.copy_to_host().astype(np.float64)


def mohr_accelerations_cuda_binned(
    pos: np.ndarray,
    types: np.ndarray,
    matrix: np.ndarray,
    r_max: float,
    beta: float,
    world: np.ndarray,
    cell: float,
    grid: tuple[int, int],
    neighborhood: int = 3,
    gain: float = 1.0,
) -> np.ndarray:
    """Compute binned Mohr forces on CUDA using CPU-built frozen hash tables."""
    if not cuda_available():
        raise RuntimeError("CUDA is unavailable; install the cuda extra and check the driver")
    from numba import cuda

    from .bins import snapshot_and_scatter

    pos = np.ascontiguousarray(pos, dtype=np.float32)
    types = np.ascontiguousarray(types, dtype=np.int32)
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    world = np.ascontiguousarray(world, dtype=np.float32)
    grid = tuple(int(size) for size in grid)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("CUDA Mohr currently requires positions with shape (N, 2)")
    if types.shape != (len(pos),) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("types and matrix must have compatible shapes")
    if np.any(types < 0) or np.any(types >= matrix.shape[0]):
        raise ValueError("types must index rows and columns of matrix")
    if world.shape != (2,) or np.any(world <= 0.0):
        raise ValueError("world must contain two positive lengths")
    if len(grid) != 2 or any(size < 1 for size in grid):
        raise ValueError("grid must contain two positive dimensions")
    if r_max <= 0.0 or not 0.0 < beta < 1.0:
        raise ValueError("r_max must be positive and beta must be in (0, 1)")
    if neighborhood not in (3, 5):
        raise ValueError("neighborhood must be 3 or 5")
    neighborhood_radius = neighborhood // 2
    bin_widths = world / np.asarray(grid, dtype=np.float32)
    if np.any(bin_widths + 1e-7 < r_max / neighborhood_radius):
        raise ValueError("world/grid cells are too small for the requested neighborhood")
    if len(pos) == 0:
        return np.zeros_like(pos, dtype=np.float64)

    tables = snapshot_and_scatter(pos, cell, grid, world)
    d_pos = cuda.to_device(pos)
    d_types = cuda.to_device(types)
    d_matrix = cuda.to_device(matrix)
    d_world = cuda.to_device(world)
    d_widths = cuda.to_device(np.ascontiguousarray(bin_widths))
    d_ranges = cuda.to_device(np.ascontiguousarray(tables["ranges"], dtype=np.int64))
    d_order = cuda.to_device(np.ascontiguousarray(tables["sorted_indices"], dtype=np.int64))
    d_accel = cuda.device_array(pos.shape, dtype=np.float32)
    threads = 128
    _get_cuda_binned_kernel()[(len(pos) + threads - 1) // threads, threads](
        d_pos,
        d_types,
        d_matrix,
        d_world,
        d_widths,
        d_ranges,
        d_order,
        grid[0],
        grid[1],
        neighborhood_radius,
        np.float32(r_max),
        np.float32(beta),
        np.float32(gain),
        d_accel,
    )
    return d_accel.copy_to_host().astype(np.float64)


def step_cuda(state: MohrState, cfg: SimConfig) -> MohrState:
    if cfg.hash:
        accel = mohr_accelerations_cuda_binned(
            state.pos,
            state.types,
            cfg.matrix,
            float(cfg.mohr["r_max"]),
            float(cfg.mohr["beta"]),
            cfg.world,
            float(cfg.hash["cell"]),
            tuple(cfg.hash["grid"]),
            int(cfg.hash.get("neighborhood", 3)),
            float(cfg.mohr.get("gain", 1.0)),
        )
    else:
        accel = mohr_accelerations_cuda(
            state.pos,
            state.types,
            cfg.matrix,
            float(cfg.mohr["r_max"]),
            float(cfg.mohr["beta"]),
            cfg.world,
            float(cfg.mohr.get("gain", 1.0)),
        )
    pos, vel = symplectic_euler(
        state.pos,
        state.vel,
        accel,
        cfg.dt,
        float(cfg.mohr.get("lambda", 0.0)),
        cfg.world,
    )
    return MohrState(pos=pos, vel=vel, types=state.types, frame=state.frame + 1)


def _get_cuda_resident_kernels():
    global _CUDA_RESIDENT_KERNELS
    if _CUDA_RESIDENT_KERNELS is None:
        from numba import cuda

        @cuda.jit
        def clear_counts(counts):
            index = cuda.grid(1)
            if index < counts.size:
                counts[index] = 0

        @cuda.jit
        def count_bins(pos, counts, widths, grid_x, grid_y):
            i = cuda.grid(1)
            if i >= pos.shape[0]:
                return
            x = int(math.floor(pos[i, 0] / widths[0])) % grid_x
            y = int(math.floor(pos[i, 1] / widths[1])) % grid_y
            cuda.atomic.add(counts, y * grid_x + x, 1)

        @cuda.jit
        def scan_bins(counts, offsets, ranges, cursors):
            if cuda.grid(1) != 0:
                return
            total = 0
            for bin_id in range(counts.size):
                offsets[bin_id] = total
                ranges[bin_id, 0] = total
                total += counts[bin_id]
                ranges[bin_id, 1] = total
                cursors[bin_id] = offsets[bin_id]

        @cuda.jit
        def scatter_bins(pos, widths, grid_x, grid_y, cursors, sorted_indices):
            i = cuda.grid(1)
            if i >= pos.shape[0]:
                return
            x = int(math.floor(pos[i, 0] / widths[0])) % grid_x
            y = int(math.floor(pos[i, 1] / widths[1])) % grid_y
            bin_id = y * grid_x + x
            slot = cuda.atomic.add(cursors, bin_id, 1)
            sorted_indices[slot] = i

        @cuda.jit
        def force_bins(pos, types, matrix, world, widths, ranges, sorted_indices,
                       grid_x, grid_y, radius, r_max, beta, gain, accel):
            i = cuda.grid(1)
            if i >= pos.shape[0]:
                return
            bx = int(math.floor(pos[i, 0] / widths[0])) % grid_x
            by = int(math.floor(pos[i, 1] / widths[1])) % grid_y
            ax = 0.0
            ay = 0.0
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx = (bx + dx) % grid_x
                    ny = (by + dy) % grid_y
                    bin_id = ny * grid_x + nx
                    for slot in range(ranges[bin_id, 0], ranges[bin_id, 1]):
                        j = sorted_indices[slot]
                        if j == i:
                            continue
                        delta_x = pos[j, 0] - pos[i, 0]
                        delta_y = pos[j, 1] - pos[i, 1]
                        delta_x -= world[0] * math.floor(delta_x / world[0] + 0.5)
                        delta_y -= world[1] * math.floor(delta_y / world[1] + 0.5)
                        distance = math.sqrt(delta_x * delta_x + delta_y * delta_y)
                        if distance < 1e-15 or distance > r_max:
                            continue
                        normalized = distance / r_max
                        a = matrix[types[i], types[j]]
                        if normalized < beta:
                            force = normalized / beta - 1.0
                        else:
                            force = a * (1.0 - abs(2.0 * normalized - 1.0 - beta) / (1.0 - beta))
                        scale = gain * force / distance
                        ax += scale * delta_x
                        ay += scale * delta_y
            accel[i, 0] = ax
            accel[i, 1] = ay

        @cuda.jit
        def integrate(pos, vel, accel, next_pos, next_vel, world, dt, decay):
            i = cuda.grid(1)
            if i >= pos.shape[0]:
                return
            vx = decay * vel[i, 0] + accel[i, 0] * dt
            vy = decay * vel[i, 1] + accel[i, 1] * dt
            x = pos[i, 0] + vx * dt
            y = pos[i, 1] + vy * dt
            x -= world[0] * math.floor(x / world[0])
            y -= world[1] * math.floor(y / world[1])
            next_pos[i, 0] = x
            next_pos[i, 1] = y
            next_vel[i, 0] = vx
            next_vel[i, 1] = vy

        _CUDA_RESIDENT_KERNELS = (clear_counts, count_bins, scan_bins, scatter_bins, force_bins, integrate)
    return _CUDA_RESIDENT_KERNELS


def run_cuda_resident(cfg: SimConfig, frames: int) -> MohrState:
    """Run hashed 2D Mohr with resident state and stable CuPy bin planning."""
    if not cuda_available():
        raise RuntimeError("CUDA is unavailable; install the cuda extra and check the driver")
    if frames < 0:
        raise ValueError("frames must be non-negative")
    if cfg.world.size != 2 or not cfg.hash:
        raise ValueError("resident CUDA Mohr requires a 2D config with spatial hash settings")
    from .cupy_field_life import cupy_available

    if not cupy_available():
        raise RuntimeError("stable GPU hash planning requires the cuda-fft extra")
    from numba import cuda
    import cupy as cp

    from .step_mohr import seed_state

    state = seed_state(cfg)
    n = len(state.pos)
    if n == 0:
        state.frame = frames
        return state
    grid = tuple(int(size) for size in cfg.hash["grid"])
    grid_x, grid_y = grid
    bins = grid_x * grid_y
    widths = np.asarray(cfg.world, dtype=np.float32) / np.asarray(grid, dtype=np.float32)
    neighborhood = int(cfg.hash.get("neighborhood", 3))
    if neighborhood not in (3, 5):
        raise ValueError("hash neighborhood must be 3 or 5")
    radius = neighborhood // 2
    r_max = float(cfg.mohr["r_max"])
    if np.any(widths + 1e-7 < r_max / radius):
        raise ValueError("world/grid cells are too small for the configured hash neighborhood")

    d_pos = cuda.to_device(np.ascontiguousarray(state.pos, dtype=np.float32))
    d_vel = cuda.to_device(np.ascontiguousarray(state.vel, dtype=np.float32))
    d_types = cuda.to_device(np.ascontiguousarray(state.types, dtype=np.int32))
    d_matrix = cuda.to_device(np.ascontiguousarray(cfg.matrix, dtype=np.float32))
    d_world = cuda.to_device(np.ascontiguousarray(cfg.world, dtype=np.float32))
    d_widths = cuda.to_device(widths)
    d_accel = cuda.device_array((n, 2), dtype=np.float32)
    d_next_pos = cuda.device_array((n, 2), dtype=np.float32)
    d_next_vel = cuda.device_array((n, 2), dtype=np.float32)

    _, _, _, _, force_bins, integrate = _get_cuda_resident_kernels()
    threads = 128
    particle_blocks = (n + threads - 1) // threads
    lam = float(cfg.mohr.get("lambda", 0.0))
    decay = np.exp(-lam * cfg.dt)
    device_positions = cp.asarray(d_pos)
    for _ in range(frames):
        bin_x = cp.floor(device_positions[:, 0] / widths[0]).astype(cp.int32) % grid_x
        bin_y = cp.floor(device_positions[:, 1] / widths[1]).astype(cp.int32) % grid_y
        bin_ids = bin_y * grid_x + bin_x
        sorted_indices = cp.argsort(bin_ids, kind="stable").astype(cp.int64)
        counts = cp.bincount(bin_ids, minlength=bins).astype(cp.int64)
        offsets = cp.empty_like(counts)
        offsets[0] = 0
        if bins > 1:
            offsets[1:] = cp.cumsum(counts[:-1], dtype=cp.int64)
        ranges = cp.stack((offsets, offsets + counts), axis=1)
        force_bins[particle_blocks, threads](
            d_pos, d_types, d_matrix, d_world, d_widths, ranges, sorted_indices,
            grid_x, grid_y, radius, np.float32(r_max), np.float32(cfg.mohr["beta"]),
            np.float32(cfg.mohr.get("gain", 1.0)), d_accel,
        )
        integrate[particle_blocks, threads](
            d_pos, d_vel, d_accel, d_next_pos, d_next_vel, d_world,
            np.float32(cfg.dt), np.float32(decay),
        )
        d_pos, d_next_pos = d_next_pos, d_pos
        d_vel, d_next_vel = d_next_vel, d_vel
        device_positions = cp.asarray(d_pos)

    state.pos = d_pos.copy_to_host().astype(np.float64)
    state.vel = d_vel.copy_to_host().astype(np.float64)
    state.frame = frames
    return state