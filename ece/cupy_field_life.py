"""Optional CuPy/cuFFT Field Life updates with resident 3D state."""

from __future__ import annotations

import numpy as np

from .config import SimConfig
from .field_life import FieldLifeState, seed_state
from .mace import ring_kernel_3d

_TRANSPORT_KERNELS = None


def _get_transport_kernels(cp):
    global _TRANSPORT_KERNELS
    if _TRANSPORT_KERNELS is None:
        local_maximum = cp.RawKernel(
            r"""
            extern "C" __global__ void local_maximum(
                const float* affinity, float* maxima,
                int channels, int depth, int height, int width)
            {
                int index = blockDim.x * blockIdx.x + threadIdx.x;
                int total = channels * depth * height * width;
                if (index >= total) return;
                int plane = height * width;
                int volume = depth * plane;
                int channel = index / volume;
                int z = (index / plane) % depth;
                int y = (index / width) % height;
                int x = index % width;
                float maximum = -3.402823466e+38F;
                for (int dz = -1; dz <= 1; ++dz) {
                    int zz = (z + dz + depth) % depth;
                    for (int dy = -1; dy <= 1; ++dy) {
                        int yy = (y + dy + height) % height;
                        for (int dx = -1; dx <= 1; ++dx) {
                            int xx = (x + dx + width) % width;
                            int neighbor = channel * volume + zz * plane + yy * width + xx;
                            maximum = fmaxf(maximum, affinity[neighbor]);
                        }
                    }
                }
                maxima[index] = maximum;
            }
            """,
            "local_maximum",
        )
        local_denominators = cp.RawKernel(
            r"""
            extern "C" __global__ void local_denominators(
                const float* affinity, const float* maxima, float* denominators,
                int channels, int depth, int height, int width, float beta)
            {
                int index = blockDim.x * blockIdx.x + threadIdx.x;
                int total = channels * depth * height * width;
                if (index >= total) return;
                int plane = height * width;
                int volume = depth * plane;
                int channel = index / volume;
                int z = (index / plane) % depth;
                int y = (index / width) % height;
                int x = index % width;
                float maximum = maxima[index];
                float denominator = 0.0F;
                for (int dz = -1; dz <= 1; ++dz) {
                    int zz = (z + dz + depth) % depth;
                    for (int dy = -1; dy <= 1; ++dy) {
                        int yy = (y + dy + height) % height;
                        for (int dx = -1; dx <= 1; ++dx) {
                            int xx = (x + dx + width) % width;
                            int neighbor = channel * volume + zz * plane + yy * width + xx;
                            denominator += expf(beta * (affinity[neighbor] - maximum));
                        }
                    }
                }
                denominators[index] = denominator;
            }
            """,
            "local_denominators",
        )
        local_gather = cp.RawKernel(
            r"""
            extern "C" __global__ void local_gather(
                const float* rho, const float* affinity, const float* maxima,
                const float* denominators, float* output,
                int channels, int depth, int height, int width, float beta)
            {
                int index = blockDim.x * blockIdx.x + threadIdx.x;
                int total = channels * depth * height * width;
                if (index >= total) return;
                int plane = height * width;
                int volume = depth * plane;
                int channel = index / volume;
                int z = (index / plane) % depth;
                int y = (index / width) % height;
                int x = index % width;
                float result = 0.0F;
                for (int dz = -1; dz <= 1; ++dz) {
                    int zz = (z + dz + depth) % depth;
                    for (int dy = -1; dy <= 1; ++dy) {
                        int yy = (y + dy + height) % height;
                        for (int dx = -1; dx <= 1; ++dx) {
                            int xx = (x + dx + width) % width;
                            int source = channel * volume + zz * plane + yy * width + xx;
                            float transition = expf(beta * (affinity[index] - maxima[source]));
                            result += rho[source] * transition / denominators[source];
                        }
                    }
                }
                output[index] = result;
            }
            """,
            "local_gather",
        )
        _TRANSPORT_KERNELS = (local_maximum, local_denominators, local_gather)
    return _TRANSPORT_KERNELS


def cupy_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def mace_step_cupy(
    rho,
    matrix: np.ndarray,
    kernel: np.ndarray,
    strength: float,
    crowding_lambda: float,
    transport_beta: float,
):
    """Run one 3D affinity/FFT/MaCE update on CuPy arrays."""
    import cupy as cp

    if rho.ndim != 4:
        raise ValueError("CuPy Field Life requires (channels, depth, height, width) data")
    if kernel.ndim != 3 or any(size % 2 != 1 for size in kernel.shape):
        raise ValueError("3D ring kernel must have odd dimensions")
    channels = rho.shape[0]
    if matrix.shape != (channels, channels):
        raise ValueError("matrix must be square with one row per channel")
    spatial = rho.shape[1:]
    if any(k > size for k, size in zip(kernel.shape, spatial)):
        raise ValueError("kernel dimensions must not exceed field dimensions")

    kernel_gpu = cp.asarray(kernel, dtype=cp.float32)
    matrix_gpu = cp.asarray(matrix, dtype=cp.float32)
    padded = cp.zeros(spatial, dtype=cp.float32)
    padded[tuple(slice(0, size) for size in kernel.shape)] = kernel_gpu
    padded = cp.roll(
        padded,
        shift=tuple(-(size // 2) for size in kernel.shape),
        axis=(0, 1, 2),
    )
    kernel_spectrum = cp.fft.rfftn(padded, axes=(0, 1, 2))
    density_spectrum = cp.fft.rfftn(rho, axes=(1, 2, 3))
    convolved = cp.fft.irfftn(
        density_spectrum * kernel_spectrum[None, ...],
        s=spatial,
        axes=(1, 2, 3),
    ).real

    affinity = cp.einsum("cd,dzyx->czyx", matrix_gpu, convolved)
    affinity = strength * affinity - crowding_lambda * rho
    maxima = cp.empty_like(rho)
    denominators = cp.empty_like(rho)
    output = cp.empty_like(rho)
    channels, depth, height, width = rho.shape
    total = rho.size
    threads = 128
    blocks = (total + threads - 1) // threads
    maximum_kernel, denominator_kernel, gather_kernel = _get_transport_kernels(cp)
    maximum_kernel((blocks,), (threads,), (affinity, maxima, channels, depth, height, width))
    denominator_kernel(
        (blocks,),
        (threads,),
        (affinity, maxima, denominators, channels, depth, height, width, np.float32(transport_beta)),
    )
    gather_kernel(
        (blocks,),
        (threads,),
        (rho, affinity, maxima, denominators, output,
         channels, depth, height, width, np.float32(transport_beta)),
    )
    return output


def run_cupy(cfg: SimConfig, frames: int) -> FieldLifeState:
    if frames < 0:
        raise ValueError("frames must be non-negative")
    if not cupy_available():
        raise RuntimeError("CuPy CUDA FFT backend is unavailable; install the cuda-fft extra")
    if len(cfg.field_cfg.get("dims", ())) != 3:
        raise ValueError("CuPy Field Life runner currently expects three spatial dimensions")
    if cfg.field_cfg.get("kernel", "ring") != "ring":
        raise ValueError("CuPy Field Life currently supports the ring kernel")

    import cupy as cp

    state = seed_state(cfg)
    rho = cp.asarray(state.rho, dtype=cp.float32)
    kernel = ring_kernel_3d()
    for _ in range(frames):
        rho = mace_step_cupy(
            rho,
            cfg.matrix,
            kernel,
            float(cfg.field_cfg.get("strength", 1.0)),
            float(cfg.field_cfg.get("crowding_lambda", 0.0)),
            float(cfg.field_cfg.get("transport_beta", 1.0)),
        )
    return FieldLifeState(rho=cp.asnumpy(rho).astype(np.float64), frame=frames)