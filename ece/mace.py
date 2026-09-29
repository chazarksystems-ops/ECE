"""MaCE mass-conserving transport — two-pass, torus wrap.

Pass A — denominators
    Z[x] = sum_{y in N(x)} exp(beta * A[y])

Pass B — gather
    rho'[x] = sum_{y in N(x)} rho[y] * exp(beta * A[x]) / Z[y]

N(x) is the 9-cell (2D) or 27-cell (3D) neighborhood including self,
wrapped on the torus. Every unit of mass that leaves a cell is assigned
to a destination; sum(rho') == sum(rho) up to float rounding.

Do not clip neighbors at the border. Clipping leaks mass.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def convolve_periodic(fields: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve channel-first 2D or 3D fields with periodic boundaries."""
    from numpy.fft import fftn, ifftn

    fields = np.asarray(fields, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if fields.ndim not in (3, 4) or kernel.ndim != fields.ndim - 1:
        raise ValueError("fields must be channel-first 2D or 3D data matching the kernel")
    spatial = fields.shape[1:]
    if any(k > size for k, size in zip(kernel.shape, spatial)):
        raise ValueError("kernel dimensions must not exceed field dimensions")
    kpad = np.zeros(spatial, dtype=np.float64)
    kpad[tuple(slice(0, size) for size in kernel.shape)] = kernel
    shifts = tuple(-(size // 2) for size in kernel.shape)
    k_hat = fftn(np.roll(kpad, shift=shifts, axis=tuple(range(len(spatial)))))
    convolved = np.empty_like(fields)
    for channel in range(fields.shape[0]):
        convolved[channel] = np.real(ifftn(fftn(fields[channel]) * k_hat))
    return convolved


def neighborhood_offsets(ndim: int) -> list[tuple[int, ...]]:
    if ndim == 2:
        return [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    if ndim == 3:
        return [
            (dz, dy, dx)
            for dz in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        ]
    raise ValueError("MaCE reference supports 2D or 3D only")


def _wrap_index(coord: Iterable[int], shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(c) % s for c, s in zip(coord, shape))


def _transport_layout(rank: int):
    if rank == 2:
        return neighborhood_offsets(2), (0, 1), 0
    if rank == 3:
        return neighborhood_offsets(2), (1, 2), 1
    if rank == 4:
        return neighborhood_offsets(3), (1, 2, 3), 1
    raise ValueError(f"unsupported affinity rank {rank}")


def _origin_maxima(affinity: np.ndarray, offsets, axes, channel_axis: int) -> np.ndarray:
    maxima = np.full_like(affinity, -np.inf)
    for offset in offsets:
        shift = tuple(-value for value in offset)
        neighbor = np.roll(affinity, shift=shift, axis=axes)
        np.maximum(maxima, neighbor, out=maxima)
    return maxima


def mace_affinity(
    rho: np.ndarray,
    matrix: np.ndarray,
    kernel: np.ndarray,
    strength: float,
    crowding_lambda: float,
) -> np.ndarray:
    """A_c = strength * sum_d M[c,d] (K * rho_d) - lambda * rho_c.

    ``rho`` is (C, *spatial). ``kernel`` is a 2D or 3D stencil.
    Convolution uses periodic (torus) boundaries via wrap.
    """
    rho = np.asarray(rho, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    c = rho.shape[0]
    spatial = rho.shape[1:]
    conv = convolve_periodic(rho, kernel)
    affinity = np.zeros_like(rho)
    for ch in range(c):
        mixed = np.zeros(spatial, dtype=np.float64)
        for d in range(c):
            mixed += matrix[ch, d] * conv[d]
        affinity[ch] = strength * mixed - crowding_lambda * rho[ch]
    return affinity


def mace_denominators(affinity: np.ndarray, transport_beta: float) -> np.ndarray:
    """Z per cell, independent of channel if A is already per-channel.

    ``affinity`` shape (C, *spatial) or (*spatial). Returns same shape.
    """
    A = np.asarray(affinity, dtype=np.float64)
    offsets, axes, channel_axis = _transport_layout(A.ndim)
    maxima = _origin_maxima(A, offsets, axes, channel_axis)
    denominators = np.zeros_like(A)
    for offset in offsets:
        shift = tuple(-value for value in offset)
        neighbor = np.roll(A, shift=shift, axis=axes)
        denominators += np.exp(transport_beta * (neighbor - maxima))
    return denominators


def mace_gather(
    rho: np.ndarray,
    affinity: np.ndarray,
    denominators: np.ndarray,
    transport_beta: float,
) -> np.ndarray:
    """Gather mass. ``rho``, ``affinity``, ``denominators`` same shape."""
    rho = np.asarray(rho, dtype=np.float64)
    A = np.asarray(affinity, dtype=np.float64)
    Z = np.asarray(denominators, dtype=np.float64)
    if rho.shape != A.shape or rho.shape != Z.shape:
        raise ValueError("rho, affinity, Z must share shape")
    if np.any(Z <= 0.0):
        raise ValueError("MaCE denominator must be strictly positive")
    offsets, axes, channel_axis = _transport_layout(rho.ndim)
    maxima = _origin_maxima(A, offsets, axes, channel_axis)
    out = np.zeros_like(rho)
    for offset in offsets:
        shift = tuple(offset)
        source_rho = np.roll(rho, shift=shift, axis=axes)
        source_z = np.roll(Z, shift=shift, axis=axes)
        source_max = np.roll(maxima, shift=shift, axis=axes)
        out += source_rho * np.exp(transport_beta * (A - source_max)) / source_z
    return out


def ring_kernel_2d(radius: int = 3, ring: float = 0.6, width: float = 0.25) -> np.ndarray:
    """Simple signed ring: negative near 0, positive on a shell. Sum ~ 0."""
    s = 2 * radius + 1
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    r = np.sqrt(xx * xx + yy * yy) / max(radius, 1)
    k = np.exp(-((r - ring) ** 2) / (2.0 * width * width)) - 0.35 * np.exp(
        -(r**2) / (2.0 * (width * 0.5) ** 2)
    )
    k -= k.mean()
    return k.astype(np.float64)


def ring_kernel_3d(radius: int = 3, ring: float = 0.6, width: float = 0.25) -> np.ndarray:
    """Signed spherical ring stencil for periodic 3D Field Life."""
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    zz, yy, xx = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    radial = np.sqrt(xx * xx + yy * yy + zz * zz) / max(radius, 1)
    kernel = np.exp(-((radial - ring) ** 2) / (2.0 * width * width)) - 0.35 * np.exp(
        -(radial**2) / (2.0 * (width * 0.5) ** 2)
    )
    kernel -= kernel.mean()
    return kernel.astype(np.float64)


def mace_step(
    rho: np.ndarray,
    matrix: np.ndarray,
    kernel: np.ndarray,
    strength: float,
    crowding_lambda: float,
    transport_beta: float,
) -> np.ndarray:
    """Full Field-Life field update: affinity → Z → gather."""
    A = mace_affinity(rho, matrix, kernel, strength, crowding_lambda)
    Z = mace_denominators(A, transport_beta)
    return mace_gather(rho, A, Z, transport_beta)
