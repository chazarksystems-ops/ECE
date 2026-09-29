import numpy as np
import pytest

from ece.mace import (
    mace_affinity,
    mace_denominators,
    mace_gather,
    mace_step,
    ring_kernel_2d,
)


def _random_rho(rng, channels=3, h=16, w=16):
    rho = rng.random((channels, h, w))
    rho /= rho.sum()  # unit total mass across all channels for readability
    rho *= 100.0
    return rho


def test_mass_conserved_per_channel():
    rng = np.random.default_rng(1)
    rho = _random_rho(rng)
    k = 3
    matrix = rng.uniform(-1, 1, size=(k, k))
    kernel = ring_kernel_2d(radius=2)
    out = mace_step(rho, matrix, kernel, strength=1.0, crowding_lambda=0.1, transport_beta=4.0)
    np.testing.assert_allclose(out.sum(axis=(1, 2)), rho.sum(axis=(1, 2)), rtol=1e-12, atol=1e-10)
    assert np.all(out >= -1e-12)


def test_uniform_field_is_fixed_point():
    rho = np.ones((2, 8, 8))
    matrix = np.eye(2)
    kernel = ring_kernel_2d(radius=2)
    out = mace_step(rho, matrix, kernel, 1.0, 0.0, 4.0)
    np.testing.assert_allclose(out, rho, rtol=1e-12, atol=1e-12)


def test_torus_does_not_leak_at_border():
    """A blob in the corner must not lose mass — clip-and-skip would leak."""
    rho = np.zeros((1, 12, 12))
    rho[0, 0, 0] = 7.0
    rho[0, 0, 11] = 3.0
    rho[0, 11, 0] = 2.0
    matrix = np.array([[0.5]])
    kernel = ring_kernel_2d(radius=2)
    out = mace_step(rho, matrix, kernel, 1.0, 0.05, 3.0)
    assert out.sum() == pytest.approx(12.0, rel=1e-12, abs=1e-10)


def test_two_pass_matches_manual_softmax():
    """On a 2x2 single-channel grid, gather equals explicit softmax routing."""
    rho = np.array([[1.0, 2.0], [3.0, 4.0]])
    A = np.array([[0.1, -0.2], [0.3, 0.0]])
    beta = 2.0
    Z = mace_denominators(A, beta)
    got = mace_gather(rho, A, Z, beta)

    # Manual: every cell sends to its 9 wrapped neighbors (2x2 → all cells,
    # self counted once, others counted more than once because of wrap
    # of the 3x3 stencil). Use the same stencil.
    h, w = 2, 2
    expA = np.exp(beta * (A - A.max()))
    manual = np.zeros_like(rho)
    for y in range(h):
        for x in range(w):
            # destination (y,x) receives from each neighbor origin
            acc = 0.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    oy, ox = (y + dy) % h, (x + dx) % w
                    acc += rho[oy, ox] * expA[y, x] / Z[oy, ox]
            manual[y, x] = acc
    np.testing.assert_allclose(got, manual, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got.sum(), rho.sum(), rtol=1e-12)


def test_affinity_crowding_penalizes_self():
    rng = np.random.default_rng(2)
    rho = _random_rho(rng, channels=2, h=8, w=8)
    matrix = np.zeros((2, 2))
    kernel = ring_kernel_2d(radius=1)
    A0 = mace_affinity(rho, matrix, kernel, strength=0.0, crowding_lambda=0.0)
    A1 = mace_affinity(rho, matrix, kernel, strength=0.0, crowding_lambda=1.0)
    np.testing.assert_allclose(A0, 0.0, atol=1e-12)
    np.testing.assert_allclose(A1, -rho)


def test_transport_weights_stay_finite_for_large_affinity():
    rho = np.ones((1, 8, 8))
    affinity = np.full_like(rho, 1000.0)
    denominator = mace_denominators(affinity, transport_beta=4.0)
    result = mace_gather(rho, affinity, denominator, transport_beta=4.0)

    assert np.isfinite(denominator).all()
    np.testing.assert_allclose(result, rho, rtol=1e-12, atol=1e-12)
