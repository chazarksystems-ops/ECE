import numpy as np
import pytest

from ece.mohr import mohr_accelerations, mohr_force_scalar, wrapped_delta


def test_wall_is_species_independent_and_repulsive():
    r = np.array([0.0, 0.15, 0.29])
    for a in (-1.0, 0.0, 1.0):
        f = mohr_force_scalar(r, np.full_like(r, a), beta=0.3)
        assert np.all(f < 0.0)
        # Wall value does not depend on matrix entry.
        f2 = mohr_force_scalar(r, np.full_like(r, -a if a else 0.5), beta=0.3)
        np.testing.assert_allclose(f, f2)


def test_tent_peak_at_midpoint():
    beta = 0.3
    peak_r = 0.5 * (1.0 + beta)
    r = np.array([beta, peak_r, 1.0])
    f = mohr_force_scalar(r, np.array([0.8, 0.8, 0.8]), beta)
    assert f[1] == pytest.approx(0.8)
    assert f[0] == pytest.approx(0.0, abs=1e-12)
    assert f[1] > f[0] and f[1] > f[2]


def test_zero_beyond_horizon():
    r = np.array([1.0 + 1e-9, 2.0, 10.0])
    f = mohr_force_scalar(r, np.ones_like(r), beta=0.3)
    np.testing.assert_array_equal(f, 0.0)


def test_force_independent_of_rmax_scale():
    """Same geometry in normalized units → same acceleration (gain=1)."""
    rng = np.random.default_rng(0)
    n = 12
    world = np.array([1.0, 1.0])
    pos = rng.random((n, 2))
    types = rng.integers(0, 3, size=n)
    matrix = rng.uniform(-1, 1, size=(3, 3))
    a_small = mohr_accelerations(pos, types, matrix, r_max=0.1, beta=0.3, world=world)
    # Stretch world and positions by 4, stretch r_max by 4. Normalized
    # r is unchanged, so acceleration vectors must match.
    scale = 4.0
    a_big = mohr_accelerations(
        pos * scale,
        types,
        matrix,
        r_max=0.1 * scale,
        beta=0.3,
        world=world * scale,
    )
    np.testing.assert_allclose(a_small, a_big, rtol=1e-10, atol=1e-12)


def test_asymmetric_matrix_breaks_momentum():
    pos = np.array([[0.4, 0.5], [0.5, 0.5]])
    types = np.array([0, 1])
    matrix = np.array([[0.0, 1.0], [-1.0, 0.0]])  # 0 chases 1, 1 flees 0
    world = np.array([1.0, 1.0])
    a = mohr_accelerations(pos, types, matrix, r_max=0.3, beta=0.3, world=world)
    # Both accelerations point the same way (+x), so total momentum grows.
    assert a[0, 0] > 0.0
    assert a[1, 0] > 0.0
    assert abs(a[:, 0].sum()) > abs(a[0, 0]) * 0.5


def test_wrapped_delta_minimum_image():
    world = np.array([1.0, 1.0])
    d = wrapped_delta(np.array([0.05, 0.0]), np.array([0.95, 0.0]), world)
    np.testing.assert_allclose(d, np.array([0.10, 0.0]))
