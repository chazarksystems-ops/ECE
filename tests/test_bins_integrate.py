import numpy as np
import pytest

from ece.bins import bin_index, exclusive_scan, snapshot_and_scatter
from ece.integrate import friction_decay, mohr_gamma_to_lambda, symplectic_euler


def test_exclusive_scan_identity():
    c = np.array([3, 0, 4, 1], dtype=np.int64)
    off = exclusive_scan(c)
    np.testing.assert_array_equal(off, [0, 3, 3, 7])
    assert off[-1] + c[-1] == c.sum()


def test_scatter_snapshot_is_frozen():
    rng = np.random.default_rng(3)
    pos = rng.random((200, 2))
    tables = snapshot_and_scatter(pos, cell=0.25, grid=(4, 4))
    # Every particle appears exactly once.
    assert sorted(tables["sorted_indices"].tolist()) == list(range(200))
    # Reconstruction: particles in bin b sit in ranges[b].
    for b, (lo, hi) in enumerate(tables["ranges"]):
        ids = tables["sorted_indices"][lo:hi]
        assert len(ids) == tables["counts"][b]
        if len(ids):
            assert np.all(tables["bin_of"][ids] == b)


def test_world_aligned_bins_keep_torus_neighbors_adjacent():
    pos = np.array([[0.95, 0.5], [0.02, 0.5]])
    indices = bin_index(pos, cell=0.08, grid=(12, 12), world=np.array([1.0, 1.0]))
    x_bins = indices % 12
    assert (x_bins[0] - x_bins[1]) % 12 in (1, 11)


def test_friction_recovers_mohr_gamma_at_60hz():
    gamma = 0.9
    lam = mohr_gamma_to_lambda(gamma, hz=60.0)
    # One 1/60 s tick should multiply velocity by gamma.
    assert friction_decay(1.0 / 60.0, lam) == pytest.approx(gamma)


def test_wrap_stays_in_torus():
    pos = np.array([[0.99, 0.01]])
    vel = np.array([[1.0, -1.0]])
    acc = np.zeros((1, 2))
    world = np.array([1.0, 1.0])
    p, v = symplectic_euler(pos, vel, acc, dt=0.05, lam=0.0, world=world)
    assert np.all(p >= 0.0) and np.all(p < 1.0)
    # No friction → velocity unchanged.
    np.testing.assert_allclose(v, vel)
