import numpy as np
import pytest

from ece.pic import deposit_particles, sample_fields


def test_deposit_conserves_mass_per_species():
    pos = np.array([[0.99, 0.02], [0.1, 0.8], [0.45, 0.35]])
    types = np.array([0, 1, 0])
    masses = np.array([2.0, 3.0, 4.0])
    rho = deposit_particles(pos, types, np.array([1.0, 1.0]), (8, 8), 2, masses)

    np.testing.assert_allclose(rho.sum(axis=(1, 2)), [6.0, 3.0], atol=1e-14)


def test_deposit_and_sample_are_adjoint_across_torus_edges():
    pos = np.array([[0.99, 0.02], [0.1, 0.8], [0.45, 0.35]])
    types = np.array([0, 1, 0])
    masses = np.array([2.0, 3.0, 4.0])
    world = np.array([1.0, 1.0])
    rho = deposit_particles(pos, types, world, (8, 8), 2, masses)
    field_values = np.arange(2 * 8 * 8, dtype=np.float64).reshape(2, 8, 8) / 17.0
    sampled = sample_fields(field_values, pos, world)
    left = np.sum(rho * field_values)
    right = np.sum(masses * sampled[np.arange(len(types)), types])

    assert left == pytest.approx(right, rel=1e-13, abs=1e-13)


def test_sample_of_constant_field_is_constant_at_wrapped_positions():
    fields = np.full((3, 4, 8), 2.5)
    pos = np.array([[-0.01, 1.02], [1.0, 0.0]])
    sampled = sample_fields(fields, pos, np.array([1.0, 1.0]))

    np.testing.assert_allclose(sampled, 2.5)