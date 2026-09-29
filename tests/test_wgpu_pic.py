import numpy as np
import pytest

from ece.pic import deposit_particles, sample_fields
from ece.wgpu_mohr import webgpu_available
from ece.wgpu_pic import deposit_particles_wgpu, sample_fields_wgpu


pytestmark = pytest.mark.skipif(not webgpu_available(), reason="WebGPU adapter is unavailable")


def test_wgpu_pic_matches_cpu_and_conserves_species_mass():
    pos = np.array([[0.99, 0.02], [0.1, 0.8], [0.45, 0.35]])
    types = np.array([0, 1, 0])
    masses = np.array([2.0, 3.0, 4.0])
    world = np.array([1.0, 1.0])
    cpu_rho = deposit_particles(pos, types, world, (8, 8), 2, masses)
    gpu_rho = deposit_particles_wgpu(pos, types, world, (8, 8), 2, masses)
    np.testing.assert_allclose(gpu_rho, cpu_rho, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gpu_rho.sum(axis=(1, 2)), [6.0, 3.0], atol=1e-6)

    fields = np.arange(2 * 8 * 8, dtype=np.float64).reshape(2, 8, 8) / 17.0
    np.testing.assert_allclose(
        sample_fields_wgpu(fields, pos, world),
        sample_fields(fields, pos, world),
        rtol=1e-6,
        atol=1e-6,
    )