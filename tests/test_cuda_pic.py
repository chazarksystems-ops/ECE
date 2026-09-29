from pathlib import Path

import numpy as np
import pytest

from ece.cuda_mohr import cuda_available
from ece.cuda_pic import deposit_particles_cuda, sample_fields_cuda
from ece.pic import deposit_particles, sample_fields


pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA device is unavailable")


def test_cuda_pic_matches_cpu_and_conserves_species_mass():
    pos = np.array([[0.99, 0.02], [0.1, 0.8], [0.45, 0.35]])
    types = np.array([0, 1, 0])
    masses = np.array([2.0, 3.0, 4.0])
    world = np.array([1.0, 1.0])
    cpu_rho = deposit_particles(pos, types, world, (8, 8), 2, masses)
    gpu_rho = deposit_particles_cuda(pos, types, world, (8, 8), 2, masses)
    np.testing.assert_allclose(gpu_rho, cpu_rho, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gpu_rho.sum(axis=(1, 2)), [6.0, 3.0], atol=1e-6)

    fields = np.arange(2 * 8 * 8, dtype=np.float64).reshape(2, 8, 8) / 17.0
    sampled = sample_fields_cuda(fields, pos, world)
    np.testing.assert_allclose(sampled, sample_fields(fields, pos, world), rtol=1e-6, atol=1e-6)