"""Entelechy Continuum Engine — corrected reference kernels.

These implementations encode the review corrections to the catalog:

* Mohr tent force has no extra ``r_max`` scale on the unit vector.
* Friction is an explicit exponential decay, not Mohr's 60 Hz power.
* MaCE is two passes (denominator, then gather) with torus wrap.
* Spatial-bin scatter snapshots counts before the exclusive scan.
"""

from .mohr import mohr_force_scalar, mohr_accelerations
from .integrate import friction_decay, symplectic_euler
from .mace import mace_affinity, mace_denominators, mace_gather, mace_step
from .bins import bin_index, exclusive_scan, snapshot_and_scatter
from .config import SimConfig, load_config
from .matrix import make_matrix
from .step_mohr import MohrState, run as run_mohr, seed_state, step as step_mohr
from .field_life import FieldLifeState, run as run_field_life
from .lenia import LeniaState, run as run_lenia
from .pic import deposit_particles, sample_fields
from .cuda_pic import deposit_particles_cuda, sample_fields_cuda
from .hybrid import HybridState, run as run_hybrid
from .particle_lenia import ParticleLeniaState, run as run_particle_lenia
from .fixedpoint_mohr import FixedMohrState, run_fixed as run_mohr_fixed

__all__ = [
    "mohr_force_scalar",
    "mohr_accelerations",
    "friction_decay",
    "symplectic_euler",
    "mace_affinity",
    "mace_denominators",
    "mace_gather",
    "mace_step",
    "bin_index",
    "exclusive_scan",
    "snapshot_and_scatter",
    "SimConfig",
    "load_config",
    "make_matrix",
    "MohrState",
    "run_mohr",
    "seed_state",
    "step_mohr",
    "FieldLifeState",
    "run_field_life",
    "LeniaState",
    "run_lenia",
    "deposit_particles",
    "sample_fields",
    "deposit_particles_cuda",
    "sample_fields_cuda",
    "HybridState",
    "run_hybrid",
    "ParticleLeniaState",
    "run_particle_lenia",
    "FixedMohrState",
    "run_mohr_fixed",
]
