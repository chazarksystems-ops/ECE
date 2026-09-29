from pathlib import Path

import numpy as np
import pytest

from ece.config import load_config
from ece.__main__ import main
from ece.cuda_mohr import cuda_available, run_cuda_resident, step_cuda
from ece.step_mohr import seed_state, step


CFG = Path(__file__).resolve().parents[1] / "configs" / "particle_life_6.toml"
pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA device is unavailable")


def test_cuda_mohr_step_matches_cpu_oracle():
    cfg = load_config(CFG)
    cfg.mohr["particles"] = 32
    state = seed_state(cfg)

    gpu = step_cuda(state, cfg)
    cpu = step(type(state)(state.pos.copy(), state.vel.copy(), state.types.copy()), cfg, use_bins=False)

    np.testing.assert_allclose(gpu.pos, cpu.pos, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(gpu.vel, cpu.vel, rtol=2e-5, atol=2e-5)
    assert gpu.frame == cpu.frame == 1


def test_headless_cli_cuda_smoke(capsys):
    main([str(CFG), "--cuda", "--particles", "16", "--frames", "1"])
    assert capsys.readouterr().out.strip() == "frame=1 particles=16"


def test_cuda_256_particles_stay_finite_and_on_torus_for_200_frames():
    cfg = load_config(CFG)
    cfg.mohr["particles"] = 256
    state = run_cuda_resident(cfg, frames=200)

    assert state.frame == 200
    assert np.isfinite(state.pos).all()
    assert np.isfinite(state.vel).all()
    assert np.all(state.pos >= 0.0)
    assert np.all(state.pos < cfg.world)


def test_cuda_resident_one_step_matches_cpu_oracle():
    cfg = load_config(CFG)
    cfg.mohr["particles"] = 32
    initial = seed_state(cfg)
    cpu = step(type(initial)(initial.pos.copy(), initial.vel.copy(), initial.types.copy()), cfg)
    gpu = run_cuda_resident(cfg, frames=1)

    np.testing.assert_allclose(gpu.pos, cpu.pos, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(gpu.vel, cpu.vel, rtol=2e-5, atol=2e-5)


def test_cuda_resident_run_repeats_identically_on_same_device():
    cfg = load_config(CFG)
    cfg.mohr["particles"] = 64
    first = run_cuda_resident(cfg, frames=20)
    second = run_cuda_resident(cfg, frames=20)

    np.testing.assert_array_equal(first.pos, second.pos)
    np.testing.assert_array_equal(first.vel, second.vel)