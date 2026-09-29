from pathlib import Path

import numpy as np
import pytest

from ece.config import load_config
from ece.__main__ import main
from ece.step_mohr import seed_state, step
from ece.wgpu_mohr import step_wgpu, webgpu_available


CFG = Path(__file__).resolve().parents[1] / "configs" / "particle_life_6.toml"
pytestmark = pytest.mark.skipif(not webgpu_available(), reason="WebGPU adapter is unavailable")


def test_wgpu_mohr_step_matches_cpu_oracle():
    cfg = load_config(CFG)
    cfg.mohr["particles"] = 32
    state = seed_state(cfg)

    gpu = step_wgpu(state, cfg)
    cpu = step(type(state)(state.pos.copy(), state.vel.copy(), state.types.copy()), cfg, use_bins=False)

    np.testing.assert_allclose(gpu.pos, cpu.pos, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(gpu.vel, cpu.vel, rtol=3e-5, atol=3e-5)
    assert gpu.frame == cpu.frame == 1


def test_headless_cli_wgpu_smoke(capsys):
    main([str(CFG), "--wgpu", "--particles", "16", "--frames", "1"])
    assert capsys.readouterr().out.strip() == "frame=1 particles=16"


def test_wgpu_256_particles_stay_finite_and_on_torus_for_200_frames():
    cfg = load_config(CFG)
    cfg.mohr["particles"] = 256
    state = seed_state(cfg)

    for _ in range(200):
        state = step_wgpu(state, cfg)

    assert state.frame == 200
    assert np.isfinite(state.pos).all()
    assert np.isfinite(state.vel).all()
    assert np.all(state.pos >= 0.0)
    assert np.all(state.pos < cfg.world)


def test_wgpu_matches_cpu_each_step_for_200_frames():
    cfg = load_config(CFG)
    cfg.mohr["particles"] = 256
    reference = seed_state(cfg)

    for _ in range(200):
        gpu = step_wgpu(reference, cfg)
        cpu = step(type(reference)(reference.pos.copy(), reference.vel.copy(), reference.types.copy()), cfg)
        np.testing.assert_allclose(gpu.pos, cpu.pos, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gpu.vel, cpu.vel, rtol=1e-6, atol=1e-6)
        reference = cpu