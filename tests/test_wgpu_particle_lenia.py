from pathlib import Path

import numpy as np
import pytest

from ece.config import load_config
from ece.__main__ import main
from ece.particle_lenia import run, seed_state
from ece.wgpu_mohr import webgpu_available
from ece.wgpu_particle_lenia import run_wgpu, step_wgpu


CFG = Path(__file__).resolve().parents[1] / "configs" / "particle_lenia.toml"
pytestmark = pytest.mark.skipif(not webgpu_available(), reason="WebGPU adapter is unavailable")


def test_wgpu_particle_lenia_matches_cpu_step():
    cfg = load_config(CFG)
    cfg.particle_lenia["particles"] = 32
    initial = seed_state(cfg)
    cpu = run(cfg, frames=1)
    gpu = step_wgpu(initial, cfg)

    np.testing.assert_allclose(gpu.pos, cpu.pos, rtol=4e-5, atol=4e-6)
    np.testing.assert_allclose(gpu.vel, cpu.vel, rtol=4e-4, atol=4e-5)


def test_wgpu_particle_lenia_stays_finite_and_wrapped():
    cfg = load_config(CFG)
    cfg.particle_lenia["particles"] = 64
    state = run_wgpu(cfg, frames=100)

    assert state.frame == 100
    assert np.isfinite(state.pos).all() and np.isfinite(state.vel).all()
    assert np.all(state.pos >= 0.0) and np.all(state.pos < cfg.world)


def test_wgpu_particle_lenia_cli_writes_state(tmp_path, capsys):
    output = tmp_path / "particle-lenia.npz"
    main([str(CFG), "--wgpu", "--particles", "24", "--frames", "1", "--output", str(output)])

    with np.load(output) as state:
        assert state["pos"].shape == (24, 2)
        assert state["frame"].item() == 1
    assert capsys.readouterr().out.strip() == "frame=1 particles=24"