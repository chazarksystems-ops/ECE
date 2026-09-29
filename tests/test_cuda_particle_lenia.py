from pathlib import Path

import numpy as np
import pytest

from ece.config import load_config
from ece.__main__ import main
from ece.__main__ import main
from ece.cuda_mohr import cuda_available
from ece.cuda_particle_lenia import run_cuda
from ece.particle_lenia import run


CFG = Path(__file__).resolve().parents[1] / "configs" / "particle_lenia.toml"
pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA device is unavailable")


def test_cuda_particle_lenia_matches_cpu_one_step():
    cfg = load_config(CFG)
    cfg.particle_lenia["particles"] = 32
    cpu = run(cfg, frames=1)
    gpu = run_cuda(cfg, frames=1)

    np.testing.assert_allclose(gpu.pos, cpu.pos, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(gpu.vel, cpu.vel, rtol=3e-4, atol=3e-5)


def test_cuda_particle_lenia_stays_finite_and_wrapped():
    cfg = load_config(CFG)
    cfg.particle_lenia["particles"] = 64
    state = run_cuda(cfg, frames=100)

    assert state.frame == 100
    assert np.isfinite(state.pos).all() and np.isfinite(state.vel).all()
    assert np.all(state.pos >= 0.0) and np.all(state.pos < cfg.world)


def test_cuda_particle_lenia_cli_writes_state(tmp_path, capsys):
    output = tmp_path / "particle-lenia.npz"
    main([str(CFG), "--cuda", "--particles", "24", "--frames", "1", "--output", str(output)])

    with np.load(output) as state:
        assert state["pos"].shape == (24, 2)
        assert state["frame"].item() == 1
    assert capsys.readouterr().out.strip() == "frame=1 particles=24"