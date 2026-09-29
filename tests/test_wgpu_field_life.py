from pathlib import Path

import numpy as np
import pytest

from ece.config import load_config
from ece.__main__ import main
from ece.field_life import run, seed_state, step
from ece.wgpu_field_life import step_wgpu
from ece.wgpu_mohr import webgpu_available


CFG = Path(__file__).resolve().parents[1] / "configs" / "field_life.toml"
pytestmark = pytest.mark.skipif(not webgpu_available(), reason="WebGPU adapter is unavailable")


def test_wgpu_field_life_matches_cpu_and_conserves_channel_mass():
    cfg = load_config(CFG)
    cfg.field_cfg["dims"] = [16, 16]
    initial = seed_state(cfg)

    gpu = step_wgpu(initial, cfg)
    cpu = step(initial, cfg)

    np.testing.assert_allclose(gpu.rho, cpu.rho, rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(
        gpu.rho.sum(axis=(1, 2)),
        initial.rho.sum(axis=(1, 2)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_wgpu_field_life_cli_writes_output(tmp_path, capsys):
    config_path = tmp_path / "field_life.toml"
    config_path.write_text(CFG.read_text().replace("[128, 128]", "[16, 16]"))
    output = tmp_path / "field.npz"
    main([str(config_path), "--wgpu", "--frames", "1", "--output", str(output)])

    with np.load(output) as state:
        assert state["rho"].shape == (4, 16, 16)
        assert state["frame"].item() == 1
    assert capsys.readouterr().out.strip() == "frame=1 channels=4 dims=16x16"


def test_wgpu_field_life_remains_finite_over_three_steps():
    cfg = load_config(CFG)
    cfg.field_cfg["dims"] = [16, 16]
    cpu = run(cfg, frames=3)
    state = seed_state(cfg)
    for _ in range(3):
        state = step_wgpu(state, cfg)

    assert np.isfinite(state.rho).all()
    np.testing.assert_allclose(state.rho, cpu.rho, rtol=5e-4, atol=5e-5)