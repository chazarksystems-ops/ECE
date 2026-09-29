from pathlib import Path

import numpy as np
import pytest

from ece.config import load_config
from ece.__main__ import main
from ece.hybrid import run
from ece.wgpu_mohr import webgpu_available


CFG = Path(__file__).resolve().parents[1] / "configs" / "hybrid_pic.toml"
pytestmark = pytest.mark.skipif(not webgpu_available(), reason="WebGPU adapter is unavailable")


def test_wgpu_hybrid_step_matches_cpu_and_preserves_species_mass():
    cfg = load_config(CFG)
    cfg.mohr["particles"] = 24
    cfg.field_cfg["dims"] = [16, 16]
    cpu = run(cfg, frames=1)
    gpu = run(cfg, frames=1, use_wgpu=True)
    expected = np.bincount(gpu.particles.types, minlength=cfg.species_count)

    np.testing.assert_allclose(gpu.particles.pos, cpu.particles.pos, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(gpu.rho, cpu.rho, rtol=4e-4, atol=4e-5)
    np.testing.assert_allclose(gpu.sampled, cpu.sampled, rtol=5e-4, atol=5e-5)
    np.testing.assert_allclose(gpu.rho.sum(axis=(1, 2)), expected, rtol=1e-5, atol=1e-5)


def test_wgpu_hybrid_cli_writes_all_state(tmp_path, capsys):
    config_path = tmp_path / "hybrid.toml"
    config_path.write_text(CFG.read_text().replace("[128, 128]", "[16, 16]"))
    output = tmp_path / "hybrid.npz"
    main(
        [str(config_path), "--wgpu", "--particles", "24", "--frames", "1", "--output", str(output)]
    )

    with np.load(output) as state:
        assert state["pos"].shape == (24, 2)
        assert state["rho"].shape == (4, 16, 16)
        assert state["sampled"].shape == (24, 4)
        assert state["frame"].item() == 1
    assert capsys.readouterr().out.strip() == "frame=1 particles=24 channels=4"