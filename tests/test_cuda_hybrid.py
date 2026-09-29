from pathlib import Path

import numpy as np
import pytest

from ece.__main__ import main
from ece.config import load_config
from ece.cuda_hybrid import run_cuda_resident
from ece.cuda_mohr import cuda_available
from ece.hybrid import run


CFG = Path(__file__).resolve().parents[1] / "configs" / "hybrid_pic.toml"
pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA device is unavailable")


def test_cuda_hybrid_step_preserves_species_mass():
    cfg = load_config(CFG)
    cfg.mohr["particles"] = 24
    cfg.field_cfg["dims"] = [16, 16]
    cpu = run(cfg, frames=1)
    gpu = run(cfg, frames=1, use_cuda=True)
    expected = np.bincount(gpu.particles.types, minlength=cfg.species_count)

    np.testing.assert_allclose(gpu.particles.pos, cpu.particles.pos, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(gpu.rho, cpu.rho, rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(gpu.sampled, cpu.sampled, rtol=4e-4, atol=4e-5)
    np.testing.assert_allclose(gpu.rho.sum(axis=(1, 2)), expected, rtol=1e-5, atol=1e-5)
    assert np.isfinite(gpu.sampled).all()
    assert gpu.frame == 1


def test_cuda_resident_hybrid_matches_cpu_and_preserves_mass():
    cfg = load_config(CFG)
    cfg.mohr["particles"] = 24
    cfg.field_cfg["dims"] = [16, 16]
    cpu = run(cfg, frames=1)
    gpu = run_cuda_resident(cfg, frames=1)
    expected = np.bincount(gpu.particles.types, minlength=cfg.species_count)

    np.testing.assert_allclose(gpu.particles.pos, cpu.particles.pos, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(gpu.particles.vel, cpu.particles.vel, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(gpu.rho, cpu.rho, rtol=4e-4, atol=4e-5)
    np.testing.assert_allclose(gpu.sampled, cpu.sampled, rtol=5e-4, atol=5e-5)
    np.testing.assert_allclose(gpu.rho.sum(axis=(1, 2)), expected, rtol=1e-5, atol=1e-5)


def test_cuda_hybrid_cli_writes_resident_state(tmp_path, capsys):
    config_path = tmp_path / "hybrid.toml"
    config_path.write_text(CFG.read_text().replace("[128, 128]", "[16, 16]"))
    output = tmp_path / "hybrid.npz"
    main(
        [str(config_path), "--cuda", "--particles", "24", "--frames", "1", "--output", str(output)]
    )

    with np.load(output) as state:
        assert state["pos"].shape == (24, 2)
        assert state["rho"].shape == (4, 16, 16)
        assert state["sampled"].shape == (24, 4)
        assert state["frame"].item() == 1
    assert capsys.readouterr().out.strip() == "frame=1 particles=24 channels=4"


def test_cuda_resident_hybrid_stays_finite_for_100_frames():
    cfg = load_config(CFG)
    cfg.mohr["particles"] = 128
    cfg.field_cfg["dims"] = [16, 16]
    state = run_cuda_resident(cfg, frames=100)

    assert state.frame == 100
    assert np.isfinite(state.particles.pos).all()
    assert np.isfinite(state.particles.vel).all()
    assert np.isfinite(state.rho).all()
    assert np.isfinite(state.sampled).all()
    assert np.all(state.particles.pos >= 0.0)
    assert np.all(state.particles.pos < cfg.world)


def test_cuda_resident_hybrid_repeats_identically_on_same_device():
    cfg = load_config(CFG)
    cfg.mohr["particles"] = 64
    cfg.field_cfg["dims"] = [16, 16]
    first = run_cuda_resident(cfg, frames=3)
    second = run_cuda_resident(cfg, frames=3)

    np.testing.assert_array_equal(first.particles.pos, second.particles.pos)
    np.testing.assert_array_equal(first.particles.vel, second.particles.vel)
    np.testing.assert_array_equal(first.rho, second.rho)
    np.testing.assert_array_equal(first.sampled, second.sampled)