from pathlib import Path

import numpy as np
import pytest

from ece.config import load_config
from ece.__main__ import main
from ece.cupy_field_life import cupy_available, run_cupy
from ece.field_life import run


CFG = Path(__file__).resolve().parents[1] / "configs" / "field_life.toml"
pytestmark = pytest.mark.skipif(not cupy_available(), reason="CuPy CUDA FFT is unavailable")


def test_cupy_3d_field_life_matches_cpu_and_conserves_mass():
    cfg = load_config(CFG)
    cfg.world = np.array([1.0, 1.0, 1.0])
    cfg.field_cfg["dims"] = [16, 16, 16]
    cpu = run(cfg, frames=3)
    gpu = run_cupy(cfg, frames=3)

    np.testing.assert_allclose(gpu.rho, cpu.rho, rtol=4e-4, atol=4e-5)
    np.testing.assert_allclose(
        gpu.rho.sum(axis=(1, 2, 3)),
        cpu.rho.sum(axis=(1, 2, 3)),
        rtol=2e-5,
        atol=2e-5,
    )
    assert np.isfinite(gpu.rho).all()


def test_cupy_3d_field_life_cli_writes_volume(tmp_path, capsys):
    raw = CFG.read_text().replace("[128, 128]", "[16, 16, 16]").replace(
        "world = [1.0, 1.0]", "world = [1.0, 1.0, 1.0]"
    )
    config_path = tmp_path / "field3d.toml"
    config_path.write_text(raw)
    output = tmp_path / "field3d.npz"
    main([str(config_path), "--cuda", "--frames", "1", "--output", str(output)])

    with np.load(output) as state:
        assert state["rho"].shape == (4, 16, 16, 16)
        assert state["frame"].item() == 1
    assert capsys.readouterr().out.strip() == "frame=1 channels=4 dims=16x16x16"