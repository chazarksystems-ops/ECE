from pathlib import Path

import numpy as np
import pytest

from ece.config import load_config
from ece.__main__ import main
from ece.lenia import seed_state, step
from ece.wgpu_lenia import step_wgpu
from ece.wgpu_mohr import webgpu_available


CFG = Path(__file__).resolve().parents[1] / "configs" / "lenia_orbium.toml"
pytestmark = pytest.mark.skipif(not webgpu_available(), reason="WebGPU adapter is unavailable")


def test_wgpu_lenia_matches_cpu_and_stays_bounded():
    cfg = load_config(CFG)
    cfg.field_cfg["dims"] = [16, 16]
    initial = seed_state(cfg)

    gpu = step_wgpu(initial, cfg)
    cpu = step(initial, cfg)

    np.testing.assert_allclose(gpu.rho, cpu.rho, rtol=4e-4, atol=4e-5)
    assert np.isfinite(gpu.rho).all()
    assert np.all(gpu.rho >= 0.0)
    assert np.all(gpu.rho <= 1.0)


def test_wgpu_lenia_cli_writes_output(tmp_path, capsys):
    config_path = tmp_path / "lenia.toml"
    config_path.write_text(CFG.read_text().replace("[128, 128]", "[16, 16]"))
    output = tmp_path / "lenia.npz"
    main([str(config_path), "--wgpu", "--frames", "1", "--output", str(output)])

    with np.load(output) as state:
        assert state["rho"].shape == (1, 16, 16)
        assert state["frame"].item() == 1
    assert capsys.readouterr().out.strip() == "frame=1 channels=1 dims=16x16"