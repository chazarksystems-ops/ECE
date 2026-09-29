from pathlib import Path

import numpy as np

from ece.config import load_config
from ece.__main__ import main
from ece.field_life import run, seed_state


CFG = Path(__file__).resolve().parents[1] / "configs" / "field_life.toml"


def test_config_driven_field_life_conserves_each_channel():
    cfg = load_config(CFG)
    cfg.field_cfg["dims"] = [16, 16]
    initial = seed_state(cfg)
    result = run(cfg, frames=3)

    assert result.rho.shape == (cfg.species_count, 16, 16)
    assert result.frame == 3
    assert np.all(result.rho >= 0.0)
    np.testing.assert_allclose(
        result.rho.sum(axis=(1, 2)),
        initial.rho.sum(axis=(1, 2)),
        rtol=1e-12,
        atol=1e-10,
    )


def test_headless_cli_writes_field_state(tmp_path, capsys):
    output = tmp_path / "field.npz"
    main([str(CFG), "--frames", "2", "--output", str(output)])

    with np.load(output) as state:
        assert state["rho"].shape == (4, 128, 128)
        assert state["frame"].item() == 2
    assert capsys.readouterr().out.strip() == "frame=2 channels=4 dims=128x128"


def test_3d_cpu_field_life_conserves_each_channel():
    cfg = load_config(CFG)
    cfg.world = np.array([1.0, 1.0, 1.0])
    cfg.field_cfg["dims"] = [8, 8, 8]
    initial = seed_state(cfg)
    result = run(cfg, frames=2)

    assert result.rho.shape == (cfg.species_count, 8, 8, 8)
    np.testing.assert_allclose(
        result.rho.sum(axis=(1, 2, 3)),
        initial.rho.sum(axis=(1, 2, 3)),
        rtol=1e-12,
        atol=1e-10,
    )