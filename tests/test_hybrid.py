from pathlib import Path

import numpy as np

from ece.config import load_config
from ece.hybrid import run
from ece.__main__ import main


CFG = Path(__file__).resolve().parents[1] / "configs" / "hybrid_pic.toml"


def test_hybrid_run_preserves_deposited_species_mass():
    cfg = load_config(CFG)
    cfg.mohr["particles"] = 24
    cfg.field_cfg["dims"] = [16, 16]
    state = run(cfg, frames=1)
    expected = np.bincount(state.particles.types, minlength=cfg.species_count)

    np.testing.assert_allclose(state.rho.sum(axis=(1, 2)), expected, rtol=1e-12, atol=1e-10)
    assert state.sampled.shape == (24, cfg.species_count)
    assert state.frame == 1


def test_hybrid_cli_writes_particle_field_and_sample_data(tmp_path, capsys):
    config_path = tmp_path / "hybrid.toml"
    config_path.write_text(CFG.read_text().replace("[128, 128]", "[16, 16]"))
    output = tmp_path / "hybrid.npz"
    main([str(config_path), "--particles", "24", "--frames", "1", "--output", str(output)])

    with np.load(output) as state:
        assert state["pos"].shape == (24, 2)
        assert state["rho"].shape == (4, 16, 16)
        assert state["sampled"].shape == (24, 4)
        assert state["frame"].item() == 1
    assert capsys.readouterr().out.strip() == "frame=1 particles=24 channels=4"