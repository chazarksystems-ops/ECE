from pathlib import Path

import numpy as np
import pytest

from ece.__main__ import main
from ece.config import load_config
from ece.lenia import ring_kernel_2d, run, seed_state


CFG = Path(__file__).resolve().parents[1] / "configs" / "lenia_orbium.toml"


def test_ring_kernel_is_normalized():
    kernel = ring_kernel_2d(radius=8)
    assert kernel.shape == (17, 17)
    assert np.all(kernel >= 0.0)
    assert kernel.sum() == pytest.approx(1.0)


def test_lenia_run_is_deterministic_and_bounded():
    cfg = load_config(CFG)
    cfg.field_cfg["dims"] = [32, 32]
    first = run(cfg, frames=8)
    second = run(cfg, frames=8)

    np.testing.assert_array_equal(first.rho, second.rho)
    assert first.frame == 8
    assert np.isfinite(first.rho).all()
    assert np.all(first.rho >= 0.0)
    assert np.all(first.rho <= 1.0)
    assert seed_state(cfg).rho.max() > 0.0


def test_headless_cli_writes_lenia_state(tmp_path, capsys):
    output = tmp_path / "lenia.npz"
    main([str(CFG), "--frames", "2", "--output", str(output)])

    with np.load(output) as state:
        assert state["rho"].shape == (1, 128, 128)
        assert state["frame"].item() == 2
    assert capsys.readouterr().out.strip() == "frame=2 channels=1 dims=128x128"


def test_configured_annular_seed_extinguishes_by_frame_100():
    cfg = load_config(CFG)
    result = run(cfg, frames=100)
    assert np.isfinite(result.rho).all()
    assert np.count_nonzero(result.rho) == 0