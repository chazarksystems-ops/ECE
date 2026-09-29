from pathlib import Path

import numpy as np

from ece.__main__ import main
from ece.config import load_config
from ece.fixedpoint_mohr import run_fixed, to_mohr_state
from ece.step_mohr import run


CFG = Path(__file__).resolve().parents[1] / "configs" / "particle_life_6.toml"


def test_fixedpoint_mohr_is_bitwise_repeatable():
    cfg = load_config(CFG)
    cfg.mohr["particles"] = 48
    first = run_fixed(cfg, frames=40)
    second = run_fixed(cfg, frames=40)

    np.testing.assert_array_equal(first.pos_q16, second.pos_q16)
    np.testing.assert_array_equal(first.vel_q24, second.vel_q24)
    np.testing.assert_array_equal(first.types, second.types)


def test_fixedpoint_mohr_tracks_float_oracle_for_one_step():
    cfg = load_config(CFG)
    cfg.mohr["particles"] = 32
    fixed = to_mohr_state(run_fixed(cfg, frames=1))
    reference = run(cfg, frames=1)

    np.testing.assert_allclose(fixed.pos, reference.pos, rtol=0.0, atol=3e-5)
    np.testing.assert_allclose(fixed.vel, reference.vel, rtol=0.0, atol=3e-4)
    assert fixed.frame == 1


def test_fixedpoint_positions_remain_on_torus():
    cfg = load_config(CFG)
    cfg.mohr["particles"] = 64
    result = to_mohr_state(run_fixed(cfg, frames=100))

    assert np.isfinite(result.pos).all() and np.isfinite(result.vel).all()
    assert np.all(result.pos >= 0.0)
    assert np.all(result.pos < cfg.world)


def test_fixedpoint_cli_exports_integer_and_decoded_state(tmp_path, capsys):
    output = tmp_path / "fixed.npz"
    main([str(CFG), "--fixedpoint", "--particles", "24", "--frames", "2", "--output", str(output)])

    with np.load(output) as state:
        assert state["pos_q16"].shape == (24, 2)
        assert state["vel_q24"].shape == (24, 2)
        assert state["pos"].shape == (24, 2)
        assert state["frame"].item() == 2
    assert capsys.readouterr().out.strip() == "fixedpoint frame=2 particles=24"