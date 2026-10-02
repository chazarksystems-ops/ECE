from pathlib import Path

import numpy as np
import pytest

from ece.__main__ import main
from ece.config import load_config
from ece.matrix import parse_matrix_values
from ece.step_mohr import run, seed_state, step

CFG = Path(__file__).resolve().parents[1] / "configs" / "particle_life_chase.toml"


def test_positions_stay_in_torus():
    cfg = load_config(CFG)
    cfg.mohr["particles"] = 64
    st = run(cfg, frames=25, use_bins=True)
    assert np.all(st.pos >= 0.0)
    assert np.all(st.pos < cfg.world)
    assert st.frame == 25


def test_bins_match_dense_on_tiny_n():
    cfg = load_config(CFG)
    cfg.mohr["particles"] = 24
    a = seed_state(cfg)
    b = type(a)(a.pos.copy(), a.vel.copy(), a.types.copy(), a.frame)
    s_bin = step(a, cfg, use_bins=True)
    s_den = step(b, cfg, use_bins=False)
    np.testing.assert_allclose(s_bin.pos, s_den.pos, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(s_bin.vel, s_den.vel, rtol=1e-9, atol=1e-9)


def test_bins_match_dense_over_200_frames():
    cfg = load_config(CFG.with_name("particle_life_6.toml"))
    cfg.mohr["particles"] = 128
    binned = seed_state(cfg)

    for _ in range(200):
        dense = type(binned)(binned.pos.copy(), binned.vel.copy(), binned.types.copy())
        next_binned = step(binned, cfg, use_bins=True)
        next_dense = step(dense, cfg, use_bins=False)
        np.testing.assert_allclose(next_binned.pos, next_dense.pos, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(next_binned.vel, next_dense.vel, rtol=1e-6, atol=1e-6)
        binned = next_binned

    assert binned.frame == 200
    assert np.all(binned.pos >= 0.0)
    assert np.all(binned.pos < cfg.world)


def test_headless_cli_writes_final_state(tmp_path, capsys):
    output = tmp_path / "state.npz"
    main(
        [
            str(CFG),
            "--frames",
            "2",
            "--particles",
            "16",
            "--output",
            str(output),
        ]
    )

    with np.load(output) as state:
        assert state["pos"].shape == (16, 2)
        assert state["frame"].item() == 2
    assert capsys.readouterr().out.strip() == "frame=2 particles=16"


def test_headless_cli_rejects_unsupported_rule_combination(tmp_path):
    raw = CFG.with_name("hybrid_pic.toml").read_text().replace(
        '["mohr", "field_life"]', '["mohr", "lenia"]'
    )
    unsupported_config = tmp_path / "unsupported.toml"
    unsupported_config.write_text(raw)
    with pytest.raises(SystemExit, match="2"):
        main([str(unsupported_config)])


def test_matrix_editor_parser_validates_and_preserves_values():
    matrix = parse_matrix_values([["1", "-0.25"], ["0.5", "0"]], size=2)
    np.testing.assert_array_equal(matrix, [[1.0, -0.25], [0.5, 0.0]])
    with pytest.raises(ValueError, match="between -1 and 1"):
        parse_matrix_values([["1.1"]], size=1)


def test_bins_match_dense_on_minimal_three_by_three_grid():
    cfg = load_config(CFG)
    cfg.mohr["particles"] = 96
    cfg.mohr["r_max"] = 0.3
    cfg.hash["cell"] = 1.0 / 3.0
    cfg.hash["grid"] = [3, 3]
    a = seed_state(cfg)
    b = type(a)(a.pos.copy(), a.vel.copy(), a.types.copy(), a.frame)
    s_bin = step(a, cfg, use_bins=True)
    s_den = step(b, cfg, use_bins=False)
    np.testing.assert_allclose(s_bin.vel, s_den.vel, rtol=1e-9, atol=1e-12)


def test_owner_chunks_respect_pair_budget():
    from ece.step_mohr import _owner_chunks

    counts = np.array([5, 5, 5, 20, 1, 1, 0, 3])
    chunks = _owner_chunks(counts, budget=10)
    assert chunks[0][0] == 0 and chunks[-1][1] == len(counts)
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))
    for start, end in chunks:
        # Over budget only when a single owner alone exceeds it.
        assert counts[start:end].sum() <= 10 or end - start == 1


def test_clustered_particles_match_dense_across_pair_budget_chunks(monkeypatch):
    import sys

    # ece/__init__.py re-exports the step function as `step_mohr`, which shadows
    # the submodule attribute, so fetch the module itself.
    step_mohr = sys.modules["ece.step_mohr"]

    cfg = load_config(CFG.with_name("particle_life_6.toml"))
    cfg.mohr["particles"] = 256
    clustered = seed_state(cfg)
    # Pack every particle into one hash bin, the case that used to allocate ~N² pairs.
    clustered.pos[:] = 0.04 + np.random.default_rng(7).random(clustered.pos.shape) * 0.03
    dense = type(clustered)(clustered.pos.copy(), clustered.vel.copy(), clustered.types.copy())
    monkeypatch.setattr(step_mohr, "_PAIR_BUDGET", 1000)
    s_bin = step(clustered, cfg, use_bins=True)
    s_den = step(dense, cfg, use_bins=False)
    np.testing.assert_allclose(s_bin.vel, s_den.vel, rtol=1e-9, atol=1e-12)
