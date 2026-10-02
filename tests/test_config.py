from pathlib import Path

import pytest

from ece.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"


def test_default_mohr_config_loads():
    cfg = load_config(CFG / "particle_life_6.toml")
    assert cfg.species_count == 6
    assert cfg.matrix.shape == (6, 6)
    assert cfg.mohr["beta"] == 0.3
    assert "mohr" in cfg.rules


def test_chase_is_cyclic():
    cfg = load_config(CFG / "particle_life_chase.toml")
    m = cfg.matrix
    assert m[0, 1] > 0 and m[1, 0] < 0


def test_field_channels_must_match_species():
    raw = (CFG / "field_life.toml").read_text()
    raw = raw.replace("channels = 4", "channels = 3")
    p = CFG.parent / "tests" / "_bad_field.toml"
    p.write_text(raw)
    with pytest.raises(ValueError, match="channels"):
        load_config(p)
    p.unlink()


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "x.toml"
    p.write_text("[simulation]\ndt=0.01\nworld=[1,1]\nseed=0\n[species]\ncount=2\n[nope]\nx=1\n")
    with pytest.raises(ValueError, match="unknown"):
        load_config(p)


def test_hash_grid_must_fit_three_by_three_neighborhood(tmp_path):
    raw = (CFG / "particle_life_6.toml").read_text().replace("grid = [12, 12]", "grid = [13, 13]")
    p = tmp_path / "bad_grid.toml"
    p.write_text(raw)
    with pytest.raises(ValueError, match="world/grid cells"):
        load_config(p)


def test_unknown_key_inside_section_rejected(tmp_path):
    raw = (CFG / "particle_life_6.toml").read_text().replace("lambda = ", "lamda = ")
    p = tmp_path / "typo.toml"
    p.write_text(raw)
    with pytest.raises(ValueError, match=r"unknown keys in \[mohr\]: \['lamda'\]"):
        load_config(p)


def test_hash_grid_smaller_than_neighborhood_rejected(tmp_path):
    raw = (
        (CFG / "particle_life_6.toml").read_text()
        .replace("r_max = 0.08", "r_max = 0.4")
        .replace("cell = 0.08", "cell = 0.5")
        .replace("grid = [12, 12]", "grid = [2, 2]")
    )
    p = tmp_path / "tiny_grid.toml"
    p.write_text(raw)
    with pytest.raises(ValueError, match="at least 3 bins per axis"):
        load_config(p)
