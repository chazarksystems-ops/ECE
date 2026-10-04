from pathlib import Path
import json

import pytest

from ece.config import (
    _ALLOWED_FIELD_KERNELS,
    _ALLOWED_KEYS,
    _ALLOWED_RULES,
    _ALLOWED_TOP,
    _MAX_SPECIES_COUNT,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"


def test_schema_and_loader_stay_in_sync():
    schema = json.loads((ROOT / "schemas" / "sim.schema.json").read_text())
    properties = schema["properties"]
    assert schema["additionalProperties"] is False
    assert set(properties) == _ALLOWED_TOP == set(_ALLOWED_KEYS)
    for section, allowed in _ALLOWED_KEYS.items():
        assert properties[section]["additionalProperties"] is False
        assert set(properties[section]["properties"]) == allowed
    sim = properties["simulation"]["properties"]
    species = properties["species"]["properties"]
    assert set(sim["rules"]["items"]["enum"]) == _ALLOWED_RULES
    assert set(properties["field"]["properties"]["kernel"]["enum"]) == _ALLOWED_FIELD_KERNELS
    assert sim["seed"]["type"] == "integer"
    assert sim["seed"]["minimum"] == 0
    assert species["count"]["type"] == "integer"
    assert species["count"]["minimum"] == 1
    assert species["count"]["maximum"] == _MAX_SPECIES_COUNT


@pytest.mark.parametrize("seed", ["-1", "1.5", "true", '\"2\"'])
def test_seed_must_be_a_nonnegative_integer(tmp_path, seed):
    path = tmp_path / "seed.toml"
    path.write_text(f"[simulation]\ndt=0.01\nworld=[1,1]\nseed={seed}\n[species]\ncount=2\n")
    with pytest.raises(ValueError, match=r"simulation\.seed must be a non-negative integer"):
        load_config(path)


@pytest.mark.parametrize("count", ["0", "-1", "129", "2.5", "true", '\"2\"'])
def test_species_count_must_be_an_integer_in_schema_range(tmp_path, count):
    path = tmp_path / "count.toml"
    path.write_text(f"[simulation]\ndt=0.01\nworld=[1,1]\nseed=0\n[species]\ncount={count}\n")
    with pytest.raises(ValueError, match=r"species\.count must be an integer between 1 and 128"):
        load_config(path)


@pytest.mark.parametrize("count", [1, 128])
def test_seed_and_species_count_boundaries_load(tmp_path, count):
    path = tmp_path / "boundary.toml"
    path.write_text(f"[simulation]\ndt=0.01\nworld=[1,1]\nseed=0\n[species]\ncount={count}\n")
    cfg = load_config(path)
    assert cfg.seed == 0
    assert cfg.species_count == count
    assert cfg.matrix.shape == (count, count)


@pytest.mark.parametrize("kernel", ['\"gaussian\"', '\"disk\"', '\"unknown\"', "false", '[\"ring\"]'])
def test_unsupported_field_kernel_rejected_at_load(tmp_path, kernel):
    path = tmp_path / "kernel.toml"
    path.write_text(
        f"[simulation]\ndt=0.01\nworld=[1,1]\nseed=0\n[species]\ncount=2\n[field]\nkernel={kernel}\n"
    )
    with pytest.raises(ValueError, match=r"field\.kernel must be 'ring'"):
        load_config(path)


@pytest.mark.parametrize("name", sorted(p.name for p in CFG.glob("*.toml")))
def test_all_shipped_configs_load(name):
    load_config(CFG / name)


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
