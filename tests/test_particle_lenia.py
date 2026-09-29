from pathlib import Path

import numpy as np
import pytest

from ece.config import load_config
from ece.particle_lenia import (
    particle_lenia_energy,
    particle_lenia_velocity,
    run,
    seed_state,
    shell_kernel_weight,
)
from ece.__main__ import main


CFG = Path(__file__).resolve().parents[1] / "configs" / "particle_lenia.toml"


def test_shell_kernel_has_unit_integral():
    mu, sigma = 4.0, 1.0
    weight = shell_kernel_weight(mu, sigma)
    radii = np.linspace(0.0, mu + 8.0 * sigma, 32769)
    values = weight * np.exp(-((radii - mu) / sigma) ** 2) * radii
    integral = 2.0 * np.pi * np.sum((values[:-1] + values[1:]) * 0.5 * np.diff(radii))
    assert integral == pytest.approx(1.0, rel=1e-8)


def test_velocity_is_negative_local_energy_gradient():
    cfg = load_config(CFG)
    positions = np.array([[7.0, 8.0], [7.4, 8.3], [10.5, 9.0]])
    velocity = particle_lenia_velocity(positions, cfg.world, cfg.particle_lenia)
    epsilon = 1e-5
    numeric = np.zeros(2)
    for axis in range(2):
        plus = positions[0].copy()
        minus = positions[0].copy()
        plus[axis] += epsilon
        minus[axis] -= epsilon
        energy_plus = particle_lenia_energy(plus, positions, cfg.world, cfg.particle_lenia)
        energy_minus = particle_lenia_energy(minus, positions, cfg.world, cfg.particle_lenia)
        numeric[axis] = -(energy_plus - energy_minus) / (2.0 * epsilon)
    np.testing.assert_allclose(velocity[0], numeric, rtol=2e-4, atol=2e-5)


def test_particle_lenia_run_is_deterministic_and_wraps():
    cfg = load_config(CFG)
    cfg.particle_lenia["particles"] = 32
    first = run(cfg, frames=20)
    second = run(cfg, frames=20)
    np.testing.assert_array_equal(first.pos, second.pos)
    assert first.frame == 20
    assert np.isfinite(first.vel).all()
    assert np.all(first.pos >= 0.0)
    assert np.all(first.pos < cfg.world)
    assert seed_state(cfg).pos.shape == (32, 2)


def test_particle_lenia_cli_writes_particle_state(tmp_path, capsys):
    output = tmp_path / "particle-lenia.npz"
    main([str(CFG), "--particles", "24", "--frames", "2", "--output", str(output)])

    with np.load(output) as state:
        assert state["pos"].shape == (24, 2)
        assert state["vel"].shape == (24, 2)
        assert state["frame"].item() == 2
    assert capsys.readouterr().out.strip() == "frame=2 particles=24"