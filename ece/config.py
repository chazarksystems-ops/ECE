"""Load and validate TOML configs against the documented rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib

from .matrix import make_matrix
import numpy as np


@dataclass
class SimConfig:
    dt: float
    world: np.ndarray
    seed: int
    rules: list[str]
    species_count: int
    matrix_mode: str
    matrix_range: tuple[float, float]
    matrix_path: str | None
    matrix: np.ndarray
    mohr: dict = field(default_factory=dict)
    field_cfg: dict = field(default_factory=dict)
    lenia: dict = field(default_factory=dict)
    particle_lenia: dict = field(default_factory=dict)
    hash: dict = field(default_factory=dict)
    io: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


_ALLOWED_TOP = {"simulation", "species", "mohr", "field", "lenia", "particle_lenia", "hash", "io"}
_ALLOWED_RULES = {"mohr", "field_life", "lenia", "particle_lenia", "pps"}


def load_config(path: str | Path) -> SimConfig:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    extra = set(raw) - _ALLOWED_TOP
    if extra:
        raise ValueError(f"unknown top-level keys: {sorted(extra)}")
    sim = raw.get("simulation") or {}
    spec = raw.get("species") or {}
    for req in ("dt", "world", "seed"):
        if req not in sim:
            raise ValueError(f"simulation.{req} is required")
    if "count" not in spec:
        raise ValueError("species.count is required")
    dt = float(sim["dt"])
    if dt <= 0:
        raise ValueError("dt must be > 0")
    world = np.asarray(sim["world"], dtype=np.float64)
    if world.ndim != 1 or world.size not in (2, 3) or np.any(world <= 0):
        raise ValueError("world must be 2 or 3 positive lengths")
    seed = int(sim["seed"])
    rules = list(sim.get("rules") or [])
    bad = set(rules) - _ALLOWED_RULES
    if bad:
        raise ValueError(f"unknown rules: {sorted(bad)}")
    k = int(spec["count"])
    mode = spec.get("matrix", "random")
    rng_range = tuple(spec.get("range") or [-1.0, 1.0])
    mpath = spec.get("matrix_path")
    rng = np.random.default_rng(seed)
    matrix = make_matrix(mode, k, rng, rng_range[0], rng_range[1], mpath)

    mohr = dict(raw.get("mohr") or {})
    if mohr:
        beta = float(mohr.get("beta", 0.3))
        if not (0.0 < beta < 1.0):
            raise ValueError("mohr.beta must be in (0, 1)")
        if float(mohr.get("r_max", 1.0)) <= 0:
            raise ValueError("mohr.r_max must be > 0")
        if float(mohr.get("lambda", 0.0)) < 0:
            raise ValueError("mohr.lambda must be >= 0")

    hsh = dict(raw.get("hash") or {})
    if hsh and mohr:
        cell = float(hsh.get("cell", mohr.get("r_max", 0.0)))
        r_max = float(mohr.get("r_max", 0.0))
        neigh = int(hsh.get("neighborhood", 3))
        if neigh == 3 and cell + 1e-12 < r_max:
            raise ValueError("hash.cell must be >= mohr.r_max for neighborhood=3")
        grid = hsh.get("grid")
        if world.size == 2 and grid is not None and len(grid) == 2 and neigh == 3:
            widths = world / np.asarray(grid, dtype=np.float64)
            if np.any(widths + 1e-12 < r_max):
                raise ValueError("world/grid cells must be >= mohr.r_max for neighborhood=3")

    field = dict(raw.get("field") or {})
    if field and "channels" in field and int(field["channels"]) != k:
        # Allow mismatch only if rules don't include field_life.
        if "field_life" in rules:
            raise ValueError("field.channels must equal species.count")

    particle_lenia = dict(raw.get("particle_lenia") or {})
    if "particle_lenia" in rules:
        if world.size != 2:
            raise ValueError("particle_lenia currently requires a 2D world")
        if k != 1:
            raise ValueError("particle_lenia currently requires exactly one species")
        required = {"particles", "kernel_mu", "kernel_sigma", "growth_mu", "growth_sigma", "c_rep"}
        missing = required - particle_lenia.keys()
        if missing:
            raise ValueError(f"particle_lenia is missing: {', '.join(sorted(missing))}")
        if int(particle_lenia["particles"]) < 1:
            raise ValueError("particle_lenia.particles must be at least 1")
        scales = np.asarray(
            [particle_lenia["kernel_mu"], particle_lenia["kernel_sigma"],
             particle_lenia["growth_mu"], particle_lenia["growth_sigma"],
             particle_lenia["c_rep"]],
            dtype=np.float64,
        )
        if not np.isfinite(scales).all():
            raise ValueError("particle_lenia parameters must be finite")
        kernel_mu, kernel_sigma, _, growth_sigma, c_rep = scales
        if kernel_mu <= 0.0 or kernel_sigma <= 0.0 or growth_sigma <= 0.0 or c_rep < 0.0:
            raise ValueError("particle_lenia scales must be positive and c_rep non-negative")
        if np.min(world) < 2.0 * (kernel_mu + 4.0 * kernel_sigma):
            raise ValueError("particle_lenia world must span at least the shell diameter")

    return SimConfig(
        dt=dt,
        world=world,
        seed=seed,
        rules=rules,
        species_count=k,
        matrix_mode=mode,
        matrix_range=(float(rng_range[0]), float(rng_range[1])),
        matrix_path=mpath,
        matrix=matrix,
        mohr=mohr,
        field_cfg=field,
        lenia=dict(raw.get("lenia") or {}),
        particle_lenia=particle_lenia,
        hash=hsh,
        io=dict(raw.get("io") or {}),
        raw=raw,
    )
