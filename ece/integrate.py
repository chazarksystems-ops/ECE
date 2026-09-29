"""Integration with explicit exponential friction.

Catalog used Mohr's display-rate form ``v *= gamma ** (60 * dt)``.
That silently couples physics to an assumed 60 Hz tick. Use a
time-constant instead:

    v <- v * exp(-lambda * dt)

``lambda = 0`` is undamped. ``lambda = -ln(gamma) * 60`` recovers Mohr
if you really want the old feel.
"""

from __future__ import annotations

import numpy as np


def friction_decay(dt: float, lam: float) -> float:
    if dt < 0.0:
        raise ValueError("dt must be >= 0")
    if lam < 0.0:
        raise ValueError("lambda (friction rate) must be >= 0")
    return float(np.exp(-lam * dt))


def mohr_gamma_to_lambda(gamma: float, hz: float = 60.0) -> float:
    """Convert Mohr's per-frame gamma at ``hz`` into an exponential rate."""
    if not (0.0 < gamma <= 1.0):
        raise ValueError("gamma must be in (0, 1]")
    return float(-np.log(gamma) * hz)


def symplectic_euler(
    pos: np.ndarray,
    vel: np.ndarray,
    accel: np.ndarray,
    dt: float,
    lam: float,
    world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """v <- decay * v + a * dt; x <- wrap(x + v * dt)."""
    decay = friction_decay(dt, lam)
    vel = vel * decay + accel * dt
    pos = pos + vel * dt
    world = np.asarray(world, dtype=np.float64)
    pos = pos - np.floor(pos / world) * world
    return pos, vel
