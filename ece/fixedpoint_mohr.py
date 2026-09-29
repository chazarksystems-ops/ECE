"""Deterministic integer Mohr reference using the documented replay formats."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
import math

import numpy as np

from .config import SimConfig
from .step_mohr import MohrState, seed_state


POSITION_BITS = 16
VELOCITY_BITS = 24
MATRIX_BITS = 15
UNIT_BITS = 15
GAIN_BITS = 16
TIME_BITS = 24

POSITION_SCALE = 1 << POSITION_BITS
VELOCITY_SCALE = 1 << VELOCITY_BITS
MATRIX_SCALE = 1 << MATRIX_BITS
UNIT_SCALE = 1 << UNIT_BITS
GAIN_SCALE = 1 << GAIN_BITS
TIME_SCALE = 1 << TIME_BITS


@dataclass
class FixedMohrState:
    pos_q16: np.ndarray
    vel_q24: np.ndarray
    types: np.ndarray
    frame: int = 0


def _quantize(value: float, scale: int) -> int:
    return int((Decimal(str(value)) * scale).to_integral_value(rounding=ROUND_HALF_EVEN))


def _round_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    magnitude = abs(numerator)
    rounded = (magnitude + denominator // 2) // denominator
    return rounded if numerator >= 0 else -rounded


def _friction_decay_q24(dt: float, lam: float) -> int:
    with localcontext() as context:
        context.prec = 80
        exponent = -Decimal(str(dt)) * Decimal(str(lam))
        decay = exponent.exp() * TIME_SCALE
        return int(decay.to_integral_value(rounding=ROUND_HALF_EVEN))


def seed_fixed_state(cfg: SimConfig) -> FixedMohrState:
    initial = seed_state(cfg)
    positions = np.rint(initial.pos * POSITION_SCALE).astype(np.int64)
    velocities = np.rint(initial.vel * VELOCITY_SCALE).astype(np.int64)
    return FixedMohrState(positions, velocities, initial.types.copy(), frame=0)


def step_fixed(state: FixedMohrState, cfg: SimConfig) -> FixedMohrState:
    if cfg.world.size != 2:
        raise ValueError("fixed-point Mohr currently supports 2D worlds")
    world = [_quantize(value, POSITION_SCALE) for value in cfg.world]
    r_max = _quantize(float(cfg.mohr["r_max"]), POSITION_SCALE)
    beta = _quantize(float(cfg.mohr["beta"]), POSITION_SCALE)
    gain = _quantize(float(cfg.mohr.get("gain", 1.0)), GAIN_SCALE)
    dt = _quantize(cfg.dt, TIME_SCALE)
    decay = _friction_decay_q24(cfg.dt, float(cfg.mohr.get("lambda", 0.0)))
    matrix = np.rint(np.asarray(cfg.matrix) * MATRIX_SCALE).astype(np.int64)
    next_positions = np.empty_like(state.pos_q16)
    next_velocities = np.empty_like(state.vel_q24)
    count = len(state.pos_q16)

    for i in range(count):
        accel_x_q46 = 0
        accel_y_q46 = 0
        for j in range(count):
            if i == j:
                continue
            delta_x = int(state.pos_q16[j, 0]) - int(state.pos_q16[i, 0])
            delta_y = int(state.pos_q16[j, 1]) - int(state.pos_q16[i, 1])
            if 2 * delta_x > world[0]:
                delta_x -= world[0]
            elif 2 * delta_x < -world[0]:
                delta_x += world[0]
            if 2 * delta_y > world[1]:
                delta_y -= world[1]
            elif 2 * delta_y < -world[1]:
                delta_y += world[1]
            distance = math.isqrt(delta_x * delta_x + delta_y * delta_y)
            if distance == 0 or distance > r_max:
                continue
            normalized = _round_div(distance * POSITION_SCALE, r_max)
            if normalized < beta:
                force_q15 = _round_div(normalized * MATRIX_SCALE, beta) - MATRIX_SCALE
            else:
                width = POSITION_SCALE - beta
                tent_offset = abs(2 * normalized - POSITION_SCALE - beta)
                shape_q16 = POSITION_SCALE - _round_div(tent_offset * POSITION_SCALE, width)
                force_q15 = _round_div(
                    int(matrix[int(state.types[i]), int(state.types[j])]) * shape_q16,
                    POSITION_SCALE,
                )
            unit_x_q15 = _round_div(delta_x * UNIT_SCALE, distance)
            unit_y_q15 = _round_div(delta_y * UNIT_SCALE, distance)
            accel_x_q46 += gain * force_q15 * unit_x_q15
            accel_y_q46 += gain * force_q15 * unit_y_q15

        accel_x_q16 = _round_div(accel_x_q46, 1 << (GAIN_BITS + MATRIX_BITS + UNIT_BITS - POSITION_BITS))
        accel_y_q16 = _round_div(accel_y_q46, 1 << (GAIN_BITS + MATRIX_BITS + UNIT_BITS - POSITION_BITS))
        velocity_x = _round_div(int(state.vel_q24[i, 0]) * decay, TIME_SCALE)
        velocity_y = _round_div(int(state.vel_q24[i, 1]) * decay, TIME_SCALE)
        velocity_x += _round_div(accel_x_q16 * dt, POSITION_SCALE)
        velocity_y += _round_div(accel_y_q16 * dt, POSITION_SCALE)
        position_x = int(state.pos_q16[i, 0]) + _round_div(velocity_x * dt, TIME_SCALE * TIME_SCALE // POSITION_SCALE)
        position_y = int(state.pos_q16[i, 1]) + _round_div(velocity_y * dt, TIME_SCALE * TIME_SCALE // POSITION_SCALE)
        next_positions[i, 0] = position_x % world[0]
        next_positions[i, 1] = position_y % world[1]
        next_velocities[i, 0] = velocity_x
        next_velocities[i, 1] = velocity_y

    return FixedMohrState(next_positions, next_velocities, state.types, state.frame + 1)


def run_fixed(cfg: SimConfig, frames: int) -> FixedMohrState:
    if frames < 0:
        raise ValueError("frames must be non-negative")
    state = seed_fixed_state(cfg)
    for _ in range(frames):
        state = step_fixed(state, cfg)
    return state


def to_mohr_state(state: FixedMohrState) -> MohrState:
    return MohrState(
        pos=state.pos_q16.astype(np.float64) / POSITION_SCALE,
        vel=state.vel_q24.astype(np.float64) / VELOCITY_SCALE,
        types=state.types.copy(),
        frame=state.frame,
    )