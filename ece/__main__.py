"""Run a headless single-rule simulation from a TOML config."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import load_config
from .cuda_mohr import cuda_available, run_cuda_resident
from .cuda_field_life import run_cuda as run_field_life_cuda
from .cuda_hybrid import run_cuda_resident as run_cuda_hybrid
from .cuda_lenia import run_cuda as run_lenia_cuda
from .cuda_particle_lenia import run_cuda as run_particle_lenia_cuda
from .cupy_field_life import cupy_available, run_cupy as run_field_life_cupy
from .field_life import run as run_field_life, seed_state as seed_field_life_state
from .fixedpoint_mohr import run_fixed as run_mohr_fixed
from .hybrid import run as run_hybrid
from .lenia import run as run_lenia, seed_state as seed_lenia_state
from .particle_lenia import run as run_particle_lenia
from .step_mohr import run as run_mohr, seed_state as seed_mohr_state
from .wgpu_mohr import step_wgpu, webgpu_available
from .wgpu_field_life import step_wgpu as step_field_life_wgpu
from .wgpu_lenia import step_wgpu as step_lenia_wgpu
from .wgpu_particle_lenia import run_wgpu as run_particle_lenia_wgpu


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a headless simulation")
    parser.add_argument("config", type=Path, help="path to a TOML simulation config")
    parser.add_argument("--frames", type=_positive_int, help="override configured frame count")
    parser.add_argument("--particles", type=_positive_int, help="override configured particle count")
    parser.add_argument("--cuda", action="store_true", help="run supported kernels on CUDA")
    parser.add_argument("--wgpu", action="store_true", help="run Mohr force steps on WebGPU")
    parser.add_argument("--fixedpoint", action="store_true", help="run integer Mohr replay on the CPU")
    parser.add_argument("--output", type=Path, help="write final state to a compressed NPZ file")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if sum((args.cuda, args.wgpu, args.fixedpoint)) > 1:
        parser.error("choose only one backend")
    frames = args.frames if args.frames is not None else int(cfg.io.get("frames", 1))
    if frames < 1:
        parser.error("configured io.frames must be at least 1")
    is_hybrid = len(cfg.rules) == 2 and set(cfg.rules) == {"mohr", "field_life"}
    if args.wgpu and cfg.rules not in (["mohr"], ["field_life"], ["lenia"], ["particle_lenia"]) and not is_hybrid:
        parser.error("--wgpu currently supports Mohr, Field Life, and Lenia simulations")
    if args.fixedpoint and cfg.rules != ["mohr"]:
        parser.error("--fixedpoint currently supports Mohr configs only")
    if args.cuda and cfg.rules not in (["mohr"], ["field_life"], ["lenia"], ["particle_lenia"]) and not is_hybrid:
        parser.error("--cuda currently supports Mohr, Field Life, Lenia, and Mohr/Field Life hybrid runs")

    if cfg.rules == ["mohr"]:
        missing = {"particles", "r_max", "beta"} - cfg.mohr.keys()
        if missing:
            parser.error(f"Mohr config is missing: {', '.join(sorted(missing))}")
        if args.particles is not None:
            cfg.mohr["particles"] = args.particles
        if args.fixedpoint:
            state = run_mohr_fixed(cfg, frames=frames)
            output_state = {
                "pos": state.pos_q16.astype(np.float64) / (1 << 16),
                "vel": state.vel_q24.astype(np.float64) / (1 << 24),
                "pos_q16": state.pos_q16,
                "vel_q24": state.vel_q24,
                "types": state.types,
                "frame": np.asarray(state.frame),
            }
            summary = f"fixedpoint frame={state.frame} particles={len(state.types)}"
        elif args.wgpu:
            if not webgpu_available():
                parser.error("WebGPU is unavailable; install the webgpu extra and check the adapter")
            state = seed_mohr_state(cfg)
            for _ in range(frames):
                state = step_wgpu(state, cfg)
        elif args.cuda:
            if not cuda_available():
                parser.error("CUDA is unavailable; install the cuda extra and check the driver")
            if not cupy_available():
                parser.error("stable CUDA Mohr planning requires the cuda-fft extra")
            state = run_cuda_resident(cfg, frames=frames)
        else:
            state = run_mohr(cfg, frames=frames)
        if not args.fixedpoint:
            output_state = {
                "pos": state.pos,
                "vel": state.vel,
                "types": state.types,
                "frame": np.asarray(state.frame),
            }
            summary = f"frame={state.frame} particles={len(state.pos)}"
    elif cfg.rules == ["field_life"]:
        if args.particles is not None:
            parser.error("--particles is only supported for Mohr simulations")
        field_dims = tuple(cfg.field_cfg.get("dims", ()))
        if args.wgpu and len(field_dims) != 2:
            parser.error("WebGPU Field Life currently requires a 2D field")
        if args.cuda:
            if len(field_dims) == 3:
                if not cupy_available():
                    parser.error("3D Field Life requires CuPy/cuFFT; install the cuda-fft extra")
                state = run_field_life_cupy(cfg, frames=frames)
            else:
                if not cuda_available():
                    parser.error("CUDA is unavailable; install the cuda extra and check the driver")
                state = run_field_life_cuda(cfg, frames=frames)
        elif args.wgpu:
            if not webgpu_available():
                parser.error("WebGPU is unavailable; install the webgpu extra and check the adapter")
            state = seed_field_life_state(cfg)
            for _ in range(frames):
                state = step_field_life_wgpu(state, cfg)
        else:
            state = run_field_life(cfg, frames=frames)
        output_state = {"rho": state.rho, "frame": np.asarray(state.frame)}
        dimensions = "x".join(str(size) for size in state.rho.shape[1:])
        summary = f"frame={state.frame} channels={state.rho.shape[0]} dims={dimensions}"
    elif cfg.rules == ["lenia"]:
        if args.particles is not None:
            parser.error("--particles is only supported for Mohr simulations")
        missing = {"mu", "sigma"} - cfg.lenia.keys()
        if missing:
            parser.error(f"Lenia config is missing: {', '.join(sorted(missing))}")
        if args.cuda:
            if not cuda_available():
                parser.error("CUDA is unavailable; install the cuda extra and check the driver")
            state = run_lenia_cuda(cfg, frames=frames)
        elif args.wgpu:
            if not webgpu_available():
                parser.error("WebGPU is unavailable; install the webgpu extra and check the adapter")
            state = seed_lenia_state(cfg)
            for _ in range(frames):
                state = step_lenia_wgpu(state, cfg)
        else:
            state = run_lenia(cfg, frames=frames)
        output_state = {"rho": state.rho, "frame": np.asarray(state.frame)}
        summary = f"frame={state.frame} channels={state.rho.shape[0]} dims={state.rho.shape[1]}x{state.rho.shape[2]}"
    elif cfg.rules == ["particle_lenia"]:
        if args.particles is not None:
            cfg.particle_lenia["particles"] = args.particles
        if args.cuda:
            state = run_particle_lenia_cuda(cfg, frames=frames)
        elif args.wgpu:
            state = run_particle_lenia_wgpu(cfg, frames=frames)
        else:
            state = run_particle_lenia(cfg, frames=frames)
        output_state = {
            "pos": state.pos,
            "vel": state.vel,
            "types": state.types,
            "frame": np.asarray(state.frame),
        }
        summary = f"frame={state.frame} particles={len(state.pos)}"
    elif is_hybrid:
        missing = {"particles", "r_max", "beta"} - cfg.mohr.keys()
        if missing:
            parser.error(f"Mohr config is missing: {', '.join(sorted(missing))}")
        if args.particles is not None:
            cfg.mohr["particles"] = args.particles
        if args.cuda and not cuda_available():
            parser.error("CUDA is unavailable; install the cuda extra and check the driver")
        if args.cuda and not cupy_available():
            parser.error("resident CUDA hybrid planning requires the cuda-fft extra")
        if args.wgpu and not webgpu_available():
            parser.error("WebGPU is unavailable; install the webgpu extra and check the adapter")
        if args.cuda:
            state = run_cuda_hybrid(cfg, frames=frames)
        else:
            state = run_hybrid(cfg, frames=frames, use_wgpu=args.wgpu)
        output_state = {
            "pos": state.particles.pos,
            "vel": state.particles.vel,
            "types": state.particles.types,
            "rho": state.rho,
            "sampled": state.sampled,
            "frame": np.asarray(state.frame),
        }
        summary = f"frame={state.frame} particles={len(state.particles.pos)} channels={state.rho.shape[0]}"
    else:
        parser.error('currently supports Mohr, Field Life, Lenia, and Mohr/Field Life hybrid configs')

    if args.output is not None:
        with args.output.open("wb") as output_file:
            np.savez_compressed(output_file, **output_state)

    print(summary)


if __name__ == "__main__":
    main()