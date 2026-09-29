"""Measure resident CUDA Mohr throughput after a JIT warmup run."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ece.config import load_config
from ece.cuda_mohr import cuda_available, run_cuda_resident


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "particle_life_6.toml",
    )
    parser.add_argument("--particles", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--constant-density",
        action="store_true",
        help="scale world and hash grid with particle count to keep density comparable",
    )
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0 or any(count < 1 for count in args.particles):
        parser.error("particles and frames must be positive; warmup must be non-negative")
    if not cuda_available():
        parser.error("CUDA is unavailable; install the cuda extra and check the driver")

    print("particles,frames,elapsed_seconds,steps_per_second,particle_steps_per_second,grid,world,state_valid")
    for particle_count in args.particles:
        cfg = load_config(args.config)
        base_particles = int(cfg.mohr["particles"])
        cfg.mohr["particles"] = particle_count
        if args.constant_density:
            if not cfg.hash:
                parser.error("--constant-density requires hash settings in the config")
            scale = np.sqrt(particle_count / base_particles)
            cfg.world = cfg.world * scale
            cell = float(cfg.hash["cell"])
            cfg.hash["grid"] = [max(1, int(np.floor(length / cell))) for length in cfg.world]
        if args.warmup:
            run_cuda_resident(cfg, frames=args.warmup)
        started = time.perf_counter()
        state = run_cuda_resident(cfg, frames=args.frames)
        elapsed = time.perf_counter() - started
        if not np.isfinite(state.pos).all() or not np.isfinite(state.vel).all():
            raise RuntimeError(f"non-finite state after {args.frames} frames at N={particle_count}")
        if np.any(state.pos < 0.0) or np.any(state.pos >= cfg.world):
            raise RuntimeError(f"positions escaped the torus after {args.frames} frames at N={particle_count}")
        steps_per_second = args.frames / elapsed
        particle_steps_per_second = particle_count * steps_per_second
        grid = "x".join(str(size) for size in cfg.hash.get("grid", ()))
        print(
            f"{particle_count},{args.frames},{elapsed:.6f},"
            f"{steps_per_second:.2f},{particle_steps_per_second:.0f},"
            f"{grid},{cfg.world[0]:.4f}x{cfg.world[1]:.4f},true"
        )


if __name__ == "__main__":
    main()