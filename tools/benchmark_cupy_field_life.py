"""Measure warmed CuPy/cuFFT 3D Field Life throughput and device-pool use."""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ece.config import load_config
from ece.cupy_field_life import cupy_available, run_cupy
from ece.field_life import seed_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "field_life.toml")
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if any(size < 7 for size in args.sizes) or args.frames < 1 or args.warmup < 0:
        parser.error("sizes must be >= 7, frames positive, and warmup non-negative")
    if not cupy_available():
        parser.error("CuPy/cuFFT is unavailable; install the cuda-fft extra")

    import cupy as cp

    print("size,channels,frames,seconds_per_frame,voxels_per_second,max_channel_mass_rel_error,pool_bytes,process_peak_rss_mb")
    for size in args.sizes:
        cfg = load_config(args.config)
        cfg.world = np.ones(3, dtype=np.float64)
        cfg.field_cfg["dims"] = [size, size, size]
        if args.warmup:
            run_cupy(cfg, frames=args.warmup)
        initial = seed_state(cfg)
        initial_mass = initial.rho.sum(axis=(1, 2, 3))
        del initial
        cp.cuda.get_current_stream().synchronize()
        started = time.perf_counter()
        result = run_cupy(cfg, frames=args.frames)
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        seconds_per_frame = elapsed / args.frames
        voxels_per_second = size**3 / seconds_per_frame
        final_mass = result.rho.sum(axis=(1, 2, 3))
        relative_mass_error = np.max(
            np.abs(final_mass - initial_mass) / np.maximum(initial_mass, 1e-30)
        )
        pool_bytes = cp.get_default_memory_pool().total_bytes()
        process_peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        print(
            f"{size},{cfg.species_count},{args.frames},{seconds_per_frame:.6f},"
            f"{voxels_per_second:.0f},{relative_mass_error:.8g},{pool_bytes},"
            f"{process_peak_rss_mb:.1f}"
        )


if __name__ == "__main__":
    main()