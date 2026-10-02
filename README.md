# ECE corrected kernels

Reference implementation and document set for the Entelechy Continuum
Engine after the catalog review.

The original catalog mixed a correct lineage write-up with an
over-specified GPU product. This tree keeps the lineage, fixes the
math, and cuts the build to something that can finish.

## Quick start

CI runs the CPU suite on every pull request (`.github/workflows/ci.yml`).


```bash
python3 -m pip install numpy pytest
python3 -m pytest tests/ -q
```

## Run a headless Mohr simulation

```bash
python3 -m ece configs/particle_life_6.toml --frames 100 \
	--output /tmp/particle-state.npz
```

The frame count defaults to `io.frames` in the config. Use `--particles`
to override the configured particle count for a small smoke run. The NPZ
contains final positions, velocities, species types, and frame number.

For an integer Mohr replay reference, pass `--fixedpoint`. Its NPZ also
includes the raw `pos_q16` and `vel_q24` arrays; this CPU prototype does not
claim cross-GPU replay.

For optional CUDA Field Life, Lenia, and Particle Lenia, install Numba with
`python3 -m pip install '.[cuda]'`. Stable-sort CUDA Mohr and resident CUDA
hybrid also require `python3 -m pip install '.[cuda-fft]'` for CuPy. Mohr
builds hashes and keeps state resident; the hybrid keeps particle/field state
and stable planning on-device. On the GB10, a constant-density one-million-particle
run measured about 9.0 steps/s over a 10-frame probe and about 5.4 steps/s over
100 frames; higher sustained throughput and system-memory accounting remain
future work.

Benchmark stable-sort resident CUDA Mohr after JIT warmup with
`python3 tools/benchmark_cuda_mohr.py --particles 4096 16384 65536 --frames 3`.
Use `--constant-density` when scaling counts so the world/hash grid grows with
the population; e.g. `--particles 4096 65536 1048576 --frames 10 --constant-density`.

The optional WebGPU compute paths, including Particle Lenia, and native window use
`python3 -m pip install '.[webgpu]'`. Add `--wgpu` to the headless runner to
use WebGPU compute. Mohr hash tables are built on the CPU.

CUDA hybrid runs keep particle, velocity, and field state on-device and use
stable GPU sorting for hash/deposit planning. The shipped 1,024-particle/128²
hybrid config measured about 746 steps/s over 100 warmed frames on the GB10.

Launch the native WebGPU window with:

```bash
python3 -m ece.wgpu_preview configs/particle_life_6.toml
```

It uses direct screen presentation and supports Mohr particles, Field Life,
Lenia fields, and Particle Lenia particles. Pass `--frames` to set the run
limit.

## Interactive CPU preview

```bash
python3 -m ece.preview configs/particle_life_6.toml --particles 512
```

The preview runs for `io.frames` frames (1,000 in this config) and then stops;
use `--frames` to choose a shorter or longer limit. It supports start/pause,
single-step, reset/restart, and live edits to the species interaction matrix.
Add `--backend cuda` or `--backend wgpu` to run the force step on an available
GPU while keeping the Tk canvas as renderer. The speed control advances up to
four simulation steps per screen update.

## Run Field Life on the CPU

```bash
python3 -m ece configs/field_life.toml --frames 20 \
	--output /tmp/field-state.npz
```

The saved NPZ contains the final per-channel density field and frame number.
The CPU reference supports 2D and 3D. For 3D CUDA/cuFFT, install
`python3 -m pip install '.[cuda-fft]'` and pass `--cuda`.
Benchmark volumes with:

```bash
python3 tools/benchmark_cupy_field_life.py --sizes 64 128 256 --frames 3
python3 tools/benchmark_cupy_field_life.py --sizes 512 --frames 10 --warmup 1
```

The ten-frame 512³ run is not a long-run stability or unified memory-headroom guarantee.

## Run the PIC hybrid

```bash
python3 -m ece configs/hybrid_pic.toml --frames 20 --particles 512 \
	--output /tmp/hybrid-state.npz
```

The hybrid output includes particle state, the evolved field, and sampled
per-channel field values at particle positions. Add `--cuda` or `--wgpu` to
use the corresponding compute backend.

## Run Lenia on the CPU

```bash
python3 -m ece configs/lenia_orbium.toml --frames 100 \
	--output /tmp/lenia-state.npz
```

The current CPU reference starts from a deterministic annular seed and clips
the growth update to `[0, 1]`. With the supplied parameters this approximate
seed grows initially but becomes extinct by frame 100; it is not a canonical
Orbium initial pattern.

## Run Particle Lenia

```bash
python3 -m ece configs/particle_lenia.toml --frames 1000 \
	--output /tmp/particle-lenia.npz
```

This uses the distinct energy-gradient rule on a periodic 16×16 domain; it is
not a Mohr matrix simulation.

## What lives here

| Area | Path |
|---|---|
| Design docs | `docs/` |
| Spec deltas | `CORRECTIONS.md` |
| Kernels | `ece/` |
| WGSL fragments | `shaders/` |
| Configs | `configs/` |
| JSON Schema | `schemas/sim.schema.json` |
| Tests | `tests/` |
| Handbook | `docs/handbook.pdf` |

Start at `docs/00-overview.md`, then `docs/13-file-index.md`. Open work is
in `docs/09-roadmap.md` as self-contained task cards; contributors and
coding agents should read `AGENTS.md` first.

## Corrections in one line each

1. Mohr: `accel += gain * F * hat(Δx)` — no extra `r_max`.
2. Friction: `v *= exp(-λ dt)`.
3. MaCE: store `Z`, gather, wrap the torus, test `Σρ`.
4. Hash: snapshot counts; do not scan a live cursor.
5. Scope: M1 is 2D Mohr. VLM, 512³, and the IDE wait.

## Status

M0–M4 are complete; M5 is partial (see `docs/09-roadmap.md`). CPU Mohr,
Field Life, Lenia, Particle Lenia, and PIC reference paths are available, with
optional CUDA/CuPy/WebGPU backends, a Tk preview, and a native Qt/WebGPU
window. The hybrid couples one way: particles deposit into the field each
frame, but sampled field values do not yet feed back into particle forces.
The CUDA fixed-point replay port and the evaluator are not built.

The CPU binned Mohr stepper is vectorized (about 0.5 s per frame at 4,096
particles); use `--particles` for quick CPU smoke runs. The `--fixedpoint`
replay path is pure-Python O(N²) and is meant for small N.
