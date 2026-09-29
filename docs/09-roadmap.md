# Roadmap

Cut from the original catalog. The catalog described a company-sized
surface. This is the build order that can actually finish.

## M0 — done

- Corrected Mohr tent (no extra `r_max`)
- Exponential friction
- Two-pass MaCE + torus wrap
- Snapshot hash tables
- Invariant tests (15 passing)
- This document tree, configs, schemas, WGSL fragments

## M1 — 2D Mohr preview

- [x] Headless CPU frame runner calling `ece.step_mohr`
- [x] Six-species binned-vs-dense comparison over 200 frames
- [x] Interactive Tk preview with live K×K matrix editor
- [x] CPU, CUDA, and WGSL force backends selectable in the preview
- [x] 200-frame WGSL-vs-CPU per-step oracle comparison on 256 particles
- [x] Native Qt/WebGPU presentation surface with direct screen rendering
- [x] Device-resident native Mohr compute/integrate/render loop

Exit: a 6-species soup that matches the Python oracle on N ≤ 256
for a few hundred steps within f32 tolerance.

## M2 — Field Life 2D

- [x] Config-driven CPU runner using convolution + MaCE passes
- [x] Per-channel mass-invariance test and NPZ output
- [x] CUDA ring convolution, MaCE passes, and per-channel readback test
- [x] WGSL MaCE passes, CPU parity, and headless CLI output
- [x] Native WebGPU field presentation for Field Life
- [x] CuPy/cuFFT 3D Field Life path with CPU parity at 16³ and 256³ smoke run

Exit: per-channel mass holds; creatures do not evaporate.

## M3 — Lenia growth on the same field buffers

- [x] CPU growth-bell update and normalized radial ring kernel
- [x] Deterministic annular seed and `lenia_orbium.toml` runner; current seed goes extinct by frame 100
- [x] CUDA ring convolution/growth and CPU parity test
- [x] WGSL ring convolution/growth and CPU parity test
- [x] Native WebGPU field presentation for Lenia

Exit: Orbium persists or dies for documented reasons, not because
of a broken clip.

## M4 — PIC coupling

- [x] CPU periodic bilinear particle deposit and field sample references
- [x] Per-species mass and deposit/sample adjoint tests
- [x] Deterministic CUDA segmented deposit and bilinear sample passes
- [x] Deterministic WGSL segmented deposit and bilinear sample passes
- [x] CPU/CUDA/WebGPU hybrid scheduler: Mohr → deposit → Field Life → sample
- [x] CUDA hybrid keeps particle/velocity/field state and stable hash/PIC planning on-device
- [x] Same-device repeated CUDA hybrid runs produce identical state
- [x] 1024-particle/128² CUDA hybrid baseline: ~746 steps/s over 100 frames (GB10)
- [x] Single-species CPU Particle Lenia energy-gradient rule and periodic config
- [x] CUDA and WGSL Particle Lenia kernels with CPU oracle tests
- [x] Native Particle Lenia visualization
- [x] Keep Particle Lenia as a standalone direct-KDE rule; it does not require PIC deposit/sample

## M5 — CUDA path on Spark

- [x] Optional CUDA Mohr force kernel with GPU-built hash tables and oracle comparison
- [x] CUDA hash count/scan/scatter and persistent device-resident Mohr integration
- [x] Reproducible stable-sort CUDA benchmark and 4096-particle baseline (GB10: ~600 steps/s, 100 frames)
- [x] Same-device deterministic standalone and hybrid CUDA repeatability tests
- [x] Stable-sort CUDA Mohr scaling probes: 16K at 135 steps/s; 65K at 10.6 steps/s
- [x] Constant-density 1M-particle CUDA run: 10 frames, ~9.0 steps/s, finite torus state
- [ ] Optimize standalone CUDA Mohr throughput beyond this baseline
- [x] cuFFT-backed 3D field update prototype
- [x] CPU parity and per-channel mass at 16³; 256³ and 512³ feasibility probes
- [x] Fused four-channel cuFFT MaCE benchmark at 64³/128³/256³ (2.6 ms/24 ms/0.21 s per frame; ~1.2e-8 mass drift)
- [x] Fused 512³ four-channel, ten-frame run (1.15 s/frame; 1.05e-8 mass drift; ~18.8 GB CuPy pool)
- [ ] Longer 512³ production run and unified system-memory accounting
- [x] CPU integer Mohr replay prototype in Q16.16/Q8.24/Q1.15 with CLI export and bitwise tests
- [ ] CUDA fixed-point port, versioned LUTs, and cross-box replay certification
- [ ] Sustained 256³ default and 512³ opt-in workloads with full memory accounting

## Explicitly not scheduled

- Co-local 7B VLM steering (design only: `docs/12-ai-evaluator-deferred.md`)
- Full egui IDE / DAG node editor
- crates.io release matrix
- 64-species food webs
- 1024³ volumes
