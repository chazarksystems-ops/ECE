# Roadmap

Cut from the original catalog. The catalog described a company-sized
surface. This is the build order that can actually finish.

This file has two parts:

1. **Open work** — task cards written so a coding agent (or a new
   contributor) can pick one up cold. Each card says why it matters,
   which files to touch, what "done" means, how to verify it, and what
   hardware the verification needs.
2. **Completed milestones** — what already exists, so nobody rebuilds it.

Read `AGENTS.md` first for repo conventions, invariants, and the
definition of done that applies to every card.

## How to use the task cards

- Pick a card whose **Depends on** items are done and whose
  **Hardware** you have. If the card says **Decision needed**, stop and
  get the owner's answer before writing code; do not pick a default
  silently.
- Keep a change to one card. Do not fold in unrelated fixes; add a new
  card instead.
- When a card is finished, move its line to the matching milestone
  under **Completed milestones** as `- [x]`, with any measured numbers
  and the hardware they were measured on, and delete the card.
- If the work reveals new problems, add a card for each using the same
  fields. Never leave a known problem only in a commit message.

### Hardware tags

| Tag | Meaning |
|---|---|
| `cpu` | Fully verifiable with `numpy` + `pytest`; works in a cloud container |
| `cuda` | Needs an NVIDIA GPU with Numba CUDA (and CuPy for resident/3D paths) |
| `wgpu` | Needs a WebGPU adapter (`wgpu` package); native window also needs Qt + a display |
| `gb10` | Needs the DGX Spark itself: benchmarks and memory numbers are only meaningful there |

GPU tests skip cleanly without a device, so a green CPU run says
**nothing** about GPU correctness. For `cuda`/`wgpu`/`gb10` cards
without that hardware: make the change, run the CPU suite and
`python3 -m py_compile ece/*.py`, and state in the PR exactly which GPU
commands the owner must run before the card counts as done. Do not tick
the roadmap box until those commands pass.

### Priority and order

| ID | Title | Priority | Hardware | Depends on |
|---|---|---|---|---|
| R-02 | Schema / loader drift test | P1 | cpu | — |
| R-03 | CLI backend-availability consistency | P2 | cpu | — |
| R-04 | MaCE input-shape ambiguity | P2 | cpu | — |
| R-05 | Canonical Orbium for Lenia (M3 exit) | P1 | cpu | — |
| R-06 | State hash and `io.hash_every` | P1 | cpu | — |
| R-07 | PPS rule (CPU reference) | P2 | cpu | R-02 |
| R-08 | Two-way hybrid coupling | P1 | cpu, then cuda/wgpu | **Decision needed** |
| R-09 | Remove dead CUDA hash kernels | P2 | cuda to verify | — |
| R-10 | CuPy 3D index-overflow guard | P1 | cpu guard, cuda to verify | — |
| R-11 | WebGPU parameter block and GPU hash | P2 | wgpu | — |
| R-12 | CUDA Mohr throughput (FP64 promotion) | P1 | cuda, gb10 | — |
| R-13 | Vectorize CPU fixed-point Mohr | P3 | cpu | — |
| R-14 | CUDA fixed-point replay port | P2 | cuda | R-06 |
| R-15 | Long 512³ run and memory accounting | P2 | gb10 | R-10 |
| R-16 | WebGPU tests in CI on a software adapter | P3 | cpu | — |

CPU CI runs on every pull request (`.github/workflows/ci.yml`), so a
`cpu` card's PR must be green before merge. The `cpu` cards can run in
parallel; they touch different files.

## Open work

### R-02 — Schema / loader drift test

- **Why:** `schemas/sim.schema.json` is documentation only; the loader
  keeps its own key sets in `ece/config.py` (`_ALLOWED_TOP`,
  `_ALLOWED_RULES`, `_ALLOWED_KEYS`). They already disagree in places:
  the schema allows `field.kernel = "gaussian" | "disk"` but every runner
  rejects anything but `"ring"`; schema `seed` must be ≥ 0 and
  `species.count` ≤ 128, the loader checks neither.
- **Files:** `tests/test_config.py`, `ece/config.py`,
  `schemas/sim.schema.json`, `docs/11-config-schema.md`.
- **Do:** add a test that reads the schema JSON with the standard library
  and asserts its per-object `properties` keys equal `_ALLOWED_KEYS`, its
  top-level keys equal `_ALLOWED_TOP`, and its rules enum equals
  `_ALLOWED_RULES`. Then make them agree: drop `gaussian`/`disk` from
  the schema (or implement them as their own card), and add the
  `seed >= 0` and `count <= 128` checks to the loader. Do not add a
  `jsonschema` dependency.
- **Done when:** the drift test passes and fails if either side gains a
  key alone.
- **Verify:** `python3 -m pytest tests/test_config.py -q`.

### R-03 — CLI backend-availability consistency

- **Why:** in `ece/__main__.py`, `--cuda`/`--wgpu` for Particle Lenia
  skip the `cuda_available()`/`webgpu_available()` checks every other
  rule has, so users get a raw exception instead of a clean error. Some
  `parser.error` messages also omit Particle Lenia from the supported
  list.
- **Files:** `ece/__main__.py`, `tests/test_particle_lenia.py`.
- **Done when:** every backend flag on every rule gives a
  `parser.error` (exit code 2) naming the missing extra when the device
  is absent, and the messages list all supported rules.
- **Verify:** CPU tests that run the CLI with `--cuda` and `--wgpu` on
  `configs/particle_lenia.toml` and expect `SystemExit(2)` when no device
  is present (mark them to skip when a device *is* present).

### R-04 — MaCE input-shape ambiguity

- **Why:** `ece.mace._transport_layout` reads any rank-3 array as
  `(channels, H, W)`. A single-channel 3D field passed without a
  channel axis silently gets 2D transport, while the docstrings say
  `(*spatial)` is accepted.
- **Files:** `ece/mace.py`, `tests/test_mace.py`.
- **Do:** require the channel axis everywhere (rank 3 means 2D with
  channels, rank 4 means 3D with channels) and update the docstrings, or
  add an explicit `spatial_dims` argument. Prefer the first; it is the
  smaller change.
- **Done when:** the docstrings match behavior and a test shows rank-2
  input is rejected with a clear message.
- **Verify:** `python3 -m pytest tests/test_mace.py tests/test_field_life.py -q`.

### R-05 — Canonical Orbium for Lenia (M3 exit criterion)

- **Why:** M3's exit is "Orbium persists or dies for documented
  reasons". Today `configs/lenia_orbium.toml` uses an approximate annular
  seed that goes extinct by frame 100, and `ece/lenia.py` uses a Gaussian
  ring kernel (peak 0.5, width 0.15, radius `min(dims)//8`), not the
  kernel the published Orbium was found with.
- **Files:** `ece/lenia.py`, `configs/lenia_orbium.toml`,
  `tests/test_lenia.py`, the CUDA/WGSL Lenia modules if the kernel
  changes, `docs/02-rules-catalog.md`.
- **Do:** implement the kernel and seed from the published Lenia
  reference (Bert Wang-Chak Chan's Lenia papers and code), including the
  Orbium cell pattern and its radius. **Obtain the pattern and
  parameters from that source and cite it in the config comment; do not
  reconstruct them from memory.** Keep the existing annular seed as a
  separate, honestly named config if tests still need it.
- **Done when:** a test shows the Orbium keeps total mass within a
  stated band and its centre of mass moves over at least 1,000 frames;
  the README and `docs/02` describe the result. If it still dies, the
  docs state the measured reason.
- **Verify:** `python3 -m pytest tests/test_lenia.py -q`; owner reruns
  `tests/test_cuda_lenia.py` and `tests/test_wgpu_lenia.py` if the kernel
  changed.

### R-06 — State hash and `io.hash_every`

- **Why:** `docs/07-determinism.md` specifies a state hash over frame,
  integer particle state, field, and matrix, but nothing implements it,
  and the `io.hash_every` config key is accepted and ignored. Replay
  work (R-14) needs a hash to compare runs across machines.
- **Files:** new `ece/state_hash.py`, `ece/__main__.py`,
  `docs/07-determinism.md`, new `tests/test_state_hash.py`.
- **Do:** hash the byte layout given in `docs/07` with
  `hashlib.blake2b` (standard library; record the algorithm choice in
  `docs/07`, which currently says xxHash3 or BLAKE3). Start with the
  fixed-point Mohr state, which is already integer. When `io.hash_every
  = N > 0`, the CLI prints `frame=<f> hash=<hex>` every N frames.
- **Done when:** identical states hash identically, any one-unit change
  in any field changes the hash, the layout is little-endian and
  documented, and the CLI emits hashes on schedule.
- **Verify:** `python3 -m pytest tests/test_state_hash.py tests/test_fixedpoint_mohr.py -q`.

### R-07 — PPS rule (CPU reference)

- **Why:** `"pps"` is an accepted rule name in the loader and schema,
  but no runner exists; a PPS config loads and then the CLI refuses it.
- **Files:** new `ece/pps.py`, `ece/config.py`, `ece/__main__.py`,
  `schemas/sim.schema.json`, new `configs/pps.toml`, new
  `tests/test_pps.py`, `docs/02-rules-catalog.md`.
- **Do:** implement Schmickl's primordial particle system per
  `docs/02` (`Δφ = α + β·N·sign(R − L)`, constant speed, periodic 2D
  world), reusing `ece.bins` for neighbour counts. Add a `[pps]` config
  table (radius, alpha, beta, speed, particles) to the loader key sets
  and the schema together (R-02's drift test will enforce this).
- **Done when:** a binned-vs-dense agreement test and a determinism test
  pass, the CLI runs the config and writes NPZ, and `docs/02` points at
  the implementation.
- **Alternative:** if PPS is not wanted, remove `"pps"` from the rules
  set and schema instead. Ask the owner which.

### R-08 — Two-way hybrid coupling

- **Why:** the hybrid (`ece/hybrid.py`, `ece/cuda_hybrid.py`) rebuilds
  the field from particle deposits every frame, applies one MaCE step,
  and exports `sampled` values, but the field never acts on particle
  motion. It is a diagnostic, not a coupled system.
- **Decision needed:** the coupling law. Before writing code, ask the
  owner to choose. Candidates:
  1. Gradient drive: `a_i += g · Σ_c W[c_i, c] ∇ρ_c(x_i)`, with a new
     species-by-channel matrix `W` and gain `g`.
  2. Affinity drive: particles climb the MaCE affinity of their own
     channel, `a_i += g ∇A_{c_i}(x_i)`, reusing the Field Life matrix.
  Also decide whether the field persists across frames (deposit adds to
  an evolving `ρ`) or is rebuilt each frame as now.
- **Files:** `ece/hybrid.py`, `ece/pic.py` (gradient sampling),
  `ece/config.py` + schema (new keys), `configs/hybrid_pic.toml`,
  `tests/test_hybrid.py`; GPU versions in follow-up cards.
- **Do (CPU first):** implement the chosen law in the CPU hybrid only,
  with the gain defaulting to 0 so existing behavior and tests are
  unchanged. Add the finite-difference test pattern used in
  `tests/test_particle_lenia.py` for the sampled gradient.
- **Done when:** with gain 0 the hybrid is bit-identical to today; with
  gain > 0 a test shows particles respond to a fixed synthetic field in
  the expected direction; docs no longer say "one way".
- **Then:** open separate cards for CUDA and WebGPU parity.

### R-09 — Remove dead CUDA hash kernels

- **Why:** `clear_counts`, `count_bins`, `scan_bins` (single-threaded
  serial scan), and `scatter_bins` in `ece/cuda_mohr.py` are compiled
  but never launched; resident planning uses CuPy `argsort`/`bincount`.
- **Files:** `ece/cuda_mohr.py`, `ece/cuda_hybrid.py` (both unpack the
  kernel tuple).
- **Do:** return only `force_bins` and `integrate` and fix the unpacking.
- **Done when:** no unused kernels remain; CPU suite passes.
- **Verify (owner, cuda):** `python3 -m pytest tests/test_cuda_mohr.py tests/test_cuda_hybrid.py -q`.

### R-10 — CuPy 3D index-overflow guard

- **Why:** the raw kernels in `ece/cupy_field_life.py` index with 32-bit
  `int`. `channels × depth × height × width ≥ 2³¹` overflows; 512³ × 16
  channels is exactly 2³¹, and `docs/08` lists 512³ × 16 as a research
  config. Overflow reads and writes the wrong memory.
- **Files:** `ece/cupy_field_life.py`, `tests/test_cupy_field_life.py`.
- **Do:** either switch kernel indices to `long long` (pass sizes as
  `np.int64`) or raise a clear `ValueError` before launch when the total
  reaches 2³¹. Prefer the guard if the 64-bit version is slower on GB10.
- **Done when:** oversized inputs fail loudly (or work), and normal sizes
  are unchanged.
- **Verify:** owner runs `tests/test_cupy_field_life.py` and
  `python3 tools/benchmark_cupy_field_life.py --sizes 64 128 --frames 3`
  and confirms timings did not regress.

### R-11 — WebGPU parameter block and GPU hash

- **Why:** every WGSL module passes counts and sizes through an
  `array<f32>` params buffer, so integers above 2²⁴ (~16.7 M) lose
  precision. The binned WebGPU Mohr path also rebuilds hash tables on the
  CPU every frame and uploads them.
- **Files:** `ece/wgpu_mohr.py`, `ece/wgpu_field_life.py`,
  `ece/wgpu_lenia.py`, `ece/wgpu_particle_lenia.py`, `ece/wgpu_pic.py`,
  `ece/wgpu_preview.py`.
- **Do:** (a) replace the f32 params array with a uniform struct holding
  `u32` counts and `f32` physical values; (b) as a second change, build
  count/scan/scatter hash tables in WGSL following `docs/05` (frozen
  counts, exclusive scan, separate cursor). Keep (a) and (b) in separate
  PRs.
- **Done when:** all `tests/test_wgpu_*.py` pass on an adapter; for (b),
  the 200-frame WGSL-vs-CPU Mohr parity test still passes.

### R-12 — CUDA Mohr throughput (FP64 promotion)

- **Why:** the Numba kernels mix float32 arrays with Python float
  literals (`0.0`, `0.5`, `1.0`), which promotes the arithmetic to FP64.
  FP64 throughput on GB10-class hardware is low; this is the likeliest
  cause of the 1M-particle rate (~5–9 steps/s). It is a hypothesis, not
  a measurement.
- **Files:** `ece/cuda_mohr.py`, `ece/cuda_hybrid.py`,
  `ece/cuda_field_life.py`, `ece/cuda_lenia.py`,
  `ece/cuda_particle_lenia.py`, `ece/cuda_pic.py`.
- **Do:** first measure: record `tools/benchmark_cuda_mohr.py` numbers
  and inspect the kernels' typed IR (Numba `inspect_types`) for
  `float64`. Then make constants and accumulators `float32`. Do not
  enable `fastmath` in the same change; it alters results and needs its
  own parity check.
- **Done when:** before/after benchmark numbers are recorded in the M5
  list with the command and hardware, and all CUDA parity tests pass at
  their current tolerances.
- **Verify (owner, gb10):** `python3 -m pytest tests/test_cuda_*.py -q`;
  `python3 tools/benchmark_cuda_mohr.py --particles 4096 65536 1048576 --frames 10 --constant-density`.

### R-13 — Vectorize CPU fixed-point Mohr

- **Why:** `ece/fixedpoint_mohr.py` is a pure-Python O(N²) double loop,
  about 13 s per frame at 4,096 particles.
- **Files:** `ece/fixedpoint_mohr.py`, `tests/test_fixedpoint_mohr.py`.
- **Do:** vectorize with `numpy.int64`, keeping every rounding step
  identical (`_round_div` is round-half-away-from-zero). Check headroom:
  `gain(Q16) × force(Q15) × unit(Q15)` is 2⁴⁶ per pair, so a sum over
  more than about 2¹⁷ neighbours overflows int64; bin the pairs or keep
  Python ints for the accumulation.
- **Done when:** the new implementation is bit-identical to the current
  one on the existing tests plus a 512-particle, 50-frame comparison,
  and is at least 10× faster.

### R-14 — CUDA fixed-point replay port

- **Why:** cross-machine replay (`docs/07`) needs integer kernels on
  CUDA with versioned, host-generated lookup tables instead of
  transcendental functions.
- **Depends on:** R-06 (state hash to compare runs).
- **Files:** new `ece/cuda_fixedpoint_mohr.py`, tests, `docs/07`.
- **Do:** port `ece/fixedpoint_mohr.py` integer-for-integer, including
  neighbour traversal order. Only friction uses `exp`, and it is
  computed once on the host; keep it that way.
- **Done when:** the CUDA port produces state hashes identical to the
  CPU prototype for at least 100 frames on the GB10, and on a second
  GPU model if one is available.

### R-15 — Long 512³ run and memory accounting

- **Why:** the 512³ evidence is a ten-frame run (~1.15 s/frame, ~18.8 GB
  CuPy pool). There is no long-run stability or system-memory number.
- **Depends on:** R-10.
- **Do (gb10):** run 512³ × 4 channels for at least 1,000 frames;
  record per-channel mass drift, peak CuPy pool, and peak system memory
  (unified memory). Add a `--report-memory` option to
  `tools/benchmark_cupy_field_life.py` if needed.
- **Done when:** numbers are recorded below with the exact command; the
  same for the sustained 256³ default.

### R-16 — WebGPU tests in CI on a software adapter

- **Why:** CI installs no GPU extras, so all `tests/test_wgpu_*.py`
  skip. A software Vulkan driver (Mesa lavapipe) may let `wgpu` find an
  adapter on a GitHub runner, which would put the WGSL-vs-CPU parity
  tests under CI. Unverified; this card is to find out.
- **Files:** `.github/workflows/ci.yml` (a separate, optional job).
- **Do:** in a new job, install `mesa-vulkan-drivers` and
  `pip install wgpu` (not PySide6), then run
  `python -m pytest tests/test_wgpu_*.py -q -rs`. Exclude
  `tests/test_wgpu_preview.py`, which needs a display.
- **Done when:** the job reports the WebGPU compute tests as run (not
  skipped) and passing; or, if no adapter appears, the card records that
  finding and is closed. Mark the job `continue-on-error` until it has
  been green for a few PRs.

## Completed milestones

### M0 — corrected kernels

- [x] Corrected Mohr tent (no extra `r_max`)
- [x] Exponential friction
- [x] Two-pass MaCE + torus wrap
- [x] Snapshot hash tables
- [x] Invariant tests, document tree, configs, schema, WGSL fragments
- [x] CPU CI on GitHub Actions: Python 3.11 and 3.12, full suite with GPU tests skipping, module compile check, every shipped config loads (R-01)

### M1 — 2D Mohr preview

- [x] Headless CPU frame runner calling `ece.step_mohr` (vectorized binned forces)
- [x] Six-species binned-vs-dense comparison over 200 frames
- [x] Interactive Tk preview with live K×K matrix editor
- [x] CPU, CUDA, and WGSL force backends selectable in the preview
- [x] 200-frame WGSL-vs-CPU per-step oracle comparison on 256 particles
- [x] Native Qt/WebGPU presentation surface with direct screen rendering
- [x] Device-resident native Mohr compute/integrate/render loop (dense O(N²))
- [x] Hash grid validated against the neighbourhood walk on every backend

Exit (met): a 6-species soup that matches the Python oracle on N ≤ 256
for a few hundred steps within f32 tolerance.

### M2 — Field Life 2D

- [x] Config-driven CPU runner using convolution + MaCE passes
- [x] Per-channel mass-invariance test and NPZ output
- [x] CUDA ring convolution, MaCE passes, and per-channel readback test
- [x] WGSL MaCE passes, CPU parity, and headless CLI output
- [x] Native WebGPU field presentation for Field Life
- [x] CuPy/cuFFT 3D Field Life path with CPU parity at 16³ and 256³ smoke run

Exit (met): per-channel mass holds; creatures do not evaporate.

### M3 — Lenia growth on the same field buffers

- [x] CPU growth-bell update and normalized radial ring kernel
- [x] Deterministic annular seed and `lenia_orbium.toml` runner; current seed goes extinct by frame 100
- [x] CUDA ring convolution/growth and CPU parity test
- [x] WGSL ring convolution/growth and CPU parity test
- [x] Native WebGPU field presentation for Lenia

Exit (**not met**, see R-05): Orbium persists or dies for documented
reasons, not because of a broken clip.

### M4 — PIC coupling

- [x] CPU periodic bilinear particle deposit and field sample references
- [x] Per-species mass and deposit/sample adjoint tests
- [x] Deterministic CUDA segmented deposit and bilinear sample passes
- [x] Deterministic WGSL segmented deposit and bilinear sample passes
- [x] CPU/CUDA/WebGPU hybrid scheduler: Mohr → deposit → Field Life → sample (one-way; see R-08)
- [x] CUDA hybrid keeps particle/velocity/field state and stable hash/PIC planning on-device
- [x] Same-device repeated CUDA hybrid runs produce identical state
- [x] 1024-particle/128² CUDA hybrid baseline: ~746 steps/s over 100 frames (GB10)
- [x] Single-species CPU Particle Lenia energy-gradient rule and periodic config
- [x] CUDA and WGSL Particle Lenia kernels with CPU oracle tests
- [x] Native Particle Lenia visualization
- [x] Particle Lenia kept as a standalone direct-KDE rule; it does not use PIC deposit/sample

### M5 — CUDA path on Spark (partial; open items are R-10 to R-15)

- [x] Optional CUDA Mohr force kernel (CPU-built hash tables) with oracle comparison
- [x] Device-side hash planning (CuPy stable sort + bincount + scan) and persistent device-resident Mohr integration
- [x] Reproducible stable-sort CUDA benchmark and 4096-particle baseline (GB10: ~600 steps/s, 100 frames)
- [x] Same-device deterministic standalone and hybrid CUDA repeatability tests
- [x] Stable-sort CUDA Mohr scaling probes: 16K at 135 steps/s; 65K at 10.6 steps/s
- [x] Constant-density 1M-particle CUDA run: 10 frames, ~9.0 steps/s, finite torus state; ~5.4 steps/s over 100 frames
- [x] cuFFT-backed 3D field update prototype
- [x] CPU parity and per-channel mass at 16³; 256³ and 512³ feasibility probes
- [x] Fused four-channel cuFFT MaCE benchmark at 64³/128³/256³ (2.6 ms/24 ms/0.21 s per frame; ~1.2e-8 mass drift)
- [x] Fused 512³ four-channel, ten-frame run (1.15 s/frame; 1.05e-8 mass drift; ~18.8 GB CuPy pool)
- [x] CPU integer Mohr replay prototype in Q16.16/Q8.24/Q1.15 with CLI export and bitwise tests

## Explicitly not scheduled

Do not start these, even as "small" extensions; they need an owner
decision to enter the roadmap.

- Co-local 7B VLM steering (design only: `docs/12-ai-evaluator-deferred.md`)
- Full egui IDE / DAG node editor (rule plugin surface in `docs/06` is
  logical only)
- crates.io release matrix
- 64-species food webs
- 1024³ volumes
- Bit-identical replay in WGSL floats (`CORRECTIONS.md` §5)
