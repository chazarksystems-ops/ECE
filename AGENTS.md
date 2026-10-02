# Agent guide

Instructions for coding agents and new contributors working in this repo.
Open work lives in `docs/09-roadmap.md` as task cards (R-01, R-02, …).
Pick one card, read it fully, and follow this guide while doing it.

## What this repo is

Reference kernels for the Entelechy Continuum Engine, an artificial-life
runtime: Mohr Particle Life, Field Life (MaCE mass-conserving transport),
Lenia, Particle Lenia, and a particle-in-cell (PIC) hybrid. The Python
CPU code in `ece/` is the **source of truth**; CUDA (Numba, CuPy) and
WebGPU backends must match it. `docs/00-overview.md` is the design entry
point; `docs/13-file-index.md` maps every file.

## Setup and commands

```bash
python3 -m pip install numpy pytest          # enough for every CPU test
python3 -m pytest tests -q                   # full suite; GPU tests skip without a device
python3 -m pytest tests/test_mace.py -q      # one area
python3 -m ece configs/particle_life_6.toml --frames 5 --particles 256   # CLI smoke run
python3 -m py_compile ece/*.py               # syntax check for GPU modules you cannot run
python3 tools/build_handbook.py              # rebuild docs/handbook.pdf (needs reportlab)
```

Optional extras (`pyproject.toml`): `.[cuda]` (Numba), `.[cuda-fft]`
(CuPy), `.[webgpu]` (wgpu + PySide6). Never make the CPU suite depend on
them; import them lazily inside functions, as the existing backends do.

A cloud container usually has no GPU. The CPU suite then reports about
52 passed and 40 skipped. Skipped GPU tests prove nothing about GPU
code: see "GPU work without a GPU" below.

## Invariants (do not break; tests lock most of them)

1. **Mohr force:** `accel += gain * F(r/r_max, a) * unit(Δx)`. Never
   multiply by `r_max`. The wall (`r < beta`) is species-independent and
   repulsive. See `CORRECTIONS.md` §1, `docs/03-mohr.md`.
2. **Friction:** `v *= exp(-lambda * dt)`, never `gamma ** (60 * dt)`.
3. **MaCE:** two passes (denominator, then gather), torus wrap, no
   border clipping, local-maximum rescaling in both passes. Per-channel
   mass is conserved to float rounding. Momentum is **not** conserved
   and must not be tested as if it were.
4. **Spatial hash:** frozen `counts` snapshot, exclusive scan, separate
   scatter cursor. Every binned path must call
   `ece.bins.validate_hash_grid`; a grid with fewer bins per axis than
   the neighbourhood walk double-counts particles.
5. **Torus:** positions stay in `[0, world)`; distances use the
   minimum image (`ece.mohr.wrapped_delta`).
6. **Determinism:** same config + seed gives the same CPU result. Seeds
   come from `simulation.seed` plus a fixed per-rule offset (see the
   `seed_state` functions); do not change offsets, it changes every
   saved run.
7. **Config strictness:** unknown keys are errors at every level. A new
   key must be added to `_ALLOWED_KEYS` in `ece/config.py`,
   `schemas/sim.schema.json`, and `docs/11-config-schema.md` in the same
   change.
8. **Scope:** do not build anything under "Explicitly not scheduled" in
   the roadmap or `CORRECTIONS.md` §5.

## How to add things

- **New rule:** CPU reference module in `ece/` with `seed_state`,
  `step`, `run`; config keys (invariant 7); a branch in
  `ece/__main__.py`; a config in `configs/`; tests for its conservation
  or symmetry properties, determinism, and CLI NPZ output; a section in
  `docs/02-rules-catalog.md`. GPU versions come later, as their own cards.
- **New backend for an existing rule:** a module named `cuda_<rule>.py`
  / `cupy_<rule>.py` / `wgpu_<rule>.py`; an `*_available()` check; a test
  file with `pytestmark = pytest.mark.skipif(not <backend>_available(), …)`
  that compares against the CPU reference on a small problem (copy an
  existing `tests/test_cuda_*.py` or `tests/test_wgpu_*.py`); CLI flag
  wiring with a clean `parser.error` when the device is absent.
- **Tolerances:** float64 CPU-vs-CPU comparisons use about `1e-9`;
  float32 GPU-vs-CPU parity tests use `rtol` between `1e-6` and `5e-4`
with small `atol` (see the existing `tests/test_cuda_*.py`). Do not
  loosen an existing tolerance to make a test pass; find the cause.

## GPU work without a GPU

If a card's hardware tag is `cuda`, `wgpu`, or `gb10` and you have no
such device:

1. Make the change and keep it minimal.
2. Run the CPU suite and `python3 -m py_compile ece/*.py`.
3. In the PR description, list the exact GPU commands the owner must run
   (test files and benchmark commands from the card), and say plainly
   that the GPU paths were not executed.
4. Do not tick the roadmap box or record performance numbers you did not
   measure.

## Definition of done (every card)

- The card's "Done when" criteria are met and its "Verify" commands pass
  (or, for GPU cards without hardware, are listed for the owner).
- New behaviour has tests; fixed bugs have a regression test that fails
  without the fix.
- `python3 -m pytest tests -q` passes with no new failures or errors,
  and the CI check on the PR is green.
- Docs updated in the same change, as applicable:
  - `docs/09-roadmap.md`: delete the card, add a `- [x]` line to the
    milestone with measured numbers and hardware;
  - `docs/13-file-index.md` for new files;
  - `docs/15-testing.md` for new test files and the test count;
  - `docs/11-config-schema.md` and the schema for config keys;
  - `README.md` if a user-facing command or status changes;
  - `CHANGELOG.md` under "Unreleased".
- If any doc under `docs/` changed, rebuild `docs/handbook.pdf`.

## Things that have gone wrong before

- Docs claiming features or numbers the code did not have. State only
  what you ran; label every benchmark with frame count and hardware.
- Test files importing GUI modules (`tkinter`, Qt) at the top level,
  which breaks collection on headless machines. Keep GUI imports inside
  the GUI modules.
- Python float literals in Numba kernels promoting float32 math to
  float64 (see R-12).
- Raw CUDA kernels indexing with 32-bit `int` on very large volumes
  (see R-10).

## Git and PRs

- One card per branch and PR; reference the card ID in the PR title,
  e.g. `R-06: state hash and io.hash_every`.
- Commit messages say what changed and why. Do not mention which model
  or agent wrote the change in code comments or docs.
