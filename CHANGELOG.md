# Changelog

## Unreleased

- Spatial hash: reject grids with fewer bins per axis than the neighborhood
  walk (they double-counted wrapped bins); validated by the loader and by
  the CPU, CUDA, and WebGPU binned force paths.
- Config: unknown keys inside any table are now errors, matching the schema.
- CPU binned Mohr force is vectorized (~19× faster at 4,096 particles).
- `tests/test_step_mohr.py` no longer requires `tkinter`; the matrix parser
  moved to `ece.matrix`.
- Roadmap rewritten as task cards (R-01 to R-15) with files, acceptance
  criteria, verification commands, and hardware needs; added `AGENTS.md`
  (conventions, invariants, definition of done) and `CLAUDE.md`.
- CPU CI on GitHub Actions (R-01): Python 3.11 and 3.12, full test suite
  without GPU extras, module compile check, and config loading.
- Docs reconciled with the code: status, hybrid one-way coupling, config
  example, hash rules, file index, shader notes, benchmark figures.

## 0.1.0 — 2026-09-28

- Implemented review corrections as executable kernels.
- Mohr tent: removed extra `r_max` scale; added `gain`.
- Friction: `exp(-λ dt)` with Mohr-gamma conversion helper.
- MaCE: two-pass denom + gather, torus wrap, per-channel mass tests.
- Spatial hash: frozen snapshot, exclusive scan, separate cursor.
- Document set `docs/00`–`13`, JSON schema, five TOML configs.
- CPU Mohr stepper with binned vs dense agreement test.
- WGSL fragments matching the corrected math.
- Printable handbook builder.
