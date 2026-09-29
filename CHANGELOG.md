# Changelog

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
