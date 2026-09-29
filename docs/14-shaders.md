# Shader fragments

These are math-correct snippets, not a full wgpu project.

## `shaders/mohr_force.wgsl`

- `mohr_force(r, a, beta)` — dimensionless tent
- `wrapped_delta_2d` — minimum image
- Force loop body multiplies by `params.gain` only
- Integrate uses `exp(-lambda * dt)`

## `shaders/mace_2pass.wgsl`

- `mace_denom` writes `Z`
- `mace_gather` reads `Z` and wraps with `(x % n + n) % n`
- One channel shown; dispatch once per channel or flatten `C*X*Y`

## Bindings the host must supply

Mohr force pass: `SimParams`, positions, sorted indices, ranges,
matrix, accel out.

MaCE: `SimParams` with `field_w/h` and `transport_beta`, `affinity`,
`rho_in`, `denom`, `rho_out`.

## Not provided

Prefix-sum shaders, FFT, deposit CAS, render. Those belong to M1–M4
hosts, not this reference.
