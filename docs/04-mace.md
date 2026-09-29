# MaCE / Field Life — two-pass torus transport

Field Life keeps Mohr’s matrix, drops the dots, and moves mass on a
grid so numerical friction cannot evaporate creatures.

## Sense (affinity)

Each color `c` is a field `ρ_c ≥ 0`.

```
A_c = strength * Σ_d M[c,d] (K * ρ_d) - λ_crowd ρ_c
```

Default `K` is a signed ring (negative near 0, positive on a shell)
so hollow shells beat solid blobs. Convolution is periodic.

## Move (MaCE)

**Pass A — denominators**

```
Z[x] = Σ_{y ∈ N(x)} exp(β A[y])
```

**Pass B — gather**

```
ρ'[x] = Σ_{y ∈ N(x)} ρ[y] * exp(β A[x]) / Z[y]
```

`N(x)` is the 9-neighborhood (2D) or 27-neighborhood (3D),
**including self**, wrapped on the torus.

Implementations evaluate each origin's neighborhood weights after subtracting
that neighborhood's maximum affinity. This log-sum-exp rescaling cancels
between numerator and denominator, preserving the transition exactly while
preventing overflow and local denominator underflow.

## What was wrong in the catalog shader

1. Recomputing neighbor softmax denominators inside the gather
   without a stored `Z`.
2. `continue` on out-of-bounds instead of wrap — mass leaks at the
   box face.
3. No unit test that `Σ ρ'_c = Σ ρ_c`.

## Invariants

- Per-channel mass is bit-stable within float rounding.
- Momentum is not conserved.
- A uniform field is a fixed point when the kernel mean is 0
  and crowding is applied evenly.
- `Z[x] > 0` always (`exp` is positive; neighborhood is non-empty).

## Implementation

Python: `ece.mace.mace_step`.
WGSL: `shaders/mace_2pass.wgsl` (`mace_denom`, `mace_gather`).

Multi-channel: run A and B per channel. `Z` is computed from that
channel’s affinity, not shared across colors, unless you deliberately
couple transport.

## 3D

Same formulas, 27 neighbors, and periodic rolls on three spatial axes. A CPU
reference and CuPy/cuFFT path are available; see `tools/benchmark_cupy_field_life.py`
for current target-specific volume timings.
