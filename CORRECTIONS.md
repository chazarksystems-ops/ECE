# Corrections applied to the catalog / ECE spec

These replace the corresponding formulas in `particlelifesim.txt`.
The Python package `ece/` is the executable source of truth.

## 1. Mohr force — drop the extra `r_max`

**Was (catalog acceleration):**

```text
accel += F * r_max * unit(other - me)
```

**Correct:**

```text
r = dist / r_max
F = tent(r, a, beta)          # dimensionless
accel += gain * F * unit(other - me)
```

`gain` defaults to `1`. Peak mid-range force is then exactly the matrix
entry `a`, independent of world scale. Scaling the world and `r_max`
together must not change accelerations (see `test_force_independent_of_rmax_scale`).

Contact wall (`r < beta`) stays species-independent and strictly
repulsive. Clusters-style per-pair `(r1,r2,f1,f2)` may be added later;
`f1` must not be attractive.

## 2. Friction — explicit time constant

**Was:** `v *= gamma ** (60 * dt)`  (assumes 60 Hz)

**Correct:** `v *= exp(-lambda * dt)`

Conversion if you want Mohr's feel: `lambda = -ln(gamma) * 60`.

## 3. MaCE — two passes, torus wrap, no border clip

**Pass A** — denominators

```
Z[x] = sum_{y in N(x)} exp(beta * A[y])
```

**Pass B** — gather

```
rho'[x] = sum_{y in N(x)} rho[y] * exp(beta * A[x]) / Z[y]
```

`N(x)` is the 9-cell (2D) or 27-cell (3D) neighborhood **including
self**, wrapped on the torus. Do not `continue` on out-of-bounds.
Clipping leaks mass at the box face.

Invariant: `sum(rho'_c) == sum(rho_c)` per channel, up to float
rounding. Momentum is not conserved.

## 4. Spatial hash — freeze the snapshot

Keep four distinct arrays. Never scan a counter you later reuse as a
write cursor.

| Buffer | Role |
|---|---|
| `counts` | frozen particle-per-bin snapshot |
| `offsets` | exclusive prefix sum of `counts` |
| `ranges` | `(offsets[b], offsets[b] + counts[b])` |
| `cursor` | scatter write head, discarded after the pass |

`cell_size >= r_max` if you walk a 3×3 neighborhood.

## 5. Scope cut (do not implement from the catalog yet)

M1 is 2D Mohr + matrix + these tests. Not in M1:

- 3D 512³ volumes
- 64-species Field Life
- CAS atomic f32 deposit
- co-local VLM
- full egui IDE / release matrix

wgpu is the interactive preview. Bit-identical replay, if required,
lives on a fixed-point CUDA path later — not in WGSL floats.

## 6. DGX Spark budget for the first real run

Default: `256²` or `256³ × 8` channels, 1–4 M particles, cuFFT when
kernel radius > ~6 in 3D. 128 GB is capacity. GB10-class bandwidth
is not an H100. Locality and mixed precision first.
