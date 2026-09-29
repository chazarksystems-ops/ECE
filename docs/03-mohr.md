# Mohr Particle Life — corrected force

Particles: position `x`, velocity `v`, species `c ∈ {0..K-1}`.
One global `r_max`, one global wall `beta ∈ (0,1)`, one matrix
`A[K,K] ∈ [-1,1]`, optional scalar `gain` (default 1).

## Tent

Normalize `r = ||x_j - x_i||_wrap / r_max`.

```
F(r, a) =
    r/beta - 1                              if r < beta
    a * (1 - |2r - 1 - beta| / (1-beta))    if beta ≤ r ≤ 1
    0                                       if r > 1
```

- `r < beta`: everyone repels everyone. `F` does not depend on `a`.
- Mid-range peak is at `r = (1+beta)/2` and equals `a`.
- Horizon is `r_max`. Wrapped shortest vector on the torus.

## Acceleration (the correction)

```
a_i = gain * Σ_{j≠i} F(r_ij, A[c_i,c_j]) * hat(x_j - x_i)
```

Do **not** multiply by `r_max`. That factor made peak force depend
on world units and broke “same matrix, different r_max” comparisons.

`ece.mohr.mohr_accelerations` is the O(N²) oracle used by tests.
Production uses the spatial hash in `docs/05-bins-integrate.md`.

## Integration

```
v ← exp(-λ dt) v + a dt
x ← wrap(x + v dt)
```

`λ = 0` is undamped. To recover Mohr’s `gamma**(60 dt)` feel:

```
λ = -ln(gamma) * 60
```

See `ece.integrate`.

## Asymmetry

`A` is not required to be symmetric. `A[i,j] ≠ A[j,i]` is the
mechanism for orbits and chase rings. Total momentum is not a
conserved quantity and must not be treated as a test invariant.

## Clusters extension (later)

Replace the single tent with per-pair `(r1, r2, f1, f2)` and optional
view-radius / delay. Keep `f1` non-attractive so particles cannot
collapse through the wall.

## Tests that lock this

- wall independent of matrix entry and strictly negative
- tent peak equals `a` at midpoint
- F = 0 beyond horizon
- scale world and `r_max` together → identical accelerations
- asymmetric chase pair produces net momentum
