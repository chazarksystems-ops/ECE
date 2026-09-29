# Rules catalog

State, interaction, and conservation. Implementation parameters live
in `configs/` and `schemas/sim.schema.json`.

## Family table

| Family | State | Interaction | Conserves mass? |
|---|---|---|---|
| Life-like CA | discrete cells | neighbor count | no |
| SmoothLife / Lenia | continuous field | convolution + growth | no (unless Flow/MaCE) |
| Clusters / Particle Life | colored points | pairwise force matrix | no |
| Field Life | colored density fields | kernel + matrix + transport | yes, per color |
| Particle Lenia | points | energy gradient | no, unless added |
| PPS / Boids | heading agents | local geometry | no |

## Side-by-side

| System | Substrate | State | Topology | Primary parameters | Conserves | Cost per step |
|---|---|---|---|---|---|---|
| GoL / Life-like | discrete grid | {0,1} | Moore 8 | B / S counts | no | O(N) |
| SmoothLife | field | [0,1] | disk + annulus | radii, birth/survive intervals, α | no | O(N log N) FFT |
| Lenia | field | [0,1] | radial kernel | shells β, (μ,σ), dt | no | O(N log N) |
| Flow-Lenia / MaCE-Lenia | field | [0,1] | kernel + flux | Lenia + transport β | **yes** | O(N log N)+O(N) |
| Clusters | particles | x,v,c | pairwise, directed | per-pair (r1,r2,f1,f2), γ, delay | no | O(M k) bins |
| Particle Life (Mohr) | particles | x,v,c | pairwise ≤ r_max | matrix A, β, λ | no | O(M k) hash |
| Field Life | density grid | ρ_c ≥ 0 | conv + matrix + MaCE | K, M, λ_crowd, β_tr | **yes** | O(K N log N) |
| Particle Lenia | particles | x | radial shell KDE + energy gradient | μ_K, σ_K, μ_G, σ_G, c_rep | no | O(M²) reference |
| PPS | agents | x, φ | left/right count | r, α, β, v | no | O(M k) |
| Boids / Swarm Chemistry | agents | x, v | neighborhood means | w_sep, w_align, w_coh | no | O(M k) |

## Update rules (compressed)

### Conway / Life-like

Birth on count set B, survive on S. Classic B3/S23.

### SmoothLife

Inner disk fill `m`, outer annulus fill `n`, logistic mix of birth
interval `[b1,b2]` and survival `[d1,d2]`. Famous glider:
birth [0.278, 0.365], survive [0.267, 0.445], α_n ≈ 0.028,
α_m ≈ 0.147.

### Lenia

`U = K * A`, `G(u) = 2 exp(-(u-μ)² / 2σ²) - 1`,
`A ← clip(A + dt G(U), 0, 1)`. Orbium: ring near r=0.5,
μ≈0.15, σ≈0.015.

### Mohr Particle Life (corrected)

See `docs/03-mohr.md`. Tent + matrix. No extra `r_max` on the unit
vector.

### Field Life / MaCE (corrected)

See `docs/04-mace.md`. Two-pass softmax transport on a torus.

### Particle Lenia

The CPU reference follows the author energy-gradient rule. For particle
positions `p_i`, define the normalized radial-shell density
`U(x) = Σ_i K(||x-p_i||)`, where
`K(r) = w_K exp(-((r-μ_K)/σ_K)^2)`. Growth is
`G(u) = exp(-((u-μ_G)/σ_G)^2)`. Short-range repulsion has potential
`R(x) = c_rep/2 Σ_{j:p_j != x} max(1-||x-p_j||,0)^2`; local energy is
`E(x) = R(x) - G(U(x))`, and overdamped motion is
`dp_i/dt = -∇E(p_i)` with other particles held fixed.

ECE adapts the continuous reference to its periodic 2D world using
minimum-image distances and wrapped positions. The separate
`particle_lenia.toml` config uses a 16×16 world so the shell radius is not
collapsed by the unit-torus scale. This rule has one species and no Mohr
matrix, inertial velocity, or PIC field state.

The equations follow the authors' [Particle Lenia description](https://google-research.github.io/self-organising-systems/particle-lenia/)
and [reference notebook](https://github.com/google-research/self-organising-systems/blob/master/notebooks/particle_lenia.ipynb);
periodic boundaries and ECE config integration are local adaptations.

### PPS

`Δφ = α + β N sign(R-L)`, constant speed. Default life-like:
r=5, α=180°, β=17°, v=0.67.

### Boids

Separation, alignment, cohesion. Swarm Chemistry assigns different
weight triples per species — same “matrix of types” idea, geometric
steering instead of a radial tent.

## DAG passes per rule

| Rule | Passes |
|---|---|
| Mohr / Clusters | Forces, Integrate |
| Field Life | Convolve, TransportDenom, TransportGather |
| Lenia / SmoothLife | Convolve, Growth |
| Flow-Lenia / MaCE-Lenia | Convolve, Growth, Transport* |
| Particle Lenia | Shell KDE, EnergyGradient, Integrate |
| PPS / Boids | Steer, Integrate |
| Hybrid PIC | Deposit + any of the above + Sample |
