# Config schema

Canonical JSON Schema: `schemas/sim.schema.json`.
Human configs: `configs/*.toml`.

## Required block

```toml
[simulation]
dt = 0.01
world = [1.0, 1.0]          # 2D torus
seed = 1

[species]
count = 6
matrix = "random"           # or "identity", "chase", "path"
range = [-1.0, 1.0]

[mohr]
particles = 4096
r_max = 0.08
beta = 0.3
gain = 1.0
lambda = 6.321             # -ln(0.9)*60 ≈ Mohr gamma 0.9 @ 60 Hz
```

## Optional blocks

```toml
[field]
dims = [256, 256]
channels = 6
strength = 1.0
crowding_lambda = 0.1
transport_beta = 4.0
kernel = "ring"

[lenia]
mu = 0.15
sigma = 0.015
dt = 0.1

[hash]
cell = 0.08                 # >= r_max
grid = [13, 13]

[io]
headless = true
frames = 1000
hash_every = 0
```

Unknown keys are errors. Do not silently ignore a misspelled
`r_max`. The schema uses `"additionalProperties": false` on each
object.

## Matrix modes

| Mode | Fill |
|---|---|
| `random` | U(range), seed from `simulation.seed` |
| `identity` | +1 diagonal, small negative off-diagonal |
| `chase` | cyclic +attract / -flee |
| `path` | load `matrix_path` as row-major f32 text |

## Validation rules

- `beta ∈ (0,1)`
- `r_max > 0`
- `lambda ≥ 0`
- `dt > 0`
- `species.count == matrix side` if a file is loaded
- `hash.cell >= r_max` unless `hash.neighborhood` is set to 5
