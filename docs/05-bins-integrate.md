# Spatial hash and integration

Neighbor search must be O(M k), not O(M²). Uniform bins are enough
for a torus with a single `r_max`.

## Cell size

`cell_size >= r_max` lets a 3×3 (2D) or 3×3×3 (3D) walk cover the
horizon. To refine bins, use `cell >= r_max / 2` and walk 5×5.
Document the choice; do not mix them at runtime.

The grid needs at least `neighborhood` bins per axis. With fewer, the
torus wrap maps two walk offsets onto the same bin and its particles
are counted twice. `ece.bins.validate_hash_grid` enforces this for the
loader and every backend.

## Four arrays, not two

| Buffer | Role |
|---|---|
| `counts` | frozen particle-per-bin snapshot |
| `offsets` | exclusive prefix sum of `counts` |
| `ranges` | `(start, end) = (offsets[b], offsets[b]+counts[b])` |
| `cursor` | scatter write head, discarded after the pass |

`sorted_indices[start:end]` are the particles in that bin.

Never exclusive-scan a counter you later `atomicAdd` as a cursor.
The catalog mixed those roles.

## Algorithm

1. Clear `counts`.
2. Count: `counts[bin(x_i)] += 1`.
3. Snapshot `counts`. Exclusive scan → `offsets`.
4. Pack `ranges`.
5. Scatter into `sorted_indices` using a copy of `offsets` as cursor.
6. Force pass walks neighbor bins via `ranges`.
7. Integrate.

`ece.bins.snapshot_and_scatter` is the CPU oracle.

## Particle-in-cell exchange

`ece.pic.deposit_particles` bilinearly scatters each particle's mass into
its species channel on a periodic grid. `ece.pic.sample_fields` gathers all
channels back at particle positions with the same weights. The pair preserves
per-species deposited mass and satisfies the deposit/sample adjoint identity.
These are CPU reference operations. `ece.hybrid.run` schedules Mohr,
deposit, Field Life, and sample passes. Coupling is currently one way:
the field is rebuilt from the deposit each frame, and the sampled values
are exported but do not yet act on particle forces. CUDA hybrid keeps particle, velocity,
field, and scratch buffers on-device and uses stable CuPy sorts for hash and
deposit planning. WebGPU currently transfers pass state per frame. Both PIC
GPU paths reduce target-cell segments without floating-point atomics.

## Integration

Symplectic Euler + wrap:

```
decay = exp(-λ dt)
v ← decay v + a dt
x ← x + v dt
x ← x - floor(x / world) * world
```

World is a unit torus unless a config says otherwise. Positions
must stay in `[0, world)`.

## Determinism note

Float GPU atomics and vendor `exp` make wgpu preview non-replayable.
If replay logs matter, freeze:

- particle order after scatter (already sorted by bin then arrival)
- neighborhood walk order (fixed nested dx,dy)
- a fixed-point / LUT path on CUDA later

Hash state = frame counter + positions + types + matrix.
Do not hash `accel`, `Z`, or `ranges`.
