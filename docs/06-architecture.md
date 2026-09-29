# Architecture

## Planes

```
                Entelechy graph scheduler (later)
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    Eulerian plane   Coupling plane   Lagrangian plane
    multi-channel    deposit/sample   SoA particles
    fields 2D/3D     scatter/gather   spatial hash
          │               │               │
          └───────────────┼───────────────┘
                          ▼
                   Rule plugins
```

## Shared state (logical)

```
time, dt
euler:   [C, X, Y, (Z)] f32 ping-pong   or absent
lagrange: pos, vel, type, mass, heading  or absent
matrix:  K×K f32, optional per-pair radii
kernels: radial samples + optional FFT plan
hash:    counts, offsets, ranges, sorted_indices
```

## Frame DAG

```
Clear → Count → Scan → Scatter → PackRanges
  → Forces → Integrate
  → Deposit → Convolve → Growth
  → TransportDenom → TransportGather
  → Sample → Render
```

Compile only the passes the active rules declare.

| Rule | Uses |
|---|---|
| Mohr | hash + Forces + Integrate |
| Field Life | Convolve + MaCE pair |
| Lenia | Convolve + Growth |
| Particle Lenia | Deposit + Convolve + Sample + Integrate |
| Hybrid | union |

## Plugin surface (logical, not yet code)

```
Rule.passes() -> [Pass]
EulerianRule.convolve / update
LagrangianRule.forces / integrate
HybridRule.deposit / sample
```

## Backends

| Backend | Role |
|---|---|
| Python `ece/` | oracle, tests, small N |
| wgpu / WGSL | interactive preview |
| CUDA on DGX Spark | capacity + future replay |

Do not share one float shader between preview and bit-identical
logs. Preview may use f32 `exp`. Replay, if required, uses LUTs
and integer state on CUDA.

## Bind-group partition (wgpu)

WebGPU cap is 8 storage buffers per stage. Group by pass family:

- Group 0 uniforms (`SimParams`)
- Group 1 particles
- Group 2 fields
- Group 3 matrix / kernel
- Group 4 hash tables

Rebind between passes. CUDA ignores this limit.

## What is implemented now

CPU Mohr, Field Life, Lenia, and Particle Lenia runners; Tk and native
WebGPU previews; PIC deposit/sample reference operations; and an optional
CUDA Mohr force kernel with GPU-built hash tables, CUDA Field Life passes, and deterministic
CUDA PIC deposit/sample primitives. WGSL compute paths cover Mohr, Field
Life, Lenia, Particle Lenia, and PIC exchange. CPU/CUDA/WebGPU hybrid schedulers expose
sampled fields to rule plugins. The native Qt/WebGPU presentation host
supports Mohr particles, Particle Lenia, Field Life, and Lenia fields. CUDA
Mohr preview and hybrid hash/PIC planning remain device-resident. See
`docs/09-roadmap.md`.
