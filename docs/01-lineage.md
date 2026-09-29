# Lineage

Tom **Mohr** of particle-life.com, not “Tom Moore.” Clusters (Jeffrey
Ventrella) came first. Mohr simplified the per-pair piecewise fields
into one global tent plus one matrix. Field Life moves that matrix
onto density fields and adds MaCE so mass does not evaporate. Lenia
is a different line: Conway’s Life made continuous.

```
GoL (B3/S23)
  ├─ Life-like CA (B/S notation)
  ├─ Larger-than-Life / Bugs
  ├─ SmoothLife (disk + annulus + birth/survive intervals)
  │    └─ SmootherLife (Gaussian kernels)
  └─ Lenia (radial kernel + growth bell)
       ├─ multi-channel / multi-kernel
       ├─ Glaberish (genesis + persistence)
       ├─ Flow-Lenia (mass-conserving flow)
       ├─ MaCE-Lenia
       └─ Particle Lenia (energy on dots)

Clusters (Ventrella): per-pair (r1,r2,f1,f2) + asymmetric horizons
  ├─ Particle Life (Mohr): one tent + one matrix
  │    ├─ hunar4321 / najarro.science / par-particle-life
  │    └─ Field Life = matrix + kernel + MaCE
  └─ Clusters X / Atomic Clusters (ciphrd)

PPS (Schmickl): Δφ = α + β N sign(R − L)
Boids (Reynolds): separate / align / cohere
NCA: learned local net
Reaction-diffusion: Gray–Scott, FitzHugh–Nagumo
```

## What each ancestor contributes

| Ancestor | Keep |
|---|---|
| Clusters | Asymmetric matrix, optional per-pair radii later |
| Mohr | Global tent, one `r_max`, one `beta`, spatial bins |
| Field Life | Convolution + matrix + conservative transport |
| Lenia | Kernel + unimodal growth, optional multi-channel |
| MaCE | Softmax-9 / softmax-27 mass routing |
| PPS / Boids | Heading agents; same bins, different force pass |

Particle Lenia is the Lagrangian reading of Lenia’s energy, not a
Mohr variant. ECE implements its overdamped energy-gradient rule as a
separate single-species periodic CPU reference; see the catalog for its
equation and torus adaptation.
