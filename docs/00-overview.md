# Entelechy Continuum Engine — Overview

ECE is a hybrid Eulerian–Lagrangian artificial-life runtime. It does
not collapse every rule family into one equation. It runs them as
plugins on a shared state:

- Eulerian fields (Lenia, SmoothLife, Field Life, MaCE, RD)
- Lagrangian particles (Mohr Particle Life, Clusters, PPS, Boids)
- Particle-in-Cell coupling (deposit / sample) for hybrids

This repository ships corrected CPU reference kernels, Tk and native
WebGPU previews, and optional CUDA/WebGPU compute backends with oracle tests.
Fully device-resident hybrid execution and co-local AI remain deferred.

## Two axes that must not be mixed

1. **Lagrangian vs Eulerian** — points moving through space vs density
   on a grid.
2. **Conserving vs open-growth** — mass invariant (MaCE, Flow-Lenia,
   Field Life) vs source/sink (classic Lenia, GoL, Mohr).

Asymmetric species matrices violate momentum conservation on purpose.
That is how chase rings and organelles appear.

## Source of truth

| Layer | File |
|---|---|
| Spec deltas vs the original catalog | `CORRECTIONS.md` |
| Executable math | `ece/*.py` |
| GPU fragments | `shaders/*.wgsl` |
| Invariants | `tests/` |
| Family catalog | `docs/02-rules-catalog.md` |
| Build cut | `docs/09-roadmap.md` |

## Default first run (M1)

2D Mohr Particle Life, torus world, 6 species, live matrix, spatial
hash, exponential friction. No 3D, no VLM, no FFT.
