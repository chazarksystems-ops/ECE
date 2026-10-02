# Testing

```bash
python3 -m pytest tests/ -q
```

92 tests after the M1 preview, field runners, PIC references, replay tests,
and hash/config guards. Without a GPU, 52 run and 40 skip; no test needs
`tkinter`.

| File | Locks |
|---|---|
| `test_mohr.py` | wall, tent peak, horizon, scale invariance, asymmetry, wrap |
| `test_mace.py` | per-channel mass, uniform fixed point, corner torus, local-max softmax, crowding |
| `test_bins_integrate.py` | exclusive scan, frozen snapshot, torus-aligned bins, grid ≥ neighborhood, Mohr-gamma conversion, wrap |
| `test_config.py` | TOML load, chase matrix, channel mismatch, invalid hash grid, unknown top-level and in-table keys |
| `test_hybrid.py` | CPU hybrid scheduling, per-species mass, sampled values, NPZ output |
| `test_cuda_mohr.py` | optional CUDA-vs-CPU step, resident stability, CLI smoke, same-device repeatability (skips without device) |
| `test_cuda_field_life.py` | optional CUDA-vs-CPU three-step field parity, mass readback, and CLI NPZ output |
| `test_cuda_hybrid.py` | optional CUDA hybrid parity, mass readback, 100-frame stability, CLI, and same-device repeatability |
| `test_cuda_lenia.py` | optional CUDA-vs-CPU growth, boundedness, and CLI NPZ output |
| `test_cuda_particle_lenia.py` | optional CUDA-vs-CPU energy gradient, wrap stability, and CLI output |
| `test_cuda_pic.py` | optional CUDA segmented deposit and sample parity (skips without device) |
| `test_wgpu_mohr.py` | optional WGSL-vs-CPU 200-frame parity at 256 particles, CLI smoke, and stability (skips without adapter) |
| `test_wgpu_field_life.py` | optional WGSL-vs-CPU three-step field parity, mass readback, and CLI NPZ output |
| `test_wgpu_lenia.py` | optional WGSL-vs-CPU growth, bounds, and CLI NPZ output |
| `test_wgpu_particle_lenia.py` | optional WGSL-vs-CPU energy gradient, wrap stability, and CLI output |
| `test_wgpu_pic.py` | optional WGSL segmented deposit/sample parity and mass conservation |
| `test_wgpu_hybrid.py` | optional one-frame hybrid parity and full-state CLI output |
| `test_wgpu_preview.py` | optional native-surface modes plus device-resident Mohr step/reset |
| `test_field_life.py` | config-driven runner, per-channel mass, field NPZ output |
| `test_cupy_field_life.py` | optional 3D CuPy/cuFFT parity, mass conservation, and CLI volume output |
| `test_lenia.py` | kernel normalization, bounded growth, configured extinction, Lenia NPZ output |
| `test_particle_lenia.py` | shell normalization, energy-gradient finite difference, periodic run, CLI state export |
| `test_fixedpoint_mohr.py` | fixed-point oracle tolerance, integer repeatability, wrap bounds, CLI Q-state export |
| `test_pic.py` | per-species deposit mass, torus sampling, deposit/sample adjoint identity |
| `test_step_mohr.py` | torus stay-in, per-frame binned vs dense agreement (incl. minimal 3×3 grid), headless CLI, matrix validation |

GPU tests skip cleanly when their adapter is unavailable. Small-N CPU oracle
parity and per-channel mass readback are the primary backend gates.
