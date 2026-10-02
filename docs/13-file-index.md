# File index

| Path | What it is |
|---|---|
| `README.md` | Entry point |
| `AGENTS.md` | Conventions, invariants, and definition of done for coding agents |
| `CLAUDE.md` | Claude Code entry point; imports `AGENTS.md` |
| `.github/workflows/ci.yml` | CPU test suite on every pull request |
| `CORRECTIONS.md` | Deltas vs the original catalog |
| `CHANGELOG.md` | Dated notes |
| `pyproject.toml` | Package metadata |
| `pytest.ini` | Test path |
| `justfile` | Local commands |
| `ece/__init__.py` | Public surface |
| `ece/__main__.py` | Headless CLI runner (`python3 -m ece`) |
| `ece/mohr.py` | Tent + O(N²) accelerations |
| `ece/integrate.py` | Friction + Euler + wrap |
| `ece/mace.py` | Affinity + two-pass MaCE, ring kernels, periodic FFT convolution |
| `ece/bins.py` | Snapshot hash + grid/neighborhood validation |
| `ece/config.py` | TOML loader + validation |
| `ece/matrix.py` | Matrix generators + editor-text parser |
| `ece/step_mohr.py` | CPU Mohr stepper (M1 host), vectorized binned forces |
| `ece/fixedpoint_mohr.py` | CPU integer replay prototype |
| `ece/field_life.py` | CPU Field Life runner |
| `ece/lenia.py` | CPU Lenia runner |
| `ece/particle_lenia.py` | CPU Particle Lenia energy-gradient rule |
| `ece/pic.py` | CPU deposit / sample |
| `ece/hybrid.py` | CPU/CUDA/WebGPU hybrid scheduler |
| `ece/cuda_*.py` | Numba CUDA backends (Mohr, Field Life, Lenia, Particle Lenia, PIC, resident hybrid) |
| `ece/cupy_field_life.py` | CuPy/cuFFT 3D Field Life |
| `ece/wgpu_*.py` | WebGPU compute backends with inline WGSL |
| `ece/preview.py` | Tk preview with matrix editor |
| `ece/wgpu_preview.py` | Native Qt/WebGPU preview |
| `shaders/mohr_force.wgsl` | Corrected force reference fragment |
| `shaders/mace_2pass.wgsl` | Denom + gather reference fragment |
| `tests/test_*.py` | Invariants and backend oracle tests; see `docs/15-testing.md` |
| `tools/benchmark_*.py` | CUDA Mohr and CuPy Field Life benchmarks |
| `configs/*.toml` | Named soups |
| `schemas/sim.schema.json` | Config JSON Schema |
| `docs/00`–`15` | Design set |
| `docs/handbook.pdf` | Printable compilation |
| `tools/build_handbook.py` | Handbook builder |
