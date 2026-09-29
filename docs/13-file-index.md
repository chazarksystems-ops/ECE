# File index

| Path | What it is |
|---|---|
| `README.md` | Entry point |
| `CORRECTIONS.md` | Deltas vs the original catalog |
| `CHANGELOG.md` | Dated notes |
| `pyproject.toml` | Package metadata |
| `pytest.ini` | Test path |
| `justfile` | Local commands |
| `ece/__init__.py` | Public surface |
| `ece/mohr.py` | Tent + O(N²) accelerations |
| `ece/integrate.py` | Friction + Euler + wrap |
| `ece/mace.py` | Affinity + two-pass MaCE |
| `ece/bins.py` | Snapshot hash |
| `ece/config.py` | TOML loader + validation |
| `ece/matrix.py` | Matrix generators |
| `ece/step_mohr.py` | CPU Mohr stepper (M1 host) |
| `shaders/mohr_force.wgsl` | Corrected force fragment |
| `shaders/mace_2pass.wgsl` | Denom + gather |
| `tests/test_mohr.py` | Force invariants |
| `tests/test_mace.py` | Mass / torus / softmax |
| `tests/test_bins_integrate.py` | Hash + friction |
| `tests/test_config.py` | Schema / TOML |
| `tests/test_step_mohr.py` | Stepper smoke |
| `configs/*.toml` | Named soups |
| `schemas/sim.schema.json` | Config JSON Schema |
| `docs/00`–`15` | Design set |
| `docs/handbook.pdf` | Printable compilation |
| `tools/build_handbook.py` | Handbook builder |
