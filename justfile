default:
    @just --list

test:
    python3 -m pytest tests/ -q

test-verbose:
    python3 -m pytest tests/ -vv

check-configs:
    python3 -c "from pathlib import Path; from ece.config import load_config; \
        [print(p.name, load_config(p).species_count) for p in Path('configs').glob('*.toml')]"

handbook:
    python3 tools/build_handbook.py
