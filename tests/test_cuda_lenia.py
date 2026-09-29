from pathlib import Path

import numpy as np
import pytest

from ece.config import load_config
from ece.__main__ import main
from ece.cuda_lenia import run_cuda
from ece.cuda_mohr import cuda_available
from ece.lenia import run


CFG = Path(__file__).resolve().parents[1] / "configs" / "lenia_orbium.toml"
pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA device is unavailable")


def test_cuda_lenia_matches_cpu_and_stays_bounded():
    cfg = load_config(CFG)
    cfg.field_cfg["dims"] = [16, 16]
    cpu = run(cfg, frames=1)
    gpu = run_cuda(cfg, frames=1)

    np.testing.assert_allclose(gpu.rho, cpu.rho, rtol=3e-4, atol=3e-5)
    assert np.isfinite(gpu.rho).all()
    assert np.all(gpu.rho >= 0.0)
    assert np.all(gpu.rho <= 1.0)
    assert gpu.frame == 1


def test_cuda_lenia_cli_writes_field_state(tmp_path, capsys):
    config_path = tmp_path / "lenia.toml"
    config_path.write_text(CFG.read_text().replace("[128, 128]", "[16, 16]"))
    output = tmp_path / "cuda-lenia.npz"
    main([str(config_path), "--cuda", "--frames", "1", "--output", str(output)])

    with np.load(output) as state:
        assert state["rho"].shape == (1, 16, 16)
        assert state["frame"].item() == 1
    assert capsys.readouterr().out.strip() == "frame=1 channels=1 dims=16x16"