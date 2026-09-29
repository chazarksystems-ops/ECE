import importlib.util
import os
from pathlib import Path

import pytest
import numpy as np

from ece.config import load_config
from ece.wgpu_mohr import webgpu_available
from ece.wgpu_preview import NativePreview


CFG_DIR = Path(__file__).resolve().parents[1] / "configs"
HAS_NATIVE_DISPLAY = (
    importlib.util.find_spec("PySide6") is not None
    and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    and webgpu_available()
)
pytestmark = pytest.mark.skipif(not HAS_NATIVE_DISPLAY, reason="native Qt/WebGPU display is unavailable")


@pytest.mark.parametrize(
    "config_name",
    ["particle_life_6.toml", "field_life.toml", "lenia_orbium.toml", "particle_lenia.toml"],
)
def test_native_window_presents_one_frame(config_name):
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    cfg = load_config(CFG_DIR / config_name)
    if cfg.field_cfg:
        cfg.field_cfg["dims"] = [16, 16]
    if cfg.mohr:
        cfg.mohr["particles"] = 32
    if cfg.particle_lenia:
        cfg.particle_lenia["particles"] = 32
    preview = NativePreview(cfg, max_frames=2)
    preview.window.show()
    app.processEvents()
    app.processEvents()
    assert preview.renderer is not None

    preview.step_once()
    preview.canvas.force_draw()
    app.processEvents()
    image = app.primaryScreen().grabWindow(preview.canvas.winId()).toImage()
    assert not image.isNull()
    assert image.width() > 0 and image.height() > 0
    preview.window.close()
    app.processEvents()


def test_native_mohr_state_stays_on_device_and_reset_reseeds():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    cfg = load_config(CFG_DIR / "particle_life_6.toml")
    cfg.mohr["particles"] = 32
    preview = NativePreview(cfg, max_frames=3)
    initial_cpu_positions = preview.state.pos.copy()
    preview.window.show()
    app.processEvents()
    app.processEvents()

    renderer = preview.renderer
    assert renderer is not None
    initial_device_positions = np.frombuffer(
        renderer.device.queue.read_buffer(renderer.mohr_position_buffers[renderer.mohr_active_index]),
        dtype=np.float32,
    ).copy()
    preview.step_once()
    updated_device_positions = np.frombuffer(
        renderer.device.queue.read_buffer(renderer.mohr_position_buffers[renderer.mohr_active_index]),
        dtype=np.float32,
    ).copy()

    assert preview.state.frame == 1
    np.testing.assert_array_equal(preview.state.pos, initial_cpu_positions)
    assert not np.array_equal(updated_device_positions, initial_device_positions)
    preview.reset()
    assert preview.state.frame == 0
    np.testing.assert_array_equal(preview.state.pos, initial_cpu_positions)
    preview.window.close()
    app.processEvents()