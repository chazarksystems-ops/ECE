"""Native Qt/WebGPU presentation for the supported 2D simulation rules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .config import load_config


_PALETTE = np.asarray(
    [
        [0.94, 0.49, 0.24],
        [0.24, 0.78, 0.58],
        [0.91, 0.31, 0.36],
        [0.29, 0.61, 0.91],
        [0.91, 0.80, 0.31],
        [0.65, 0.43, 0.82],
    ],
    dtype=np.float32,
)

_PARTICLE_SHADER = """
struct VertexInput {
    @location(0) corner: vec2<f32>,
    @location(1) rect: vec4<f32>,
    @location(2) color: vec3<f32>,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};
@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(input.rect.xy + input.corner * input.rect.zw, 0.0, 1.0);
    output.color = input.color;
    return output;
}
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}
"""

_MOHR_RESIDENT_SHADER = """
@group(0) @binding(0) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> types: array<u32>;
@group(0) @binding(2) var<storage, read> matrix: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<f32>;
@group(0) @binding(4) var<storage, read_write> next_positions: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> velocities: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> instances: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let count = u32(params[0]);
    let species_count = u32(params[1]);
    if (i >= count) { return; }
    let world = vec2<f32>(params[7], params[8]);
    var acceleration = vec2<f32>(0.0, 0.0);
    for (var j = 0u; j < count; j = j + 1u) {
        if (i == j) { continue; }
        var delta = positions[j] - positions[i];
        delta -= world * round(delta / world);
        let distance = length(delta);
        if (distance < 1e-6 || distance > params[2]) { continue; }
        let r = distance / params[2];
        let a = matrix[types[i] * species_count + types[j]];
        var force = 0.0;
        if (r < params[3]) {
            force = r / params[3] - 1.0;
        } else {
            force = a * (1.0 - abs(2.0 * r - 1.0 - params[3]) / (1.0 - params[3]));
        }
        acceleration += params[4] * force * delta / distance;
    }
    let decay = exp(-params[5] * params[6]);
    let velocity = decay * velocities[i] + acceleration * params[6];
    var position = positions[i] + velocity * params[6];
    position -= world * floor(position / world);
    next_positions[i] = position;
    velocities[i] = velocity;

    let scale = vec2<f32>(params[11], params[12]);
    let clip_center = vec2<f32>(
        (position.x / world.x * 2.0 - 1.0) * scale.x,
        (1.0 - position.y / world.y * 2.0) * scale.y,
    );
    let base = i * 7u;
    instances[base] = clip_center.x;
    instances[base + 1u] = clip_center.y;
    instances[base + 2u] = params[9];
    instances[base + 3u] = params[10];
    var color = vec3<f32>(0.94, 0.49, 0.24);
    switch types[i] {
        case 1u: { color = vec3<f32>(0.24, 0.78, 0.58); }
        case 2u: { color = vec3<f32>(0.91, 0.31, 0.36); }
        case 3u: { color = vec3<f32>(0.29, 0.61, 0.91); }
        case 4u: { color = vec3<f32>(0.91, 0.80, 0.31); }
        case 5u: { color = vec3<f32>(0.65, 0.43, 0.82); }
        default: {}
    }
    instances[base + 4u] = color.r;
    instances[base + 5u] = color.g;
    instances[base + 6u] = color.b;
}
"""

_FIELD_SHADER = """
@group(0) @binding(0) var field_texture: texture_2d<f32>;
@group(0) @binding(1) var field_sampler: sampler;
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let position = positions[index];
    var output: VertexOutput;
    output.position = vec4<f32>(position, 0.0, 1.0);
    output.uv = vec2<f32>((position.x + 1.0) * 0.5, (1.0 - position.y) * 0.5);
    return output;
}
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(field_texture, field_sampler, input.uv);
}
"""


class WebGPURenderer:
    def __init__(self, canvas, kind: str, channels: int, world: np.ndarray, particle_count: int = 0):
        import wgpu

        self.wgpu = wgpu
        self.canvas = canvas
        self.kind = kind
        self.world = np.asarray(world, dtype=np.float32)
        adapter = wgpu.gpu.request_adapter_sync(
            canvas=canvas,
            power_preference="high-performance",
        )
        if adapter is None:
            raise RuntimeError("No WebGPU adapter can present to this window")
        self.device = adapter.request_device_sync()
        self.context = canvas.get_wgpu_context()
        self.format = self.context.get_preferred_format(adapter)
        self.context.configure(device=self.device, format=self.format)
        self.pipeline = self._create_pipeline(channels)
        self.particle_capacity = max(particle_count, 1)
        self.instance_buffer = None
        self.corner_buffer = None
        self.mohr_position_buffers = None
        self.mohr_velocity_buffer = None
        self.mohr_bind_groups = None
        self.mohr_active_index = 0
        self.field_texture = None
        self.field_bind_group = None
        if kind == "particles":
            self._create_particle_buffers(self.particle_capacity)
        else:
            self.field_sampler = self.device.create_sampler(
                mag_filter="nearest",
                min_filter="nearest",
            )

    def _create_pipeline(self, channels: int):
        wgpu = self.wgpu
        if self.kind == "particles":
            module = self.device.create_shader_module(code=_PARTICLE_SHADER)
            return self.device.create_render_pipeline(
                layout="auto",
                vertex={
                    "module": module,
                    "entry_point": "vs_main",
                    "buffers": [
                        {
                            "array_stride": 8,
                            "step_mode": "vertex",
                            "attributes": [{"format": "float32x2", "offset": 0, "shader_location": 0}],
                        },
                        {
                            "array_stride": 28,
                            "step_mode": "instance",
                            "attributes": [
                                {"format": "float32x4", "offset": 0, "shader_location": 1},
                                {"format": "float32x3", "offset": 16, "shader_location": 2},
                            ],
                        },
                    ],
                },
                primitive={"topology": "triangle-list", "cull_mode": "none"},
                fragment={
                    "module": module,
                    "entry_point": "fs_main",
                    "targets": [{"format": self.format}],
                },
            )
        module = self.device.create_shader_module(code=_FIELD_SHADER)
        return self.device.create_render_pipeline(
            layout="auto",
            vertex={"module": module, "entry_point": "vs_main"},
            primitive={"topology": "triangle-list"},
            fragment={
                "module": module,
                "entry_point": "fs_main",
                "targets": [{"format": self.format}],
            },
        )

    def _create_particle_buffers(self, capacity: int) -> None:
        wgpu = self.wgpu
        corners = np.asarray(
            [[-1, -1], [1, -1], [-1, 1], [-1, 1], [1, -1], [1, 1]],
            dtype=np.float32,
        )
        self.corner_buffer = self.device.create_buffer_with_data(
            data=corners,
            usage=wgpu.BufferUsage.VERTEX,
        )
        self.instance_buffer = self.device.create_buffer(
            size=capacity * 7 * np.dtype(np.float32).itemsize,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE,
        )
        self.particle_capacity = capacity

    def setup_mohr_device_simulation(self, cfg, state) -> None:
        if self.kind != "particles":
            raise ValueError("Mohr simulation requires the particle renderer")
        self.mohr_cfg = cfg
        self.particle_count = len(state.pos)
        self.mohr_position_buffers = [
            self.device.create_buffer_with_data(
                data=np.ascontiguousarray(state.pos, dtype=np.float32),
                usage=(
                    self.wgpu.BufferUsage.STORAGE
                    | self.wgpu.BufferUsage.COPY_DST
                    | self.wgpu.BufferUsage.COPY_SRC
                ),
            ),
            self.device.create_buffer(
                size=max(self.particle_count, 1) * 2 * 4,
                usage=(
                    self.wgpu.BufferUsage.STORAGE
                    | self.wgpu.BufferUsage.COPY_DST
                    | self.wgpu.BufferUsage.COPY_SRC
                ),
            ),
        ]
        self.mohr_velocity_buffer = self.device.create_buffer_with_data(
            data=np.ascontiguousarray(state.vel, dtype=np.float32),
            usage=self.wgpu.BufferUsage.STORAGE | self.wgpu.BufferUsage.COPY_DST,
        )
        self.mohr_types_buffer = self.device.create_buffer_with_data(
            data=np.ascontiguousarray(state.types, dtype=np.uint32),
            usage=self.wgpu.BufferUsage.STORAGE,
        )
        self.mohr_matrix_buffer = self.device.create_buffer_with_data(
            data=np.ascontiguousarray(cfg.matrix, dtype=np.float32),
            usage=self.wgpu.BufferUsage.STORAGE | self.wgpu.BufferUsage.COPY_DST,
        )
        self.mohr_params_buffer = self.device.create_buffer(
            size=13 * 4,
            usage=self.wgpu.BufferUsage.STORAGE | self.wgpu.BufferUsage.COPY_DST,
        )
        module = self.device.create_shader_module(code=_MOHR_RESIDENT_SHADER)
        self.mohr_pipeline = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": module, "entry_point": "main"},
        )
        self.mohr_bind_groups = [
            self._create_mohr_bind_group(0, 1),
            self._create_mohr_bind_group(1, 0),
        ]
        self.mohr_active_index = 0
        self._write_mohr_params()

    def _create_mohr_bind_group(self, source_index: int, target_index: int):
        return self.device.create_bind_group(
            layout=self.mohr_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": self.mohr_position_buffers[source_index]},
                {"binding": 1, "resource": self.mohr_types_buffer},
                {"binding": 2, "resource": self.mohr_matrix_buffer},
                {"binding": 3, "resource": self.mohr_params_buffer},
                {"binding": 4, "resource": self.mohr_position_buffers[target_index]},
                {"binding": 5, "resource": self.mohr_velocity_buffer},
                {"binding": 6, "resource": self.instance_buffer},
            ],
        )

    def _write_mohr_params(self) -> None:
        width, height = self.canvas.get_physical_size()
        width = max(width, 1)
        height = max(height, 1)
        cfg = self.mohr_cfg
        params = np.asarray(
            [
                self.particle_count,
                cfg.species_count,
                cfg.mohr["r_max"],
                cfg.mohr["beta"],
                cfg.mohr.get("gain", 1.0),
                cfg.mohr.get("lambda", 0.0),
                cfg.dt,
                cfg.world[0],
                cfg.world[1],
                5.0 / width,
                5.0 / height,
                min(1.0, width / height),
                min(1.0, height / width),
            ],
            dtype=np.float32,
        )
        self.device.queue.write_buffer(self.mohr_params_buffer, 0, params)

    def step_mohr_device(self) -> None:
        self._write_mohr_params()
        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.mohr_pipeline)
        compute_pass.set_bind_group(0, self.mohr_bind_groups[self.mohr_active_index])
        compute_pass.dispatch_workgroups(max((self.particle_count + 63) // 64, 1))
        compute_pass.end()
        self.device.queue.submit([encoder.finish()])
        self.mohr_active_index = 1 - self.mohr_active_index
        self.canvas.request_draw()

    def reset_mohr_device(self, state) -> None:
        if self.mohr_position_buffers is None:
            return
        self.device.queue.write_buffer(
            self.mohr_position_buffers[0], 0, np.ascontiguousarray(state.pos, dtype=np.float32)
        )
        self.device.queue.write_buffer(
            self.mohr_position_buffers[1], 0, np.zeros_like(state.pos, dtype=np.float32)
        )
        self.device.queue.write_buffer(
            self.mohr_velocity_buffer, 0, np.ascontiguousarray(state.vel, dtype=np.float32)
        )
        self.mohr_active_index = 0
        self._upload_particles(state)

    def update(self, state) -> None:
        if self.kind == "particles":
            self._upload_particles(state)
        else:
            self._upload_field(state.rho)
        self.canvas.request_draw()

    def _upload_particles(self, state) -> None:
        count = len(state.pos)
        if count > self.particle_capacity:
            self._create_particle_buffers(count)
        width, height = self.canvas.get_physical_size()
        width = max(width, 1)
        height = max(height, 1)
        radius_x = 5.0 / width
        radius_y = 5.0 / height
        scale_x = min(1.0, width / height)
        scale_y = min(1.0, height / width)
        normalized = np.asarray(state.pos, dtype=np.float32) / self.world
        rectangles = np.empty((count, 7), dtype=np.float32)
        rectangles[:, 0] = (normalized[:, 0] * 2.0 - 1.0) * scale_x
        rectangles[:, 1] = (1.0 - normalized[:, 1] * 2.0) * scale_y
        rectangles[:, 2] = radius_x
        rectangles[:, 3] = radius_y
        for species in range(int(state.types.max()) + 1 if count else 0):
            rectangles[state.types == species, 4:7] = _PALETTE[species % len(_PALETTE)]
        if count:
            self.device.queue.write_buffer(self.instance_buffer, 0, rectangles)
        self.particle_count = count

    def _field_rgba(self, rho: np.ndarray) -> np.ndarray:
        if rho.shape[0] == 1:
            density = np.clip(rho[0], 0.0, 1.0)[..., None]
            rgb = np.concatenate(
                [0.12 + 0.82 * density, 0.13 + 0.48 * density, 0.16 + 0.22 * density],
                axis=2,
            )
        else:
            palette = _PALETTE[: rho.shape[0]]
            density = np.clip(rho, 0.0, 1.0)
            rgb = np.einsum("chw,ck->hwk", density, palette, optimize=True)
            peak = np.maximum(rgb.max(axis=2, keepdims=True), 1.0)
            rgb = rgb / peak
        rgba = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = np.asarray(np.clip(rgb, 0.0, 1.0) * 255.0, dtype=np.uint8)
        rgba[..., 3] = 255
        return rgba

    def _upload_field(self, rho: np.ndarray) -> None:
        wgpu = self.wgpu
        rgba = self._field_rgba(rho)
        height, width = rgba.shape[:2]
        if self.field_texture is None or self.field_texture_size != (width, height):
            if self.field_texture is not None:
                self.field_texture.destroy()
            self.field_texture = self.device.create_texture(
                size=(width, height, 1),
                format="rgba8unorm",
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            )
            self.field_texture_size = (width, height)
            self.field_bind_group = self.device.create_bind_group(
                layout=self.pipeline.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": self.field_texture.create_view()},
                    {"binding": 1, "resource": self.field_sampler},
                ],
            )
        row_bytes = width * 4
        padded_bytes = ((row_bytes + 255) // 256) * 256
        padded = np.zeros((height, padded_bytes), dtype=np.uint8)
        padded[:, :row_bytes] = rgba.reshape(height, row_bytes)
        self.device.queue.write_texture(
            {"texture": self.field_texture, "mip_level": 0, "origin": (0, 0, 0)},
            padded,
            {"offset": 0, "bytes_per_row": padded_bytes, "rows_per_image": height},
            (width, height, 1),
        )

    def draw(self) -> None:
        wgpu = self.wgpu
        view = self.context.get_current_texture().create_view()
        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": view,
                    "resolve_target": None,
                    "clear_value": (0.035, 0.055, 0.065, 1.0),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ]
        )
        render_pass.set_pipeline(self.pipeline)
        if self.kind == "particles":
            render_pass.set_vertex_buffer(0, self.corner_buffer)
            render_pass.set_vertex_buffer(1, self.instance_buffer)
            render_pass.draw(6, self.particle_count)
        elif self.field_bind_group is not None:
            render_pass.set_bind_group(0, self.field_bind_group)
            render_pass.draw(3, 1)
        render_pass.end()
        self.device.queue.submit([encoder.finish()])


class NativePreview:
    def __init__(self, config, max_frames: int):
        import PySide6
        from PySide6.QtCore import QTimer, Qt
        from PySide6.QtWidgets import (
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QPushButton,
            QSlider,
            QVBoxLayout,
            QWidget,
        )
        from rendercanvas.qt import QRenderWidget

        self._qt = Qt
        self._QTimer = QTimer
        self.cfg = config
        self.max_frames = max_frames
        self.running = False
        self.steps_per_update = 1
        self.rule = config.rules[0]
        if self.rule == "mohr":
            from .step_mohr import seed_state

            self.state = seed_state(config)
            self.kind = "particles"
            particle_count = len(self.state.pos)
            channels = config.species_count
        elif self.rule == "field_life":
            from .field_life import seed_state

            self.state = seed_state(config)
            self.kind = "field"
            particle_count = 0
            channels = config.species_count
        elif self.rule == "lenia":
            from .lenia import seed_state

            self.state = seed_state(config)
            self.kind = "field"
            particle_count = 0
            channels = 1
        elif self.rule == "particle_lenia":
            from .particle_lenia import seed_state

            self.state = seed_state(config)
            self.kind = "particles"
            particle_count = len(self.state.pos)
            channels = 1
        else:
            raise ValueError("native WebGPU preview supports Mohr, Field Life, and Lenia")

        self.window = QMainWindow()
        self.window.setWindowTitle("ECE | Native WebGPU Preview")
        self.window.resize(1140, 820)
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(20)
        controls = QVBoxLayout()
        controls.setSpacing(10)
        title = QLabel("ECE / 2D")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #e5ece9")
        self.status = QLabel()
        self.status.setStyleSheet("color: #9ab0ac; font-size: 13px")
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.toggle_running)
        step_button = QPushButton("Step")
        step_button.clicked.connect(self.step_once)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset)
        speed_label = QLabel("Steps per update")
        self.speed_slider = QSlider(self._qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 4)
        self.speed_slider.setValue(1)
        self.speed_slider.valueChanged.connect(self._set_speed)
        for widget in (title, self.status, self.start_button, step_button, reset_button, speed_label, self.speed_slider):
            controls.addWidget(widget)
        controls.addStretch(1)

        self.canvas = QRenderWidget(
            present_method="screen",
            update_mode="continuous",
            max_fps=60,
        )
        self.canvas.setMinimumSize(620, 620)
        layout.addLayout(controls, 0)
        layout.addWidget(self.canvas, 1)
        self.window.setCentralWidget(root)
        self.window.setStyleSheet(
            "QMainWindow, QWidget { background: #10181b; }"
            "QPushButton { background: #26383b; color: #e5ece9; padding: 9px; text-align: left; }"
            "QPushButton:hover { background: #345052; }"
            "QSlider::groove:horizontal { height: 4px; background: #34464b; }"
            "QSlider::handle:horizontal { background: #f0a35e; width: 12px; margin: -5px 0; }"
        )
        self.renderer = None
        self.timer = QTimer(self.window)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.advance)
        QTimer.singleShot(0, lambda: self._initialize_renderer(channels, particle_count))
        self._update_status()

    def _initialize_renderer(self, channels: int, particle_count: int) -> None:
        self.renderer = WebGPURenderer(self.canvas, self.kind, channels, self.cfg.world, particle_count)
        self.renderer.update(self.state)
        if self.rule == "mohr":
            self.renderer.setup_mohr_device_simulation(self.cfg, self.state)

    def _set_speed(self, value: int) -> None:
        self.steps_per_update = value

    def _update_status(self) -> None:
        self.status.setText(f"{self.rule.replace('_', ' ').upper()}\nFrame {self.state.frame} / {self.max_frames}")

    def toggle_running(self) -> None:
        if self.state.frame >= self.max_frames:
            self.reset()
        self.running = not self.running
        self.start_button.setText("Pause" if self.running else "Start")
        if self.running:
            self.timer.start()
        else:
            self.timer.stop()

    def step_once(self) -> None:
        if self.state.frame >= self.max_frames:
            return
        self._step()

    def _step(self) -> None:
        if self.rule == "mohr":
            self.renderer.step_mohr_device()
            self.state.frame += 1
        elif self.rule == "field_life":
            from .wgpu_field_life import step_wgpu

            self.state = step_wgpu(self.state, self.cfg)
        elif self.rule == "lenia":
            from .wgpu_lenia import step_wgpu

            self.state = step_wgpu(self.state, self.cfg)
        else:
            from .wgpu_particle_lenia import step_wgpu

            self.state = step_wgpu(self.state, self.cfg)
        if self.rule != "mohr":
            self.renderer.update(self.state)
        self._update_status()
        if self.state.frame >= self.max_frames:
            self.running = False
            self.timer.stop()
            self.start_button.setText("Restart")

    def advance(self) -> None:
        for _ in range(min(self.steps_per_update, self.max_frames - self.state.frame)):
            self._step()
            if not self.running:
                break

    def reset(self) -> None:
        self.running = False
        self.timer.stop()
        if self.rule == "mohr":
            from .step_mohr import seed_state
        elif self.rule == "field_life":
            from .field_life import seed_state
        elif self.rule == "particle_lenia":
            from .particle_lenia import seed_state
        else:
            from .lenia import seed_state
        self.state = seed_state(self.cfg)
        self.start_button.setText("Start")
        if self.renderer is not None:
            if self.rule == "mohr":
                self.renderer.reset_mohr_device(self.state)
            else:
                self.renderer.update(self.state)
        self._update_status()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Native Qt/WebGPU preview")
    parser.add_argument("config", type=Path, help="path to a TOML simulation config")
    parser.add_argument("--frames", type=int, help="override configured frame limit")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    max_frames = args.frames if args.frames is not None else int(cfg.io.get("frames", 1000))
    if max_frames < 1:
        parser.error("frame limit must be at least 1")
    if cfg.rules not in (["mohr"], ["field_life"], ["lenia"], ["particle_lenia"]):
        parser.error("native preview supports Mohr, Field Life, Lenia, and Particle Lenia configs")

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    preview = NativePreview(cfg, max_frames)
    preview.window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())