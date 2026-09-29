"""Deterministic WebGPU PIC operations with stable host-side segment planning."""

from __future__ import annotations

import numpy as np

from .bins import exclusive_scan
from .pic import _particle_grid_coordinates
from .wgpu_mohr import webgpu_available


_DEVICE = None
_DEPOSIT_PIPELINE = None
_SAMPLE_PIPELINE = None

_DEPOSIT_SHADER = """
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read> counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let cell = id.x;
    if (cell >= arrayLength(&output)) { return; }
    var total = 0.0;
    for (var item = offsets[cell]; item < offsets[cell] + counts[cell]; item = item + 1u) {
        total += values[item];
    }
    output[cell] = total;
}
"""

_SAMPLE_SHADER = """
@group(0) @binding(0) var<storage, read> fields: array<f32>;
@group(0) @binding(1) var<storage, read> coords: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> params: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let particle = id.x;
    let particle_count = u32(params[0]);
    let channels = u32(params[1]);
    let height = u32(params[2]);
    let width = u32(params[3]);
    if (particle >= particle_count) { return; }
    let coordinate = coords[particle];
    let x0 = u32(coordinate.x);
    let y0 = u32(coordinate.y);
    let fx = coordinate.z;
    let fy = coordinate.w;
    for (var channel = 0u; channel < channels; channel = channel + 1u) {
        var value = 0.0;
        for (var dy = 0u; dy < 2u; dy = dy + 1u) {
            let wy = select(1.0 - fy, fy, dy == 1u);
            let y = (y0 + dy) % height;
            for (var dx = 0u; dx < 2u; dx = dx + 1u) {
                let wx = select(1.0 - fx, fx, dx == 1u);
                let x = (x0 + dx) % width;
                let index = channel * height * width + y * width + x;
                value += fields[index] * wx * wy;
            }
        }
        output[particle * channels + channel] = value;
    }
}
"""


def _get_pipelines():
    global _DEVICE, _DEPOSIT_PIPELINE, _SAMPLE_PIPELINE
    if _DEPOSIT_PIPELINE is None or _SAMPLE_PIPELINE is None:
        import wgpu

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError("no WebGPU adapter is available")
        _DEVICE = adapter.request_device_sync()
        deposit_module = _DEVICE.create_shader_module(code=_DEPOSIT_SHADER)
        sample_module = _DEVICE.create_shader_module(code=_SAMPLE_SHADER)
        _DEPOSIT_PIPELINE = _DEVICE.create_compute_pipeline(
            layout="auto", compute={"module": deposit_module, "entry_point": "main"}
        )
        _SAMPLE_PIPELINE = _DEVICE.create_compute_pipeline(
            layout="auto", compute={"module": sample_module, "entry_point": "main"}
        )
    return _DEVICE, _DEPOSIT_PIPELINE, _SAMPLE_PIPELINE


def deposit_particles_wgpu(
    pos: np.ndarray,
    types: np.ndarray,
    world: np.ndarray,
    shape: tuple[int, int],
    channels: int,
    masses: np.ndarray | None = None,
) -> np.ndarray:
    if not webgpu_available():
        raise RuntimeError("WebGPU is unavailable; install the webgpu extra and check the adapter")
    import wgpu

    pos = np.asarray(pos, dtype=np.float64)
    types = np.asarray(types, dtype=np.int64)
    if types.shape != (len(pos),) or channels < 1 or np.any(types < 0) or np.any(types >= channels):
        raise ValueError("particle types must match particles and index valid channels")
    if masses is None:
        masses = np.ones(len(pos), dtype=np.float64)
    else:
        masses = np.asarray(masses, dtype=np.float64)
    if masses.shape != (len(pos),) or not np.isfinite(masses).all() or np.any(masses < 0.0):
        raise ValueError("masses must be finite, non-negative, and match particle count")

    height, width = shape
    x0, y0, fx, fy = _particle_grid_coordinates(pos, world, shape)
    target_parts = []
    value_parts = []
    for dy, wy in ((0, 1.0 - fy), (1, fy)):
        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            target_parts.append(types * height * width + ((y0 + dy) % height) * width + (x0 + dx) % width)
            value_parts.append(masses * wx * wy)
    targets = np.concatenate(target_parts)
    values = np.concatenate(value_parts)
    order = np.argsort(targets, kind="stable")
    sorted_values = np.ascontiguousarray(values[order], dtype=np.float32)
    total_cells = channels * height * width
    counts = np.bincount(targets, minlength=total_cells).astype(np.uint32)
    offsets = exclusive_scan(counts).astype(np.uint32)

    device, pipeline, _ = _get_pipelines()
    storage = wgpu.BufferUsage.STORAGE
    input_buffers = [
        device.create_buffer_with_data(data=data, usage=storage)
        for data in (sorted_values, offsets, counts)
    ]
    output = device.create_buffer(size=total_cells * 4, usage=storage | wgpu.BufferUsage.COPY_SRC)
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[{"binding": i, "resource": buffer} for i, buffer in enumerate([*input_buffers, output])],
    )
    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((total_cells + 63) // 64)
    compute_pass.end()
    device.queue.submit([encoder.finish()])
    result = device.queue.read_buffer(output)
    return np.frombuffer(result, dtype=np.float32).reshape(channels, height, width).astype(np.float64)


def sample_fields_wgpu(fields: np.ndarray, pos: np.ndarray, world: np.ndarray) -> np.ndarray:
    if not webgpu_available():
        raise RuntimeError("WebGPU is unavailable; install the webgpu extra and check the adapter")
    import wgpu

    fields = np.ascontiguousarray(fields, dtype=np.float32)
    pos = np.asarray(pos, dtype=np.float64)
    if fields.ndim != 3:
        raise ValueError("fields must have shape (channels, height, width)")
    channels, height, width = fields.shape
    x0, y0, fx, fy = _particle_grid_coordinates(pos, world, (height, width))
    if len(pos) == 0:
        return np.zeros((0, channels), dtype=np.float64)
    coords = np.ascontiguousarray(np.column_stack((x0, y0, fx, fy)), dtype=np.float32)
    params = np.asarray([len(pos), channels, height, width], dtype=np.float32)
    device, _, pipeline = _get_pipelines()
    storage = wgpu.BufferUsage.STORAGE
    field_buffer = device.create_buffer_with_data(data=fields, usage=storage)
    coords_buffer = device.create_buffer_with_data(data=coords, usage=storage)
    params_buffer = device.create_buffer_with_data(data=params, usage=storage)
    output = device.create_buffer(size=len(pos) * channels * 4, usage=storage | wgpu.BufferUsage.COPY_SRC)
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": field_buffer},
            {"binding": 1, "resource": coords_buffer},
            {"binding": 2, "resource": params_buffer},
            {"binding": 3, "resource": output},
        ],
    )
    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((len(pos) + 63) // 64)
    compute_pass.end()
    device.queue.submit([encoder.finish()])
    result = device.queue.read_buffer(output)
    return np.frombuffer(result, dtype=np.float32).reshape(len(pos), channels).astype(np.float64)