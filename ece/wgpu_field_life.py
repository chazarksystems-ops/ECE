"""Optional WebGPU implementation of the 2D Field Life update."""

from __future__ import annotations

import numpy as np

from .mace import ring_kernel_2d
from .wgpu_mohr import webgpu_available


_DEVICE = None
_PIPELINE = None

_SHADER = """
@group(0) @binding(0) var<storage, read> rho: array<f32>;
@group(0) @binding(1) var<storage, read> matrix: array<f32>;
@group(0) @binding(2) var<storage, read> kernel: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<f32>;
@group(0) @binding(4) var<storage, read_write> affinity: array<f32>;
@group(0) @binding(5) var<storage, read_write> denominators: array<f32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;
@group(0) @binding(7) var<storage, read_write> maxima: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let channels = u32(params[0]);
    let height = u32(params[1]);
    let width = u32(params[2]);
    let radius = i32(params[3]);
    let mode = u32(params[4]);
    if (index >= channels * height * width) { return; }
    let x = index % width;
    let y = (index / width) % height;
    let channel = index / (height * width);

    if (mode == 0u) {
        var mixed = 0.0;
        for (var source = 0u; source < channels; source = source + 1u) {
            var convolution = 0.0;
            for (var ky = 0u; ky < u32(2 * radius + 1); ky = ky + 1u) {
                let yy = u32((i32(y) + i32(ky) - radius + i32(height)) % i32(height));
                for (var kx = 0u; kx < u32(2 * radius + 1); kx = kx + 1u) {
                    let xx = u32((i32(x) + i32(kx) - radius + i32(width)) % i32(width));
                    let kernel_index = ky * u32(2 * radius + 1) + kx;
                    convolution += kernel[kernel_index] * rho[source * height * width + yy * width + xx];
                }
            }
            mixed += matrix[channel * channels + source] * convolution;
        }
        affinity[index] = params[5] * mixed - params[6] * rho[index];
    } else if (mode == 1u) {
        var maximum = -3.402823e+38;
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            let yy = u32((i32(y) + dy + i32(height)) % i32(height));
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let xx = u32((i32(x) + dx + i32(width)) % i32(width));
                let neighbor = channel * height * width + yy * width + xx;
                maximum = max(maximum, affinity[neighbor]);
            }
        }
        maxima[index] = maximum;
    } else if (mode == 2u) {
        var denominator = 0.0;
        let maximum = maxima[index];
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            let yy = u32((i32(y) + dy + i32(height)) % i32(height));
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let xx = u32((i32(x) + dx + i32(width)) % i32(width));
                let neighbor = channel * height * width + yy * width + xx;
                denominator += exp(params[7] * (affinity[neighbor] - maximum));
            }
        }
        denominators[index] = denominator;
    } else {
        var value = 0.0;
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            let yy = u32((i32(y) + dy + i32(height)) % i32(height));
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let xx = u32((i32(x) + dx + i32(width)) % i32(width));
                let neighbor = channel * height * width + yy * width + xx;
                let transition = exp(params[7] * (affinity[index] - maxima[neighbor]));
                value += rho[neighbor] * transition / denominators[neighbor];
            }
        }
        output[index] = value;
    }
}
"""


def _get_pipeline():
    global _DEVICE, _PIPELINE
    if _PIPELINE is None:
        import wgpu

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError("no WebGPU adapter is available")
        _DEVICE = adapter.request_device_sync()
        module = _DEVICE.create_shader_module(code=_SHADER)
        _PIPELINE = _DEVICE.create_compute_pipeline(
            layout="auto",
            compute={"module": module, "entry_point": "main"},
        )
    return _DEVICE, _PIPELINE


def mace_step_wgpu(
    rho: np.ndarray,
    matrix: np.ndarray,
    kernel: np.ndarray,
    strength: float,
    crowding_lambda: float,
    transport_beta: float,
) -> np.ndarray:
    if _PIPELINE is None and not webgpu_available():
        raise RuntimeError("WebGPU is unavailable; install the webgpu extra and check the adapter")
    import wgpu

    rho = np.ascontiguousarray(rho, dtype=np.float32)
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    kernel = np.ascontiguousarray(kernel, dtype=np.float32)
    if rho.ndim != 3 or matrix.shape != (rho.shape[0], rho.shape[0]):
        raise ValueError("rho must be (channels, height, width) and matrix must be square by channel")
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or kernel.shape[0] % 2 != 1:
        raise ValueError("kernel must be a square 2D stencil with odd width")
    channels, height, width = rho.shape
    radius = kernel.shape[0] // 2
    if height < kernel.shape[0] or width < kernel.shape[1]:
        raise ValueError("kernel dimensions must not exceed field dimensions")

    device, pipeline = _get_pipeline()
    storage = wgpu.BufferUsage.STORAGE
    rho_buffer = device.create_buffer_with_data(data=rho, usage=storage)
    matrix_buffer = device.create_buffer_with_data(data=matrix, usage=storage)
    kernel_buffer = device.create_buffer_with_data(data=kernel, usage=storage)
    affinity_buffer = device.create_buffer(size=rho.nbytes, usage=storage)
    maxima_buffer = device.create_buffer(size=rho.nbytes, usage=storage)
    denominator_buffer = device.create_buffer(size=rho.nbytes, usage=storage)
    output_buffer = device.create_buffer(size=rho.nbytes, usage=storage | wgpu.BufferUsage.COPY_SRC)
    groups = []
    for mode in range(4):
        params = np.asarray(
            [channels, height, width, radius, mode, strength, crowding_lambda, transport_beta],
            dtype=np.float32,
        )
        params_buffer = device.create_buffer_with_data(data=params, usage=storage)
        groups.append(
            device.create_bind_group(
                layout=pipeline.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": rho_buffer},
                    {"binding": 1, "resource": matrix_buffer},
                    {"binding": 2, "resource": kernel_buffer},
                    {"binding": 3, "resource": params_buffer},
                    {"binding": 4, "resource": affinity_buffer},
                    {"binding": 5, "resource": denominator_buffer},
                    {"binding": 6, "resource": output_buffer},
                    {"binding": 7, "resource": maxima_buffer},
                ],
            )
        )

    encoder = device.create_command_encoder()
    for bind_group in groups:
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups((rho.size + 63) // 64)
        compute_pass.end()
    device.queue.submit([encoder.finish()])
    result = device.queue.read_buffer(output_buffer)
    return np.frombuffer(result, dtype=np.float32).reshape(rho.shape).astype(np.float64)


def step_wgpu(state, cfg):
    if cfg.field_cfg.get("kernel", "ring") != "ring":
        raise ValueError("WebGPU Field Life currently supports the ring kernel")
    rho = mace_step_wgpu(
        state.rho,
        cfg.matrix,
        ring_kernel_2d(),
        float(cfg.field_cfg.get("strength", 1.0)),
        float(cfg.field_cfg.get("crowding_lambda", 0.0)),
        float(cfg.field_cfg.get("transport_beta", 1.0)),
    )
    return type(state)(rho=rho, frame=state.frame + 1)