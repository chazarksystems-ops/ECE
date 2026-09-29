"""Optional WebGPU implementation of the 2D Lenia update."""

from __future__ import annotations

import numpy as np

from .lenia import LeniaState, ring_kernel_2d
from .wgpu_mohr import webgpu_available


_DEVICE = None
_PIPELINE = None

_SHADER = """
@group(0) @binding(0) var<storage, read> rho: array<f32>;
@group(0) @binding(1) var<storage, read> kernel: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let height = u32(params[0]);
    let width = u32(params[1]);
    let radius = i32(params[2]);
    if (index >= height * width) { return; }
    let x = index % width;
    let y = index / width;
    var potential = 0.0;
    for (var ky = 0u; ky < u32(2 * radius + 1); ky = ky + 1u) {
        let yy = u32((i32(y) + i32(ky) - radius + i32(height)) % i32(height));
        for (var kx = 0u; kx < u32(2 * radius + 1); kx = kx + 1u) {
            let xx = u32((i32(x) + i32(kx) - radius + i32(width)) % i32(width));
            let kernel_index = ky * u32(2 * radius + 1) + kx;
            potential += kernel[kernel_index] * rho[yy * width + xx];
        }
    }
    let mu = params[3];
    let sigma = params[4];
    let dt = params[5];
    let growth = 2.0 * exp(-0.5 * ((potential - mu) / sigma) * ((potential - mu) / sigma)) - 1.0;
    var value = rho[index] + dt * growth;
    if (value < 0.0) { value = 0.0; }
    if (value > 1.0) { value = 1.0; }
    output[index] = value;
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


def lenia_step_wgpu(
    rho: np.ndarray,
    kernel: np.ndarray,
    mu: float,
    sigma: float,
    dt: float,
) -> np.ndarray:
    if _PIPELINE is None and not webgpu_available():
        raise RuntimeError("WebGPU is unavailable; install the webgpu extra and check the adapter")
    import wgpu

    rho = np.ascontiguousarray(rho, dtype=np.float32)
    kernel = np.ascontiguousarray(kernel, dtype=np.float32)
    if rho.ndim != 3 or rho.shape[0] != 1:
        raise ValueError("WebGPU Lenia currently requires one channel of 2D field data")
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or kernel.shape[0] % 2 != 1:
        raise ValueError("kernel must be a square 2D stencil with odd width")
    if sigma <= 0.0 or dt <= 0.0:
        raise ValueError("sigma and dt must be positive")
    height, width = rho.shape[1:]
    radius = kernel.shape[0] // 2
    if height < kernel.shape[0] or width < kernel.shape[1]:
        raise ValueError("kernel dimensions must not exceed field dimensions")

    device, pipeline = _get_pipeline()
    storage = wgpu.BufferUsage.STORAGE
    rho_buffer = device.create_buffer_with_data(data=rho, usage=storage)
    kernel_buffer = device.create_buffer_with_data(data=kernel, usage=storage)
    params = np.asarray([height, width, radius, mu, sigma, dt], dtype=np.float32)
    params_buffer = device.create_buffer_with_data(data=params, usage=storage)
    output_buffer = device.create_buffer(size=rho.nbytes, usage=storage | wgpu.BufferUsage.COPY_SRC)
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": rho_buffer},
            {"binding": 1, "resource": kernel_buffer},
            {"binding": 2, "resource": params_buffer},
            {"binding": 3, "resource": output_buffer},
        ],
    )
    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((height * width + 63) // 64)
    compute_pass.end()
    device.queue.submit([encoder.finish()])
    result = device.queue.read_buffer(output_buffer)
    return np.frombuffer(result, dtype=np.float32).reshape(rho.shape).astype(np.float64)


def step_wgpu(state: LeniaState, cfg) -> LeniaState:
    if cfg.field_cfg.get("kernel", "ring") != "ring":
        raise ValueError("WebGPU Lenia currently supports the ring kernel")
    radius = max(2, min(state.rho.shape[1:]) // 8)
    rho = lenia_step_wgpu(
        state.rho,
        ring_kernel_2d(radius),
        float(cfg.lenia["mu"]),
        float(cfg.lenia["sigma"]),
        float(cfg.lenia.get("dt", cfg.dt)),
    )
    return LeniaState(rho=rho, frame=state.frame + 1)