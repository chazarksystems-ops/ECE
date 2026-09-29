"""WebGPU Particle Lenia using the energy-gradient rule."""

from __future__ import annotations

import numpy as np

from .particle_lenia import ParticleLeniaState, seed_state, shell_kernel_weight
from .wgpu_mohr import webgpu_available


_DEVICE = None
_PIPELINE = None

_SHADER = """
@group(0) @binding(0) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read_write> next_positions: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> velocities: array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let count = u32(params[0]);
    if (i >= count) { return; }
    let world = vec2<f32>(params[1], params[2]);
    let kernel_mu = params[3];
    let kernel_sigma = params[4];
    let growth_mu = params[5];
    let growth_sigma = params[6];
    let c_rep = params[7];
    let kernel_weight = params[8];
    let dt = params[9];
    var density = 0.0;
    var grad_density = vec2<f32>(0.0, 0.0);
    var repulsion = vec2<f32>(0.0, 0.0);
    for (var j = 0u; j < count; j = j + 1u) {
        var delta = positions[i] - positions[j];
        delta = delta - world * round(delta / world);
        let distance = length(delta);
        let radial = (distance - kernel_mu) / kernel_sigma;
        let kernel = kernel_weight * exp(-(radial * radial));
        density += kernel;
        if (distance > 1e-6) {
            let derivative = -2.0 * (distance - kernel_mu) / (kernel_sigma * kernel_sigma) * kernel;
            grad_density += derivative * delta / distance;
            if (distance < 1.0) {
                repulsion += c_rep * (1.0 - distance) * delta / distance;
            }
        }
    }
    let growth = exp(-((density - growth_mu) / growth_sigma) * ((density - growth_mu) / growth_sigma));
    let growth_derivative = -2.0 * (density - growth_mu) / (growth_sigma * growth_sigma) * growth;
    let velocity = repulsion + growth_derivative * grad_density;
    let next_position = positions[i] + dt * velocity;
    next_positions[i] = next_position - world * floor(next_position / world);
    velocities[i] = velocity;
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


def step_wgpu(state: ParticleLeniaState, cfg) -> ParticleLeniaState:
    if _PIPELINE is None and not webgpu_available():
        raise RuntimeError("WebGPU is unavailable; install the webgpu extra and check the adapter")
    import wgpu

    positions = np.ascontiguousarray(state.pos, dtype=np.float32)
    params = cfg.particle_lenia
    kernel_weight = shell_kernel_weight(float(params["kernel_mu"]), float(params["kernel_sigma"]))
    uniform = np.asarray(
        [
            len(positions), cfg.world[0], cfg.world[1],
            params["kernel_mu"], params["kernel_sigma"], params["growth_mu"],
            params["growth_sigma"], params["c_rep"], kernel_weight,
            params.get("dt", cfg.dt),
        ],
        dtype=np.float32,
    )
    device, pipeline = _get_pipeline()
    storage = wgpu.BufferUsage.STORAGE
    pos_buffer = device.create_buffer_with_data(data=positions, usage=storage)
    params_buffer = device.create_buffer_with_data(data=uniform, usage=storage)
    next_buffer = device.create_buffer(size=positions.nbytes, usage=storage | wgpu.BufferUsage.COPY_SRC)
    velocity_buffer = device.create_buffer(size=positions.nbytes, usage=storage | wgpu.BufferUsage.COPY_SRC)
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": pos_buffer},
            {"binding": 1, "resource": params_buffer},
            {"binding": 2, "resource": next_buffer},
            {"binding": 3, "resource": velocity_buffer},
        ],
    )
    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((len(positions) + 63) // 64)
    compute_pass.end()
    device.queue.submit([encoder.finish()])
    new_pos = np.frombuffer(device.queue.read_buffer(next_buffer), dtype=np.float32).reshape(positions.shape)
    velocity = np.frombuffer(device.queue.read_buffer(velocity_buffer), dtype=np.float32).reshape(positions.shape)
    return ParticleLeniaState(
        pos=new_pos.astype(np.float64),
        vel=velocity.astype(np.float64),
        types=state.types,
        frame=state.frame + 1,
    )


def run_wgpu(cfg, frames: int) -> ParticleLeniaState:
    if frames < 0:
        raise ValueError("frames must be non-negative")
    state = seed_state(cfg)
    for _ in range(frames):
        state = step_wgpu(state, cfg)
    return state