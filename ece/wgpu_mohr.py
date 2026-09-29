"""Optional WebGPU implementation of the corrected 2D Mohr force kernel."""

from __future__ import annotations

import numpy as np

from .config import SimConfig
from .integrate import symplectic_euler
from .step_mohr import MohrState


_DEVICE = None
_PIPELINE = None

_SHADER = """
@group(0) @binding(0) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> types: array<u32>;
@group(0) @binding(2) var<storage, read> matrix: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<f32>;
@group(0) @binding(4) var<storage, read_write> accelerations: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> ranges: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read> sorted_indices: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = u32(params[0]);
    let species_count = u32(params[1]);
    if (i >= n) { return; }
    let r_max = params[2];
    let beta = params[3];
    let gain = params[4];
    let world = vec2<f32>(params[5], params[6]);
    let origin = positions[i];
    var accel = vec2<f32>(0.0, 0.0);
    if (params[7] < 0.5) {
        for (var j = 0u; j < n; j = j + 1u) {
            if (j == i) { continue; }
            var delta = positions[j] - origin;
            delta = delta - world * round(delta / world);
            let distance = length(delta);
            if (distance < 1e-6 || distance > r_max) { continue; }
            let r = distance / r_max;
            let a = matrix[types[i] * species_count + types[j]];
            var force = 0.0;
            if (r < beta) {
                force = r / beta - 1.0;
            } else {
                force = a * (1.0 - abs(2.0 * r - 1.0 - beta) / (1.0 - beta));
            }
            accel = accel + gain * force * delta / distance;
        }
    } else {
        let grid_x = u32(params[8]);
        let grid_y = u32(params[9]);
        let width_x = params[10];
        let width_y = params[11];
        let radius = i32(params[12]);
        let bx = i32(floor(origin.x / width_x));
        let by = i32(floor(origin.y / width_y));
        for (var dy = -radius; dy <= radius; dy = dy + 1) {
            for (var dx = -radius; dx <= radius; dx = dx + 1) {
                let nx = u32((bx + dx + i32(grid_x)) % i32(grid_x));
                let ny = u32((by + dy + i32(grid_y)) % i32(grid_y));
                let cell_range = ranges[ny * grid_x + nx];
                for (var slot = cell_range.x; slot < cell_range.y; slot = slot + 1u) {
                    let j = sorted_indices[slot];
                    if (j == i) { continue; }
                    var delta = positions[j] - origin;
                    delta = delta - world * round(delta / world);
                    let distance = length(delta);
                    if (distance < 1e-6 || distance > r_max) { continue; }
                    let r = distance / r_max;
                    let a = matrix[types[i] * species_count + types[j]];
                    var force = 0.0;
                    if (r < beta) {
                        force = r / beta - 1.0;
                    } else {
                        force = a * (1.0 - abs(2.0 * r - 1.0 - beta) / (1.0 - beta));
                    }
                    accel = accel + gain * force * delta / distance;
                }
            }
        }
    }
    accelerations[i] = accel;
}
"""


def webgpu_available() -> bool:
    try:
        import wgpu

        return wgpu.gpu.request_adapter_sync(power_preference="high-performance") is not None
    except Exception:
        return False


def _get_pipeline():
    global _DEVICE, _PIPELINE
    if _PIPELINE is None:
        import wgpu

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError("no WebGPU adapter is available")
        _DEVICE = adapter.request_device_sync()
        shader = _DEVICE.create_shader_module(code=_SHADER)
        _PIPELINE = _DEVICE.create_compute_pipeline(
            layout="auto",
            compute={"module": shader, "entry_point": "main"},
        )
    return _DEVICE, _PIPELINE


def mohr_accelerations_wgpu(
    pos: np.ndarray,
    types: np.ndarray,
    matrix: np.ndarray,
    r_max: float,
    beta: float,
    world: np.ndarray,
    gain: float = 1.0,
    hash_cfg: dict | None = None,
) -> np.ndarray:
    """Compute dense 2D Mohr accelerations on WebGPU."""
    if _PIPELINE is None and not webgpu_available():
        raise RuntimeError("WebGPU is unavailable; install the webgpu extra and check the adapter")
    import wgpu

    pos = np.ascontiguousarray(pos, dtype=np.float32)
    types = np.ascontiguousarray(types, dtype=np.uint32)
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    world = np.ascontiguousarray(world, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("WebGPU Mohr currently requires positions with shape (N, 2)")
    if types.shape != (len(pos),) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("types and matrix must have compatible shapes")
    if np.any(types >= matrix.shape[0]):
        raise ValueError("types must index rows and columns of matrix")
    if world.shape != (2,) or np.any(world <= 0.0):
        raise ValueError("world must contain two positive lengths")
    if r_max <= 0.0 or not 0.0 < beta < 1.0:
        raise ValueError("r_max must be positive and beta must be in (0, 1)")
    if len(pos) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    device, pipeline = _get_pipeline()
    storage = wgpu.BufferUsage.STORAGE
    input_buffers = [
        device.create_buffer_with_data(data=data, usage=storage)
        for data in (pos, types, matrix)
    ]
    if hash_cfg:
        from .bins import snapshot_and_scatter

        grid = tuple(int(size) for size in hash_cfg["grid"])
        tables = snapshot_and_scatter(pos.astype(np.float64), float(hash_cfg["cell"]), grid, world)
        ranges = np.ascontiguousarray(tables["ranges"], dtype=np.uint32)
        sorted_indices = np.ascontiguousarray(tables["sorted_indices"], dtype=np.uint32)
        widths = world / np.asarray(grid, dtype=np.float32)
        use_bins = 1.0
        radius = int(hash_cfg.get("neighborhood", 3)) // 2
    else:
        grid = (1, 1)
        ranges = np.zeros((1, 2), dtype=np.uint32)
        sorted_indices = np.zeros(1, dtype=np.uint32)
        widths = np.ones(2, dtype=np.float32)
        use_bins = 0.0
        radius = 0
    params = np.asarray(
        [len(pos), matrix.shape[0], r_max, beta, gain, world[0], world[1], use_bins,
         grid[0], grid[1], widths[0], widths[1], radius],
        dtype=np.float32,
    )
    params_buffer = device.create_buffer_with_data(data=params, usage=storage)
    ranges_buffer = device.create_buffer_with_data(data=ranges, usage=storage)
    indices_buffer = device.create_buffer_with_data(data=sorted_indices, usage=storage)
    output_buffer = device.create_buffer(
        size=pos.nbytes,
        usage=storage | wgpu.BufferUsage.COPY_SRC,
    )
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": index, "resource": buffer}
            for index, buffer in enumerate(
                [*input_buffers, params_buffer, output_buffer, ranges_buffer, indices_buffer]
            )
        ],
    )
    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((len(pos) + 63) // 64)
    compute_pass.end()
    device.queue.submit([encoder.finish()])
    result = device.queue.read_buffer(output_buffer)
    return np.frombuffer(result, dtype=np.float32).reshape(pos.shape).astype(np.float64)


def step_wgpu(state: MohrState, cfg: SimConfig) -> MohrState:
    accel = mohr_accelerations_wgpu(
        state.pos,
        state.types,
        cfg.matrix,
        float(cfg.mohr["r_max"]),
        float(cfg.mohr["beta"]),
        cfg.world,
        float(cfg.mohr.get("gain", 1.0)),
        cfg.hash,
    )
    pos, vel = symplectic_euler(
        state.pos,
        state.vel,
        accel,
        cfg.dt,
        float(cfg.mohr.get("lambda", 0.0)),
        cfg.world,
    )
    return MohrState(pos=pos, vel=vel, types=state.types, frame=state.frame + 1)