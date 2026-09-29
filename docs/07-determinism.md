# Determinism

BYTES_ARE_LAW applies to replay logs, not to the first interactive
preview.

## Why GPU floats are not enough

Vendors fuse FMA differently, approximate `exp`/`sin` differently,
and schedule reductions in different orders. Two GPUs will diverge.

## Preview vs replay

| Path | Arithmetic | Promise |
|---|---|---|
| Python oracle | IEEE f64, fixed loops | tests match on one machine |
| wgpu preview | f32 WGSL | looks right, not a log |
| Python fixed-point Mohr prototype | Q16.16 position, Q8.24 velocity, Q1.15 matrix | repeatable for identical quantized input |
| CUDA fixed-point (future) | integer kernels and fixed LUTs | cross-box replay; not implemented |

## Current prototype boundary

`ece.fixedpoint_mohr` stores persistent state as integers and uses integer
distance, force, and integration operations. Friction decay is quantized
using decimal arithmetic. This is a CPU replay prototype, not a CUDA backend
or a cross-box certification. Serialize the integer state to replay an exact
starting condition.

## Target persistent formats

| Quantity | Format |
|---|---|
| Position | Q16.16 |
| Velocity | Q8.24 |
| Matrix | Q1.15 |
| Field mass | Q16.16 |

The future CUDA port must replace transcendental functions with versioned,
host-generated LUTs and keep neighborhood traversal order fixed.

## State hash

xxHash3 (or BLAKE3) over:

```
frame u64 LE
particles  N × (i32 x, i32 y, [i32 z], u32 type)
field      C × X × Y × [Z] × i32   (if present)
matrix     K² × i16
```

Derived buffers are functions of that state and are not hashed.

## Tests that exist today

Python and same-device replay invariants live in `tests/`. They lock integer
state repeatability, not cross-vendor GPU execution. A reproducibility
registry awaits the CUDA fixed-point port.
