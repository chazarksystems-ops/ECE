# DGX Spark notes

128 GB unified memory is a **capacity** win. GB10-class bandwidth
is not an H100. Design for locality, mixed precision, and graphs.
Do not plan interactive `1024³ × 64` channels.

## Honest first budget

| Item | Size |
|---|---|
| 256² × 8 ch × 2 buffers f32 | ~4 MB |
| 256³ × 8 ch × 2 buffers f32 | ~1 GB |
| 4 M particles × 32 B | ~128 MB |
| cuFFT scratch (256³) | hundreds of MB |
| 7B Q4 VLM (later) | ~5 GB |

512³ × 16 is a research config, not the default.

## What unified memory is for

- Host-side clustering / BUNCH / graph tools reading particle
  buffers without a PCIe staging stall.
- Dumping replay frames to NVMe while the sim runs.
- Co-locating an evaluator model later.

It does not make 3D convolution cheap.

## Stack split

| Work | Tool |
|---|---|
| Field FFT ≥ radius ~6 in 3D | CUDA / cuFFT / CuPy |
| 2D preview, UI | wgpu |
| Oracle / CI | Python `ece/` |

Dawn-on-Spark is extra risk. Use CUDA on that box for the heavy
field path.

## 3D memory warning

| Grid | One f32 volume |
|---|---|
| 256³ | 64 MB |
| 512³ | 512 MB |
| 1024³ | 4 GB |

Double-buffer × channels × scratch multiplies fast. Cap channels
and resolution independently in config.

## Bandwidth habits

- bf16 / fp16 for convolution payloads; keep MaCE mass in f32.
- Spatial hash with Morton order if the force kernel is cache-cold.
- CUDA graphs once the pass list is stable.
- Persistent kernels only after M1 is correct.
