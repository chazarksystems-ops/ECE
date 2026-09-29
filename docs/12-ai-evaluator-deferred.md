# Co-local AI evaluator — design only

Not scheduled. Written down so the idea does not get rebuilt from
memory when M5 is actually done.

## Why it waits

A VLM cannot debug a wrong tent, a leaking MaCE gather, or a
scrambled hash table. Hand-edit the matrix until you can name a
creature without a model.

## Contract (when built)

Local multimodal model only. No network.

```
generate_multimodal(system, user, image, opts) -> text
last_embedding() -> optional [f32]
```

Stub backend returns score 0 and no deltas so the sim runs without
a model.

## Prompt shape

System: ALife critic. Score 0–1 against a target. Propose sparse
`(i,j,δ)` with `δ ∈ [-0.1, 0.1]`. JSON only.

User: target sentence, a few numeric stats (mean speed, spatial
entropy, cluster count), 256² thumbnail.

## Steering

```
v ← β v + (1-β) δ
M ← clip(M + η v, -1, 1)
```

Default `η=0.5`, `β=0.7`. Saturation decay if a cell is pushed the
same way for N evals. Reject updates that collapse `std(M)` below
0.1.

## Targets that work

Visible structure: “ring of chasers”, “stable bilayer”, “hollow
shell with a core”.

Targets that do not work: periods and counts the still frame cannot
show (“divides every 100 frames”).

## Cadence

Default every 300 frames. Adaptive: double interval on plateau,
halve on a jump. Evaluator thread separate from the GPU tick.

## Memory on Spark (if enabled later)

7B Q4 ~5 GB plus KV cache. Leaves the 256³ sim untouched. Full
frame readback is for offline analysis, not the live loop.
