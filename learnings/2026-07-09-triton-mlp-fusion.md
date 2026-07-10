# What I learned today writing GPU kernels from scratch

Spent the evening on a DGX Spark (GB10 / Blackwell SM121) with a simple goal: write a fused MLP kernel in Triton that beats cuBLAS. The target was Qwen2.5-1.5B's MLP layer — gate=SiLU(x@W_gate), up=x@W_up, out=(gate*up)@W_down. Three matmuls, a SiLU, and a multiply. Native PyTorch does this in ~0.34ms.

## Attempt 1: Naive Triton — 0.26x cuBLAS

One program per output element. 8,960 launches, each doing a scalar dot product over the hidden dimension. Each program does almost nothing, so launch overhead dominates. **4x slower.**

```python
# What NOT to do
grid = (I,)  # one program per output element
pid = tl.program_id(0)
i_idx = pid
acc = 0.0
for d in range(0, D, BLOCK_D):
    acc += tl.sum(x_tile * w_tile)
```

## Attempt 2: Tiled Triton — 1.02x cuBLAS

BLOCK_M=128, so 70 programs instead of 8,960. Each handles 128 output elements simultaneously, accumulating into a block accumulator. The autotuner picked BLOCK_D=128, BLOCK_M=128. **Triton matched cuBLAS.**

```python
# The pattern that works
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
m_offs = m_start + tl.arange(0, BLOCK_M)
acc = tl.zeros([BLOCK_M], dtype=tl.float32)
for d in range(0, D, BLOCK_D):
    x_tile = tl.load(x_ptr + d_offs, mask=d_mask, other=0.0)
    w_tile = tl.load(w_ptr + d_offs[:, None] * M + m_offs[None, :], ...)
    acc += tl.sum(x_tile[:, None] * w_tile, axis=0)
```

**Lesson: tiling is not an optimization, it's the entire game.** The difference between 0.26x and 1.02x is purely grid structure — same math, different program count.

## Attempt 3: Tiled + Fused — 0.90x cuBLAS

Reuse the input vector load across both gate and up projections in one kernel. This should save memory bandwidth (load x once instead of twice). Result: **slower than cuBLAS.**

The problem: loading both W_gate and W_up weight tiles in the same kernel doubles register pressure per program, which reduces occupancy. cuBLAS avoids this by doing them sequentially with separate launches. The launch overhead (~0.01ms per launch) is negligible compared to the occupancy loss from doubled register usage.

**Lesson: fusion is not universally good.** For batch-1 GEMV where matmuls are near-instant and bandwidth is the ceiling, the occupancy cost of dual weight loads outweighs the bandwidth savings from reusing inputs. Fusion shines where intermediates are much larger than inputs — like attention, not MLP layers.

## The real insight: you can't beat bandwidth

For batch=1 inference, GEMV operations are memory-bandwidth-bound. The math: ~27M FLOPs moving ~27MB of data = ~1 FLOP per byte. On GB10 with ~500 GB/s bandwidth, that's ~500M FLOP/s theoretical max — which cuBLAS already hits at ~0.1ms per GEMV. You can't compute your way out of a bandwidth problem. You need to move fewer bytes.

That means **quantization.**

## The NVFP4 rabbit hole

Blackwell's tensor cores have native 4-bit floating point (NVFP4) support. PyTorch 2.12 exposes `torch.float4_e2m1fn_x2` as a dtype — but doesn't wire up compute ops. The standalone `fp4-cuda-kernel` library (85-129 TFLOPS on DGX Spark) can't build because CUTLASS 3.8 lacks SM120/SM121 arch tags.

The only working path right now: [vLLM-Moet](https://github.com/kacper-daftcode/vllm-Moet) by Kacper Daftcode — hand-written SM120 SASS kernels using a reverse-engineered ISA database and custom assembler. This is literally assembly-level GPU programming below PTX. DeepSeek V4 Flash at 38.5 tok/s on a single RTX 5090.

The stack I wanted:

```
Triton → CUTLASS → NVFP4
```

The stack that exists:

```
reverse-engineered ISA → custom assembler → SASS kernels
```

Bleeding edge in the literal sense.

## What I'd tell someone starting this tomorrow

**Step 1:** Write the tiled Triton GEMV kernel (100 lines, autotuned). You'll learn more in that hour about GPU memory hierarchies, occupancy, and launch overhead than in a week of reading papers. The code is at `scripts/tiled_gemv_bench.py`.

**Step 2:** Profile with `ncu` and `nsys`. The PyTorch profiler shows what's slow; Nsight Compute shows WHY. On GB10 it's always memory bandwidth.

**Step 3:** Check if NVFP4 tooling has caught up. If CUTLASS adds SM120 tags or PyTorch wires up `float4_e2m1fn_x2` compute ops, you get 2x inference speedup for free. If not, you now understand why it's hard — and you have the profiling infrastructure to measure the moment it becomes available.

## Concrete artifacts

| Artifact | Path | What it does |
|----------|------|-------------|
| Kernel profiler | `scripts/kernel_profiler.py` | Load model, benchmark, profile with PyTorch/ncu/nsys |
| Fused MLP kernel | `scripts/fused_qwen_mlp.py` | Autotuned tiled+fused Triton MLP for Qwen2 |
| Tiled GEMV benchmark | `scripts/tiled_gemv_bench.py` | Standalone tiled vs cuBLAS comparison |
| Skill (reference) | `mlops/gpu-kernel-recompilation` | Full pipeline, patterns, pitfalls for future sessions |

## Key numbers

| Metric | Value |
|--------|-------|
| Qwen2.5-1.5B throughput (fp16) | 50 tok/s |
| Single GEMV (1536×8960, fp16) | 0.097ms (cuBLAS) |
| Tiled Triton vs cuBLAS | 1.02x |
| Kernel launches per generation | 225,456 |
| NVFP4 theoretical speedup | 1.3-2.3x (when tooling arrives) |
| NVFP4 memory savings | 4x vs fp16 |
| GB10 memory bandwidth | ~500 GB/s |
