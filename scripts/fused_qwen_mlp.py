"""
Fused MLP Triton kernels for Qwen2 — with proper tiling.
Combines memory fusion (reusing x loads) with tiled GEMV (matching cuBLAS).

Qwen2 MLP: gate=SiLU(x@W_gate), up=x@W_up, out=(gate*up)@W_down

Optimization: fused gate+up kernel reuses x tile loads, cutting the
memory bandwidth for the input vector in half (load x once, not twice).
Tiled at BLOCK_M=128 to match cuBLAS launch efficiency.

Usage:
    python scripts/fused_qwen_mlp.py --model /mnt/data/models/qwen2.5-1.5b-instruct
"""
import time
import torch
import triton
import triton.language as tl


# ── TILED fused gate+up ─────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 64, 'BLOCK_M': 64}),
        triton.Config({'BLOCK_D': 64, 'BLOCK_M': 128}),
        triton.Config({'BLOCK_D': 128, 'BLOCK_M': 64}),
        triton.Config({'BLOCK_D': 128, 'BLOCK_M': 128}),
    ],
    key=['D', 'I'],
)
@triton.jit
def _fused_gate_up_tiled_kernel(
    x_ptr,
    w_gate_ptr,
    w_up_ptr,
    gate_out_ptr,
    up_out_ptr,
    D: tl.constexpr,
    I: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Fused tiled GEMV: each program computes BLOCK_M gate[i] and up[i] values."""
    pid = tl.program_id(0)
    i_start = pid * BLOCK_M
    i_offs = i_start + tl.arange(0, BLOCK_M)
    i_mask = i_offs < I

    # Accumulators for gate and up: [BLOCK_M]
    gate_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    up_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        # Load x tile [BLOCK_D] — SHARED between gate and up
        x_tile = tl.load(x_ptr + d_offs, mask=d_mask, other=0.0)

        # Load weight tiles [BLOCK_D, BLOCK_M]
        wg_tile = tl.load(
            w_gate_ptr + d_offs[:, None] * I + i_offs[None, :],
            mask=d_mask[:, None] & i_mask[None, :], other=0.0,
        )
        wu_tile = tl.load(
            w_up_ptr + d_offs[:, None] * I + i_offs[None, :],
            mask=d_mask[:, None] & i_mask[None, :], other=0.0,
        )

        # Accumulate: gate[i] += sum_d(x[d] * w_gate[d,i])
        gate_acc += tl.sum(
            x_tile[:, None].to(tl.float32) * wg_tile.to(tl.float32), axis=0)
        up_acc += tl.sum(
            x_tile[:, None].to(tl.float32) * wu_tile.to(tl.float32), axis=0)

    # Apply SiLU to gate: gated = gate * sigmoid(gate)
    gate_silu = gate_acc * tl.sigmoid(gate_acc)

    # Store results
    tl.store(gate_out_ptr + i_offs, gate_silu.to(tl.float16), mask=i_mask)
    tl.store(up_out_ptr + i_offs, up_acc.to(tl.float16), mask=i_mask)


# ── TILED gated × down ──────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_I': 64, 'BLOCK_D': 64}),
        triton.Config({'BLOCK_I': 64, 'BLOCK_D': 128}),
        triton.Config({'BLOCK_I': 128, 'BLOCK_D': 64}),
        triton.Config({'BLOCK_I': 128, 'BLOCK_D': 128}),
    ],
    key=['I', 'D'],
)
@triton.jit
def _gated_down_tiled_kernel(
    gated_ptr,
    w_down_ptr,
    out_ptr,
    I: tl.constexpr,
    D: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Tiled GEMV: out = gated @ W_down. Each program handles BLOCK_D output elements."""
    pid = tl.program_id(0)
    d_start = pid * BLOCK_D
    d_offs = d_start + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for i_start in range(0, I, BLOCK_I):
        i_offs = i_start + tl.arange(0, BLOCK_I)
        i_mask = i_offs < I

        # Load gated tile [BLOCK_I]
        g_tile = tl.load(gated_ptr + i_offs, mask=i_mask, other=0.0)

        # Load weight tile [BLOCK_I, BLOCK_D]
        w_tile = tl.load(
            w_down_ptr + i_offs[:, None] * D + d_offs[None, :],
            mask=i_mask[:, None] & d_mask[None, :], other=0.0,
        )

        acc += tl.sum(
            g_tile[:, None].to(tl.float32) * w_tile.to(tl.float32), axis=0)

    tl.store(out_ptr + d_offs, acc.to(tl.float16), mask=d_mask)


# ── Python API ──────────────────────────────────────────────────────

def fused_qwen_mlp(
    hidden_states: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> torch.Tensor:
    """Tiled fused Qwen2 MLP. Matches cuBLAS speed while reusing x loads."""
    x = hidden_states.squeeze(0).contiguous()
    D, I = x.shape[0], gate_proj.shape[1]

    # Stage 1: fused gate+up (reuses x loads — loads x once, not twice)
    gate_out = torch.empty(I, dtype=x.dtype, device=x.device)
    up_out = torch.empty(I, dtype=x.dtype, device=x.device)
    grid1 = lambda meta: (triton.cdiv(I, meta['BLOCK_M']),)
    _fused_gate_up_tiled_kernel[grid1](x, gate_proj, up_proj, gate_out, up_out, D, I)

    # Element-wise: gated = silu(gate) * up  (gate already has SiLU applied)
    gated = gate_out * up_out

    # Stage 2: gated × down
    out = torch.empty(D, dtype=x.dtype, device=x.device)
    grid2 = lambda meta: (triton.cdiv(D, meta['BLOCK_D']),)
    _gated_down_tiled_kernel[grid2](gated, down_proj, out, I, D)

    return out.unsqueeze(0)


def native_qwen_mlp(hidden_states, gate_proj, up_proj, down_proj):
    """Native PyTorch reference."""
    import torch.nn.functional as F
    gate = F.silu(hidden_states @ gate_proj)
    up = hidden_states @ up_proj
    return (gate * up) @ down_proj


# ── Benchmark ───────────────────────────────────────────────────────

def benchmark_vs_native(model_path: str, n_warmup: int = 20, n_runs: int = 200):
    """Load model, extract MLP weights, compare native vs tiled-fused."""
    from transformers import AutoModelForCausalLM

    print(f"Loading {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16,
        device_map="cuda:0", trust_remote_code=True,
    )
    model.eval()

    mlp = model.model.layers[0].mlp
    gate_proj = mlp.gate_proj.weight.data.T.contiguous()
    up_proj = mlp.up_proj.weight.data.T.contiguous()
    down_proj = mlp.down_proj.weight.data.T.contiguous()

    D, I = gate_proj.shape
    print(f"  Hidden={D}, Intermediate={I}")

    torch.manual_seed(42)
    x = torch.randn(1, D, dtype=torch.float16, device="cuda:0")

    def time_it(name, fn):
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        avg = sum(times) / len(times) * 1000
        best = min(times) * 1000
        return avg, best

    navg, nbest = time_it("Native", lambda: native_qwen_mlp(x, gate_proj, up_proj, down_proj))
    favg, fbest = time_it("Fused ", lambda: fused_qwen_mlp(x, gate_proj, up_proj, down_proj))

    native_out = native_qwen_mlp(x, gate_proj, up_proj, down_proj)
    fused_out = fused_qwen_mlp(x, gate_proj, up_proj, down_proj)
    error = (native_out - fused_out).abs().max().item()

    print(f"\n{'='*60}")
    print(f"Qwen2 MLP — cuBLAS vs Tiled+Fused Triton")
    print(f"{'='*60}")
    print(f"  {'Native cuBLAS':<22} {navg:>8.4f} ms (best {nbest:.4f})")
    print(f"  {'Tiled Fused Triton':<22} {favg:>8.4f} ms (best {fbest:.4f})")
    print(f"  {'Speedup':<22} {navg/favg:>8.2f}x")
    print(f"  {'Error':<22} {error:>8.6f}")

    grids = {}
    if hasattr(_fused_gate_up_tiled_kernel, 'best_config'):
        cfg = _fused_gate_up_tiled_kernel.best_config
        grids['gate+up'] = f"BLOCK_D={cfg.kwargs['BLOCK_D']}, BLOCK_M={cfg.kwargs['BLOCK_M']}"
    if hasattr(_gated_down_tiled_kernel, 'best_config'):
        cfg = _gated_down_tiled_kernel.best_config
        grids['down'] = f"BLOCK_I={cfg.kwargs['BLOCK_I']}, BLOCK_D={cfg.kwargs['BLOCK_D']}"
    if grids:
        print(f"  {'Autotuner':<22} gate+up: {grids.get('gate+up','?')}")
        if 'down' in grids:
            print(f"  {'':<22} down: {grids['down']}")

    return {"native_ms": navg, "fused_ms": favg, "speedup": navg/favg, "error": error}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--runs", type=int, default=200)
    args = p.parse_args()
    benchmark_vs_native(args.model, n_runs=args.runs)
