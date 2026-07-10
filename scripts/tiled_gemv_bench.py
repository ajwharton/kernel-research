"""
Comparison: properly tiled Triton GEMV vs cuBLAS.
Demonstrates that Triton CAN match cuBLAS when tiled correctly.
"""
import time
import torch
import triton
import triton.language as tl


# ── Properly tiled Triton GEMV ──────────────────────────────────────
# Instead of 1 program per output element (8960 launches),
# we use BLOCK_M programs each handling BLOCK_M output elements.
# This reduces launches from 8960 to ~140 and amortizes the overhead.

@triton.jit
def _tiled_gemv_kernel(
    x_ptr,           # input vector [D]
    w_ptr,           # weight matrix [D, M]
    out_ptr,         # output vector [M]
    D: tl.constexpr,
    M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Tiled GEMV: out = x @ W. Each program handles BLOCK_M output elements."""
    pid = tl.program_id(0)
    m_start = pid * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M

    # Accumulate partial dot products: [BLOCK_M]
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        # Load x tile [BLOCK_D]
        x_tile = tl.load(x_ptr + d_offs, mask=d_mask, other=0.0)

        # Load weight tile [BLOCK_D, BLOCK_M]
        # w[d_offs, m_offs] = w_ptr + d_offs * M + m_offs
        w_tile = tl.load(
            w_ptr + d_offs[:, None] * M + m_offs[None, :],
            mask=d_mask[:, None] & m_mask[None, :],
            other=0.0,
        )

        # Accumulate: acc[m] += sum_d(x[d] * w[d, m])
        acc += tl.sum(x_tile[:, None].to(tl.float32) * w_tile.to(tl.float32), axis=0)

    # Store results
    tl.store(out_ptr + m_offs, acc.to(tl.float16), mask=m_mask)


def tiled_gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Wrapper for tiled Triton GEMV."""
    D = x.shape[0]
    M = w.shape[1]
    out = torch.empty(M, dtype=x.dtype, device=x.device)

    # Tune block sizes for GB10 (Blackwell, compute 12.1)
    BLOCK_D = 64
    BLOCK_M = 64
    grid = (triton.cdiv(M, BLOCK_M),)

    _tiled_gemv_kernel[grid](x, w, out, D, M, BLOCK_D, BLOCK_M)
    return out


# ── Autotuned version ───────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 32, 'BLOCK_M': 32}),
        triton.Config({'BLOCK_D': 32, 'BLOCK_M': 64}),
        triton.Config({'BLOCK_D': 64, 'BLOCK_M': 32}),
        triton.Config({'BLOCK_D': 64, 'BLOCK_M': 64}),
        triton.Config({'BLOCK_D': 64, 'BLOCK_M': 128}),
        triton.Config({'BLOCK_D': 128, 'BLOCK_M': 32}),
        triton.Config({'BLOCK_D': 128, 'BLOCK_M': 64}),
        triton.Config({'BLOCK_D': 128, 'BLOCK_M': 128}),
    ],
    key=['D', 'M'],
)
@triton.jit
def _autotuned_gemv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    D: tl.constexpr,
    M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Autotuned tiled GEMV — Triton picks the best block sizes."""
    pid = tl.program_id(0)
    m_start = pid * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        x_tile = tl.load(x_ptr + d_offs, mask=d_mask, other=0.0)
        w_tile = tl.load(
            w_ptr + d_offs[:, None] * M + m_offs[None, :],
            mask=d_mask[:, None] & m_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(x_tile[:, None].to(tl.float32) * w_tile.to(tl.float32), axis=0)

    tl.store(out_ptr + m_offs, acc.to(tl.float16), mask=m_mask)


def autotuned_gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Autotuned GEMV — Triton benchmarks block sizes and picks best."""
    D = x.shape[0]
    M = w.shape[1]
    out = torch.empty(M, dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
    _autotuned_gemv_kernel[grid](x, w, out, D, M)
    return out


# ── Benchmark ───────────────────────────────────────────────────────

def benchmark():
    torch.manual_seed(42)
    D = 1536   # Qwen2.5-1.5B hidden dim
    M = 8960   # intermediate dim

    x = torch.randn(D, dtype=torch.float16, device="cuda:0")
    w = torch.randn(D, M, dtype=torch.float16, device="cuda:0")

    n_warmup = 20
    n_runs = 200

    def bench(name, fn):
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
        p95 = sorted(times)[int(len(times) * 0.95)] * 1000
        return avg, best, p95

    # cuBLAS baseline
    native_fn = lambda: x @ w
    navg, nbest, np95 = bench("cuBLAS GEMV", native_fn)

    # Naive Triton (1 program per element — our first attempt)
    naive_fn = lambda: tiled_gemv(torch.ones(1, dtype=torch.float16, device="cuda:0"), torch.ones(1, 1, dtype=torch.float16, device="cuda:0"))  # dummy, just showing concept
    # Tiled Triton
    tav, tb, tp = bench("Tiled Triton", lambda: tiled_gemv(x, w))

    # Autotuned Triton
    aav, ab, ap = bench("Autotuned Triton", lambda: autotuned_gemv(x, w))

    # Correctness
    native_out = x @ w
    tiled_out = tiled_gemv(x, w)
    auto_out = autotuned_gemv(x, w)
    tiled_err = (native_out - tiled_out).abs().max().item()
    auto_err = (native_out - auto_out).abs().max().item()

    print(f"GEMV: {D} × {M} (fp16) on {torch.cuda.get_device_name()}")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'Avg (ms)':>10} {'Best (ms)':>10} {'Speedup':>10}")
    print(f"{'-'*55}")
    print(f"{'cuBLAS':<25} {navg:>10.4f} {nbest:>10.4f} {'1.00x':>10}")
    print(f"{'Triton tiled':<25} {tav:>10.4f} {tb:>10.4f} {f'{navg/tav:.2f}x':>10}")
    print(f"{'Triton autotuned':<25} {aav:>10.4f} {ab:>10.4f} {f'{navg/aav:.2f}x':>10}")
    print(f"\nCorrectness vs cuBLAS:")
    print(f"  Tiled error:     {tiled_err:.6f}")
    print(f"  Autotuned error: {auto_err:.6f}")

    # Show which config Triton picked
    cfg = _autotuned_gemv_kernel.best_config
    if cfg:
        print(f"\nAutotuner picked: BLOCK_D={cfg.kwargs['BLOCK_D']}, BLOCK_M={cfg.kwargs['BLOCK_M']}")
        print(f"  Grid size: {triton.cdiv(M, cfg.kwargs['BLOCK_M'])} programs")


if __name__ == "__main__":
    benchmark()
