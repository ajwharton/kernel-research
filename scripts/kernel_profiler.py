"""
Kernel profiling harness for GPU kernel recompilation experiments.
Loads a small model (Qwen2.5-1.5B), runs inference benchmarks,
and collects profiling data for kernel optimization.

Usage on forge:
    # Quick benchmark (no profiler)
    python scripts/kernel_profiler.py --model /mnt/data/models/qwen2.5-1.5b-instruct

    # With PyTorch profiler (shows Triton kernel names)
    python scripts/kernel_profiler.py --model ... --profile

    # With Nsight Compute (kernel-level hardware counters)
    ncu --set full -o /tmp/kernel_profile \
        python scripts/kernel_profiler.py --model ... --ncu-mode

    # With Nsight Systems (timeline, memory transfers)
    nsys profile -o /tmp/kernel_timeline \
        python scripts/kernel_profiler.py --model ...
"""
import argparse
import time
import json
import os
import sys
from pathlib import Path
from contextlib import nullcontext

import torch


# ── Model loading ───────────────────────────────────────────────────

def load_model(model_path: str):
    """Load model + tokenizer. Returns (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_path}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0",  # explicit — never "auto" on GB10
        trust_remote_code=True,
    )
    model.eval()

    elapsed = time.time() - t0
    params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded {params:.2f}B params in {elapsed:.1f}s")
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

    return model, tokenizer


# ── Benchmarking ────────────────────────────────────────────────────

def run_benchmark(model, tokenizer, prompt: str, n_warmup: int = 3,
                  n_runs: int = 10, max_new_tokens: int = 128,
                  use_compile: bool = False):
    """Run inference benchmark. Returns timing stats dict."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    # Optional torch.compile (uses Triton backend)
    if use_compile:
        print("  Compiling model with torch.compile (Triton backend)...")
        model = torch.compile(model, mode="reduce-overhead")

    # Warmup
    print(f"  Warming up ({n_warmup} runs)...")
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    # Timed runs
    print(f"  Benchmarking ({n_runs} runs)...")
    torch.cuda.synchronize()
    times = []
    for i in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    output_tokens = output.shape[1] - input_len
    avg_time = sum(times) / len(times)
    tok_per_sec = output_tokens / avg_time
    best_time = min(times)
    best_tok_per_sec = output_tokens / best_time
    p95_time = sorted(times)[int(len(times) * 0.95)]

    return {
        "input_tokens": input_len,
        "output_tokens": output_tokens,
        "avg_time_s": round(avg_time, 4),
        "best_time_s": round(best_time, 4),
        "p95_time_s": round(p95_time, 4),
        "avg_tok_per_sec": round(tok_per_sec, 1),
        "best_tok_per_sec": round(best_tok_per_sec, 1),
        "n_runs": n_runs,
        "n_warmup": n_warmup,
        "compile": use_compile,
    }


# ── Profiling ───────────────────────────────────────────────────────

def run_pytorch_profiler(model, tokenizer, prompt: str,
                         max_new_tokens: int = 64):
    """Profile with PyTorch profiler — shows Triton kernel names."""
    from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup once
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                          pad_token_id=tokenizer.eos_token_id)

    out_dir = "/tmp/kernel_profiles"
    os.makedirs(out_dir, exist_ok=True)
    trace_path = os.path.join(out_dir, "pytorch_trace")

    print(f"  Profiling with PyTorch profiler...")
    print(f"  Trace → {trace_path}.json")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=3),
        on_trace_ready=tensorboard_trace_handler(trace_path),
        with_stack=True,
        profile_memory=True,
    ) as prof:
        for _ in range(6):
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            prof.step()

    # Print top kernels by CUDA time
    print("\n  Top 10 CUDA kernels by time:")
    events = [e for e in prof.key_averages() if getattr(e, 'cuda_time_total', 0) > 0 or getattr(e, 'self_cuda_time_total', 0) > 0]
    events.sort(key=lambda e: getattr(e, 'cuda_time_total', getattr(e, 'self_cuda_time_total', 0)), reverse=True)
    for i, e in enumerate(events[:10]):
        cuda_t = getattr(e, 'cuda_time_total', getattr(e, 'self_cuda_time_total', 0))
        print(f"    {i+1}. {e.key[:80]:80s} | {cuda_t/1000:8.1f} ms | {e.count:5d} calls")

    return trace_path


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GPU kernel profiling harness for kernel recompilation experiments")
    parser.add_argument("--model", required=True,
                        help="Path to model (HuggingFace format)")
    parser.add_argument("--profile", action="store_true",
                        help="Run PyTorch profiler (shows Triton kernel names)")
    parser.add_argument("--ncu-mode", action="store_true",
                        help="Minimal output for Nsight Compute profiling")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile (Triton backend)")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max new tokens to generate")
    parser.add_argument("--prompt", default=None,
                        help="Custom prompt (default: strength coaching prompt)")
    parser.add_argument("--output", default=None,
                        help="Save benchmark results to JSON")
    args = parser.parse_args()

    # Default prompt — strength coaching (ties to Mia domain)
    if args.prompt is None:
        args.prompt = (
            "You are a strength coach. The athlete just completed "
            "Back Squat: 225x5 @ RPE 7, then 230x5 @ RPE 7.5, "
            "then 235x5 @ RPE 8. They feel solid. What's the next set? "
            "Recommend weight, reps, and reasoning."
        )

    model, tokenizer = load_model(args.model)

    if args.ncu_mode:
        # Minimal mode for Nsight Compute — one clean run
        print("NCU mode — single clean inference pass")
        inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        result_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Output: {result_text[:200]}...")
        return

    if args.profile:
        trace = run_pytorch_profiler(model, tokenizer, args.prompt,
                                     max_new_tokens=args.max_tokens)
        print(f"\nTo view: tensorboard --logdir /tmp/kernel_profiles/")
        print(f"Or open chrome://tracing with the .json file")
        return

    # Standard benchmark
    results = run_benchmark(model, tokenizer, args.prompt,
                           max_new_tokens=args.max_tokens,
                           use_compile=args.compile)

    print(f"\n{'='*60}")
    print(f"Benchmark Results")
    print(f"{'='*60}")
    print(f"  Input tokens:     {results['input_tokens']}")
    print(f"  Output tokens:    {results['output_tokens']}")
    print(f"  Avg time:         {results['avg_time_s']:.3f}s")
    print(f"  Best time:        {results['best_time_s']:.3f}s")
    print(f"  P95 time:         {results['p95_time_s']:.3f}s")
    print(f"  Avg tok/s:        {results['avg_tok_per_sec']:.1f}")
    print(f"  Best tok/s:       {results['best_tok_per_sec']:.1f}")
    print(f"  GPU:              {torch.cuda.get_device_name()}")
    print(f"  VRAM peak:        {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"  torch.compile:    {results['compile']}")
    print(f"  torch version:    {torch.__version__}")

    if args.output:
        results["gpu"] = torch.cuda.get_device_name()
        results["torch_version"] = torch.__version__
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
