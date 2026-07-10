# Kernel Research

GPU kernel profiling and recompilation research on NVIDIA Blackwell (GB10 / DGX Spark).
Targeting inference throughput on real model sizes (14B+ parameters).

## Hardware

- **GPU**: NVIDIA GB10 (Blackwell SM121, 128 GB unified LPDDR5x)
- **Host**: forge (Ubuntu 26.04, CUDA 13.0, arm64)
- **Access**: `ssh forge`

## Models

| Model | Size | Format | Status |
|-------|------|--------|--------|
| Qwen2.5-14B-Instruct | 14B | GGUF Q4_K_M | K-101 baseline |

## Experiments

| ID | Name | Status | Result |
|----|------|--------|--------|
| K-101 | Profile baseline | ✅ done | 23.6 tok/s, 89% matmul |
| K-102 | CUDA vs CPU | ✅ done | 8× GPU speedup, roofline reached |
| K-103 | Attention backend | ⏭️ skipped | Attention only 2%, Flash already active |
| K-104 | Triton residual | ⏭️ skipped | No named leftover op ≥5% |

## Planned

| ID | Name | Description |
|----|------|-------------|
| S-101 | Speculative decoding | Draft model + target model, measure acceptance rate and speedup |
| Q-101 | Quantization ladder | Benchmark Q2_K → Q8_0 throughput vs perplexity |

## Prior learnings

See `learnings/2026-07-09-triton-mlp-fusion.md` — Triton vs cuBLAS tiling, NVFP4 tooling status (from earlier session on Qwen2.5-1.5B PyTorch).
scripts/         Profiling and benchmark scripts
data/            Model paths, experiment configs
results/         Benchmark results, ncu/nsys traces
learnings/       Write-ups and session notes
docs/            Reference material
```

## Prior learnings

See `learnings/2026-07-09-triton-mlp-fusion.md` — Triton vs cuBLAS tiling, NVFP4 tooling status.

## Quick start

```bash
# On forge:
cd kernel-research
bash scripts/setup.sh           # verify tools, pull model if needed
bash scripts/k101_baseline.sh   # run profiling baseline
```
