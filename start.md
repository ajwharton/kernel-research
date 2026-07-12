# start.md — session entry for kernel-research

> **How to start:** open Grok here, say `read start.md` (or `start`), then one-line **Outcome**.  
> Thin harness: `~/.grok/docs/thin-harness.md`

## Outcome default

Profile-before-optimize on **forge GB10**; record tok/s + VRAM; write a short learning if something changes.

## Red lines

- Branch → PR → merge (no direct `main`)
- **`device_map="auto"` never** on GB10 (locks box)
- **No vLLM** for inference here — llama.cpp path
- No `rm -rf` without explicit consent
- **Profile before optimizing** — every time
- Benchmarks must log: GPU, model, batch/seq, tok/s, VRAM

## Forge

```bash
ssh vulcan@forge.local
# Repo: /home/vulcan/src/kernel-research
# GPU: GB10, 128GB unified LPDDR5x, CUDA 13.0
# llama.cpp: /mnt/data/llama.cpp
# Models: /mnt/data/models/
```

## Facts

| Item | Value |
|------|--------|
| Focus GPU | NVIDIA GB10 (Blackwell SM121) |
| Baseline model | Qwen2.5-14B-Instruct GGUF Q4_K_M |
| K-101 | ✅ ~23.6 tok/s, matmul-dominated |
| K-102 | ✅ ~8× GPU vs CPU; roofline-ish |
| Next ideas | Speculative decode (S-101), quant ladder (Q-101) — only if Outcome says so |

## Prefer artifacts

```text
learnings/     # latest write-up first
results/       # experiment outputs / traces
scripts/       # how to re-run
```

## Commands (pattern)

```bash
ssh forge
# then run the experiment script under scripts/ for the active ID
# save ncu/nsys under results/<experiment>/
```

## Pull-on-miss only

| Path | When |
|------|------|
| `learnings/*.md` | Prior conclusions (latest only) |
| `results/<exp>/` | Numbers / traces for that exp |
| `README.md` | Experiment table overview |
| `docs/` | Extra design notes if any |

## Do not

- Startup-read all learnings + all results  
- Re-open “profile baseline” without a new hypothesis  
- Import Mia RL training into this repo  

## Agent memory

Qdrant `agent-memory__*` on miss only. SSOT = learnings/ + results/.
