# AGENTS.md — Kernel Research

GPU kernel profiling and recompilation on NVIDIA Blackwell (GB10).

## Session startup
1. `ssh forge` — verify connectivity, check GPU state (`nvidia-smi`)
2. Read latest in `learnings/`
3. Check `results/` for in-progress experiments

## Governance
- Branch → PR → merge. No direct commits to `main`.
- All benchmarks recorded with: GPU, model, batch/seq, tok/s, VRAM.
- ncu/nsys traces saved to `results/<experiment>/`.

## Forge access
```
ssh vulcan@forge.local
# Repo: /home/vulcan/src/kernel-research
# GPU: GB10, 128GB unified LPDDR5x, CUDA 13.0
# llama.cpp: /mnt/data/llama.cpp
# Models: /mnt/data/models/
```

## Current focus
- K-101: Profile baseline for 14B GGUF on GB10
- K-102: compile path exploration (deferred — GGUF is already quantized)

## Red lines
- `device_map="auto"` locks GB10 — never use it
- vLLM locks GB10 — use llama.cpp for inference
- No `rm -rf` without explicit consent
- Profile before optimizing — every time
