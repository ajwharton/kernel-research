#!/bin/bash
# K-101: Profile baseline for Qwen2.5-14B-Instruct GGUF on GB10.
# Measures tok/s, VRAM, and kernel breakdown via ncu.
set -euo pipefail

MODEL_DIR="/mnt/data/models/qwen2.5-14b-instruct"
MODEL_FILE="$MODEL_DIR/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
RESULTS_DIR="/home/vulcan/src/kernel-research/results/k101-baseline"
LLAMA_DIR="/mnt/data/llama.cpp"

echo "=== K-101: Profile Baseline ==="
echo "Model: Qwen2.5-14B-Instruct Q4_K_M GGUF"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# ── Step 1: Verify tools ───────────────────────────────────────
echo "[1/3] Tools check..."
ncu --version 2>&1 | head -1
"$LLAMA_DIR/build/bin/llama-bench" --version 2>&1 || echo "  (version flag not supported)"
ls -lh "$MODEL_FILE"

# ── Step 2: Quick bench (tok/s) ─────────────────────────────────
echo "[2/3] llama-bench (prompt 512, gen 128)..."
mkdir -p "$RESULTS_DIR"
"$LLAMA_DIR/build/bin/llama-bench" \
    -m "$MODEL_FILE" \
    -p 512 -n 128 \
    -o json 2>/dev/null \
    > "$RESULTS_DIR/bench.json"

# Extract key numbers
python3 -c "
import json
with open('$RESULTS_DIR/bench.json') as f:
    data = json.load(f)
print(f'  Tok/s: {data[0][\"tokens_per_second\"]:.1f}')
print(f'  Model size: {data[0][\"model_size\"]} GB')
print(f'  VRAM used: {data[0][\"model_size\"]} GB (weights only)')
" 2>/dev/null || echo "  (bench json parse skipped — run directly for numbers)"

# ── Step 3: ncu kernel breakdown ────────────────────────────────
echo "[3/3] ncu kernel profile (prompt 64, gen 32 — small for speed)..."
echo 'Write a one-sentence story about a robot learning to paint.' > /tmp/k101_prompt.txt

ncu --set full \
    --kernel-name regex:void \
    -o "$RESULTS_DIR/ncu_profile" \
    "$LLAMA_DIR/build/bin/llama-cli" \
    -m "$MODEL_FILE" \
    -f /tmp/k101_prompt.txt \
    -n 32 \
    -t 4 \
    2>&1 | tail -5

echo "  ncu profile saved to $RESULTS_DIR/ncu_profile.ncu-rep"

# ── Step 5: Summary ─────────────────────────────────────────────
echo ""
echo "=== K-101 Complete ==="
echo "Results in: $RESULTS_DIR/"
echo ""
echo "To view: ncu --open $RESULTS_DIR/ncu_profile.ncu-rep"
echo "For nsys timeline: nsys profile -o $RESULTS_DIR/nsys_timeline llama-cli ..."
