#!/bin/bash
# K-101: Profile baseline for Qwen2.5-14B-Instruct GGUF on GB10.
# Measures tok/s, VRAM, and kernel breakdown via ncu.
set -euo pipefail

MODEL_DIR="/mnt/data/models"
MODEL_NAME="qwen2.5-14b-instruct"
MODEL_FILE="$MODEL_DIR/$MODEL_NAME/Q4_K_M.gguf"
HF_REPO="bartowski/Qwen2.5-14B-Instruct-GGUF"
HF_FILE="Qwen2.5-14B-Instruct-Q4_K_M.gguf"
RESULTS_DIR="results/k101-baseline"
LLAMA_DIR="/mnt/data/llama.cpp"

echo "=== K-101: Profile Baseline ==="
echo "Model: Qwen2.5-14B-Instruct Q4_K_M GGUF"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# ── Step 1: Check/install llama.cpp ─────────────────────────────
if [ ! -f "$LLAMA_DIR/build/bin/llama-bench" ]; then
    echo "[1/5] Building llama.cpp..."
    if [ ! -d "$LLAMA_DIR" ]; then
        git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
    fi
    cd "$LLAMA_DIR"
    cmake -B build -DGGML_CUDA=ON
    cmake --build build --config Release -j$(nproc)
else
    echo "[1/5] llama.cpp found at $LLAMA_DIR"
fi

# ── Step 2: Pull model ──────────────────────────────────────────
if [ ! -f "$MODEL_FILE" ]; then
    echo "[2/5] Downloading $HF_REPO → $MODEL_DIR/$MODEL_NAME..."
    mkdir -p "$MODEL_DIR/$MODEL_NAME"
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('$HF_REPO', '$HF_FILE', local_dir='$MODEL_DIR/$MODEL_NAME')
"
else
    echo "[2/5] Model found at $MODEL_FILE"
fi

# ── Step 3: Quick bench (tok/s) ─────────────────────────────────
echo "[3/5] llama-bench (prompt 512, gen 128)..."
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

# ── Step 4: ncu kernel breakdown ────────────────────────────────
echo "[4/5] ncu kernel profile (prompt 64, gen 32 — small for speed)..."
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
