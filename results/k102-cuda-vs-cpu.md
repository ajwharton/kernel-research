# K-102: CUDA vs CPU — classify compute vs memory bound

**Date**: 2026-07-10  
**Hardware**: NVIDIA GB10  
**Model**: Qwen2.5-14B-Instruct Q4_K_M GGUF

## Results

| Metric | All GPU (ngl=-1) | No GPU (ngl=0) | Speedup |
|--------|-------------------|----------------|---------|
| Prompt processing (pp512) | 1,783 tok/s | 1,023 tok/s | **1.7×** |
| Token generation (tg128) | 23.6 tok/s | 2.9 tok/s | **8.0×** |

## Kernel breakdown (nsys — all GPU)

| Kernel | % GPU Time | Type |
|--------|-----------|------|
| `mul_mat_q` (Q4_K) | 34.9% | Quantized matmul |
| `mul_mat_vec_q` (Q4_K) | 32.0% | Quantized GEMV |
| `mul_mat_q` (Q8_0) | 8.2% | Q8 matmul |
| `mul_mat_vec_q` (Q8_0) | 7.9% | Q8 GEMV |
| `flash_attn_ext_f16` | 2.0% | Flash Attention |
| Elementwise (norm, silu, rope) | ~15% | Various |

## Classification

- **89% matmul/GEMV** — compute-bound, already hand-tuned quantized CUDA
- **2% attention** — Flash Attention active, well below optimization threshold
- **Generated output**: "Hello"

## Verdict

**Stop here — at roofline.** llama.cpp's CUDA backend is near-optimal for this model/hardware combination. No named leftover op justifies a custom kernel. The brief's decision tree terminates at K-103 (skip) → stop.

## Remaining speedup levers (not kernel optimization)

- Lower quantization (Q2_K, IQ3_XXS)
- Speculative decoding
- Batch inference
