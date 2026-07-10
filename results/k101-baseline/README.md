# K-101: Profile Baseline

**Date**: 2026-07-10  
**Hardware**: NVIDIA GB10 (Blackwell, 128 GB LPDDR5x, CUDA 13.0)  
**Model**: Qwen2.5-14B-Instruct Q4_K_M GGUF  
**Backend**: llama.cpp (CUDA, all layers on GPU)

## Throughput

| Metric | Value |
|--------|-------|
| Prompt processing (512 tok) | **1,783 tok/s** |
| Token generation (128 tok) | **23.6 tok/s** |
| Model size (on disk) | 8.37 GB |
| Model params | 14.77B |
| Backend | CUDA, ngl=-1 (all layers GPU) |
| llama.cpp build | 049326a00 |

## Next

- [ ] ncu kernel breakdown (pending)
- [ ] Classify: attention vs GEMM vs elementwise vs launch overhead
- [ ] Compare against raw PyTorch HuggingFace path (fp16) for K-102 compile/not-compile A/B
