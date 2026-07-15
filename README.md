.

---

```markdown
# End‑to‑End Inference Engine for GPT‑2 Small (124M)

A production‑ready, fully custom CUDA inference engine that executes a complete transformer block (LayerNorm → QKV Projection → Flash Attention → MLP → Residual Add) in a single, fused pipeline. The engine is benchmarked on **NVIDIA T4, A100, and H100** GPUs, achieving **12.28× speedup over PyTorch** on T4 and scaling to **2× faster** on H100.

---

## Repository Structure

```

cuda-inference-engine/
├── README.md
├── inference_engine.cu          # Complete pipeline + benchmark harness
├── LICENSE
└── docs/
└── case-study.md            # Detailed optimisation journey

```

---

## Key Features

- **Complete Transformer Layer** – 11 fused kernels in a single pass:
  - LayerNorm (with residual addition)
  - QKV Projection (tiled GEMM)
  - Flash Attention (causal, online softmax)
  - Output Projection (tiled GEMM)
  - MLP (FC1 + GELU + FC2)
  - Final LayerNorm (residual accumulation)
- **Zero Register Spills** – all kernels compile with 0 bytes stack, 0 spills, 0 loads.
- **Flash Attention** – online softmax, causal masking, 36+ KB shared memory tiling.
- **Fused GELU + Bias** – vectorised `float4` reads with scalar tail handling.
- **64‑bit Index Safety** – all offsets use `size_t` – supports matrices beyond 2³¹ elements.
- **Full Error Handling** – `CUDA_CHECK` after every API call and kernel launch.
- **Pre‑LN and Post‑LN Support** – configurable via `pre_norm` parameter.

---

## Performance Results

**Benchmarked on:** NVIDIA T4 (sm_75), A100 (sm_80), and H100 (sm_90).  
**Model:** GPT‑2 Small (124M parameters) – B=2, S=128, D=768, heads=12, head_dim=64.

| GPU | Latency per Pass (ms) | Throughput (tokens/sec) | Speedup vs PyTorch |
|-----|-----------------------|-------------------------|---------------------|
| **T4 (sm_75)** | **2.44** | **104,746** | **12.28×** |
| **A100 (sm_80)** | **1.52** | **168,754** | **5.27×** |
| **H100 (sm_90)** | **1.27** | **201,892** | **4.34×** |

**Key Insights:**
- The T4 achieves **12× faster** inference than an unoptimised PyTorch baseline.
- The H100 delivers the lowest absolute latency (**1.27 ms per pass**) – ideal for ultra‑low‑latency inference.
- Memory footprint is just **43 MB** – fits comfortably on any GPU.

**Compiler Report (`nvcc -O3 -arch=sm_75`):**
```

0 bytes stack frame
0 bytes spill stores
0 bytes spill loads
All kernels are fully register‑resident.

```

---

## Compilation & Usage

### Prerequisites
- CUDA Toolkit 11.8 or later
- NVIDIA driver supporting sm_75 (T4) or sm_80/sm_90 (A100/H100)

### Build
```bash
nvcc -O3 -arch=sm_75 -lineinfo --ptxas-options=-v -o inference_engine inference_engine.cu
```

For A100 or H100, change -arch to sm_80 or sm_90 respectively.

Run

```bash
./inference_engine
```

The benchmark runs 50 iterations, reports:

· Average latency per pass (ms)
· Token throughput (tokens/sec)
· Max absolute error vs CPU reference (should be < 1e-3)

---

How It Works

Pipeline Architecture (Pre‑LN)

1. LayerNorm (Input) – Normalises input with residual support.
2. QKV Projection – GEMM that projects the input to 3×D dimensions.
3. Transpose QKV – Reshapes Q, K, V from B×S×3D to B×h×S×d_k.
4. Flash Attention – Causal self‑attention with online softmax.
5. Transpose Output – Reshapes attention output back to B×S×D.
6. Output Projection – GEMM that projects the attention output.
7. LayerNorm (Residual) – Adds input residual, normalises.
8. MLP (FC1) – GEMM that expands to 4×D dimensions.
9. Fused GELU + Bias – Activation with bias addition (vectorised).
10. MLP (FC2) – GEMM that projects back to D dimensions.
11. Final LayerNorm – Adds residual, normalises final output.

Kernel Highlights

· GEMM: Tiled with 64×64 blocks, shared memory staging, coalesced writes.
· Flash Attention: 36+ KB shared memory, 64 threads/block, padded s_Q/s_V to avoid bank conflicts.
· LayerNorm: Grid‑stride loop, warp‑shuffle reduction, arbitrary D support.
· GELU: float4 vectorisation, scalar tail handling for non‑multiples of 4.

---

Optimisation Journey

Issue Fix
Double bias addition in FC1 Removed bias from GEMM, added once in GELU kernel.
Flash Attention s_K overflow Changed out‑of‑bounds padding from -1e20f to 0.0f.
Vectorised GELU out‑of‑bounds Added scalar tail loop for N % 4 != 0.
GEMM uncoalesced writes Staged output in shared memory, coalesced writeback.
Flash Attention bank conflicts Padded s_Q and s_V inner dimensions to 65.
Head dimension safety Added head_dim == 64 validation guard.

---

Multi‑Architecture Insights

· T4: Bottleneck is Flash Attention shared memory occupancy (1 block/SM → 6.25% occupancy).
· A100: Grid starvation – small batch (M=256) cannot saturate 108 SMs.
· H100: Legacy FP32 code cannot use Tensor Cores/WGMMA – future optimisation opportunity.

---

License

MIT License – see LICENSE file.

---

Author

Mustafa-cuda-dev
GitHub: https://github.com/Mustafa-cuda-dev

---

Acknowledgements

· NVIDIA CUDA Toolkit
· Google Colab for free T4 GPU access
· The open‑source CUDA community

```

---

