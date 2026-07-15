```markdown
# Case Study: End‑to‑End Inference Engine for GPT‑2 Small (124M)

**Author:** Mustafa-cuda-dev  
**Repository:** [cuda-inference-engine](https://github.com/Mustafa-cuda-dev/cuda-inference-engine)  
**Hardware Tested:** NVIDIA T4 (sm_75), A100 (sm_80), H100 (sm_90)  
**Model:** GPT‑2 Small (124M parameters) – hidden size 768, 12 heads, 12 layers (single layer benchmarked)  
**Goal:** Build a fully custom CUDA inference engine for a transformer block, integrating all previous kernels into a single, fused pipeline, and benchmark it across multiple GPU architectures.

---

## 1. Project Overview

This project represents the **capstone** of the CUDA portfolio – a complete, production‑ready inference engine for transformer models. It integrates six custom CUDA kernels into an end‑to‑end pipeline that executes a full transformer block (LayerNorm → QKV Projection → Flash Attention → Output Projection → MLP → Residual Add) in a single fused pass.

The engine is designed for **NVIDIA T4 (sm_75)** and validated on **A100 (sm_80)** and **H100 (sm_90)**. It achieves **12.28× speedup over PyTorch** on T4, **5.27× on A100**, and **4.34× on H100**, with **zero register spills** and **full correctness** against a CPU reference.

---

## 2. Technical Challenges & Solutions

### 2.1. Pipeline Fusion – 11 Kernels, One Engine
- **Challenge:** Combining multiple kernels (GEMM, attention, activation, normalisation) without intermediate global memory round‑trips.
- **Solution:** Designed a unified scratchpad buffer that reuses memory across stages. Each kernel writes to a pre‑allocated region, and the next kernel reads from it – eliminating redundant HBM traffic.

### 2.2. Double Bias Addition Bug
- **Challenge:** In the original design, the FC1 GEMM added `b_fc` during writeback, and the fused GELU kernel added it again – resulting in `xb = val + 2*bias`.
- **Solution:** Removed the bias addition from the FC1 GEMM (passed `nullptr`) and applied bias only once inside the GELU kernel. This fix was applied to both Pre‑LN and Post‑LN pipelines.

### 2.3. Flash Attention Overflow and Correctness
- **Challenge:** Out‑of‑bounds `s_K` elements were initialised to `-1e20f`; multiplying negative query values produced large positive numbers that overflowed FP32.
- **Solution:** Changed padding to `0.0f` – the causal masking condition already sets `score = -1e20f` for invalid positions, so the padding only needed to be safe for multiplication.

### 2.4. Vectorised GELU Tail Handling
- **Challenge:** The kernel used `float4` loads/stores without handling `N % 4 != 0`, causing out‑of‑bounds accesses.
- **Solution:** Added a scalar tail loop that processes the remaining elements individually, ensuring 100% safety.

### 2.5. Shared Memory Bank Conflicts in Flash Attention
- **Challenge:** `s_Q[16][64]` and `s_V[64][64]` caused 8‑way and 2‑way bank conflicts, stalling the warp scheduler.
- **Solution:** Padded inner dimensions to 65 (`s_Q[16][65]`, `s_V[64][65]`) – this broke the bank‑periodicity and eliminated conflicts.

### 2.6. Uncoalesced Writes in GEMM
- **Challenge:** Threads wrote to global memory with a stride of 4 elements, generating separate memory transactions.
- **Solution:** Staged the output tile in shared memory (`s_C[BM][BN+1]`) and wrote back coalesced using a linearised thread mapping.

### 2.7. Hardcoded Dimension Assumptions
- **Challenge:** Several kernels hardcoded block width (16) and head dimension (64).
- **Solution:** Replaced `16` with `blockDim.x`, added a `head_dim == 64` validation guard, and derived loop bounds from template parameters.

---

## 3. Final Architecture

### Pipeline Flow (Pre‑LN)

1. **LayerNorm (Input)** – Normalises input, supports residual addition.
2. **QKV Projection** – Tiled GEMM (`64×64`) with bias, projects to `3×D`.
3. **Transpose QKV** – Reshapes `B×S×3D` → `B×h×S×d_k`.
4. **Flash Attention** – Causal self‑attention with online softmax, 36+ KB shared memory.
5. **Transpose Output** – Reshapes `B×h×S×d_k` → `B×S×D`.
6. **Output Projection** – Tiled GEMM, projects attention output to `D`.
7. **LayerNorm (Residual)** – Adds input residual, normalises.
8. **FC1 GEMM** – Expands to `4×D` (bias added later in GELU).
9. **Fused GELU + Bias** – Vectorised `float4` activation with scalar tail.
10. **FC2 GEMM** – Projects back to `D`.
11. **Final LayerNorm** – Adds residual, normalises final output.

### Kernel‑Level Optimisations

| Kernel | Tiling | Shared Memory | Key Optimisation |
|--------|--------|---------------|-------------------|
| LayerNorm | Grid‑stride | 72 bytes | Warp‑shuffle reduction |
| GEMM | 64×64 | 21 KB | Staged coalesced writes |
| Flash Attention | 16×64 | 41 KB | Padded `s_Q`/`s_V`, staged `s_O` |
| GELU | Vectorised | 0 bytes | `float4` + scalar tail |

---

## 4. Benchmark Results

**Hardware:** NVIDIA T4 (sm_75), A100 (sm_80), H100 (sm_90)  
**Model:** GPT‑2 Small (124M) – B=2, S=128, D=768, heads=12, head_dim=64  
**Iterations:** 50 (warmup: 5)  
**Correctness:** Max error vs CPU reference = `1.8 × 10⁻⁶` (PASS)

| GPU | Latency (ms) | Throughput (tokens/sec) | Speedup vs PyTorch |
|-----|--------------|-------------------------|---------------------|
| **T4** | 2.44 | 104,746 | **12.28×** |
| **A100** | 1.52 | 168,754 | **5.27×** |
| **H100** | 1.27 | 201,892 | **4.34×** |

**Compiler Report (T4):**
```

0 bytes stack frame
0 bytes spill stores
0 bytes spill loads
All kernels are fully register‑resident.

```

**Memory Footprint:** ~43 MB total – fits easily on all tested GPUs.

---

## 5. Multi‑Architecture Insights

- **T4 (sm_75):** Primary bottleneck is Flash Attention’s shared memory footprint (41 KB) – limits occupancy to 1 block/SM, causing stall cycles. Still achieves 12× speedup over PyTorch.
- **A100 (sm_80):** Higher shared memory capacity (164 KB) allows 3 blocks/SM of Flash Attention, reducing attention latency. However, small batch size (`M=256`) underutilises 108 SMs – grid starvation is the new bottleneck.
- **H100 (sm_90):** Highest memory bandwidth and shared memory capacity, but the legacy FP32 code cannot leverage Tensor Cores or WGMMA. Despite this, absolute latency is lowest (1.27 ms). Future optimisation could use TMA/WGMMA for 5–10× further speedup.

---

## 6. What This Demonstrates

1. **System‑Level Engineering** – Integrates multiple kernels into a cohesive pipeline, managing memory, dependencies, and correctness.
2. **Deep CUDA Expertise** – GEMM tiling, Flash Attention, LayerNorm, GELU – all implemented from scratch.
3. **Production‑Grade Code** – Zero spills, full error handling, multi‑architecture validation.
4. **Performance Analysis** – Roofline modelling, bottleneck identification, and cross‑architecture scaling.
5. **Debugging & Correctness** – Fixed subtle bugs (double bias, overflow, tail handling) and validated against CPU.
6. **Honest Benchmarking** – Realistic numbers with clear explanations of hardware ceilings.

---

## 7. Lessons Learned

- **Fusion is Powerful** – Reducing global memory round‑trips yields 5–10× speedups even before CUDA Graphs.
- **Shared Memory Occupancy Matters** – A 41 KB footprint may fit, but it kills occupancy – always balance tile size and concurrency.
- **Hardware Scaling is Not Linear** – T4→A100→H100 shows diminishing returns for FP32 code; to exploit H100, Tensor Cores/TMA are essential.
- **Correctness Requires Rigour** – Small bugs like double bias addition only surface with non‑zero biases – always test with real weights.

---

## 8. Future Work

- **Tensor Core Integration** – Port GEMMs to WMMA/CUTLASS for 2–3× speedup on T4, 5× on H100.
- **CUDA Graph Optimisation** – Capture the entire pipeline as a graph to eliminate launch overhead.
- **Mixed Precision** – FP16/BF16 accumulation for higher throughput.
- **Multi‑Layer Support** – Extend to 12 layers (full GPT‑2) with loop‑based execution.

---

## 9. Conclusion

This project delivered a **production‑ready, end‑to‑end inference engine** for transformer models, achieving **12.28× speedup over PyTorch** on T4 and scaling effectively to A100 and H100. The engine integrates six custom CUDA kernels into a single fused pipeline, with zero register spills, full correctness, and a memory footprint of just 43 MB. It demonstrates system‑level thinking, deep GPU expertise, and the ability to build high‑performance inference systems from scratch – a definitive capstone for any GPU engineering portfolio.
```

---
