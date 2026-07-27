# GLM-5.2 LoRA-SFT at 128k Context: Memory-Scaling Findings

**Author:** Zhe Li (@zhe-patronus)
**Date:** 2026-07-27
**Stack:** Public-only (`radixark/miles` + `radixark/Megatron-LM` + `Megatron-Bridge` + SGLang), built via `tools/build.sh`. No private NVFP4 fork.
**Hardware:** B200 (178.35 GB usable/GPU), 1–4 nodes × 8 GPU, RoCE.

---

## TL;DR

On the **public bf16 stack**, GLM-5.2 LoRA-SFT at 128k context fits up to **~20 transformer layers on one node (8 B200)** with full activation recomputation. 30 layers OOMs. The bottleneck is **not** parameters or expert sharding — it is the **128k activation working set**, which scales with sequence length and does not shard across nodes. CPU activation offload closes almost all of the gap (30-layer/128k came within **~0.23 GB** of fitting on 2 nodes), but the final residual is the un-shardable `[128k × vocab]` logits tensor, which no runtime flag can remove.

Crucially, **adding nodes does not help — and can hurt.** Scaling 30-layer/128k from 1→2→4 nodes shrinks base/rank exactly as predicted (expert-parallel sharding), yet it OOMs at *every* node count, and 4 nodes was *worse* than 2: the dominant per-rank activation allocation **grows with the data-parallel degree**, swamping the base savings. The best result at any node count was 2-node + offload (~0.23 GB short). Full 78-layer/744B at 128k therefore requires the private stack's memory tricks (NVFP4 base, context parallelism, chunked loss) or a chunked-loss code change — **not** more hardware.

---

## 1. Objective

Determine, empirically, the point at which GLM-5.2 LoRA-SFT stops fitting at 128k context on the public stack, and identify *what* limits it — so we know whether the target workload (full 744B/128k) is reachable without the private `patronus-ai` fork.

## 2. Model & Method

- **Model:** GLM-5.2 = 744B-A40B MoE. 78 layers (3 dense + 75 MoE), hidden 6144, 256 routed experts (8 active) + 1 shared, MLA attention (kv_lora_rank 512, q_lora_rank 2048), DeepSeek Sparse Attention (index_topk 2048, every 4th layer), vocab 154,880. "128k" = 131,072 tokens.
- **Training:** LoRA (rank 16) on the MoE expert projections. The base is **frozen bf16** — the FP8 checkpoint is dequantized to bf16 on load by the bridge (public miles has no NVFP4/QLoRA base). No optimizer/grad state on the base.
- **Probe:** rather than fight the full 744B directly, we prune the model to *N* layers (`tools/build_Nlayer.py`, keeps layers 0..N-1) and train each at 128k. Parameters scale ~linearly (~9.7B per MoE layer), so the layer count is a clean knob for base-weight memory while the 128k activation cost stays roughly constant. This isolates *which* term hits the ceiling.
- **Grid:** 1 node = TP8/PP1/EP8/ETP1 (DP=1); 2 nodes = TP8/PP1/EP16/ETP1 (DP=2); 4 nodes = TP8/PP1/EP32/ETP1 (DP=4). EP always divides the 256 experts cleanly. CP=1 (DSA does not support context parallelism on the public stack); PP=1 (miles LoRA weight-sync requires it). DSA via `tilelang` backend + `--qkv-format thd` (recompute-safe).

## 3. Results

### 3.1 Single node (8 B200), full recompute

| Layers | Params | Base/rank (bf16) | Result | Peak GPU mem |
|--------|--------|------------------|--------|--------------|
| 5  | 22.8 B  | ~6 GB  | ✅ Fits | ~148 GB |
| 10 | ~72 B   | ~18 GB | ✅ Fits | ~149 GB |
| 20 | ~172 B  | ~44 GB | ✅ Fits (edge) | ~182 / 183 GB |
| 30 | 269.6 B | ~69 GB | ❌ OOM | — |
| 40 | 368.3 B | ~90 GB | ❌ OOM (not run) | — |

**Peak ≈ base_weight/rank (grows with layers) + ~130–140 GB of ~constant 128k activation/DSA working memory.** The 20→30 transition crosses 183 GB.

### 3.2 Two nodes (16 B200), 30 layers

| Config | PyTorch allocated | Failing alloc | Free | Total overshoot | Result |
|--------|-------------------|---------------|------|-----------------|--------|
| recompute (full) | 159.6 GB | 21.56 GiB | 13.9 GB | ~11.5 GB over | ❌ OOM |
| **CPU offload, 28 layers** | 171.9 GB | **2.23 GiB** | 2.00 GB | **~0.23 GB over** | ❌ OOM (barely) |

Adding a 2nd node (EP8→EP16) halved base/rank exactly as predicted — 34.5 B → 18.06 B params/rank (~69 → ~36 GB) — yet 30L still OOM'd. The failure merely **moved from checkpoint-load to the first forward pass**: base sharding worked, activations did not shrink.

### 3.3 Four nodes (32 B200), 30 layers — more nodes made it *worse*

| Config | Base/rank | PyTorch allocated | Failing alloc | Free | Overshoot | Result |
|--------|-----------|-------------------|---------------|------|-----------|--------|
| recompute (2-node) | ~36 GB | 159.6 GB | 21.56 GiB | 13.9 GB | ~11.5 GB | ❌ OOM |
| **recompute (4-node)** | ~19.7 GB | 143.8 GB | **43.12 GiB** | 29.7 GB | **~14 GB** | ❌ OOM |
| offload (2-node) | ~36 GB | 171.9 GB | 2.23 GiB | 2.00 GB | **~0.23 GB** | ❌ OOM |
| **offload (4-node)** | ~19.7 GB | 167.9 GB | **10.33 GiB** | 6.10 GB | **~4.23 GB** | ❌ OOM |

Going 2→4 nodes (EP16→EP32) halved base/rank again — 18.06 B → 9.84 B params/rank (~36 → ~19.7 GB) — and PyTorch's *allocated* total did drop accordingly. **But the failing single allocation grew** in lockstep with the data-parallel degree (recompute: 21.56 → 43.12 GiB; offload: 2.23 → 10.33 GiB), so the overshoot got *larger*, not smaller. The base savings were real but irrelevant — a DP-scaling activation/logits allocation (most likely the MoE all-to-all dispatch buffers and/or dynamic-batch token packing, which grow with the number of DP ranks) dominates and cancels them out. **The closest fit at any node count was 2-node + offload.**

### 3.4 Throughput note

Runs at 20 layers were memory-saturated at ~17 min/step (128k, batch 1). Fewer layers ran faster. Throughput collapses as the activation working set fills VRAM.

## 4. Why 128k Is the Wall

**Fundamental law:** parameters and optimizer state are fixed regardless of context length; **activations scale linearly with the number of tokens.** At 128k, every forward tensor carries a 131,072-long sequence dimension — 32× a 4k context. Even at micro-batch-size 1, that is 131,072 tokens in flight.

Concrete per-tensor sizes at S=131,072, hidden 6144, vocab 154,880:

| Term | Size | Shards with EP / extra nodes? |
|------|------|-------------------------------|
| One hidden-state tensor (S×H, bf16) | **1.61 GB** | No (per-DP-rank) |
| Retained layer inputs, 30 layers, full recompute | ~48 GB | No |
| **Output logits (S×vocab, bf16)** | **40.6 GB** (81.2 GB if CE in fp32) | Only across TP (vocab-parallel → ~5–10 GB/rank), not EP |
| MoE dispatch (S × top-k × moe_hidden) | multi-GB/MoE layer | Partially (expert weights only) |
| Base weights (frozen bf16) | ~36–69 GB/rank | **Yes** — this is what nodes/EP shard |

The base weights are the *only* large term that expert-parallelism and extra nodes shrink. Everything driven by S — hidden states, logits, MoE dispatch, the DSA indexer scoring all 131,072 keys — stays per-data-parallel-rank. That is why a 2nd node bought ~33 GB of base headroom and **zero** activation relief.

## 5. The Offload Experiment

**Mechanism:** CPU activation offloading (`--cpu-offloading-num-layers N`) streams a layer's activations to host RAM after the forward and prefetches them back for the backward, on a side CUDA stream. It trades GPU memory for host-link bandwidth. In our Megatron fork it requires PP=1 and the TE transformer impl, and it is **mutually exclusive with activation recomputation** (`transformer_config.py:1237`) — so the offload runs drop `--recompute-granularity full` and offload nearly all layers instead.

**Result:** offloading 28 of 30 layers cut the overshoot from ~11.5 GB to **~0.23 GB**. Offload's peak *allocated* memory is higher than recompute's (it keeps activations rather than discarding them), but total demand dropped below the recompute path — it very nearly fit.

**Why the last 0.23 GB cannot be closed with flags:**

| Lever | Outcome |
|-------|---------|
| Offload 28 → 29 layers | Byte-identical OOM — offload benefit is saturated; layer count no longer moves peak |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Reached the actors (confirmed), no effect — the wall is real capacity, not fragmentation (only 398 MiB reserved-unallocated) |
| `--fp16-lm-cross-entropy` | Incompatible — asserts `args.fp16`; it is fp16-only and we train bf16 |
| `--cross-entropy-loss-fusion` | Inert — miles' `loss_function` (`backends/training_utils/loss.py:95`) receives an **already-materialized** `logits` tensor and computes its own SFT loss; it never routes through Megatron's fused CE |

The residual is the full `[128k × vocab]` logits tensor the model forward materializes before miles computes the loss. **Removing it requires a code change** — chunked log-softmax in `loss.py` (compute the SFT loss in vocab tiles), not a runtime flag. And as §3.3 shows, **adding a 4th node makes it worse, not better** — the DP-scaling activation allocation outgrows the base savings — so more hardware is not a path to fitting.

## 6. Conclusions

1. **Confirmed ceiling:** ~20 layers at 128k on 1 node (public bf16, full recompute). 30 layers is one code change (chunked CE) away.
2. **The limiter is activation memory, not weights.** Sequence length dominates; expert-parallelism and extra nodes shrink only the base.
3. **Offload is a genuine, powerful lever** — it took 30L/128k from ~11.5 GB over to ~0.2 GB over on 2 nodes, entirely on public code.
4. **More nodes do not help — and can hurt.** Tested across 1/2/4 nodes, 30L/128k OOMs at every count, and 4 nodes was worse than 2: base/rank halves with each node but a DP-scaling activation allocation grows to cancel it. The single closest configuration was **2-node + offload (~0.23 GB short)**.
5. **Full 744B/128k on the public bf16 stack is not reachable by adding nodes.** It needs the memory tricks the private fork provides: NVFP4/FP8 base (shrinks the un-shardable base further), context parallelism (shards the sequence dimension — blocked here by DSA), and/or chunked loss.

## 7. Paths Forward (not executed)

- **Chunked cross-entropy in `miles/backends/training_utils/loss.py`** — compute the SFT loss over vocab tiles so the full logits are never materialized. On 2 nodes + offload this frees well over the 0.23 GB needed. Correctness-sensitive; needs one verification run. (Note: the DP-scaling allocation in §3.3 is a *separate* term — worth profiling which tensor grows with DP if pushing to more nodes.)
- **Shorter context** — 30L (and more) fits immediately at 32k/64k; useful to confirm the activation-vs-base split.
- **Private stack** — for the real 744B/128k target, adopt the NVFP4-base + DSA-chunking path (`patronus-ai/miles-nvfp4`).

---

*Raw data and per-run memory numbers: `MEMSCALE_RESULTS.md`. Reproduction: `HANDOFF.md`. Scripts: `scripts/memscale-1node.sbatch`, `scripts/sft-128k-glm744b.sbatch` (`OFFLOAD_LAYERS`/`CE_FP16` env toggles), `scripts/memscale-30L-2node-offload.sbatch`, `scripts/memscale-30L-4node.sbatch`, `scripts/memscale-30L-4node-offload.sbatch`.*
