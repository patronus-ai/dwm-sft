# Layer-scaling @ 128k, 1 node (8×B200), grid TP8/PP1/EP8/ETP1, bf16 base (FP8 ckpt→bf16), LoRA r16, no fp8-param-gather

| Layers | Params  | Base/rank (bf16) | Result        | Peak GPU mem            |
|--------|---------|------------------|---------------|-------------------------|
| 5      | 22.8 B  | ~6 GB            | ✅ FITS        | ~148 GB (snapshot)      |
| 10     | ~72 B   | ~18 GB           | ✅ FITS        | ~149 GB (snapshot)      |
| 20     | ~172 B  | ~44 GB           | ✅ FITS (edge) | ~182 GB (near 183 ceil) |
| 30     | 269.6 B | ~69 GB           | ❌ OOM         | needed ~179 (~0.8 over) |
| 40     | 368.3 B | ~90 GB           | (killed: would OOM) | —                 |

## 2 nodes (16×B200), grid TP8/PP1/EP16/ETP1 — 30 layers
| Layers | Params/rank (base) | Result | Detail |
|--------|--------------------|--------|--------|
| 30 (EP16) | 18.06 B (~36 GB bf16) | ❌ OOM | base sharding worked (69→36 GB/rank), but OOM moved to **first forward**: 159.6 GB already allocated, tried +21.6 GB, only 13.9 GB free (178 GB cap). |

**Key takeaway:** adding the 2nd node halved base/rank exactly as predicted (EP8→EP16), yet 30L still OOMs — because the **128k activation working set (~120+ GB/rank) does NOT shard with EP**; it's per-DP-rank and dominated by sequence length, not layer count. So a 2nd node buys ~33 GB of base headroom but nothing on the activation side. Breaking point stays ~20 layers regardless of node count on this bf16-base public stack.

### 30L + CPU activation offload (2 nodes, EP16, recompute OFF, --cpu-offloading-num-layers 28)
| Config | PyTorch allocated | Failing alloc | Free | Overshoot | Result |
|--------|-------------------|---------------|------|-----------|--------|
| recompute (full) | 159.6 GB | 21.56 GiB | 13.9 GB | ~11.5 GB over | ❌ OOM |
| **offload 28L**  | 171.9 GB | **2.23 GiB** | 2.00 GB | **~0.23 GB over** | ❌ OOM (barely) |

Offload trades recompute's *discard* for *keep-then-stream-to-host*, so PyTorch's peak **allocated** went UP (160→172 GB), BUT total demand dropped: recompute needed 168.3+21.6≈190 GB, offload needed 176.3+2.2≈178.6 GB — cutting overshoot from ~11.5 GB to **~0.23 GB**. It nearly fit.

#### Chasing the last ~0.23 GB (2 nodes, EP16, offload) — flag levers EXHAUSTED
| Lever tried | Outcome |
|-------------|---------|
| `--cpu-offloading-num-layers` 28 → 29 | **byte-identical OOM** (2.23 GiB short) — offload benefit saturated; layer count no longer moves peak |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | reached actors (confirmed in env), **no effect** — wall is real capacity, not fragmentation (only 398 MiB reserved-unallocated) |
| `--fp16-lm-cross-entropy` | **incompatible** — asserts `args.fp16` (fp16-only; we run bf16). Dead end. |
| `--cross-entropy-loss-fusion` | **inert** — miles `loss_function` (training_utils/loss.py:95) takes already-materialized `logits`; miles computes its OWN SFT loss and never routes through Megatron fused CE |

**Conclusion:** on the public bf16 stack, 30L/128k sits ~0.23 GB from fitting on 2 nodes with offload, but **no flag closes it**. The residual is the full `[S, vocab]` logits tensor that miles materializes before its own loss. Closing it needs a **code change** — chunked log-softmax in `miles/backends/training_utils/loss.py` (compute the SFT loss in vocab tiles, never materializing the full logits). A 3rd node doesn't help cleanly: 256 experts don't divide evenly at 24 GPUs (EP grid breaks).

**Confirmed public-stack ceiling @128k LoRA: 20 layers (recompute) / ~30 layers is one code change (chunked CE) away.**

## Breaking point (8 GPU, bf16 base, 128k, LoRA)
- **Max that fits: ~20 layers (barely, ~182/183 GB). 30 layers OOMs. Boundary = 20–30, close to 20.**
- Peak ≈ base_weight/rank (grows w/ MoE layers) + ~130–140 GB of ~constant 128k activation/DSA working memory.
- All runs LoRA (frozen base); full-FT would OOM at far fewer layers.

## Scaling to 32 GPU (real deployment grid, EP×ETP=32 = 4× more expert sharding)
- Base/rank shrinks ~4× → breaking point ≈ 4× more layers ≈ ~60-80 layers region.
- Consistent with: full 78-layer 744B OOM'd on 32 GPU (bf16). So 32-GPU bf16 is ~at/over the edge for full 128k → need fp8/nvfp4 base (the patronus path) or fewer layers/context.
