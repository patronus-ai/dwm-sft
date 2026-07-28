# Context Parallelism for GLM-5.2 at 128k: Why It Doesn't Fit (Yet)

**Author:** Zhe Li (@zhe-patronus)
**Date:** 2026-07-28
**Stack:** Public-only (`radixark/miles` + `radixark/Megatron-LM` + `Megatron-Bridge` + SGLang), `tools/build.sh`.
**Hardware:** B200 (178.35 GB usable/GPU), 1–4 nodes × 8 GPU.
**Companion:** `FINDINGS.md` (the memory-scaling study that motivated this), `MEMSCALE_RESULTS.md` (raw data).

---

## TL;DR

Context parallelism (CP) is the *right* axis to unlock 128k — it's the only parallelism that shards the sequence dimension, which is what activation memory scales with. We removed the blanket `CP==1` guard for GLM's DSA path (scoped, opt-in) and CP **runs**: no crash, all steps `ok=true` at CP=2/64k. But it still **does not make 30-layer/128k fit**, and the reason is not activation memory or hardware — it's a **dense `O(S²)` score matrix inside the DSA "lightning indexer"**:

```
megatron/bridge/models/glm5/tilelang/tilelang_indexer_fwd.py:149
  logits = torch.empty([seq_len, seq_len_kv], dtype=torch.float32)   # 206.64 GiB at 128k
```

The indexer materializes every query×key score before selecting the top-2048, so it pays full `O(S²)` memory even though the *attention* that follows is sparse. CP shards the query dimension but the **key dimension stays full** (all keys are needed for a correct top-k), so CP shrinks this matrix only linearly — not enough. The real fix is a **fused / online-top-k indexer kernel** that never materializes the full matrix, not more nodes and not a config flag.

---

## 1. Why we tried CP

From the memory-scaling study: 30-layer/128k OOMs at **every** node count (1/2/4), because activation memory scales with sequence length and does **not** shard across expert-parallel or data-parallel ranks — only **context parallelism** shards the sequence itself. 30L/**64k** fit on a single node, proving the wall is the 128k activation working set. CP=k puts `128k/k` tokens per rank, so CP=2 should reproduce the 64k profile at the full 128k context.

## 2. The blocker, and the scoped change

GLM-5.2 uses DSA (`experimental_attention_variant == "dsa"`), and Megatron hard-asserts against CP for it:
```
megatron/core/transformer/transformer_config.py:1073
  assert self.context_parallel_size == 1, "Currently context parallelism is not supported by DSAttention!"
```
This is a **shared, conservative guard** (the generic Megatron `dsa.py` indexer isn't CP-safe). We scoped it to an explicit opt-in so default behavior is unchanged:
```python
elif self.experimental_attention_variant == "dsa":
    import os as _os
    assert (self.context_parallel_size == 1 or _os.environ.get("DSA_ALLOW_CP") == "1"), \
        "... (set DSA_ALLOW_CP=1 to opt into GLM's CP-aware plugin path)"
```
`DSA_ALLOW_CP` is threaded into the multi-node actor env passthrough (`--train-env-vars`).

## 3. Experiments and results

All 30-layer, LoRA r16, bf16 base, `dsa-attention-backend tilelang --qkv-format thd`.

| Run | Nodes | Grid | Seq | Outcome |
|-----|-------|------|-----|---------|
| CP=1 (4493) | 2 | TP8/CP1/EP16 | 64k | ✅ ran, rc=0, all steps `ok=true` |
| CP=2 (4494) | 2 | TP8/CP2/EP8 | 64k | ✅ ran, rc=0, **no assert, no crash** — CP plumbing works |
| CP=2 (4495) | 2 | TP8/CP2/EP8 | **128k** | ❌ OOM at first forward, **~0.22 GB short** (176.76 GB used, +1.80 GiB, 1.58 free) — activation wall |
| CP=4 (4496) | 4 | TP8/CP4/EP8 | **128k** | ❌ OOM on a **single 206.64 GiB allocation**, with **106 GB free** — the indexer |

Two operational facts worth keeping:
- **CP runs for GLM.** CP=2/64k completed end-to-end; CP never asserted or crashed in the plumbing.
- **Loss was NaN in every run** (decapitated 30-layer model + synthetic data), so numerical **correctness of CP was never verified** — a separate open item. "Runs" ≠ "correct."

## 4. The real issue: an `O(S²)` dense indexer matrix

The CP=4 OOM was categorically different from a normal memory-edge OOM: it tried to allocate **more than an entire GPU (206 GiB) with 106 GB free**. The traceback lands in the DSA indexer:

```
forward_step → … → transformer_block custom_forward
  → bridge/models/glm5/tilelang/tilelang_mla.py: forward → _tilelang_forward → _tilelang_topk
    → tilelang/indexer.py: lighting_indexer
      → tilelang/tilelang_indexer_fwd.py:149  logits = torch.empty([seq_len, seq_len_kv], float32)
```

**What the indexer does:** DSA's promise is that each query attends to only its top-2048 keys. But this "lightning indexer" selects them the naive way:
1. compute **all** query×key scores into a **fully materialized** dense buffer `[seq_len × seq_len_kv]` (fp32),
2. mask cross-sample entries to `-inf` (`clean_logits_kernel`),
3. **then** take top-2048.

So the top-k is cheap in FLOPs (tiny index heads, a tilelang kernel) but pays **full `O(S²)` memory** to get there. `206.64 GiB / 4 bytes ≈ 5.5×10¹⁰ ≈ (235k)²` — a dense pairwise score grid over the packed ~128k-token sequence. The sparse attention that follows never sees the savings; the indexer already blew the budget.

## 5. Why CP cannot fix this

- The dense matrix size is `seq_len × seq_len_kv`, and it is **dominated by `seq_len_kv` — the full key sequence**. A correct top-k must score every query against **all** keys, so under CP the keys are **gathered back to full length**.
- CP shards the **query** dimension (`seq_len`) but **not** the key dimension of this matrix. So the matrix shrinks only **linearly** with CP, and at 128k it's still ~200 GiB.
- That's why CP=4 *exposed* it: halving activation (relative to CP=2) freed enough memory to get *further* into the forward and actually reach the indexer allocation. CP=2 would hit the same wall if it hadn't OOM'd earlier on activation.

**CP shards the wrong dimension for this bottleneck.** More nodes / higher CP only chip at it linearly and can't bring an `O(S²)` term under the GPU limit at 128k within any realistic node count.

## 6. A correction to record

Earlier in the investigation I concluded GLM's DSA path was "CP-aware" based on CP key-gather logic in **miles' `glm5.py` plugin**. But the training forward actually runs **`megatron.bridge`'s glm5 tilelang path** (a port of the slime inline DSA path) — a *different* module. That is the code that executes, and it materializes the dense indexer matrix. Whether it even gathers keys correctly across CP is **unverified** (the NaN loss hid any correctness signal). Lesson: confirm which module the traceback actually runs before reasoning about its properties.

## 7. Reconciliation with prior results

- **5/10/20-layer fit at 128k without CP** — there the indexer matrix was small/transient enough to fit alongside fewer base weights (and full recompute frees it between layers).
- **30L/128k OOM'd on activation** *before* reaching the indexer wall (1/2/4 nodes, recompute and offload).
- **CP got past the activation wall** → exposed the deeper `O(S²)` indexer ceiling.

All consistent: the ceiling you hit depends on how far into the forward you get before running out of memory.

## 8. The real fix

Not a config, not more hardware: a **fused / tiled online-top-k indexer**. Compute query×key scores in **tiles**, keep only the running top-2048 per query, and never materialize `[S × S_kv]` — the same principle FlashAttention uses to avoid the `O(S²)` score matrix. This is almost certainly what the private `patronus-ai/miles-nvfp4` DSA-CP path provides. Options, roughly in order of effort:
1. **Check for an existing tiled path** in the tilelang indexer that we could switch to (the current `indexer_fwd_interface` always allocates the dense buffer — but the kernels may support a fused variant).
2. **Implement online-top-k** in `tilelang_indexer_fwd.py` (real kernel work, correctness-sensitive).
3. **Private fork** for the 744B/128k target.

Meanwhile the public-stack answer stands: **run at ≤64k** (fits and trains today), or take on the indexer kernel change for 128k.

---

## Artifacts

- Scoped assert: `Megatron-LM/megatron/core/transformer/transformer_config.py` (gated on `DSA_ALLOW_CP`). *Note: this lives in the git-ignored clone; captured as an overlay/patch in `build.sh` if we keep it.*
- Scripts: `scripts/cp-test-30L-64k-cp1.sbatch`, `cp-test-30L-64k-cp2.sbatch`, `cp-test-30L-128k-cp2.sbatch`, `cp-test-30L-128k-cp4.sbatch`.
- Key code: `.venv/.../megatron/bridge/models/glm5/tilelang/tilelang_indexer_fwd.py:149` (the dense allocation); `tilelang_mla.py` (`_tilelang_topk`); `transformer_config.py:1070` (the assert).
