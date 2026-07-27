# glm-sft

Independent, public-only stack for **LoRA-SFT of GLM-5.2 (744B-A40B)** at long
context on B200. Built entirely from public repos (`radixark/miles`,
`radixark/Megatron-LM`, `Megatron-Bridge`, SGLang) — no private forks.

This repo tracks only the **authored** pieces (build recipe, run scripts, tools,
docs, overlays). The heavy, reconstructable pieces (venv, upstream clones, model
weights, data, run outputs) are git-ignored and rebuilt by `tools/build.sh`.

## Layout
| Path | Tracked? | What |
|------|----------|------|
| `tools/build.sh` | ✅ | One-shot, resumable env builder (clones + patches + venv + kernels + overlays) |
| `tools/build_Nlayer.py` | ✅ | Prune GLM-5.2 to N layers (memory-scaling probe) |
| `tools/make_128k_sft_data.py` | ✅ | Generate the 128k SFT parquet |
| `scripts/*.sbatch` | ✅ | SLURM run scripts (smoke, 128k, memory-scaling, offload) |
| `overlays/` | ✅ | My files that live inside the clones; `build.sh` copies them into place |
| `env.sh` | ✅ | Runtime shell (paths, CUDA libs, NCCL/DSA env) |
| `*.md` | ✅ | `FINDINGS.md`, `MEMSCALE_RESULTS.md`, `HANDOFF.md` |
| `miles/ Megatron-LM/ src/ .venv/ wheels/` | ❌ | Reconstructed by `build.sh` |
| `models/ data/ out/ logs/` | ❌ | Weights / generated data / run artifacts |

## Frameworks

The stack is assembled from public components by `tools/build.sh` (pinned commits
shown). It splits into a training engine, an HF↔Megatron bridge, the GLM-5.2 DSA
attention kernels, and a rollout/inference path — orchestrated by Ray under SLURM.

**Training engine**
| Component | Source @ pin | Role |
|-----------|--------------|------|
| **miles** | `radixark/miles` | Top-level app. The SFT/RL loop (`train_async.py`), LoRA, the GLM-5.2 model spec (`miles_plugins/models/glm5`), the SFT loss, and rollout↔train orchestration. |
| **Megatron-LM / `megatron.core`** | `radixark/Megatron-LM` @ `miles-main` (+ DSA-dispatch patch) | Distributed training engine: TP/PP/EP parallelism, transformer layers, distributed optimizer, checkpointing, CPU activation offload. The patch wires `"dsa"` into the experimental-attention dispatcher. |
| **Transformer Engine (TE)** | `transformer_engine_cu13==2.12.0` + prebuilt wheels | FP8 kernels, fused attention, the TE transformer layers Megatron instantiates, and the `get_cpu_offload_context` used by activation offload. |
| **Apex** | NVIDIA (prebuilt wheel) | Fused kernels (RMSNorm/LayerNorm, fused optimizers) used by Megatron. |

**HF ↔ Megatron bridge (weight conversion + PEFT/LoRA)**
| Component | Source @ pin | Role |
|-----------|--------------|------|
| **Megatron-Bridge** | `radixark/Megatron-Bridge` @ `bridge` | Loads the HF checkpoint into the sharded Megatron model (`load_hf_weights` → per-rank TP/EP scatter, FP8→bf16 dequant), and exports LoRA adapters back to HF. |
| **mbridge** | `ISEEKYAN/mbridge` @ `89eb108` | Lightweight model-provider / param-mapping library the bridge builds on. |

**GLM-5.2 DSA (DeepSeek Sparse Attention) kernels**
| Component | Source @ pin | Role |
|-----------|--------------|------|
| **tilelang** | pip (abi3 wheel) | DSL + kernels for the DSA indexer. **Required** — `glm5.py` hard-imports it; the build fails without it. |
| **fast_hadamard_transform** | `Dao-AILab/fast-hadamard-transform` @ `e7706fa` | Hadamard-transform CUDA kernels used inside the DSA indexer. |
| **flash-attention (FA2 + FA3 shim)** | `Dao-AILab/flash-attention` @ `fbf24f6` (prebuilt wheels) | Dense attention backend (`--attention-backend flash`). |

**Rollout / inference path**
| Component | Source @ pin | Role |
|-----------|--------------|------|
| **SGLang** | `sgl-project/sglang` @ `sglang-miles` | Generation/rollout engine. An import-time dependency of miles `train_async` (loaded even for SFT). |
| **sglang_router / sgl-model-gateway** | prebuilt | SGLang request routing / gateway. |
| **torch_memory_saver** | `fzyzcjy/torch_memory_saver` @ `d64a639` | CUDA/KV-cache memory offload used during RL rollout. |

**Base & infra**
| Component | Detail |
|-----------|--------|
| **PyTorch + CUDA** | torch `2.11.0+cu130` (CUDA 13.0); TE/Megatron dlopen the bundled `nvidia/*/lib` (see `env.sh`). |
| **Prebuilt kernel wheels** | `yueming-yuan/miles-wheels` @ `cu130-x86_64-v0.5.12` — apex, TE, flash-attn, sglang_router, gateway (no local compile). |
| **Ray** | Actor orchestration across nodes (train + rollout actors; per-job isolated Ray on a unique port). |
| **SLURM** | Cluster scheduler; see `scripts/*.sbatch`. |

## Setup
```bash
bash tools/build.sh          # build the stack (idempotent; resumes via .build-stamps)
source env.sh                # activate the runtime
```
`WANDB_API_KEY` is read from the inherited environment at runtime — it is never
stored in this repo.

## Findings
See **`FINDINGS.md`** — the memory-scaling study (public bf16 stack tops out at
~20 layers / 128k; activation memory, not weights, is the wall; CPU offload came
within ~0.23 GB of 30 layers). Raw data in `MEMSCALE_RESULTS.md`.
