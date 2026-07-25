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
