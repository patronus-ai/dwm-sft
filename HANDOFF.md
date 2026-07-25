# GLM-5.2 LoRA-SFT — reproduction / handoff

Independent, **public-only**, editable Miles stack for **GLM-5.2 (744B-A40B MoE)** LoRA/QLoRA
SFT on the Patronus B200 cluster (driver CUDA 13.0). No private forks, no dependence on any
other user's tree. Validated: 5-layer LoRA-SFT smoke, `rc=0`, loss 12.6→6.8, adapter saved.

Original build under `/workspace/users/zhe-li/repos/glm-sft/`. Artifacts: `build.sh` (env
builder, resumable), `env.sh` (runtime shell), `smoke-5layer-validate.sbatch` (validation).

---

## 0. Prerequisites (cluster-provided — verify, don't install)
- **Hardware:** node-[0-3], each 8× B200 (183 GB). SLURM partitions `gpu-low` (preemptible)
  and `gpu-high` (not preempted — use for runs that must finish).
- **`uv`** on PATH (`/usr/bin/uv`), Python 3.12.
- **Your per-user profile** `/workspace/users/<you>/env/activate.sh` (routes all caches to the
  shared `/workspace/.cache`). Every user has one.
- **Shared caches** `/workspace/.cache/{uv,pip,cargo}` — reused across users; makes rebuilds fast.
- **Shared HF cache** `/workspace/.cache/huggingface/hub` — already holds `zai-org/GLM-5.2-FP8`
  (705 GB), `nvidia/GLM-5.2-NVFP4` (433 GB). Do NOT re-download; point `--hf-checkpoint` at a
  snapshot dir there.
- **RDMA libs** already system-wide (`ldconfig -p | grep -E 'libibverbs|librdmacm'`) → the build
  does NOT apt-install anything.
- **GitHub:** only PUBLIC repos are cloned (no private auth needed).

## 1. Per-colleague substitutions (the only things to change)
Replace `zhe-li` everywhere with your own user:
- `ROOT=/workspace/users/<you>/repos/glm-sft` (top of `build.sh`, `env.sh`, the sbatch)
- `source /workspace/users/<you>/env/activate.sh` (in `build.sh` + `env.sh`)
- Ray scoping string `ray_zheli_$JOBID` → `ray_<you>_$JOBID` (in the sbatch trap/temp-dir)
- W&B: your own `WANDB_API_KEY` in env; set `--wandb-project`/entity as you like
- SLURM `--output` log path → your dir
Everything else is cluster-shared or public and needs no change.

## 2. Build (≈30–60 min, resumable)
```
export ROOT=/workspace/users/<you>/repos/glm-sft
mkdir -p "$ROOT"
git clone https://github.com/radixark/miles "$ROOT/miles"       # public: LoRA + glm5 spec + SFT
cp build.sh env.sh "$ROOT"/                                      # then edit ROOT/activate paths
bash "$ROOT/build.sh"                                            # stamp-based; safe to re-run
```
`build.sh` clones the rest (all PUBLIC) and builds `$ROOT/.venv`:
- `radixark/Megatron-LM@miles-main` (editable) + one-line DSA-dispatch patch
- `radixark/Megatron-Bridge@bridge`, `ISEEKYAN/mbridge`, `torch_memory_saver`
- prebuilt kernel wheels `yueming-yuan/miles-wheels@cu130-x86_64-v0.5.12` (flash-attn 2/3, apex, TE)
- `fast-hadamard-transform` (cu13 compile), **tilelang + apache-tvm-ffi**, `pulp`
- `sgl-project/sglang@sglang-miles` (editable; import-time dep), `miles` (editable)
Ends with import checks: `megatron OK`, `tilelang OK`, `glm5 spec import OK`, `[build.sh] COMPLETE`.

## 3. Validate
```
sbatch "$ROOT/smoke-5layer-validate.sbatch"     # 1 GPU, gpu-high
```
Expect `[validate] done rc=0` with a decreasing loss and a saved LoRA adapter under
`$ROOT/out/.../checkpoints/iter_*/adapter`. (Uses a 5-layer GLM-5.2 test model + a tiny SFT
parquet; swap in your own model path / data.)

## 4. Fixes already baked in (do NOT rediscover — `radixark/miles@main` gaps)
1. `nvidia` is a namespace pkg → `nvidia.__file__` is None; use `nvidia.__path__[0]` for CU13.
2. Install `nvidia-cuda-nvcc` on its OWN pip line (bundling a failing pkg under `|| true`
   silently skipped nvcc). cu13 nvcc lands at `nvidia/cu13/bin/nvcc`.
3. `pulp` missing — modelopt is installed `--no-deps` (to protect the torch pin) and imports
   `pulp` via the bridge. → `pip install pulp`.
4. `--dsa-attention-backend` defaults to `tilelang` (needs `--qkv-format thd` + packed seqs).
5. `megatron` DSA backend + `bshd` is NOT recompute-safe (cross-layer index-share reads stale
   top-k under activation recompute).

## 5. Which DSA config to use (IMPORTANT)
| Scenario | backend | qkv-format | recompute | notes |
|---|---|---|---|---|
| Small / 5-layer smoke | `megatron` | `bshd` + `--micro-batch-size 1` | OFF | validated here |
| **744B / long ctx (128k)** | **`tilelang`** | **`thd`** (+ `--use-dynamic-batch-size`) | **ON** | recompute-safe + packing-efficient; **not yet validated at train time — do a smoke first** |

Rationale: at 128k on 744B, activation recompute is mandatory (memory), so the recompute-unsafe
`megatron`+`bshd` path can't be used — you must use `thd`+`tilelang` (rides top-k on
`packed_seq_params`, recompute-safe; packing avoids padding waste on long sequences).

## 6. Feasibility ceilings (this cluster)
- **Full-parameter SFT of 744B: infeasible** (~12 TB optimizer/master/grad vs 32×183 GB = 5.9 TB).
- **LoRA/QLoRA: feasible.** Frozen base bf16 1.4 TB / FP8 0.7 TB / NVFP4 0.43 TB + small adapter.
  With 4 nodes you have headroom for FP8/bf16 LoRA at 128k with context-parallel (CP) across nodes.
- NVFP4-QLoRA memory unlock (fits 744B on 2 nodes) lives only in the private `patronus-ai/*`
  forks; with 4 nodes it's optional.

## 7. Don'ts
- Don't copy the 1.7 TB of model weights — reference the shared HF cache / a test model.
- Don't `ray stop --force` on a shared node (kills neighbours' Ray); use per-job scoped
  `pkill -f ray_<you>_$JOBID` + a unique Ray port/temp-dir (the sbatch already does this).
- Don't write outside your `/workspace/users/<you>/` subtree.
