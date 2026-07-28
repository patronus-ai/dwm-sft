#!/usr/bin/env bash
#SBATCH --job-name=zheli-glm744b-128k-sft
#SBATCH --partition=gpu-high
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
#
# GLM-5.2 744B-A40B LoRA-SFT @ 128k context, 4 nodes × 8 B200 = 32 GPUs.
# FP8 frozen base + bf16 LoRA adapter; thd+tilelang DSA backend (recompute-safe,
# packing-efficient) + context parallel for the 128k sequence.
#
# !!! PREREQS BEFORE THIS IS SAFE TO RUN (see HANDOFF.md / notes below) !!!
#   1. thd+tilelang backend validated on the 5-layer smoke (cheap, 1 GPU).
#   2. Multi-node RoCE validated (nccl_test) — knobs below are ported from the
#      colleague's proven env but re-verify per-node GID detection.
#   3. Parallelism grid (TP/CP/EP/PP) tuned so 744B FP8 + 128k activations fit.
set -euo pipefail

ROOT=/workspace/users/zhe-li/repos/glm-sft
RUN=${RUN:-glm744b-128k-$SLURM_JOB_ID}
OUT=$ROOT/out/$RUN
mkdir -p "$OUT" "$ROOT/logs"
cd "$ROOT/miles"
export TMPDIR=/tmp no_proxy="127.0.0.1,localhost"

NNODES=${SLURM_NNODES:-4}; GPN=${GPUS_PER_NODE:-8}; WORLD=$((NNODES*GPN))
# --- parallelism grid (STARTING POINT — tune) : TP*CP*PP*DP must == WORLD(32) ---
# CP overridable via env. The DSA *SFT* path DOES have CP plumbing (slice_with_cp,
# CP-aware logits); only --use-rollout-indexer-replay asserts CP==1. So the 128k sequence
# is NOT split along the seq dim; instead PP splits the 78 layers (activation
# memory per rank ∝ layers/PP), TP+sequence-parallel split hidden/activations,
# DSA sparsity (top-2048) keeps attention cheap, and full recompute is on.
TP=${TP:-8}; CP=${CP:-1}; PP=${PP:-1}; EP=${EP:-4}; ETP=${ETP:-$TP}
# TP8 * CP1 * PP1 * DP4 = 32.  PP=1 is REQUIRED by miles LoRA (weight-sync/adapter
# path asserts PP==1). PP=1 kept all 78 layers -> OOM at 128k when the base was
# dequantized to bf16 (~1.5 TB). FIX: --fp8-param-gather keeps the frozen base in
# FP8 (~705 GB, ~half) via TE fp8_model_init, so the FP8 checkpoint loads with no
# bf16 cast and 128k activations fit. MoE grid: EP*ETP == TP*DP (32).
# ETP=TP=8 => EP=4.  Check: EP*ETP = 4*8 = 32 = TP*DP ✓ ; ETP*EP*PP = 8*4*1 = 32 ✓
SEQ_LEN=${SEQ_LEN:-131072}                       # 128k
MAX_TOK=${MAX_TOK:-$SEQ_LEN}                      # no CP split -> one rank sees full 128k/microbatch
GBS=${GBS:-8}
MODEL=${MODEL:-/workspace/.cache/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1}

# --- multi-node NCCL/RoCE (ported from the validated cluster setup) ----------
NET_IFACE=${NET_IFACE:-ens1}
export GLOO_SOCKET_IFNAME=$NET_IFACE NCCL_SOCKET_IFNAME=$NET_IFACE
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_9,mlx5_12,mlx5_13}
export NCCL_NVLS_ENABLE=0        # NVLS deadlocks 16+ rank cross-node collective here
export NCCL_CROSS_NIC=0          # 8 HCAs on 8 subnets -> rail-align the MoE all-to-all
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-0}   # host-staged (missing dmabuf verb)
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export INDEXER_ROPE_NEOX_STYLE=0 NVSHMEM_DISABLE_NCCL=1
# Per-node RoCE-v2 GID index differs (e.g. node-3=3, others=7); detect locally.
_detect_gid() { local h; h=$(echo "$NCCL_IB_HCA" | cut -d, -f1)
  local g=/sys/class/infiniband/$h/ports/1/gid_attrs/types
  [ -d "$g" ] && for i in $(seq 0 31); do grep -qi "RoCE v2" "$g/$i" 2>/dev/null && { echo "$i"; return; }; done; echo 7; }

MODEL_ARGS_SH=${MODEL_ARGS_SH:-scripts/models/glm5.2-744B-A40B.sh}   # real 78-layer arch (overridable for layer-scaling)

# args assembled below and shared with each node's ray env via --train-env-vars
build_args() {
source "$ROOT/env.sh" >/dev/null 2>&1
source "$MODEL_ARGS_SH"
CKPT_ARGS=(--hf-checkpoint "$MODEL" --load "$MODEL" --save "$OUT/checkpoints" --save-interval ${SAVE_INTERVAL:-50})
SFT_ARGS=(--rollout-function-path miles.rollout.sft_rollout.generate_rollout
  --prompt-data "${PROMPT_DATA:-$ROOT/data/sft_128k.parquet}" --input-key messages
  --rollout-shuffle --num-epoch ${NUM_EPOCH:-1} --rollout-batch-size $GBS --global-batch-size $GBS
  --loss-type sft_loss --calculate-per-token-loss
  --disable-compute-advantages-and-returns --debug-train-only)
LORA_ARGS=(--lora-rank ${LORA_RANK:-16} --lora-alpha ${LORA_ALPHA:-32} --lora-dropout 0.0
  --target-modules "gate_proj,up_proj,down_proj" --megatron-to-hf-mode bridge)
  # NOTE: public radixark/miles has NO --lora-base-precision (that's a patronus-nvfp4
  # feature). Frozen base loads in bf16 (bridge dequantizes the FP8 checkpoint) ->
  # this is bf16-LoRA, not QLoRA. Fits on 32 GPU with the TP4/PP4/EP8 sharding.
PERF_ARGS=(--tensor-model-parallel-size $TP --pipeline-model-parallel-size $PP
  --context-parallel-size $CP --expert-model-parallel-size $EP --expert-tensor-parallel-size $ETP
  --sequence-parallel --seq-length $SEQ_LEN
  --use-dynamic-batch-size --max-tokens-per-gpu $MAX_TOK
  --moe-token-dispatcher-type alltoall)
# Memory strategy — MUTUALLY EXCLUSIVE (transformer_config.py:1237):
#   OFFLOAD_LAYERS set  -> CPU activation offload of N layers to host RAM (needs PP=1, TE impl,
#                          and recompute OFF). Keeps activations, streams them to CPU.
#   otherwise           -> full activation recompute (discard + recompute in backward).
if [ -n "${OFFLOAD_LAYERS:-}" ]; then
  PERF_ARGS+=(--cpu-offloading-num-layers "$OFFLOAD_LAYERS")
  echo "[offload] CPU activation offloading ON: $OFFLOAD_LAYERS layers -> host (recompute disabled)"
else
  PERF_ARGS+=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1)
fi
# fp8-param-gather is OPTIONAL (FP8_PG=1). The layer-scaling experiment runs
# WITHOUT it to find the natural bf16-base memory breaking point.
[ "${FP8_PG:-0}" = "1" ] && PERF_ARGS+=(--fp8-param-gather)
OPT_ARGS=(--optimizer adam --lr ${LR:-1e-4} --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95)
MISC_ARGS=(--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32 --attention-backend flash --allgather-cp
  --dsa-attention-backend tilelang --qkv-format thd)     # long-ctx: recompute-safe + packed
# CE_FP16=1 -> compute the LM cross-entropy in bf16 instead of fp32. At 128k the
# per-rank vocab-parallel logits are ~10 GB in fp32; bf16 halves that (~5 GB saved).
[ "${CE_FP16:-0}" = "1" ] && MISC_ARGS+=(--fp16-lm-cross-entropy)
# W&B ON, securely: actors get WANDB_API_KEY from the inherited env (RAY_ENV ->
# env.sh -> activate.sh), NOT via --train-env-vars, so the key never hits the log.
WANDB_ARGS=(--use-wandb --wandb-project "${WANDB_PROJECT:-glm744b-128k-zheli}"
  --wandb-group "${WANDB_GROUP:-memscale}" --wandb-run-id "$RUN")
}
build_args

TENV='["INDEXER_ROPE_NEOX_STYLE","NVSHMEM_DISABLE_NCCL","GLOO_SOCKET_IFNAME","NCCL_SOCKET_IFNAME","NCCL_IB_DISABLE","NCCL_IB_HCA","NCCL_NVLS_ENABLE","NCCL_CROSS_NIC","NCCL_NET_GDR_LEVEL","NCCL_DEBUG","PYTORCH_CUDA_ALLOC_CONF","DSA_ALLOW_CP"]'
TRAIN_ENV_JSON=$(TENV="$TENV" python3 -c 'import os,json;print(json.dumps({k:os.environ[k] for k in json.loads(os.environ["TENV"]) if k in os.environ}))')

TRAIN_CMD=(python3 -u train_async.py --actor-num-nodes $NNODES --actor-num-gpus-per-node $GPN
  --train-env-vars "$TRAIN_ENV_JSON"
  "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${SFT_ARGS[@]}" "${LORA_ARGS[@]}"
  "${PERF_ARGS[@]}" "${OPT_ARGS[@]}" "${MISC_ARGS[@]}" "${WANDB_ARGS[@]}")

# --- start Ray on the fabric IP (10.65.x), head on node0 + workers via srun ----
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); head=${nodes[0]}
head_ip=$(srun --overlap --nodes=1 --ntasks=1 -w "$head" bash -lc "ip -o -4 addr show $NET_IFACE | awk '{print \$4}' | cut -d/ -f1")
RAY_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))     # unique per-job -> no stale-Redis session clash
echo "[128k] nodes=${nodes[*]} head=$head ip=$head_ip port=$RAY_PORT TP=$TP CP=$CP EP=$EP PP=$PP seq=$SEQ_LEN max_tok=$MAX_TOK"

# UP-FRONT cleanup: kill MY orphaned Ray from prior (dead) jobs on every allocated
# node, scoped to my temp-dir prefix (never a neighbour's Ray). Runs before this
# job's Ray exists, so it can't kill our own. Exit trap does the same on the way out.
for n in "${nodes[@]}"; do
  srun --overlap --nodes=1 --ntasks=1 -w "$n" bash -lc "pkill -9 -f ray_zheli_ 2>/dev/null; rm -rf /tmp/ray_zheli_* 2>/dev/null" || true
done
trap 'for n in "${nodes[@]}"; do srun --overlap --nodes=1 --ntasks=1 -w "$n" bash -lc "pkill -9 -f ray_zheli_'"$SLURM_JOB_ID"' 2>/dev/null; rm -rf /tmp/ray_zheli_'"$SLURM_JOB_ID"' 2>/dev/null" 2>/dev/null || true; done' EXIT
sleep 3

RAY_ENV="cd '$ROOT/miles' && source '$ROOT/env.sh' >/dev/null 2>&1 && export NCCL_IB_GID_INDEX=\$($(declare -f _detect_gid); _detect_gid)"
srun --overlap --nodes=1 --ntasks=1 -w "$head" bash -lc "$RAY_ENV && exec ray start --head --node-ip-address='$head_ip' --port=$RAY_PORT --num-gpus $GPN --num-cpus 64 --temp-dir /tmp/ray_zheli_$SLURM_JOB_ID --disable-usage-stats --block" &
sleep 20
for w in "${nodes[@]:1}"; do
  srun --overlap --nodes=1 --ntasks=1 -w "$w" bash -lc "$RAY_ENV && exec ray start --address='$head_ip:$RAY_PORT' --num-gpus $GPN --num-cpus 64 --temp-dir /tmp/ray_zheli_$SLURM_JOB_ID --block" &
done
sleep 25
export RAY_ADDRESS="$head_ip:$RAY_PORT"
echo "[128k] launching driver on $head"
srun --overlap --nodes=1 --ntasks=1 -w "$head" "${TRAIN_CMD[@]}"
rc=$?
echo "[128k] done rc=$rc"; exit $rc
