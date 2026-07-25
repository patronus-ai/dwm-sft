#!/usr/bin/env bash
#SBATCH --job-name=zheli-memscale-1node
#SBATCH --partition=gpu-high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
#
# Layer-scaling @ 128k on a SINGLE node (8 B200). No multi-node RoCE, no 4-node
# contention. Grid: TP8/PP1/EP8/ETP1 (CP=1 for DSA; PP=1 for LoRA). bf16 base.
# Override MODEL_ARGS_SH + MODEL for each layer count (5/10/30/40).
set -euo pipefail
ROOT=/workspace/users/zhe-li/repos/glm-sft
RUN=${RUN:-memscale-1node-$SLURM_JOB_ID}
OUT=$ROOT/out/$RUN; mkdir -p "$OUT" "$ROOT/logs"
source "$ROOT/env.sh"; cd "$ROOT/miles"
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo TMPDIR=/tmp no_proxy="127.0.0.1,localhost"

GPUS=8
TP=${TP:-8}; PP=1; CP=1; EP=${EP:-8}; ETP=${ETP:-1}   # TP8*PP1*DP1=8 ; EP8*ETP1=8=TP*DP
SEQ_LEN=${SEQ_LEN:-131072}; MAX_TOK=${MAX_TOK:-$SEQ_LEN}
MODEL_ARGS_SH=${MODEL_ARGS_SH:?set MODEL_ARGS_SH}
MODEL=${MODEL:?set MODEL (hf checkpoint dir)}
echo "[1node] $(hostname) TP=$TP EP=$EP ETP=$ETP seq=$SEQ_LEN model=$MODEL"

source "$MODEL_ARGS_SH"
CKPT_ARGS=(--hf-checkpoint "$MODEL" --load "$MODEL" --save "$OUT/checkpoints" --save-interval 1000000)
SFT_ARGS=(--rollout-function-path miles.rollout.sft_rollout.generate_rollout
  --prompt-data "${PROMPT_DATA:-$ROOT/data/sft_128k.parquet}" --input-key messages --rollout-shuffle
  --num-rollout 6 --rollout-batch-size 8 --global-batch-size 8
  --loss-type sft_loss --calculate-per-token-loss --disable-compute-advantages-and-returns --debug-train-only)
LORA_ARGS=(--lora-rank 16 --lora-alpha 32 --lora-dropout 0.0 --target-modules "gate_proj,up_proj,down_proj" --megatron-to-hf-mode bridge)
PERF_ARGS=(--tensor-model-parallel-size $TP --pipeline-model-parallel-size $PP
  --context-parallel-size $CP --expert-model-parallel-size $EP --expert-tensor-parallel-size $ETP
  --sequence-parallel --seq-length $SEQ_LEN --use-dynamic-batch-size --max-tokens-per-gpu $MAX_TOK
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --moe-token-dispatcher-type alltoall --log-device-memory-used --log-memory-interval 1 --log-interval 1)
OPT_ARGS=(--optimizer adam --lr 1e-4 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95)
MISC_ARGS=(--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32 --attention-backend flash --allgather-cp
  --dsa-attention-backend tilelang --qkv-format thd)
WANDB_ARGS=(); [ -n "${WANDB_API_KEY:-}" ] && WANDB_ARGS=(--use-wandb --wandb-project "${WANDB_PROJECT:-glm-memscale-1node}" --wandb-group "${WANDB_GROUP:-memscale1node}" --wandb-run-id "$RUN")

# Isolated single-node Ray (loopback, unique port + scoped cleanup) — the proven path.
RAY_PORT=$(( 20000 + SLURM_JOB_ID % 20000 )); RAY_TMP=/tmp/ray_zheli_$SLURM_JOB_ID
rm -rf "$RAY_TMP"; mkdir -p "$RAY_TMP"
trap 'pkill -9 -f "ray_zheli_'"$SLURM_JOB_ID"'" 2>/dev/null || true; rm -rf "$RAY_TMP" 2>/dev/null || true' EXIT
export RAY_ADDRESS="127.0.0.1:$RAY_PORT"
ray start --head --node-ip-address 127.0.0.1 --port "$RAY_PORT" --num-gpus $GPUS --num-cpus 64 \
  --temp-dir "$RAY_TMP" --include-dashboard false --disable-usage-stats
sleep 8
set +e
python3 -u train_async.py --actor-num-nodes 1 --actor-num-gpus-per-node $GPUS \
  "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${SFT_ARGS[@]}" "${LORA_ARGS[@]}" \
  "${PERF_ARGS[@]}" "${OPT_ARGS[@]}" "${MISC_ARGS[@]}" "${WANDB_ARGS[@]}"
rc=$?; set -e
echo "[1node] done rc=$rc"; exit $rc
