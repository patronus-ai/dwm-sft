#!/usr/bin/env bash
#SBATCH --job-name=zheli-glm52-tilelang
#SBATCH --partition=gpu-high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
#
# VALIDATION smoke: prove zhe-li's INDEPENDENT build (public radixark/miles +
# self-built .venv) runs GLM-5.2 LoRA-SFT end-to-end. Only variable vs the
# earlier proven run is the ENV (this tree's venv/miles) — model + data are the
# same colleague artifacts (read-only) so a green rc=0 attributes to the env.
set -euo pipefail

ROOT=/workspace/users/zhe-li/repos/glm-sft
REF=/workspace/users/yoshinari-fujinuma/repos/glm_finetuning   # READ-ONLY: model + data only
RUN=${RUN:-glm52-5layer-tilelang-$SLURM_JOB_ID}
OUT=$ROOT/out/$RUN
mkdir -p "$OUT" "$ROOT/logs"

source "$ROOT/env.sh"          # <-- MY tree's env + venv
cd "$ROOT/miles"               # <-- MY editable miles

export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo TMPDIR=/tmp
export no_proxy="127.0.0.1,localhost"
echo "[validate] job=$SLURM_JOB_ID node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES run=$RUN"
echo "[validate] miles=$(git -C "$ROOT/miles" rev-parse --short HEAD 2>/dev/null)  venv=$ROOT/.venv"

source scripts/models/glm5.2-744B-A40B_5layer.sh        # MODEL_ARGS (mine)

CKPT_ARGS=(
  --hf-checkpoint "$REF/models/GLM-5.2_5layer"           # read-only reuse
  --load          "$REF/models/GLM-5.2_5layer"
  --save          "$OUT/checkpoints"
  --save-interval 1000000
)
SFT_ARGS=(
  --rollout-function-path miles.rollout.sft_rollout.generate_rollout
  --prompt-data "$REF/data/sft_smoke.parquet"            # same smoke data
  --input-key messages
  --rollout-shuffle --num-epoch 1
  --rollout-batch-size 32 --global-batch-size 32
  --loss-type sft_loss --calculate-per-token-loss
  --disable-compute-advantages-and-returns --debug-train-only
)
LORA_ARGS=(
  --lora-rank 16 --lora-alpha 32 --lora-dropout 0.0
  --target-modules "gate_proj,up_proj,down_proj"
  --megatron-to-hf-mode bridge
)
PERF_ARGS=(
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --context-parallel-size 1 --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  # thd+tilelang IS recompute-safe -> exercise activation recompute (the 128k path).
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --use-dynamic-batch-size --max-tokens-per-gpu 4096    # packed thd; small for the smoke
  --moe-token-dispatcher-type alltoall
)
OPTIMIZER_ARGS=(--optimizer adam --lr 1e-4 --lr-decay-style constant
  --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95)
MISC_ARGS=(--attention-dropout 0.0 --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
  --attention-backend flash --allgather-cp
  # LONG-CTX backend: fused tilelang DSA kernels + packed thd. Recompute-safe
  # (top-k rides on packed_seq_params). This is what the 744B/128k run will use.
  --dsa-attention-backend tilelang
  --qkv-format thd)

WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
  export WANDB_API_KEY
  WANDB_ARGS=(--use-wandb --wandb-project "${WANDB_PROJECT:-glm52-smoke-zheli}"
              --wandb-group "${WANDB_GROUP:-glm52-5layer-tilelang}" --wandb-run-id "$RUN")
fi

# Isolated Ray: per-job unique port + temp-dir; scoped teardown (never a global kill).
RAY_PORT=${RAY_PORT:-$(( 20000 + SLURM_JOB_ID % 20000 ))}
RAY_TMP=/tmp/ray_zheli_$SLURM_JOB_ID
rm -rf "$RAY_TMP"; mkdir -p "$RAY_TMP"
trap 'pkill -9 -f "ray_zheli_'"$SLURM_JOB_ID"'" 2>/dev/null || true; rm -rf "$RAY_TMP" 2>/dev/null || true' EXIT
export RAY_ADDRESS="127.0.0.1:$RAY_PORT"
ray start --head --node-ip-address 127.0.0.1 --port "$RAY_PORT" \
  --num-gpus 1 --num-cpus 16 --temp-dir "$RAY_TMP" --include-dashboard false --disable-usage-stats
sleep 8

set +e
python3 -u train_async.py \
  --actor-num-nodes 1 --actor-num-gpus-per-node 1 \
  "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${SFT_ARGS[@]}" "${LORA_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" "${PERF_ARGS[@]}" "${MISC_ARGS[@]}" "${WANDB_ARGS[@]}"
rc=$?
set -e
echo "[validate] done rc=$rc"
exit $rc
