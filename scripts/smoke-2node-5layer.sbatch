#!/usr/bin/env bash
#SBATCH --job-name=zheli-2node-5layer
#SBATCH --partition=gpu-high
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
#
# STAGE 3: validate MULTI-NODE (cross-node Ray + RoCE/NCCL + CP + thd/tilelang)
# on the cheap 5-layer model across 2 nodes (16 GPU) BEFORE the 4-node 744B run.
# Proves the exact multi-node machinery of sft-128k-glm744b.sbatch works here.
set -euo pipefail

ROOT=/workspace/users/zhe-li/repos/glm-sft
REF=/workspace/users/yoshinari-fujinuma/repos/glm_finetuning   # read-only: 5-layer model + smoke data
RUN=${RUN:-2node-5layer-$SLURM_JOB_ID}
OUT=$ROOT/out/$RUN; mkdir -p "$OUT" "$ROOT/logs"
cd "$ROOT/miles"; export TMPDIR=/tmp no_proxy="127.0.0.1,localhost"

NNODES=${SLURM_NNODES:-2}; GPN=8; WORLD=$((NNODES*GPN))
# NOTE: DSAttention does NOT support context parallelism (CP must be 1). Cross-node
# RoCE is still exercised by the DP all-reduce (DP=16) + EP all-to-all.
TP=1; CP=1; PP=1; EP=8; ETP=1          # TP1*CP1*PP1*DP16=16 ; EP8 shards 256 experts
SEQ_LEN=4096; MAX_TOK=2048; GBS=32

NET_IFACE=${NET_IFACE:-ens1}
export GLOO_SOCKET_IFNAME=$NET_IFACE NCCL_SOCKET_IFNAME=$NET_IFACE
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_9,mlx5_12,mlx5_13}
export NCCL_NVLS_ENABLE=0 NCCL_CROSS_NIC=0 NCCL_NET_GDR_LEVEL=0 NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export INDEXER_ROPE_NEOX_STYLE=0 NVSHMEM_DISABLE_NCCL=1
_detect_gid() { local h g i; h=$(echo "$NCCL_IB_HCA"|cut -d, -f1); g=/sys/class/infiniband/$h/ports/1/gid_attrs/types
  [ -d "$g" ] && for i in $(seq 0 31); do grep -qi "RoCE v2" "$g/$i" 2>/dev/null && { echo "$i"; return; }; done; echo 7; }

source "$ROOT/env.sh" >/dev/null 2>&1
source scripts/models/glm5.2-744B-A40B_5layer.sh    # MODEL_ARGS
CKPT_ARGS=(--hf-checkpoint "$REF/models/GLM-5.2_5layer" --load "$REF/models/GLM-5.2_5layer" --save "$OUT/checkpoints" --save-interval 1000000)
SFT_ARGS=(--rollout-function-path miles.rollout.sft_rollout.generate_rollout
  --prompt-data "$REF/data/sft_smoke.parquet" --input-key messages --rollout-shuffle
  --num-epoch 1 --rollout-batch-size $GBS --global-batch-size $GBS
  --loss-type sft_loss --calculate-per-token-loss --disable-compute-advantages-and-returns --debug-train-only)
LORA_ARGS=(--lora-rank 16 --lora-alpha 32 --lora-dropout 0.0 --target-modules "gate_proj,up_proj,down_proj" --megatron-to-hf-mode bridge)
PERF_ARGS=(--tensor-model-parallel-size $TP --pipeline-model-parallel-size $PP
  --context-parallel-size $CP --expert-model-parallel-size $EP --expert-tensor-parallel-size $ETP
  --sequence-parallel --seq-length $SEQ_LEN --use-dynamic-batch-size --max-tokens-per-gpu $MAX_TOK
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --moe-token-dispatcher-type alltoall)
OPT_ARGS=(--optimizer adam --lr 1e-4 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95)
MISC_ARGS=(--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32 --attention-backend flash --allgather-cp
  --dsa-attention-backend tilelang --qkv-format thd)
# W&B disabled for validation (read loss from logs) — avoids the key leaking into
# the args dump via --train-env-vars. Re-enable securely for the real run.
WANDB_ARGS=()

TENV='["INDEXER_ROPE_NEOX_STYLE","NVSHMEM_DISABLE_NCCL","GLOO_SOCKET_IFNAME","NCCL_SOCKET_IFNAME","NCCL_IB_DISABLE","NCCL_IB_HCA","NCCL_NVLS_ENABLE","NCCL_CROSS_NIC","NCCL_NET_GDR_LEVEL","NCCL_DEBUG","PYTORCH_CUDA_ALLOC_CONF"]'
TRAIN_ENV_JSON=$(TENV="$TENV" python3 -c 'import os,json;print(json.dumps({k:os.environ[k] for k in json.loads(os.environ["TENV"]) if k in os.environ}))')
TRAIN_CMD=(python3 -u train_async.py --actor-num-nodes $NNODES --actor-num-gpus-per-node $GPN --train-env-vars "$TRAIN_ENV_JSON"
  "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${SFT_ARGS[@]}" "${LORA_ARGS[@]}" "${PERF_ARGS[@]}" "${OPT_ARGS[@]}" "${MISC_ARGS[@]}" "${WANDB_ARGS[@]}")

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); head=${nodes[0]}
head_ip=$(srun --overlap --nodes=1 --ntasks=1 -w "$head" bash -lc "ip -o -4 addr show $NET_IFACE | awk '{print \$4}' | cut -d/ -f1")
echo "[2node] nodes=${nodes[*]} head=$head ip=$head_ip TP=$TP CP=$CP EP=$EP"
RAY_ENV="cd '$ROOT/miles' && source '$ROOT/env.sh' >/dev/null 2>&1 && export NCCL_IB_GID_INDEX=\$($(declare -f _detect_gid); _detect_gid)"
srun --overlap --nodes=1 --ntasks=1 -w "$head" bash -lc "$RAY_ENV && exec ray start --head --node-ip-address='$head_ip' --port=6379 --num-gpus $GPN --num-cpus 32 --temp-dir /tmp/ray_zheli_$SLURM_JOB_ID --disable-usage-stats --block" &
sleep 20
for w in "${nodes[@]:1}"; do
  srun --overlap --nodes=1 --ntasks=1 -w "$w" bash -lc "$RAY_ENV && exec ray start --address='$head_ip:6379' --num-gpus $GPN --num-cpus 32 --temp-dir /tmp/ray_zheli_$SLURM_JOB_ID --block" &
done
sleep 25
export RAY_ADDRESS="$head_ip:6379"
echo "[2node] launching driver on $head"
srun --overlap --nodes=1 --ntasks=1 -w "$head" "${TRAIN_CMD[@]}"
rc=$?
echo "[2node] done rc=$rc"; exit $rc
