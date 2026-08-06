#!/usr/bin/env bash
#SBATCH --job-name=zheli-full744b-128k
#SBATCH --partition=gpu-low
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
# FULL GLM-5.2 744B (78 layers) LoRA-SFT @ 128k on 4 nodes -- NON-fp8 route.
# Key: EP32 shards base to ~50 GB/rank (vs 99 at EP16), leaving room to fit the
# 128k activation of all 78 layers via CPU OFFLOAD. Grid TP8/CP1/PP1/EP32/ETP1
# (world32, DP4; EP*ETP=32=TP*DP; EP32|256 experts). Chunked DSA indexer kills
# the O(S^2) 206 GiB wall. base-scale fix makes the loaded base correct.
export MODEL_ARGS_SH=/workspace/users/zhe-li/repos/glm-sft/miles/scripts/models/glm5.2-744B-A40B.sh
export MODEL=/workspace/.cache/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1
export PROMPT_DATA=/workspace/users/zhe-li/repos/glm-sft/data/sft_128k.parquet
export SEQ_LEN=131072
export MAX_TOK=${MAX_TOK:-131072}
export TP=8 CP=1 PP=1 EP=32 ETP=1 NUM_EPOCH=1
# recompute mode: no offload (OFFLOAD_LAYERS unset -> full recompute)
export DSA_INDEXER_Q_CHUNK=${DSA_INDEXER_Q_CHUNK:-4096}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=glm-full744b WANDB_GROUP=full744b-128k RUN=full744b-128k-ep32-recompute
exec bash /workspace/users/zhe-li/repos/glm-sft/scripts/sft-128k-glm744b.sbatch
