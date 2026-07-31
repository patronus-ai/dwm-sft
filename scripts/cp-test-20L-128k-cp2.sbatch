#!/usr/bin/env bash
#SBATCH --job-name=zheli-20L-cp2-128k
#SBATCH --partition=gpu-low
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
# OPTION A: 20L (~172B) @ FULL 128k on 2 nodes, CP=2 + chunked indexer.
# 20L fit at 128k on ONE node (base ~44 GB, EP8); on 2 nodes with CP=2 (64k/rank
# activation) it fits with room. GOAL: first clean CP-at-128k run end-to-end
# + a real 2-node/128k throughput number.
export MODEL_ARGS_SH=/workspace/users/zhe-li/repos/glm-sft/miles/scripts/models/glm5.2-744B-A40B_20layer.sh
export MODEL=/workspace/users/zhe-li/repos/glm-sft/models/GLM-5.2_20layer
export SEQ_LEN=131072
export PROMPT_DATA=/workspace/users/zhe-li/repos/glm-sft/data/sft_128k.parquet
export TP=8 CP=2 PP=1 EP=8 ETP=1 NUM_EPOCH=1
export DSA_ALLOW_CP=1
export DSA_INDEXER_Q_CHUNK=4096
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=glm-cptest WANDB_GROUP=cptest RUN=cptest-20L-128k-cp2
exec bash /workspace/users/zhe-li/repos/glm-sft/scripts/sft-128k-glm744b.sbatch
