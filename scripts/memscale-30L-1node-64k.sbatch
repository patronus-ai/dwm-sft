#!/usr/bin/env bash
#SBATCH --job-name=zheli-30L-1node-64k
#SBATCH --partition=gpu-low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
# 30-layer @ 64k on ONE node (8 B200), grid TP8/PP1/EP8/ETP1, full recompute.
# Tests the activation-vs-base split: at 64k the ~seq-linear activation working
# set roughly halves vs 128k, so 30L (which OOMs at 128k on every node count)
# should FIT. base/rank ~69 GB (EP8) + ~half the 128k activation.
export MODEL_ARGS_SH=/workspace/users/zhe-li/repos/glm-sft/miles/scripts/models/glm5.2-744B-A40B_30layer.sh
export MODEL=/workspace/users/zhe-li/repos/glm-sft/models/GLM-5.2_30layer
export SEQ_LEN=65536
export PROMPT_DATA=/workspace/users/zhe-li/repos/glm-sft/data/sft_64k.parquet
export TP=8 EP=8 ETP=1
export WANDB_PROJECT=glm-memscale-64k WANDB_GROUP=memscale64k RUN=memscale-30L-1node-64k
exec bash /workspace/users/zhe-li/repos/glm-sft/scripts/memscale-1node.sbatch
