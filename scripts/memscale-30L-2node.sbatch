#!/usr/bin/env bash
#SBATCH --job-name=zheli-30L-2node
#SBATCH --partition=gpu-low
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
export MODEL_ARGS_SH=/workspace/users/zhe-li/repos/glm-sft/miles/scripts/models/glm5.2-744B-A40B_30layer.sh
export MODEL=/workspace/users/zhe-li/repos/glm-sft/models/GLM-5.2_30layer
export TP=8 PP=1 EP=16 ETP=1 NUM_EPOCH=1
export WANDB_PROJECT=glm-memscale-2node WANDB_GROUP=memscale2node RUN=memscale-30L-2node
exec bash /workspace/users/zhe-li/repos/glm-sft/scripts/sft-128k-glm744b.sbatch
