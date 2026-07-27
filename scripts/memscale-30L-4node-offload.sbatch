#!/usr/bin/env bash
#SBATCH --job-name=zheli-30L-4node-offload
#SBATCH --partition=gpu-low
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=03:00:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
# 30-layer @128k on 4 nodes (32 B200), grid TP8/PP1/EP32/ETP1 + CPU activation
# offload (recompute OFF). Base/rank ~19.7 GB (EP32) leaves ~16 GB more headroom
# than the 2-node offload run that was only ~2.23 GiB short -> expect FIT.
export MODEL_ARGS_SH=/workspace/users/zhe-li/repos/glm-sft/miles/scripts/models/glm5.2-744B-A40B_30layer.sh
export MODEL=/workspace/users/zhe-li/repos/glm-sft/models/GLM-5.2_30layer
export TP=8 PP=1 EP=32 ETP=1 NUM_EPOCH=1
export OFFLOAD_LAYERS=29                                   # max (< num_layers); recompute auto-disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=glm-memscale-4node WANDB_GROUP=memscale4node-offload RUN=memscale-30L-4node-offload
exec bash /workspace/users/zhe-li/repos/glm-sft/scripts/sft-128k-glm744b.sbatch
