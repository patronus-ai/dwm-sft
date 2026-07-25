#!/usr/bin/env bash
#SBATCH --job-name=zheli-30L-2node-offload
#SBATCH --partition=gpu-low
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=03:00:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
# 30-layer @128k on 2 nodes (EP16), same grid as the run that OOM'd by ~21.6 GB,
# but with CPU ACTIVATION OFFLOADING instead of recompute (mutually exclusive).
# Offloads 28/30 layers' activations to host RAM -> frees GPU; costs PCIe/C2C bw.
export MODEL_ARGS_SH=/workspace/users/zhe-li/repos/glm-sft/miles/scripts/models/glm5.2-744B-A40B_30layer.sh
export MODEL=/workspace/users/zhe-li/repos/glm-sft/models/GLM-5.2_30layer
export TP=8 PP=1 EP=16 ETP=1 NUM_EPOCH=1
export OFFLOAD_LAYERS=29                                   # max (< num_layers)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True    # (confirmed applied; frag not the wall)
export CE_FP16=1                                           # bf16 LM cross-entropy: frees ~5 GB/rank (logits were the residual)
export WANDB_PROJECT=glm-memscale-2node WANDB_GROUP=memscale2node-offload RUN=memscale-30L-2node-offload
exec bash /workspace/users/zhe-li/repos/glm-sft/scripts/sft-128k-glm744b.sbatch
