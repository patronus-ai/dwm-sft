#!/usr/bin/env bash
#SBATCH --job-name=zheli-30L-2node-cp2
#SBATCH --partition=gpu-low
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=03:00:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
# 30-layer @ 128k on 2 nodes (16 B200) with CONTEXT PARALLELISM CP=2.
# Grid TP8/CP2/PP1/EP8/ETP1 (world 16: TP8*CP2*DP1; EP*ETP=8=TP*DP; EP8|256).
# CP shards the SEQUENCE, so each rank holds 128k/2 = 64k of activation — the
# same profile as the 30L/64k run that FIT on 1 node, but at the full 128k context.
# This is the direct test of "context parallelism is the missing lever". Full recompute.
export MODEL_ARGS_SH=/workspace/users/zhe-li/repos/glm-sft/miles/scripts/models/glm5.2-744B-A40B_30layer.sh
export MODEL=/workspace/users/zhe-li/repos/glm-sft/models/GLM-5.2_30layer
export TP=8 CP=2 PP=1 EP=8 ETP=1 NUM_EPOCH=1
export WANDB_PROJECT=glm-memscale-cp WANDB_GROUP=memscale-cp2 RUN=memscale-30L-2node-cp2
exec bash /workspace/users/zhe-li/repos/glm-sft/scripts/sft-128k-glm744b.sbatch
