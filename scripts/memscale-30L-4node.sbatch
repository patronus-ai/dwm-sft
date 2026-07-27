#!/usr/bin/env bash
#SBATCH --job-name=zheli-30L-4node
#SBATCH --partition=gpu-low
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=03:00:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
# 30-layer @128k on 4 nodes (32 B200), grid TP8/PP1/EP32/ETP1 (EP32 | 256 experts;
# EP*ETP=32=TP*DP). Expert sharding halves base/rank again (~36->~18 GB vs 2-node),
# freeing ~18 GB — more than the ~11.5 GB the 2-node recompute run was short.
# Plain full recompute (no offload needed). Expect FIT with ~6 GB margin.
export MODEL_ARGS_SH=/workspace/users/zhe-li/repos/glm-sft/miles/scripts/models/glm5.2-744B-A40B_30layer.sh
export MODEL=/workspace/users/zhe-li/repos/glm-sft/models/GLM-5.2_30layer
export TP=8 PP=1 EP=32 ETP=1 NUM_EPOCH=1
export WANDB_PROJECT=glm-memscale-4node WANDB_GROUP=memscale4node RUN=memscale-30L-4node
exec bash /workspace/users/zhe-li/repos/glm-sft/scripts/sft-128k-glm744b.sbatch
