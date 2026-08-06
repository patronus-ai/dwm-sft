#!/usr/bin/env bash
#SBATCH --job-name=zheli-qwen3-235b-128k
#SBATCH --partition=gpu-low
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
# Qwen3-235B-A22B LoRA-SFT @ 128k on 4 nodes. Execs the GENERIC launcher (a copy of
# the GLM launcher) so the working GLM-5.2 recipe is left completely untouched.
#
# Qwen3-MoE uses standard GQA + dense-softmax attention -> NONE of the GLM-5.2 hacks
# apply: no DSA chunked indexer, no fp8 base-scale fix, no DSA_ALLOW_CP. DSA_BACKEND=none
# tells the generic launcher to drop --dsa-attention-backend.
#
# MEMORY: base ~470 GB bf16 -> ~16 GB/rank at EP32 (128 experts / 94 layers); recompute
# checkpoints ~100 GB + logits/MoE ~15 GB -> ~135 GB peak -> ~45 GB headroom (robust,
# unlike the marginal 744B). Grid TP8/CP1/PP1/EP32/ETP1 (world32, DP4; EP*ETP=32=TP*DP;
# EP32 | 128 experts). LoRA on ATTENTION only (the fix that made the 744B 20-step run work).
#
# CHECKPOINT: NOT downloaded yet. Set MODEL to the text Qwen3-235B-A22B snapshot after an
# HF pull (~470 GB bf16). The Qwen3-VL-235B on the cluster is a DIFFERENT (multimodal)
# arch (Qwen3VLMoeForConditionalGeneration) and is NOT usable with this text recipe.
export MODEL_ARGS_SH=/workspace/users/zhe-li/repos/glm-sft/miles/scripts/models/qwen3-235B-A22B.sh
export MODEL=${MODEL:?set MODEL to the Qwen3-235B-A22B checkpoint snapshot dir (not downloaded yet)}
export PROMPT_DATA=/workspace/users/zhe-li/repos/glm-sft/data/sft_128k.parquet
export SEQ_LEN=131072 MAX_TOK=131072
export TP=8 CP=1 PP=1 EP=32 ETP=1 NUM_EPOCH=${NUM_EPOCH:-1}
export TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"          # attention-only LoRA (no expert LoRA)
export DSA_BACKEND=none                                       # standard attention, not DSA
# flash_attn is a cu12 wheel; TE's cuDNN fused-attn refuses when it sees libcudart.so.12
# next to the cu13 stack ("Multiple libcudart libraries"). Force TE's flash-attn backend.
export NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export WANDB_PROJECT=qwen3-235b WANDB_GROUP=qwen3-235b-128k RUN=qwen3-235b-128k-ep32
exec bash /workspace/users/zhe-li/repos/glm-sft/scripts/sft-128k-generic.sbatch
