SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/glm5.2-744B-A40B.sh"
# 10-layer decapitated GLM-5.2 for the memory-scaling experiment.
N_MOE_LAYERS=7   # 10 total = 3 dense + 7 MoE
for ((i=0; i<${#MODEL_ARGS[@]}; i++)); do
    case "${MODEL_ARGS[$i]}" in
        --num-layers) MODEL_ARGS[$((i+1))]=$((N_DENSE_LAYERS + N_MOE_LAYERS)) ;;
        --moe-layer-freq) MODEL_ARGS[$((i+1))]="[0]*${N_DENSE_LAYERS}+[1]*${N_MOE_LAYERS}" ;;
    esac
done
