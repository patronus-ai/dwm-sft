# overlays/

Files here are my own additions/edits that live *inside* the cloned upstream
repos (which are git-ignored and reconstructed by `tools/build.sh`). Each path
mirrors its destination relative to the repo root. `build.sh` copies them into
place after cloning (see its "apply overlays" stage), so a fresh build lands a
working tree.

- `miles/scripts/models/glm5.2-744B-A40B_{10,20,30,40}layer.sh` — pruned N-layer
  GLM-5.2 arch configs for the memory-scaling experiment. Each sources the
  upstream `glm5.2-744B-A40B.sh` and overrides `--num-layers` / `--moe-layer-freq`
  (N total = 3 dense + (N-3) MoE).

Note: the Megatron-LM DSA-dispatch patch is NOT here — `build.sh` applies it
directly (inline heredoc) during the Megatron stage.
