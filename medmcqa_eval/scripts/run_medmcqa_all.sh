#!/bin/bash
set -euo pipefail

cd /projects/checkpoint/medmcqa_eval

SBATCH_CMD=(sbatch)
if [ -n "${SBATCH_NODELIST_OVERRIDE:-}" ]; then
  SBATCH_CMD+=(-w "${SBATCH_NODELIST_OVERRIDE}")
fi
if [ -n "${SBATCH_GRES_OVERRIDE:-}" ]; then
  SBATCH_CMD+=(--gres "${SBATCH_GRES_OVERRIDE}")
fi

models=(
  gemma-3-4b-it
  lora-fulltoken-s1
  lora-fulltoken-s2
  dora-fulltoken-s1
  dora-fulltoken-s2
  gemma-3-4b-med-lora-s1
  gemma-3-4b-med-lora-s2
  gemma-3-4b-med-lora-newdata-r16
  gemma-3-4b-med-lora-newdata-r32
  gemma-3-4b-med-dora-s1
  gemma-3-4b-med-dora-s2
)

for model_name in "${models[@]}"; do
  "${SBATCH_CMD[@]}" scripts/run_medmcqa_one.sh "${model_name}" "$@"
done
