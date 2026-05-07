#!/bin/bash
#SBATCH --job-name=medmcqa-bias
#SBATCH --output=logs/medmcqa_bias_%j.out
#SBATCH --error=logs/medmcqa_bias_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:pro6000:1

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch scripts/run_option_bias_diag.sh <model_name> [extra diagnose_option_permutation_bias.py args]" >&2
  exit 2
fi

MODEL_NAME="$1"
shift

cd /projects/checkpoint/medmcqa_eval

mkdir -p logs outputs results

if [ -f /home/jie018/miniconda3/etc/profile.d/conda.sh ]; then
  source /home/jie018/miniconda3/etc/profile.d/conda.sh
  conda activate finetune
fi

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

python diagnose_option_permutation_bias.py --model-name "${MODEL_NAME}" "$@"
