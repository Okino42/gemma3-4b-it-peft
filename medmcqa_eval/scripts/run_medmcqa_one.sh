#!/bin/bash
#SBATCH --job-name=medmcqa
#SBATCH --output=logs/medmcqa_%j.out
#SBATCH --error=logs/medmcqa_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:pro6000:1

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch scripts/run_medmcqa_one.sh <model_name> [extra eval_medmcqa.py args]" >&2
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

python eval_medmcqa.py --model-name "${MODEL_NAME}" "$@"
python score_medmcqa.py --model-name "${MODEL_NAME}"
