#!/bin/bash
#SBATCH --job-name=landscape-direct
#SBATCH --partition=normal
#SBATCH --gres=gpu:full:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=hgf_aso4615
#SBATCH --time=24:00:00
#SBATCH --output=/home/hgf_dkfz/hgf_aso4615/slurm_logs/%x_%j.out
#SBATCH --error=/home/hgf_dkfz/hgf_aso4615/slurm_logs/%x_%j.err

set -euo pipefail

# --- adjust as needed ---
REPO_DIR="$HOME/flexcraft_preet"
NUM_AA=100
NUM_DESIGNS=1000
SEED=42
OUT_PATH="$REPO_DIR/test_run"
# ------------------------

CONDA_ENV="flexcraft"
set +u
source ~/.bashrc
set -u
conda activate "$CONDA_ENV"


cd "$REPO_DIR"
mkdir -p "$OUT_PATH"

python -m flexcraft.pipelines.experimental.landscape \
    --num_aa "$NUM_AA" \
    --num_designs "$NUM_DESIGNS" \
    --hallucinate False \
    --seed "$SEED" \
    --out_path "$OUT_PATH"

echo "Done. Results in $OUT_PATH"
