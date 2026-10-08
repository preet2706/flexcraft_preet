#!/bin/bash
#SBATCH --job-name=landscape-hallucinate
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=~/flexcraft_preet/slurm_logs/%x_%j.out
#SBATCH --error=~/flexcraft_preet/slurm_logs/%x_%j.err

# --- adjust as needed ---
REPO_DIR="$HOME/flexcraft_preet"
NUM_AA=100
NUM_DESIGNS=100
SEED=42
OUT_PATH="$REPO_DIR/test_run_hallucinate"
# ------------------------

module purge
CONDA_ENV="flexcraft_new"
source ~/.bashrc
conda activate "$CONDA_ENV"


cd "$REPO_DIR"
mkdir -p "$OUT_PATH"

python3 -m flexcraft.pipelines.experimental.landscape \
    --num_aa "$NUM_AA" \
    --num_designs "$NUM_DESIGNS" \
    --hallucinate True \
    --seed "$SEED" \
    --out_path "$OUT_PATH"

echo "Done. Results in $OUT_PATH"
