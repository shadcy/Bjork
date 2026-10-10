#!/bin/bash
#SBATCH --job-name=knee_first_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --output=logs/knee_first_run_%j.out
#SBATCH --error=logs/knee_first_run_%j.err

set -e

# Change to repository root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
elif [ -d "$HOME/projects/Bjork" ]; then
    cd "$HOME/projects/Bjork"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "$SCRIPT_DIR"
fi

echo "=========================================================="
echo " Starting FastMRI Knee First Run on PARAM Rudra A100 GPU "
echo " Job ID: $SLURM_JOB_ID | Host: $(hostname)"
echo " Start Time: $(date)"
echo " Working Dir: $(pwd)"
echo "=========================================================="

# Create log and output directories
mkdir -p logs results/logs results/knee_first_run results/knee_first_run/checkpoints results/knee_first_run/reconstructions

# 1. Robust Conda Environment & Python Setup
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null || true)"
    conda activate bjork 2>/dev/null || conda activate "$HOME/.conda/envs/bjork" 2>/dev/null || true
fi

# Fallback to direct environment PATH
if [ -d "$HOME/.conda/envs/bjork/bin" ]; then
    export PATH="$HOME/.conda/envs/bjork/bin:$PATH"
fi

PYTHON_BIN="$(which python)"
echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" --version

# 2. Strict Check for CUDA Availability
"$PYTHON_BIN" -c "
import torch, sys
if not torch.cuda.is_available():
    print('ERROR: CUDA is not available on this compute node!', file=sys.stderr)
    sys.exit(1)
print(f'CUDA Verified: {torch.cuda.get_device_name(0)} (Count: {torch.cuda.device_count()})')
"
if [ $? -ne 0 ]; then
    echo "Aborting job: GPU is required."
    exit 1
fi

# 3. Environment & Dependency Diagnostic
"$PYTHON_BIN" scripts/check_environment.py results/knee_first_run/environment.txt

# 4. Prepare Manifests (if not already prepared)
DATA_DIR="${1:-./data/fastmri_knee/multicoil_train}"
echo "FastMRI Data Directory: $DATA_DIR"

if [ ! -f "data/manifests/knee_first_run/train_manifest.json" ]; then
    echo "Generating dataset manifests..."
    "$PYTHON_BIN" scripts/prepare_fastmri_knee.py \
        --data-dir "$DATA_DIR" \
        --manifest-dir "data/manifests/knee_first_run" \
        --seed 42
else
    echo "Using existing manifests from data/manifests/knee_first_run/"
fi

# 5. Execute 1-Epoch Experiment & Evaluation
echo "Launching 1-Epoch Experiment..."
"$PYTHON_BIN" scripts/run_first_experiment.py --config configs/knee_first_run.yaml

echo "=========================================================="
echo " Job Completed Successfully at: $(date)"
echo " Results Directory: results/knee_first_run"
echo "=========================================================="
