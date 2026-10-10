#!/bin/bash
#SBATCH --job-name=knee_first_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/knee_first_run_%j.out
#SBATCH --error=results/logs/knee_first_run_%j.err

set -e

echo "=========================================================="
echo " Starting FastMRI Knee First Run on PARAM Rudra A100 GPU "
echo " Job ID: $SLURM_JOB_ID | Host: $(hostname)"
echo " Start Time: $(date)"
echo "=========================================================="

mkdir -p results/logs results/knee_first_run

# 1. Activate Private Conda Environment on Rudra
CONDA_ENV_PATH="$HOME/.conda/envs/bjork"
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
fi

conda activate "$CONDA_ENV_PATH" || source activate "$CONDA_ENV_PATH"

echo "Using Python: $(which python)"

# 2. Strict Check for CUDA Availability
python -c "
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
python scripts/check_environment.py results/knee_first_run/environment.txt

# 4. Prepare Manifests with Volume-Level Separation
# Configurable DATA_DIR argument passed to sbatch or defaulted to /scratch/$USER/...
DATA_DIR="${1:-/scratch/$USER/fastmri_knee/multicoil_train}"
echo "FastMRI Data Directory: $DATA_DIR"

python scripts/prepare_fastmri_knee.py \
    --data-dir "$DATA_DIR" \
    --manifest-dir "data/manifests/knee_first_run" \
    --num-train 3 \
    --num-val 1 \
    --num-test 1 \
    --seed 42

# 5. Execute 1-Epoch Experiment & Evaluation
python scripts/run_first_experiment.py --config configs/knee_first_run.yaml

echo "=========================================================="
echo " Job Completed Successfully at: $(date)"
echo " Results Directory: results/knee_first_run"
echo "=========================================================="
