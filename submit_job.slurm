#!/bin/bash
#SBATCH --job-name=bjork_knee
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/bjork_knee_%j.out
#SBATCH --error=logs/bjork_knee_%j.err

echo "=========================================================="
echo " Starting BJORK Knee MRI Job on PARAM Cluster "
echo " Job ID: $SLURM_JOB_ID on $(hostname)"
echo " Start Time: $(date)"
echo "=========================================================="

mkdir -p logs checkpoints

# 1. Load Modules (Adapt to your PARAM cluster's module names)
# Check available modules with 'module avail'
if command -v module &> /dev/null; then
    echo "Loading environment modules..."
    module purge
    module load cuda/11.8 2>/dev/null || module load cuda/12.1 2>/dev/null || module load compiler/cuda/11.8 2>/dev/null || true
fi

# 2. Activate Conda Environment
# Change CONDA_ENV_NAME to your env name (e.g. bjork or mri)
CONDA_ENV_NAME="bjork"
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate $CONDA_ENV_NAME
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate $CONDA_ENV_NAME
else
    echo "Warning: Conda script not found in standard paths. Using current python."
fi

# 3. Print GPU Diagnostics
echo "GPU Status:"
nvidia-smi

echo "Python executable: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# 4. Pre-flight Verification Test
echo "Running Pre-Flight Verification Check..."
python verify_setup.py
if [ $? -ne 0 ]; then
    echo "ERROR: Pre-flight verification failed! Aborting training."
    exit 1
fi
echo "Pre-flight verification passed successfully!"

# 5. Launch Training on Knee Dataset
# Adjust DATA_PATH to where your FastMRI Knee dataset is mounted on the cluster scratch/lustre
DATA_PATH="${1:-/scratch/$USER/fastmri_knee/multicoil_train}"

echo "Launching BJORK training with data from: $DATA_PATH"
python train.py \
    --data_path "$DATA_PATH" \
    --save_dir "./checkpoints/run_${SLURM_JOB_ID}" \
    --epochs 50 \
    --batch_size 1 \
    --im_size 256 \
    --ncoils 8 \
    --num_shots 16 \
    --nfe 512 \
    --decim 4 \
    --num_blocks 4 \
    --modl_lambda 0.05 \
    --modl_tol 1e-4 \
    --cg_max_iter 8 \
    --lr_recon 1e-4 \
    --lr_traj 5e-4 \
    --g_max 4.0 \
    --s_max 15000.0 \
    --lambda_g 1e-3 \
    --lambda_s 1e-4 \
    --device cuda:0 \
    --log_interval 20

echo "=========================================================="
echo " Job Finished at: $(date)"
echo "=========================================================="
