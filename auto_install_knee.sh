#!/bin/bash
# ==============================================================================
# auto_install_knee.sh - One-Command Auto-Installer for FastMRI Knee Experiments
# ==============================================================================
# This script automates:
# 1. Environment activation (~/.conda/envs/bjork)
# 2. Dependency checks & installation (h5py, pyyaml, scikit-image, etc.)
# 3. Resumable download of FastMRI multi-coil archive via curl with recovery mode (-C -)
# 4. Selective extraction of only N volumes (default: 5) to save disk quota & time
# 5. Volume-level manifest generation (zero slice leakage across splits)
# 6. Pre-flight GPU & package diagnostics
# 7. Optional one-click SLURM job submission to PARAM Rudra A100 GPU partition
# ==============================================================================

set -e

# Change to repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "          BJORK FastMRI Knee Experiment Auto-Installer                "
echo "======================================================================"

# 1. Activate Conda Environment if available
CONDA_ENV_PATH="$HOME/.conda/envs/bjork"
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
fi

if command -v conda &> /dev/null && [ -d "$CONDA_ENV_PATH" ]; then
    echo "Activating conda environment: $CONDA_ENV_PATH"
    conda activate "$CONDA_ENV_PATH" || source activate "$CONDA_ENV_PATH"
fi

echo "Using Python: $(which python)"

# 2. Parse arguments
TARGET="${1:-multicoil_train_batch_0}"
VOLUMES="${2:-5}"
SUBMIT_FLAG=""

for arg in "$@"; do
    case "$arg" in
        --submit)
            SUBMIT_FLAG="--submit-slurm"
            ;;
        --run-local)
            SUBMIT_FLAG="--run-local"
            ;;
    esac
done

echo "Target Split  : $TARGET"
echo "Volumes to extract : $VOLUMES"

# 3. Run Python Auto-Installer
python scripts/auto_install_knee.py \
    --target "$TARGET" \
    --max-extract "$VOLUMES" \
    $SUBMIT_FLAG

echo ""
echo "======================================================================"
echo " Auto-Installer Finished Successfully!                                "
echo "======================================================================"
