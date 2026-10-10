#!/bin/bash
# ==============================================================================
# auto_install_knee.sh - One-Command Auto-Installer for FastMRI Knee Experiments
# ==============================================================================

set -e

# Change to repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "          BJORK FastMRI Knee Experiment Auto-Installer                "
echo "======================================================================"

# 1. Environment Verification / Activation
# Check if current Python or conda is already bjork
if [[ "$CONDA_DEFAULT_ENV" == "bjork" ]] || [[ "$(which python 2>/dev/null)" == *"bjork"* ]]; then
    echo "Using active conda environment: $(which python)"
else
    # Try initializing conda shell hook
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook 2> /dev/null || true)"
        conda activate bjork 2>/dev/null || conda activate "$HOME/.conda/envs/bjork" 2>/dev/null || true
    fi
    # Fallback to direct python path if conda activation didn't set it
    if [ -f "$HOME/.conda/envs/bjork/bin/python" ]; then
        export PATH="$HOME/.conda/envs/bjork/bin:$PATH"
    fi
fi

PYTHON_BIN="$(which python)"
echo "Using Python: $PYTHON_BIN"

# 2. Parse arguments
TARGET="multicoil_train_batch_0"
VOLUMES="5"
URL_ARG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --volumes|--max-extract)
            VOLUMES="$2"
            shift 2
            ;;
        --url)
            URL_ARG="--url $2"
            shift 2
            ;;
        --submit|--submit-slurm)
            EXTRA_ARGS+=("--submit-slurm")
            shift
            ;;
        --run-local)
            EXTRA_ARGS+=("--run-local")
            shift
            ;;
        --dry-run)
            EXTRA_ARGS+=("--dry-run")
            shift
            ;;
        *)
            # Support positional: $1 = TARGET, $2 = VOLUMES
            if [[ "$1" != -* ]]; then
                if [ -z "$POSITIONAL_1" ]; then
                    TARGET="$1"
                    POSITIONAL_1=true
                elif [ -z "$POSITIONAL_2" ]; then
                    VOLUMES="$1"
                    POSITIONAL_2=true
                fi
            fi
            shift
            ;;
    esac
done

echo "Target Split       : $TARGET"
echo "Volumes to extract : $VOLUMES"

# 3. Run Python Auto-Installer
"$PYTHON_BIN" scripts/auto_install_knee.py \
    --target "$TARGET" \
    --max-extract "$VOLUMES" \
    $URL_ARG \
    "${EXTRA_ARGS[@]}"

echo ""
echo "======================================================================"
echo " Auto-Installer Finished Successfully!                                "
echo "======================================================================"
