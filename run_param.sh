#!/bin/bash
# run_param.sh - Helper script to run or submit BJORK on PARAM cluster

set -e

echo "=== PARAM Cluster BJORK Runner ==="
echo "Usage: "
echo "  bash run_param.sh install               -> Auto-download, selective extract, prepare manifests"
echo "  bash run_param.sh first-run             -> Submit 1-epoch first run SLURM job"
echo "  bash run_param.sh verify                -> Run pre-flight verification only"
echo "  bash run_param.sh demo <h5_file>        -> Run quick forward/backward demo on knee slice"
echo "  bash run_param.sh reconstruct <h5_file> -> Run first knee reconstruction benchmark"
echo "  bash run_param.sh submit <data_path>    -> Submit SLURM job to cluster queue"
echo "  bash run_param.sh local <data_path>     -> Run training directly in current shell"

MODE="${1:-demo}"
DATA_PATH="${2:-}"

case "$MODE" in
    install)
        bash auto_install_knee.sh
        ;;
    first-run)
        sbatch slurm/knee_first_run.slurm "${DATA_PATH:-/scratch/$USER/fastmri_knee/multicoil_train}"
        ;;
    verify)
        python verify_setup.py
        ;;
    demo)
        if [ -n "$DATA_PATH" ]; then
            python demo.py --h5_path "$DATA_PATH" --device cuda:0
        else
            echo "Error: Please specify the path to a fastMRI knee .h5 file: bash run_param.sh demo /path/to/file.h5"
            exit 1
        fi
        ;;
    reconstruct)
        if [ -n "$DATA_PATH" ]; then
            python reconstruct_knee_sample.py --h5_path "$DATA_PATH" --out_dir ./preliminary_results --device cuda:0
        else
            echo "Error: Please specify the path to a fastMRI knee .h5 file: bash run_param.sh reconstruct /path/to/file.h5"
            exit 1
        fi
        ;;
    submit)
        sbatch submit_job.slurm "$DATA_PATH"
        ;;
    local)
        python train.py --data_path "$DATA_PATH" --save_dir ./checkpoints/local_run --epochs 30 --device cuda:0
        ;;
    *)
        echo "Unknown mode: $MODE"
        exit 1
        ;;
esac
