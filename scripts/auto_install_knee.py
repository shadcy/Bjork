"""
scripts/auto_install_knee.py - Complete FastMRI Knee Auto-Installer & Pipeline Initializer.

One-command automated workflow:
1. Verifies/Installs required Python dependencies (`requirements-fastmri.txt`).
2. Reads or accepts presigned FastMRI URLs with automatic recovery mode (`curl -C -`).
3. Selectively extracts only the required number of volumes (default: 5) to save disk quota.
4. Generates volume-level dataset manifest without slice leakage (`scripts/prepare_fastmri_knee.py`).
5. Executes environment pre-flight diagnostics (`scripts/check_environment.py`).
6. Optionally submits SLURM job on cluster or launches local test run.
"""

import os
import sys
import argparse
import subprocess
import shutil

# Root path of the repo
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cmd(cmd, desc=None, check=True):
    """Executes a system command and streams output."""
    if desc:
        print(f"\n>>> [Step] {desc}")
    print(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, cwd=REPO_ROOT, shell=isinstance(cmd, str))
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with return code {res.returncode}: {cmd}")
    return res.returncode


def install_dependencies():
    """Ensures requirements-fastmri.txt dependencies are installed."""
    req_file = os.path.join(REPO_ROOT, "requirements-fastmri.txt")
    print(f"\n[1/5] Checking and Installing Python Dependencies from {req_file}...")
    run_cmd([sys.executable, "-m", "pip", "install", "-r", req_file], desc="Pip Install Dependencies")


def download_and_extract(target, url, url_file, dest_dir, archive_dir, max_extract, skip_download=False):
    """Downloads archive with curl -C - and selectively extracts volumes."""
    print(f"\n[2/5] Downloading and Extracting FastMRI Target: {target}...")
    dl_script = os.path.join(REPO_ROOT, "scripts", "download_fastmri_knee.py")
    cmd = [
        sys.executable, dl_script,
        "--target", target,
        "--max-extract", str(max_extract),
    ]
    if url:
        cmd += ["--url", url]
    elif url_file and os.path.exists(url_file):
        cmd += ["--url-file", url_file]
    if dest_dir:
        cmd += ["--dest-dir", dest_dir]
    if archive_dir:
        cmd += ["--archive-dir", archive_dir]
    if skip_download:
        cmd += ["--skip-download"]

    run_cmd(cmd, desc=f"Download {target} and extract {max_extract} volumes")


def prepare_manifest(dest_dir, num_train=3, num_val=1, num_test=1, manifest_dir=None):
    """Generates the volume-separated dataset manifest."""
    print(f"\n[3/5] Generating Manifest with Volume-Level Separation...")
    prep_script = os.path.join(REPO_ROOT, "scripts", "prepare_fastmri_knee.py")
    manifest_dir = manifest_dir or os.path.join(REPO_ROOT, "data", "manifests", "knee_first_run")
    
    cmd = [
        sys.executable, prep_script,
        "--data-dir", dest_dir,
        "--manifest-dir", manifest_dir,
        "--num-train", str(num_train),
        "--num-val", str(num_val),
        "--num-test", str(num_test),
        "--seed", "42"
    ]
    run_cmd(cmd, desc="Prepare dataset manifests")


def verify_environment(output_path=None):
    """Runs environment diagnostic check."""
    print(f"\n[4/5] Running Pre-flight Environment Diagnostic...")
    check_script = os.path.join(REPO_ROOT, "scripts", "check_environment.py")
    output_path = output_path or os.path.join(REPO_ROOT, "results", "knee_first_run", "environment.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    run_cmd([sys.executable, check_script, output_path], desc="Check environment")


def main():
    parser = argparse.ArgumentParser(description="Auto-Installer for FastMRI Knee Experiment Pipeline")
    parser.add_argument("--target", type=str, default="multicoil_train_batch_0",
                        help="Target dataset split (default: multicoil_train_batch_0)")
    parser.add_argument("--url", type=str, default=None,
                        help="Direct presigned URL (optional; if not provided, reads data/fastmri_urls.txt)")
    parser.add_argument("--url-file", type=str, default=os.path.join(REPO_ROOT, "data", "fastmri_urls.txt"),
                        help="Path to file containing URLs")
    parser.add_argument("--max-extract", type=int, default=5,
                        help="Number of volumes to selectively extract (default: 5)")
    parser.add_argument("--dest-dir", type=str, default=None,
                        help="Destination directory for extracted .h5 files")
    parser.add_argument("--archive-dir", type=str, default=None,
                        help="Archive directory for downloaded .tar.xz file")
    parser.add_argument("--skip-pip", action="store_true",
                        help="Skip python pip dependency installation")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download if archive already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate configuration and URLs without downloading or modifying files")
    parser.add_argument("--submit-slurm", action="store_true",
                        help="Submit SLURM job to cluster queue upon completion")
    parser.add_argument("--run-local", action="store_true",
                        help="Run 1-epoch experiment locally after installation")

    args = parser.parse_args()

    # Determine default destination paths
    user = os.environ.get("USER", "user")
    scratch_base = f"/scratch/{user}/fastmri_knee" if os.path.exists(f"/scratch/{user}") else os.path.join(REPO_ROOT, "data", "fastmri_knee")
    dest_dir = args.dest_dir or os.path.join(scratch_base, "multicoil_train" if "train" in args.target else "multicoil_val")
    archive_dir = args.archive_dir or os.path.join(scratch_base, "archives")

    print("=" * 70)
    print(" FastMRI Knee Pipeline Auto-Installer ")
    print("=" * 70)
    print(f" Target Split     : {args.target}")
    print(f" Max Volumes      : {args.max_extract} (selective extraction to save space)")
    print(f" Archive Cache    : {archive_dir}")
    print(f" Data Directory   : {dest_dir}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Testing download script resolution...")
        dl_script = os.path.join(REPO_ROOT, "scripts", "download_fastmri_knee.py")
        run_cmd([
            sys.executable, dl_script,
            "--target", args.target,
            "--url-file", args.url_file,
            "--dest-dir", dest_dir,
            "--archive-dir", archive_dir,
            "--dry-run"
        ], desc="Dry-run URL validation")
        print("\n[DRY RUN] All configurations verified! Ready to run without --dry-run.")
        return

    # Step 1: Dependencies
    if not args.skip_pip:
        install_dependencies()
    else:
        print("\n[1/5] Skipping dependency installation (--skip-pip specified)")

    # Step 2: Download & Extract
    download_and_extract(
        target=args.target,
        url=args.url,
        url_file=args.url_file,
        dest_dir=dest_dir,
        archive_dir=archive_dir,
        max_extract=args.max_extract,
        skip_download=args.skip_download
    )

    # Step 3: Manifest
    prepare_manifest(dest_dir=dest_dir)

    # Step 4: Environment check
    verify_environment()

    # Step 5: Launch or Submit
    print("\n[5/5] Installation and Dataset Setup Complete!")
    print("=" * 70)
    if args.submit_slurm:
        if shutil.which("sbatch"):
            slurm_file = os.path.join(REPO_ROOT, "slurm", "knee_first_run.slurm")
            print(f"Submitting SLURM job: sbatch {slurm_file} {dest_dir}")
            run_cmd(["sbatch", slurm_file, dest_dir], desc="Submit SLURM job")
        else:
            print("Note: 'sbatch' command not found on this machine (run on HPC login node).")
    elif args.run_local:
        config_path = os.path.join(REPO_ROOT, "configs", "knee_first_run.yaml")
        exp_script = os.path.join(REPO_ROOT, "scripts", "run_first_experiment.py")
        run_cmd([sys.executable, exp_script, "--config", config_path], desc="Run Local Experiment")
    else:
        print("Ready to run! Launch commands:")
        print(f"  * On PARAM Rudra Cluster (SLURM GPU partition):")
        print(f"      sbatch slurm/knee_first_run.slurm {dest_dir}")
        print(f"  * Or run interactively:")
        print(f"      python scripts/run_first_experiment.py --config configs/knee_first_run.yaml")
    print("=" * 70)


if __name__ == "__main__":
    main()
