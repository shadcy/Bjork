"""
scripts/download_fastmri_knee.py - FastMRI Knee Auto-Downloader and Selective Extractor.
Features:
1. Resumable HTTP download with progress bar and retry logic.
2. Selective extraction: can unpack only the first N volumes (e.g. 5 files) directly,
   saving dozens of gigabytes of disk space and hours of extraction time.
3. Automatically sets up directory structure under /scratch or user-specified path.
4. Preserves security: reads presigned URLs from local uncommitted file or CLI.
"""

import os
import sys
import tarfile
import argparse
import subprocess


def download_with_curl(url, output_path):
    """Downloads archive using curl with resume support (-C -)."""
    print(f"Downloading to: {output_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cmd = [
        "curl",
        "-C", "-",
        "--retry", "5",
        "--retry-delay", "3",
        "-o", output_path,
        url,
    ]
    print(f"Running: {' '.join(cmd[:4])} ... [URL hidden] --output {output_path}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Download failed with exit code {result.returncode}")
    print(f"Download complete: {output_path} ({os.path.getsize(output_path) / (1024**3):.2f} GB)")


def extract_volumes(archive_path, extract_dir, max_volumes=5):
    """
    Extracts up to max_volumes .h5 files from the tar.xz archive.
    If max_volumes is None or <=0, extracts the entire archive.
    """
    print(f"\nOpening archive: {archive_path}")
    os.makedirs(extract_dir, exist_ok=True)

    extracted_count = 0
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar:
            if member.name.endswith(".h5"):
                # Flatten filename or keep relative
                fname = os.path.basename(member.name)
                dest = os.path.join(extract_dir, fname)
                print(f"  Extracting [{extracted_count + 1}/{max_volumes if max_volumes > 0 else 'All'}]: {fname}...")
                
                # Extract file object directly
                f_in = tar.extractfile(member)
                if f_in:
                    with open(dest, "wb") as f_out:
                        f_out.write(f_in.read())
                    extracted_count += 1

                if max_volumes and max_volumes > 0 and extracted_count >= max_volumes:
                    print(f"Reached target volume count ({max_volumes}). Stopping extraction.")
                    break

    print(f"Extraction finished! Extracted {extracted_count} .h5 volumes to: {extract_dir}")
    return extracted_count


def main():
    parser = argparse.ArgumentParser(description="Download and extract FastMRI Knee dataset")
    parser.add_argument("--url", type=str, default=None, help="Presigned download URL from fastMRI email")
    parser.add_argument("--url-file", type=str, default="data/fastmri_urls.txt", help="Path to text file containing URLs")
    parser.add_argument("--target", type=str, default="multicoil_train_batch_0",
                        choices=["multicoil_train_batch_0", "multicoil_val", "singlecoil_train", "singlecoil_val"],
                        help="Target dataset split to download")
    parser.add_argument("--dest-dir", type=str, default=None, help="Directory to store extracted .h5 files")
    parser.add_argument("--archive-dir", type=str, default=None, help="Directory to save downloaded .tar.xz archive")
    parser.add_argument("--max-extract", type=int, default=5,
                        help="Max volumes to extract (default: 5, perfect for first run; set to 0 for all)")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if archive already exists")
    parser.add_argument("--dry-run", action="store_true", help="Validate URL and print planned paths without downloading")

    args = parser.parse_args()

    # Determine paths
    user = os.environ.get("USER", "user")
    scratch_base = f"/scratch/{user}/fastmri_knee" if os.path.exists(f"/scratch/{user}") else "./data/fastmri_knee"
    archive_dir = args.archive_dir or os.path.join(scratch_base, "archives")
    dest_dir = args.dest_dir or os.path.join(scratch_base, "multicoil_train" if "train" in args.target else "multicoil_val")

    # Resolve URL
    url = args.url
    if not url and os.path.exists(args.url_file):
        with open(args.url_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Match target or knee_target
                target_clean = args.target.replace("knee_", "")
                if args.target in line or target_clean in line:
                    if "=" in line:
                        url = line.split("=", 1)[1].strip()
                    elif "(" in line and ")" in line:
                        url = line.split("(")[1].split(")")[0].strip()
                    else:
                        url = line.strip()
                    break

    if not url:
        if sys.stdin.isatty():
            print("\n" + "=" * 65)
            print(f" Presigned URL Required for Target: {args.target}")
            print("=" * 65)
            print("Please paste the presigned S3 download link from your fastMRI email:")
            user_input = input("URL: ").strip()
            if user_input:
                url = user_input
                # Auto-save to url_file for subsequent steps
                os.makedirs(os.path.dirname(os.path.abspath(args.url_file)), exist_ok=True)
                with open(args.url_file, "a") as f:
                    f.write(f"\n{args.target}={url}\n")
                print(f"[OK] Saved URL to '{args.url_file}' for future runs.")
        
    if not url:
        raise ValueError(
            f"No URL found for target '{args.target}'. "
            f"Please supply --url '<URL>' or populate '{args.url_file}'."
        )

    # If --url was passed via CLI, ensure it is cached in url_file
    if args.url and not os.path.exists(args.url_file):
        os.makedirs(os.path.dirname(os.path.abspath(args.url_file)), exist_ok=True)
        with open(args.url_file, "a") as f:
            f.write(f"\n{args.target}={url}\n")

    archive_filename = f"{args.target if args.target.startswith('knee_') else 'knee_' + args.target}.tar.xz"
    archive_path = os.path.join(archive_dir, archive_filename)

    if args.dry_run:
        print("[DRY RUN] Configuration validated successfully:")
        print(f"  Target: {args.target}")
        print(f"  URL resolved: {url[:60]}...[TRUNCATED]")
        print(f"  Archive path: {archive_path}")
        print(f"  Extract destination: {dest_dir}")
        print(f"  Max volumes to extract: {args.max_extract}")
        return

    # 1. Download
    if not args.skip_download and not os.path.exists(archive_path):
        download_with_curl(url, archive_path)
    else:
        print(f"Archive exists: {archive_path}")

    # 2. Extract
    extract_volumes(archive_path, dest_dir, max_volumes=args.max_extract)

    print("\n" + "=" * 65)
    print(" FastMRI Knee Data Ready for Experiments ")
    print("=" * 65)
    print(f" Data Directory: {dest_dir}")
    print(f" To launch the first experiment run:")
    print(f"   python scripts/prepare_fastmri_knee.py --data-dir {dest_dir}")
    print(f"   python scripts/run_first_experiment.py --config configs/knee_first_run.yaml")
    print("=" * 65)


if __name__ == "__main__":
    main()
