"""
scripts/download_fastmri_knee.py - FastMRI Knee Auto-Downloader and Selective Extractor.

Features:
1. Stream Extraction (Default): Streams directly over HTTP from fastMRI S3 URL into tarfile,
   extracting only the first N volumes (e.g. 5 files) on the fly, then immediately terminates curl!
   This downloads ~1 GB in ~3 minutes instead of downloading all 92 GB in 6.5 hours!
2. Partial Archive Extraction: If an archive was already partially downloaded, extracts files
   directly from the existing downloaded bytes without redownloading.
3. Resumable Full Download (--full-download): Option to download the complete 92GB archive if desired.
4. Preserves security: reads presigned URLs from local uncommitted file or CLI.
"""

import os
import sys
import tarfile
import argparse
import subprocess
import shutil


def download_with_curl(url, output_path):
    """Downloads archive using curl with resume support (-C -)."""
    print(f"Downloading full archive to: {output_path}")
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
    Uses sequential streaming mode ('r|*') which works on partial/in-progress archives.
    """
    print(f"\nOpening archive: {archive_path}")
    os.makedirs(extract_dir, exist_ok=True)

    extracted_count = 0
    try:
        with tarfile.open(archive_path, "r|*") as tar:
            for member in tar:
                if member.name.endswith(".h5"):
                    fname = os.path.basename(member.name)
                    dest = os.path.join(extract_dir, fname)
                    print(f"  Extracting [{extracted_count + 1}/{max_volumes if max_volumes > 0 else 'All'}]: {fname} ({member.size / (1024**2):.1f} MB)...")
                    f_in = tar.extractfile(member)
                    if f_in:
                        with open(dest, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                        extracted_count += 1

                    if max_volumes and max_volumes > 0 and extracted_count >= max_volumes:
                        print(f"Reached target volume count ({max_volumes}). Stopping extraction.")
                        break
    except Exception as e:
        print(f"Note: archive stream reached end or cutoff: {e}")

    print(f"Extraction finished! Extracted {extracted_count} .h5 volumes to: {extract_dir}")
    return extracted_count


def stream_download_and_extract(url, extract_dir, max_volumes=5):
    """
    Direct HTTP streaming: pipes curl stdout directly into tarfile without saving the 92GB archive to disk!
    Terminates curl immediately once max_volumes .h5 files are extracted.
    Saves hours of download time and 90+ GB of disk space.
    """
    print(f"\n[STREAM MODE] Streaming directly from fastMRI S3 URL...")
    print(f"Target: Extract {max_volumes} volumes directly to: {extract_dir}")
    os.makedirs(extract_dir, exist_ok=True)

    cmd = [
        "curl",
        "-sSL",
        "--retry", "5",
        "--retry-delay", "3",
        url
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    extracted_count = 0
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|*") as tar:
            for member in tar:
                if member.name.endswith(".h5"):
                    fname = os.path.basename(member.name)
                    dest = os.path.join(extract_dir, fname)
                    print(f"  Streaming & extracting [{extracted_count + 1}/{max_volumes}]: {fname} ({member.size / (1024**2):.1f} MB)...")
                    f_in = tar.extractfile(member)
                    if f_in:
                        with open(dest, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                        extracted_count += 1
                    if max_volumes and max_volumes > 0 and extracted_count >= max_volumes:
                        print(f"Target count of {max_volumes} reached! Stopping download stream.")
                        break
    except Exception as e:
        print(f"Notice: stream completed or closed ({e})")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    print(f"Stream extraction complete! Extracted {extracted_count} volumes to {extract_dir}")
    return extracted_count


def main():
    parser = argparse.ArgumentParser(description="Download and extract FastMRI Knee dataset")
    parser.add_argument("--url", type=str, default=None, help="Presigned download URL from fastMRI email")
    parser.add_argument("--url-file", type=str, default="data/fastmri_urls.txt", help="Path to text file containing URLs")
    parser.add_argument("--target", type=str, default="multicoil_train_batch_0",
                        help="Target dataset split to download")
    parser.add_argument("--dest-dir", type=str, default=None, help="Directory to store extracted .h5 files")
    parser.add_argument("--archive-dir", type=str, default=None, help="Directory to save downloaded .tar.xz archive")
    parser.add_argument("--max-extract", type=int, default=5,
                        help="Max volumes to extract (default: 5, perfect for first run; set to 0 for all)")
    parser.add_argument("--full-download", action="store_true", help="Download the full 92GB archive file before extracting")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, extract from existing archive only")
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
                os.makedirs(os.path.dirname(os.path.abspath(args.url_file)), exist_ok=True)
                with open(args.url_file, "a") as f:
                    f.write(f"\n{args.target}={url}\n")
                print(f"[OK] Saved URL to '{args.url_file}' for future runs.")

    if not url:
        raise ValueError(
            f"No URL found for target '{args.target}'. "
            f"Please supply --url '<URL>' or populate '{args.url_file}'."
        )

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

    # Check 1: Do we already have enough .h5 files in destination directory?
    existing_h5 = [f for f in os.listdir(dest_dir) if f.endswith(".h5")] if os.path.exists(dest_dir) else []
    if len(existing_h5) >= args.max_extract:
        print(f"\n[OK] Found {len(existing_h5)} .h5 volumes already in {dest_dir}. Skipping download/extraction.")
    # Check 2: Does an archive file already exist (even partially downloaded > 50MB)?
    elif os.path.exists(archive_path) and os.path.getsize(archive_path) > 50 * 1024 * 1024:
        print(f"\nFound existing archive file: {archive_path} ({os.path.getsize(archive_path) / (1024**2):.1f} MB)")
        print(f"Attempting selective extraction directly from available bytes...")
        extracted = extract_volumes(archive_path, dest_dir, max_volumes=args.max_extract)
        if extracted >= args.max_extract:
            print(f"[SUCCESS] Extracted all {args.max_extract} required volumes from existing archive!")
        else:
            print(f"Extracted {extracted}/{args.max_extract} volumes. Streaming remaining from URL...")
            stream_download_and_extract(url, dest_dir, max_volumes=args.max_extract - extracted)
    # Check 3: Full download requested?
    elif args.full_download:
        if not args.skip_download and not os.path.exists(archive_path):
            download_with_curl(url, archive_path)
        extract_volumes(archive_path, dest_dir, max_volumes=args.max_extract)
    # Check 4: Fast Stream Extraction (Default)
    else:
        stream_download_and_extract(url, dest_dir, max_volumes=args.max_extract)

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
