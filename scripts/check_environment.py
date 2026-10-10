"""
scripts/check_environment.py - Pre-experiment Environment & Dependency Diagnostic.
Verifies Python, CUDA, PyTorch, MIRTorch, torchkbnufft, and h5py on PARAM Rudra / local.
"""

import os
import sys
import platform


def check_env(save_txt=None):
    lines = []
    lines.append("=" * 65)
    lines.append("      PARAM RUDRA / LOCAL ENVIRONMENT DIAGNOSTIC      ")
    lines.append("=" * 65)

    # 1. System info
    lines.append(f"Host OS         : {platform.system()} {platform.release()} ({platform.machine()})")
    lines.append(f"Python Executable: {sys.executable}")
    lines.append(f"Python Version  : {sys.version.split()[0]}")

    # 2. PyTorch & CUDA
    try:
        import torch
        lines.append(f"PyTorch Version : {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        lines.append(f"CUDA Available  : {cuda_avail}")
        if cuda_avail:
            device_count = torch.cuda.device_count()
            lines.append(f"CUDA Devices    : {device_count}")
            for d in range(device_count):
                name = torch.cuda.get_device_name(d)
                mem = torch.cuda.get_device_properties(d).total_memory / (1024 ** 3)
                lines.append(f"  Device {d}        : {name} ({mem:.2f} GB VRAM)")
        else:
            lines.append("  [INFO] Running on CPU (no CUDA acceleration available).")
    except ImportError as e:
        lines.append(f"[FAIL] PyTorch is not installed: {e}")

    # 3. Check specific libraries
    lines.append("\nLibrary Checks:")
    packages = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("h5py", "h5py (HDF5 for fastMRI)"),
        ("torchkbnufft", "TorchKbNufft"),
        ("mirtorch", "MIRTorch"),
        ("yaml", "PyYAML"),
        ("skimage", "scikit-image (for SSIM)"),
    ]

    all_passed = True
    missing_h5py = False

    for mod_name, desc in packages:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            lines.append(f"  [PASS] {desc:<30}: {ver}")
        except ImportError:
            lines.append(f"  [MISSING] {desc:<27}: Not installed")
            all_passed = False
            if mod_name == "h5py":
                missing_h5py = True

    lines.append("=" * 65)
    if missing_h5py:
        lines.append("\n[ACTION REQUIRED] h5py is missing in this environment.")
        lines.append("To install it in your private Conda environment without sudo:")
        lines.append("    python -m pip install h5py")
        lines.append("=" * 65)

    report = "\n".join(lines)
    print(report)

    if save_txt:
        os.makedirs(os.path.dirname(os.path.abspath(save_txt)), exist_ok=True)
        with open(save_txt, "w") as f:
            f.write(report + "\n")
        print(f"Saved diagnostic report to: {save_txt}")

    return all_passed


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else None
    success = check_env(save_txt=out_file)
    sys.exit(0 if success else 1)
