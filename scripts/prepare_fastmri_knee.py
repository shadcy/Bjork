"""
scripts/prepare_fastmri_knee.py - FastMRI Multi-Coil Knee Dataset Inspection & Manifest Builder.
Strictly adheres to:
1. Volume-level separation (zero slice leakage across splits).
2. Deterministic volume selection.
3. Reading real FastMRI multicoil HDF5 metadata and complex k-space dimensions.
4. Documenting sensitivity map estimation and target references.
5. Never fabricating data, targets, or credentials.
"""

import os
import glob
import json
import argparse
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None


def inspect_fastmri_volume(fpath):
    """
    Inspect a single fastMRI multicoil knee HDF5 file.
    Returns metadata dictionary or None if corrupted/unsupported.
    """
    if h5py is None:
        raise ImportError("h5py is required. Please install with: python -m pip install h5py")

    try:
        with h5py.File(fpath, "r") as hf:
            keys = list(hf.keys())
            if "kspace" not in hf:
                print(f"  [WARN] Skipping {os.path.basename(fpath)}: Missing 'kspace' key.")
                return None

            kspace_ds = hf["kspace"]
            kspace_shape = list(kspace_ds.shape)
            kspace_dtype = str(kspace_ds.dtype)

            # Slices and coils
            num_slices = kspace_shape[0]
            num_coils = kspace_shape[1] if len(kspace_shape) >= 2 else 1
            readout_dim = kspace_shape[2] if len(kspace_shape) >= 3 else 0
            phase_dim = kspace_shape[3] if len(kspace_shape) >= 4 else 0

            # Inspect targets
            has_rss = "reconstruction_rss" in hf
            has_esc = "reconstruction_esc" in hf
            has_smaps = any(k in hf for k in ["smap", "smaps", "sensitivity_map", "maps"])

            rss_shape = list(hf["reconstruction_rss"].shape) if has_rss else None

            # Attributes / metadata
            attrs = {}
            for k in ["acquisition", "patient_id", "max", "norm", "acceleration"]:
                if k in hf.attrs:
                    val = hf.attrs[k]
                    attrs[k] = float(val) if isinstance(val, (np.floating, float)) else str(val)

            info = {
                "filepath": os.path.abspath(fpath),
                "filename": os.path.basename(fpath),
                "num_slices": num_slices,
                "num_coils": num_coils,
                "kspace_shape": kspace_shape,
                "kspace_dtype": kspace_dtype,
                "readout_dim": readout_dim,
                "phase_dim": phase_dim,
                "has_reconstruction_rss": has_rss,
                "reconstruction_rss_shape": rss_shape,
                "has_reconstruction_esc": has_esc,
                "has_provided_smaps": has_smaps,
                "attributes": attrs,
                "slice_indices": list(range(num_slices)),
            }
            return info
    except Exception as e:
        print(f"  [ERROR] Failed to read {fpath}: {e}")
        return None


def prepare_manifests(data_dir, manifest_dir, num_train=3, num_val=1, num_test=1, seed=42):
    """
    Selects a deterministic small subset of volumes and outputs manifests with volume-level splits.
    """
    print("=" * 65)
    print("   FastMRI Multi-Coil Knee Dataset Inspection & Manifest Builder   ")
    print("=" * 65)
    print(f"Input Directory  : {data_dir}")
    print(f"Manifest Output  : {manifest_dir}")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Configured fastMRI directory does not exist: {data_dir}\n"
            f"Please download official fastMRI knee files and place them in that directory."
        )

    # Find all .h5 files
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    if not all_files:
        raise FileNotFoundError(
            f"No .h5 files found in {data_dir}. "
            f"Please verify that fastMRI multicoil files are present in the directory."
        )

    print(f"Discovered {len(all_files)} total .h5 files. Inspecting structure...")

    # Inspect each volume
    valid_volumes = []
    for fpath in all_files:
        info = inspect_fastmri_volume(fpath)
        if info is not None:
            valid_volumes.append(info)

    if not valid_volumes:
        raise RuntimeError("No valid fastMRI multicoil volumes could be parsed.")

    print(f"Successfully validated {len(valid_volumes)} volumes.")

    # Deterministic split by volume filename
    rng = np.random.RandomState(seed)
    # Sort first for reproducibility
    valid_volumes.sort(key=lambda v: v["filename"])
    indices = rng.permutation(len(valid_volumes))

    total_needed = num_train + num_val + num_test
    if len(valid_volumes) < total_needed:
        print(f"[WARN] Available volumes ({len(valid_volumes)}) < requested total ({total_needed}).")
        # Adjust allocation proportionally
        num_train = max(1, len(valid_volumes) - num_val - num_test)
        if len(valid_volumes) == 1:
            num_train, num_val, num_test = 1, 0, 0
        elif len(valid_volumes) == 2:
            num_train, num_val, num_test = 1, 1, 0

    train_vols = [valid_volumes[i] for i in indices[:num_train]]
    val_vols = [valid_volumes[i] for i in indices[num_train:num_train + num_val]]
    test_vols = [valid_volumes[i] for i in indices[num_train + num_val:num_train + num_val + num_test]]

    os.makedirs(manifest_dir, exist_ok=True)

    def write_manifest(name, vol_list):
        total_slices = sum(v["num_slices"] for v in vol_list)
        fpath = os.path.join(manifest_dir, f"{name}_manifest.json")
        data = {
            "split": name,
            "num_volumes": len(vol_list),
            "total_slices": total_slices,
            "volumes": vol_list,
        }
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  [SAVED] {name:<6} manifest: {len(vol_list)} volumes ({total_slices} slices) -> {fpath}")
        return data

    train_data = write_manifest("train", train_vols)
    val_data = write_manifest("val", val_vols)
    test_data = write_manifest("test", test_vols)

    # Save master summary
    summary_path = os.path.join(manifest_dir, "manifest_summary.json")
    summary = {
        "dataset_name": "fastMRI_multicoil_knee",
        "data_dir": os.path.abspath(data_dir),
        "random_seed": seed,
        "splits": {
            "train": {"num_volumes": len(train_vols), "num_slices": train_data["total_slices"], "files": [v["filename"] for v in train_vols]},
            "val": {"num_volumes": len(val_vols), "num_slices": val_data["total_slices"], "files": [v["filename"] for v in val_vols]},
            "test": {"num_volumes": len(test_vols), "num_slices": test_data["total_slices"], "files": [v["filename"] for v in test_vols]},
        },
        "sensitivity_map_strategy": "estimated_from_acs_windowed_ifft",
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 65)
    print("Manifest creation complete. Volume-level separation strictly guaranteed.")
    print(f"Summary written to: {summary_path}")
    print("=" * 65)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare FastMRI Knee Manifests with Volume-Level Separation")
    parser.add_argument("--data-dir", type=str, default="/scratch/22b3926/fastmri_knee/multicoil_train", help="Path to fastMRI knee .h5 directory")
    parser.add_argument("--manifest-dir", type=str, default="data/manifests/knee_first_run", help="Output path for JSON manifests")
    parser.add_argument("--num-train", type=int, default=3, help="Number of training volumes")
    parser.add_argument("--num-val", type=int, default=1, help="Number of validation volumes")
    parser.add_argument("--num-test", type=int, default=1, help="Number of test volumes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic volume selection")

    args = parser.parse_args()
    prepare_manifests(
        data_dir=args.data_dir,
        manifest_dir=args.manifest_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
