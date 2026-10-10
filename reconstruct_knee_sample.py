"""
reconstruct_knee_sample.py - Turnkey script for your FIRST Knee MRI Reconstruction.
Evaluates and compares:
1. Baseline 1: Non-Cartesian Zero-Filled Adjoint (A^H y).
2. Baseline 2: Standard Radial CG-SENSE / QWLS ((A^H A + lambda I)^-1 A^H y).
3. BJORK + MoDL: Unrolled iterative deep reconstruction.

Outputs:
- Quantitative metrics: NMSE, PSNR, SSIM.
- Comparison summary table.
- Saved reconstructed magnitude arrays (.npy).
"""

import os
import argparse
import numpy as np
import torch

try:
    from skimage.metrics import structural_similarity as ssim_fn
except ImportError:
    ssim_fn = None

from dataset import FastMRIKneeDataset, estimate_sensitivity_maps
from models import BjorkMoDL
from sys_op import NuSense_om, Gram_inv
from traj_utils import generate_radial_trajectory


def compute_all_metrics(pred, target):
    """
    Compute NMSE, PSNR, and SSIM between reconstruction and ground-truth reference.
    Both inputs can be torch Tensors or numpy arrays.
    """
    if isinstance(pred, torch.Tensor):
        pred_mag = torch.abs(pred).detach().cpu().numpy().squeeze()
    else:
        pred_mag = np.abs(pred).squeeze()

    if isinstance(target, torch.Tensor):
        targ_mag = torch.abs(target).detach().cpu().numpy().squeeze()
    else:
        targ_mag = np.abs(target).squeeze()

    # Peak normalization
    max_val = max(targ_mag.max(), 1e-8)
    p_norm = pred_mag / max_val
    t_norm = targ_mag / max_val

    # NMSE
    mse = np.mean((p_norm - t_norm) ** 2)
    norm_targ = np.mean(t_norm ** 2) + 1e-8
    nmse = float(mse / norm_targ)

    # PSNR
    psnr = float(10.0 * np.log10(1.0 / (mse + 1e-12)))

    # SSIM
    if ssim_fn is not None:
        ssim_val = float(ssim_fn(t_norm, p_norm, data_range=1.0))
    else:
        # Fallback SSIM approximation if skimage is missing
        mu_x = np.mean(t_norm)
        mu_y = np.mean(p_norm)
        sig_x = np.var(t_norm)
        sig_y = np.var(p_norm)
        sig_xy = np.cov(t_norm.flatten(), p_norm.flatten())[0, 1]
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_val = float(((2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sig_x + sig_y + c2)))

    return {"nmse": nmse, "psnr": psnr, "ssim": ssim_val}


def run_single_knee_reconstruction(args):
    print("=" * 70)
    print("      BJORK KNEE MRI: FIRST SAMPLE RECONSTRUCTION & BENCHMARK      ")
    print("=" * 70)

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load Real Knee Slice from fastMRI HDF5 file
    if not args.h5_path or not os.path.exists(args.h5_path):
        raise FileNotFoundError(
            f"Please specify a valid fastMRI knee .h5 file via --h5_path. Given path: '{args.h5_path}'"
        )

    print(f"Loading real fastMRI knee HDF5 file: {args.h5_path}")
    import h5py
    with h5py.File(args.h5_path, "r") as hf:
        slice_idx = min(args.slice_idx, hf["kspace"].shape[0] - 1)
        kspace = hf["kspace"][slice_idx]
        if kspace.ndim == 2:
            kspace = kspace[None, :, :]
        elif kspace.ndim == 4:
            kspace = kspace[..., 0] + 1j * kspace[..., 1]
        kspace_c = kspace.astype(np.complex64)
        imgs_coil = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(kspace_c, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
        smap_c = estimate_sensitivity_maps(kspace_c)
        img_c = np.sum(imgs_coil * np.conj(smap_c), axis=0)

        # Crop/Pad
        th, tw = args.im_size, args.im_size
        h, w = img_c.shape
        if h >= th:
            sh = (h - th) // 2
            img_c = img_c[sh:sh+th, :]
            smap_c = smap_c[:, sh:sh+th, :]
        if w >= tw:
            sw = (w - tw) // 2
            img_c = img_c[:, sw:sw+tw]
            smap_c = smap_c[:, :, sw:sw+tw]

        norm_val = np.percentile(np.abs(img_c), 99) + 1e-8
        img_c = img_c / norm_val
    print(f"Loaded slice {slice_idx} from {args.h5_path}")

    Ireal = torch.from_numpy(img_c).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.complex64)
    smap = torch.from_numpy(smap_c).unsqueeze(0).to(device=device, dtype=torch.complex64)

    # Calculate effective acceleration factor
    # Nyquist radial spokes for N=im_size is ~ pi/2 * N
    nyquist_spokes = int(np.pi / 2.0 * args.im_size)
    acc_factor = nyquist_spokes / args.num_shots
    print(f"Sampling Setup: {args.num_shots} spokes x {args.nfe} points (Effective Accel R ~ {acc_factor:.1f}x)")

    # 2. Standard Radial Trajectory Setup
    traj_radial_np = generate_radial_trajectory(num_shots=args.num_shots, nfe=args.nfe, golden_angle=True)
    traj_radial_t = torch.from_numpy(traj_radial_np).reshape(2, -1).unsqueeze(0).to(device=device, dtype=torch.float32)

    # Forward operator with standard radial trajectory
    A_standard = NuSense_om(smap, traj_radial_t, norm="ortho", numpoints=args.numpoints, grid_size=args.grid_size)
    kunder = A_standard * Ireal

    # -------------------------------------------------------------
    # Method 1: Zero-Filled Adjoint Reconstruction
    # -------------------------------------------------------------
    print("\n[1/3] Running Zero-Filled Adjoint Reconstruction (A^H y)...")
    I_under = A_standard.H * kunder
    metrics_zf = compute_all_metrics(I_under, Ireal)
    print(f"  --> Zero-Filled:  PSNR = {metrics_zf['psnr']:.2f} dB, NMSE = {metrics_zf['nmse']:.4f}, SSIM = {metrics_zf['ssim']:.4f}")

    # -------------------------------------------------------------
    # Method 2: Standard Radial CG-SENSE / QWLS ((A'A + lambda I)^-1 A'y)
    # -------------------------------------------------------------
    print(f"\n[2/3] Running Standard Radial CG-SENSE (lambda={args.modl_lambda}, {args.cg_iter} iters)...")
    P_cgsense = Gram_inv(
        smap,
        traj_radial_t,
        lambd=args.modl_lambda,
        tol=1e-5,
        max_iter=args.cg_iter,
        norm="ortho",
        numpoints=args.numpoints,
        grid_size=args.grid_size,
        alert=False,
    )
    I_cgsense = P_cgsense * I_under
    metrics_cg = compute_all_metrics(I_cgsense, Ireal)
    print(f"  --> CG-SENSE:     PSNR = {metrics_cg['psnr']:.2f} dB, NMSE = {metrics_cg['nmse']:.4f}, SSIM = {metrics_cg['ssim']:.4f}")

    # -------------------------------------------------------------
    # Method 3: BJORK + MoDL Deep Reconstruction
    # -------------------------------------------------------------
    print(f"\n[3/3] Running BJORK + MoDL Deep Reconstruction ({args.num_blocks} blocks)...")
    model = BjorkMoDL(
        num_shots=args.num_shots,
        nfe=args.nfe,
        decim=args.decim,
        num_blocks=args.num_blocks,
        MODLLambda=args.modl_lambda,
        MODLtol=1e-4,
        cg_max_iter=args.cg_iter,
        noise_level=0.0,
        grid_size=args.grid_size,
        numpoints=args.numpoints,
        device=device,
        denoiser_filters=args.filters,
        denoiser_layers=args.layers,
    ).to(device)

    # If a trained checkpoint is provided, load weights
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading trained weights from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    model.eval()
    with torch.no_grad():
        out_bjork = model(Ireal, smap, is_train=False)
        I_bjork = out_bjork["recon"]
        grad = out_bjork["grad"]
        slew = out_bjork["slew"]

    metrics_bjork = compute_all_metrics(I_bjork, Ireal)
    print(f"  --> BJORK MoDL:   PSNR = {metrics_bjork['psnr']:.2f} dB, NMSE = {metrics_bjork['nmse']:.4f}, SSIM = {metrics_bjork['ssim']:.4f}")
    print(f"      Waveforms: Max G = {torch.max(torch.abs(grad)):.2f} G/cm, Max S = {torch.max(torch.abs(slew)):.2f} G/cm/s")

    # -------------------------------------------------------------
    # Summary Comparison Table
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("                    PRELIMINARY BENCHMARK SUMMARY                   ")
    print("=" * 70)
    print(f"Acceleration: R ~ {acc_factor:.1f}x ({args.num_shots} spokes x {args.nfe} points)")
    print("-" * 70)
    print(f"{'Method':<25} | {'PSNR (dB)':<12} | {'NMSE':<12} | {'SSIM':<10}")
    print("-" * 70)
    print(f"{'Zero-Filled Adjoint':<25} | {metrics_zf['psnr']:<12.2f} | {metrics_zf['nmse']:<12.4f} | {metrics_zf['ssim']:<10.4f}")
    print(f"{'Radial CG-SENSE':<25} | {metrics_cg['psnr']:<12.2f} | {metrics_cg['nmse']:<12.4f} | {metrics_cg['ssim']:<10.4f}")
    print(f"{'BJORK + MoDL':<25} | {metrics_bjork['psnr']:<12.2f} | {metrics_bjork['nmse']:<12.4f} | {metrics_bjork['ssim']:<10.4f}")
    print("=" * 70)

    # Save output numpy arrays for inspection or plotting
    np.save(os.path.join(args.out_dir, "target_reference.npy"), np.abs(Ireal.squeeze().cpu().numpy()))
    np.save(os.path.join(args.out_dir, "recon_zerofill.npy"), np.abs(I_under.squeeze().cpu().numpy()))
    np.save(os.path.join(args.out_dir, "recon_cgsense.npy"), np.abs(I_cgsense.squeeze().cpu().numpy()))
    np.save(os.path.join(args.out_dir, "recon_bjork.npy"), np.abs(I_bjork.squeeze().cpu().numpy()))
    print(f"\nSaved reconstructed magnitude slices (.npy) to: {args.out_dir}")

    # Generate publication-grade plots using #2373F4 and grayscale theme
    print("\nGenerating detailed visual maps (#2373F4 theme)...")
    try:
        from visualize_results import (
            plot_reconstruction_error_map,
            plot_trajectory_detailed_map,
            plot_roi_detailed_map,
        )
        recon_dict = {
            "Zero-Filled": np.abs(I_under.squeeze().cpu().numpy()),
            "Radial CG-SENSE": np.abs(I_cgsense.squeeze().cpu().numpy()),
            "BJORK MoDL": np.abs(I_bjork.squeeze().cpu().numpy()),
        }
        metrics_dict = {
            "Zero-Filled": metrics_zf,
            "Radial CG-SENSE": metrics_cg,
            "BJORK MoDL": metrics_bjork,
        }
        targ_np = np.abs(Ireal.squeeze().cpu().numpy())

        # 1. Error maps
        plot_reconstruction_error_map(
            target_img=targ_np,
            recon_dict=recon_dict,
            metrics_dict=metrics_dict,
            error_scale=args.error_scale,
            save_path=os.path.join(args.out_dir, "reconstruction_comparison_error_map.png"),
        )

        # 2. Trajectory & hardware waveforms
        plot_trajectory_detailed_map(
            traj_initial=traj_radial_np,
            traj_learned=out_bjork["ktraj"].detach().cpu().numpy(),
            grad_learned=grad.detach().cpu().numpy(),
            slew_learned=slew.detach().cpu().numpy(),
            save_path=os.path.join(args.out_dir, "trajectory_analysis.png"),
        )

        # 3. ROI Zoom Inset & 1D edge profile
        plot_roi_detailed_map(
            target_img=targ_np,
            recon_dict=recon_dict,
            save_path=os.path.join(args.out_dir, "roi_zoom_and_line_profile.png"),
        )
        print(f"Generated 3 visual maps successfully in: {args.out_dir}")
    except Exception as e:
        print(f"Warning: Plot generation encountered an error: {e}")

    print("Done! First knee reconstruction benchmark completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="First Knee MRI Reconstruction Benchmark with BJORK")
    parser.add_argument("--h5_path", type=str, default=None, help="Path to real FastMRI Knee .h5 file")
    parser.add_argument("--slice_idx", type=int, default=16, help="Slice index within volume")
    parser.add_argument("--im_size", type=int, default=256, help="Target image size")
    parser.add_argument("--ncoils", type=int, default=8, help="Number of receiver coils")
    parser.add_argument("--num_shots", type=int, default=16, help="Number of radial spokes")
    parser.add_argument("--nfe", type=int, default=512, help="Readout points per spoke")
    parser.add_argument("--decim", type=int, default=4, help="B-spline decimation factor")
    parser.add_argument("--num_blocks", type=int, default=4, help="MoDL iteration blocks")
    parser.add_argument("--modl_lambda", type=float, default=0.05, help="Regularization weight lambda")
    parser.add_argument("--cg_iter", type=int, default=8, help="CG solver iterations")
    parser.add_argument("--grid_size", type=float, default=2.0, help="NUFFT grid factor")
    parser.add_argument("--numpoints", type=int, default=6, help="NUFFT kernel points")
    parser.add_argument("--filters", type=int, default=64, help="Denoiser CNN filters")
    parser.add_argument("--layers", type=int, default=5, help="Denoiser CNN depth")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to optional trained checkpoint")
    parser.add_argument("--out_dir", type=str, default="./preliminary_results", help="Output directory")
    parser.add_argument("--error_scale", type=float, default=5.0, help="Error map magnification multiplier (default: 5.0x)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (cuda:0 or cpu)")

    args = parser.parse_args()
    run_single_knee_reconstruction(args)


if __name__ == "__main__":
    main()
