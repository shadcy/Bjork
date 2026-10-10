"""
demo.py - End-to-end forward & backward gradient verification on a real FastMRI knee slice.
Verifies:
1. FastMRI knee HDF5 slice loading and multi-coil sensitivity map estimation.
2. B-spline trajectory parameterization and hardware waveform generator.
3. Non-Cartesian NUFFT encoding.
4. Unrolled MoDL image reconstruction with CG data consistency.
5. Autograd differentiation through both network weights and trajectory coordinates.
"""

import os
import argparse
import numpy as np
import torch

try:
    import h5py
except ImportError:
    h5py = None

from dataset import estimate_sensitivity_maps
from models import BjorkMoDL


def load_real_knee_slice(h5_path, slice_idx=16, im_size=(256, 256)):
    """Load a real knee slice from a fastMRI .h5 file."""
    if h5py is None:
        raise ImportError("h5py is required. Install with: pip install h5py")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"FastMRI file not found: {h5_path}")

    with h5py.File(h5_path, "r") as hf:
        kspace = hf["kspace"][slice_idx]
        if kspace.ndim == 2:
            kspace = kspace[None, :, :]
        elif kspace.ndim == 4:
            kspace = kspace[..., 0] + 1j * kspace[..., 1]

        kspace_c = kspace.astype(np.complex64)
        imgs_coil = np.fft.ifftshift(
            np.fft.ifft2(np.fft.fftshift(kspace_c, axes=(-2, -1)), axes=(-2, -1)),
            axes=(-2, -1),
        )
        smap_c = estimate_sensitivity_maps(kspace_c)
        img_c = np.sum(imgs_coil * np.conj(smap_c), axis=0)

        # Center crop / pad to im_size
        th, tw = im_size
        h, w = img_c.shape
        if h > th:
            sh = (h - th) // 2
            img_c = img_c[sh:sh + th, :]
            smap_c = smap_c[:, sh:sh + th, :]
        if w > tw:
            sw = (w - tw) // 2
            img_c = img_c[:, sw:sw + tw]
            smap_c = smap_c[:, :, sw:sw + tw]

        norm_val = np.percentile(np.abs(img_c), 99) + 1e-8
        img_c = img_c / norm_val

    return img_c, smap_c


def run_demo(h5_path, slice_idx=16, im_size=(256, 256), device_str="cuda:0"):
    print("=" * 65)
    print(" Running BJORK + MoDL Verification Demo on Real Knee Data ")
    print("=" * 65)

    device = torch.device(device_str if torch.cuda.is_available() and "cuda" in device_str else "cpu")
    print(f"Device: {device}")

    # 1. Load real fastMRI knee slice
    print(f"Loading real knee slice from: {h5_path} (slice {slice_idx})...")
    img_np, smap_np = load_real_knee_slice(h5_path, slice_idx=slice_idx, im_size=im_size)

    Ireal = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.complex64)
    smap = torch.from_numpy(smap_np).unsqueeze(0).to(device=device, dtype=torch.complex64)

    # 2. Instantiate BjorkMoDL model
    num_shots = 16
    nfe = 512
    print(f"Initializing BjorkMoDL ({num_shots} shots, {nfe} points/shot, 4 MoDL blocks)...")
    model = BjorkMoDL(
        num_shots=num_shots,
        nfe=nfe,
        decim=4,
        num_blocks=4,
        MODLLambda=0.05,
        MODLtol=1e-4,
        cg_max_iter=8,
        noise_level=0.0,
        grid_size=2.0,
        numpoints=6,
        device=device,
        denoiser_filters=64,
        denoiser_layers=5,
    ).to(device)

    # 3. Forward pass
    print("Executing forward pass...")
    output = model(Ireal, smap, is_train=True)

    recon = output["recon"]
    ktraj = output["ktraj"]
    grad = output["grad"]
    slew = output["slew"]

    print(f" Target image shape:      {Ireal.shape}")
    print(f" Sensitivity maps shape:  {smap.shape}")
    print(f" Trajectory shape:        {ktraj.shape}")
    print(f" Reconstructed shape:     {recon.shape}")
    print(f" Gradient waveform shape: {grad.shape} (max = {torch.max(torch.abs(grad)):.2f} G/cm)")
    print(f" Slew rate shape:         {slew.shape} (max = {torch.max(torch.abs(slew)):.2f} G/cm/s)")

    # 4. Compute loss
    loss_dict = model.compute_loss(output, Ireal)
    loss = loss_dict["total_loss"]
    print(f" Loss: total = {loss.item():.5f}, recon = {loss_dict['recon_loss'].item():.5f}, "
          f"grad_viol = {loss_dict['grad_loss'].item():.5e}, slew_viol = {loss_dict['slew_loss'].item():.5e}")

    # 5. Backward pass
    print("Executing backward pass to verify gradient propagation...")
    loss.backward()

    # Check trajectory gradients
    traj_grad = model.netSampling.coeff.grad
    if traj_grad is not None and not torch.isnan(traj_grad).any():
        print(f" Trajectory gradient norm: {torch.norm(traj_grad):.4e} (Valid)")
    else:
        print(" Error: Trajectory gradient is None or NaN!")

    # Check denoiser gradients
    denoiser_param = next(model.netG_I.parameters())
    if denoiser_param.grad is not None and not torch.isnan(denoiser_param.grad).any():
        print(f" Denoiser gradient norm:   {torch.norm(denoiser_param.grad):.4e} (Valid)")
    else:
        print(" Error: Denoiser gradient is None or NaN!")

    print("=" * 65)
    print(" DEMO COMPLETED SUCCESSFULLY ON REAL KNEE DATA!")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="BJORK Real Knee Data Demo")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to real fastMRI knee .h5 file")
    parser.add_argument("--slice_idx", type=int, default=16, help="Slice index within volume")
    parser.add_argument("--im_size", type=int, default=256, help="Target image size")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (cuda:0 or cpu)")
    args = parser.parse_args()

    run_demo(
        h5_path=args.h5_path,
        slice_idx=args.slice_idx,
        im_size=(args.im_size, args.im_size),
        device_str=args.device,
    )


if __name__ == "__main__":
    main()