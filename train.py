"""
train.py - Training pipeline for BJORK on Knee MRI dataset.
Jointly optimizes:
1. B-spline sampling trajectory (subject to gradient and slew rate physical limits).
2. Deep unrolled MoDL reconstruction network.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.optim as optim

from dataset import get_knee_dataloader
from models import BjorkMoDL
from traj_utils import save_initial_trajectory


def compute_metrics(pred, target):
    """
    Compute NMSE, PSNR, and SSIM between complex reconstruction and reference image.
    Evaluates on magnitude images normalized to [0, 1].
    """
    pred_mag = torch.abs(pred).detach().cpu().numpy().squeeze()
    targ_mag = torch.abs(target).detach().cpu().numpy().squeeze()

    # Peak normalization
    max_val = max(targ_mag.max(), 1e-8)
    pred_mag = pred_mag / max_val
    targ_mag = targ_mag / max_val

    # NMSE
    mse = np.mean((pred_mag - targ_mag) ** 2)
    norm_targ = np.mean(targ_mag ** 2) + 1e-8
    nmse = mse / norm_targ

    # PSNR
    psnr = 10 * np.log10(1.0 / (mse + 1e-12))

    return {"nmse": float(nmse), "psnr": float(psnr)}


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    print(f"=== BJORK Training on Knee Dataset ===")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}, Image size: ({args.im_size}, {args.im_size})")
    print(f"Trajectory: {args.num_shots} shots, {args.nfe} points/shot, decim={args.decim}")
    print(f"MoDL blocks: {args.num_blocks}, CG max iter: {args.cg_max_iter}")

    # 1. Trajectory initialization
    init_traj_path = args.init_traj
    if init_traj_path is None or not os.path.exists(init_traj_path):
        init_traj_path = os.path.join(args.save_dir, "initial_radial_trajectory.npy")
        save_initial_trajectory(init_traj_path, traj_type="radial", num_shots=args.num_shots, nfe=args.nfe)

    # 2. Dataset and Dataloaders
    print("Loading datasets...")
    train_loader = get_knee_dataloader(
        data_dir=args.data_path,
        batch_size=args.batch_size,
        im_size=(args.im_size, args.im_size),
        ncoils=args.ncoils,
        is_train=True,
        num_workers=args.num_workers,
    )

    val_loader = get_knee_dataloader(
        data_dir=args.data_path,
        batch_size=1,
        im_size=(args.im_size, args.im_size),
        ncoils=args.ncoils,
        is_train=False,
        num_workers=0,
    )

    # 3. Model setup
    model = BjorkMoDL(
        num_shots=args.num_shots,
        nfe=args.nfe,
        decim=args.decim,
        num_blocks=args.num_blocks,
        MODLLambda=args.modl_lambda,
        MODLtol=args.modl_tol,
        cg_max_iter=args.cg_max_iter,
        noise_level=args.noise_level,
        grid_size=args.grid_size,
        numpoints=args.numpoints,
        g_max=args.g_max,
        s_max=args.s_max,
        init_traj=init_traj_path,
        device=device,
        denoiser_filters=args.filters,
        denoiser_layers=args.layers,
    ).to(device)

    # 4. Separate optimizers for trajectory and reconstruction network
    optimizer_recon = optim.Adam(model.netG_I.parameters(), lr=args.lr_recon)
    optimizer_traj = optim.Adam([model.netSampling.coeff], lr=args.lr_traj)

    scheduler_recon = optim.lr_scheduler.CosineAnnealingLR(optimizer_recon, T_max=args.epochs, eta_min=1e-6)
    scheduler_traj = optim.lr_scheduler.CosineAnnealingLR(optimizer_traj, T_max=args.epochs, eta_min=1e-6)

    # Resume checkpoint if specified
    start_epoch = 0
    best_psnr = -float("inf")
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer_recon.load_state_dict(ckpt["optim_recon"])
        optimizer_traj.load_state_dict(ckpt["optim_traj"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", -float("inf"))

    # 5. Training loop
    log_file = os.path.join(args.save_dir, "training_log.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("epoch,recon_loss,grad_loss,slew_loss,total_loss,val_psnr,val_nmse,time_sec\n")

    print("\nStarting training loop...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        running_recon_loss = 0.0
        running_grad_loss = 0.0
        running_slew_loss = 0.0
        running_total_loss = 0.0

        for step, (Ireal, smap) in enumerate(train_loader):
            Ireal = Ireal.to(device)
            smap = smap.to(device)

            optimizer_recon.zero_grad()
            optimizer_traj.zero_grad()

            output = model(Ireal, smap, is_train=True)
            loss_dict = model.compute_loss(
                output,
                Ireal,
                lambda_grad=args.lambda_g,
                lambda_slew=args.lambda_s,
            )

            total_loss = loss_dict["total_loss"]
            total_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.netG_I.parameters(), max_norm=1.0)
            if model.netSampling.coeff.grad is not None:
                torch.nn.utils.clip_grad_norm_([model.netSampling.coeff], max_norm=1.0)

            optimizer_recon.step()
            optimizer_traj.step()

            running_recon_loss += loss_dict["recon_loss"].item()
            running_grad_loss += loss_dict["grad_loss"].item()
            running_slew_loss += loss_dict["slew_loss"].item()
            running_total_loss += total_loss.item()

            if (step + 1) % args.log_interval == 0 or (step + 1) == len(train_loader):
                print(
                    f"Epoch [{epoch+1}/{args.epochs}] Step [{step+1}/{len(train_loader)}] "
                    f"Loss: {total_loss.item():.4f} (Recon: {loss_dict['recon_loss'].item():.4f}, "
                    f"G: {loss_dict['grad_loss'].item():.2e}, S: {loss_dict['slew_loss'].item():.2e})"
                )

        scheduler_recon.step()
        scheduler_traj.step()

        num_batches = max(len(train_loader), 1)
        avg_recon = running_recon_loss / num_batches
        avg_grad = running_grad_loss / num_batches
        avg_slew = running_slew_loss / num_batches
        avg_total = running_total_loss / num_batches
        epoch_time = time.time() - epoch_start

        # 6. Validation step
        model.eval()
        val_psnr_list = []
        val_nmse_list = []
        with torch.no_grad():
            for Ireal_val, smap_val in val_loader:
                Ireal_val = Ireal_val.to(device)
                smap_val = smap_val.to(device)
                val_out = model(Ireal_val, smap_val, is_train=False)
                metrics = compute_metrics(val_out["recon"], Ireal_val)
                val_psnr_list.append(metrics["psnr"])
                val_nmse_list.append(metrics["nmse"])

        mean_psnr = float(np.mean(val_psnr_list)) if val_psnr_list else 0.0
        mean_nmse = float(np.mean(val_nmse_list)) if val_nmse_list else 0.0

        print(
            f"--> Epoch {epoch+1} Complete: Val PSNR: {mean_psnr:.2f} dB, "
            f"Val NMSE: {mean_nmse:.4f}, Elapsed: {epoch_time:.1f}s"
        )

        # Log metrics
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{avg_recon:.6f},{avg_grad:.6e},{avg_slew:.6e},{avg_total:.6f},{mean_psnr:.4f},{mean_nmse:.6f},{epoch_time:.2f}\n")

        # 7. Save checkpoints
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_recon": optimizer_recon.state_dict(),
            "optim_traj": optimizer_traj.state_dict(),
            "best_psnr": best_psnr,
            "args": vars(args),
        }
        latest_path = os.path.join(args.save_dir, "checkpoint_latest.pth")
        torch.save(checkpoint, latest_path)

        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            checkpoint["best_psnr"] = best_psnr
            best_path = os.path.join(args.save_dir, "checkpoint_best.pth")
            torch.save(checkpoint, best_path)
            # Save optimized trajectory
            opt_traj = model.netSampling.coeff.detach().cpu().numpy()
            np.save(os.path.join(args.save_dir, "optimized_trajectory_coeff.npy"), opt_traj)
            print(f"*** New best validation PSNR ({best_psnr:.2f} dB)! Model saved. ***")

    print(f"\nTraining completed! Results saved to {args.save_dir}")


def main():
    parser = argparse.ArgumentParser(description="BJORK Trajectory & MoDL Reconstruction Training on Knee Dataset")
    # Data arguments
    parser.add_argument("--data_path", type=str, default="./knee_data", help="Path to FastMRI Knee directory")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--init_traj", type=str, default=None, help="Path to initial trajectory .npy file")
    parser.add_argument("--im_size", type=int, default=256, help="Target 2D image height/width")
    parser.add_argument("--ncoils", type=int, default=8, help="Number of receiver coils")
    parser.add_argument("--num_workers", type=int, default=2, help="Dataloader workers")

    # Trajectory arguments
    parser.add_argument("--num_shots", type=int, default=16, help="Number of shots (interleaves/spokes)")
    parser.add_argument("--nfe", type=int, default=512, help="Frequency encoding points per shot")
    parser.add_argument("--decim", type=int, default=4, help="B-spline decimation rate")
    parser.add_argument("--g_max", type=float, default=4.0, help="Max gradient limit in Gauss/cm")
    parser.add_argument("--s_max", type=float, default=15000.0, help="Max slew rate limit in Gauss/cm/s")
    parser.add_argument("--lambda_g", type=float, default=1e-3, help="Gradient penalty weight")
    parser.add_argument("--lambda_s", type=float, default=1e-4, help="Slew rate penalty weight")

    # MoDL & NUFFT arguments
    parser.add_argument("--num_blocks", type=int, default=4, help="Unrolled MoDL iteration blocks")
    parser.add_argument("--modl_lambda", type=float, default=0.05, help="MoDL regularization weight")
    parser.add_argument("--modl_tol", type=float, default=1e-4, help="CG solver tolerance")
    parser.add_argument("--cg_max_iter", type=int, default=8, help="CG solver max iterations")
    parser.add_argument("--noise_level", type=float, default=0.0, help="Measurement additive noise level")
    parser.add_argument("--grid_size", type=float, default=2.0, help="NUFFT oversampling factor")
    parser.add_argument("--numpoints", type=int, default=6, help="NUFFT interpolation kernel size")
    parser.add_argument("--filters", type=int, default=64, help="Denoiser CNN filters")
    parser.add_argument("--layers", type=int, default=5, help="Denoiser CNN depth")

    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (typically 1 for non-Cartesian MRI)")
    parser.add_argument("--lr_recon", type=float, default=1e-4, help="Learning rate for MoDL denoiser")
    parser.add_argument("--lr_traj", type=float, default=5e-4, help="Learning rate for trajectory coefficients")
    parser.add_argument("--device", type=str, default="cuda:0", help="PyTorch device (cuda:0, cuda, or cpu)")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging print interval")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
