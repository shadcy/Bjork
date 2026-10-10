"""
test.py - Evaluation script for BJORK on Knee MRI dataset.
Loads a trained checkpoint and evaluates reconstruction quality and trajectory metrics.
"""

import os
import argparse
import numpy as np
import torch

from dataset import get_knee_dataloader
from models import BjorkMoDL
from train import compute_metrics


def test(args):
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    print(f"=== BJORK Knee MRI Evaluation ===")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})

    im_size = args.im_size if args.im_size is not None else saved_args.get("im_size", 256)
    ncoils = args.ncoils if args.ncoils is not None else saved_args.get("ncoils", 8)
    num_shots = saved_args.get("num_shots", 16)
    nfe = saved_args.get("nfe", 512)
    decim = saved_args.get("decim", 4)
    num_blocks = saved_args.get("num_blocks", 4)

    # Instantiate model
    model = BjorkMoDL(
        num_shots=num_shots,
        nfe=nfe,
        decim=decim,
        num_blocks=num_blocks,
        MODLLambda=saved_args.get("modl_lambda", 0.05),
        MODLtol=saved_args.get("modl_tol", 1e-4),
        cg_max_iter=saved_args.get("cg_max_iter", 8),
        device=device,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_loader = get_knee_dataloader(
        data_dir=args.data_path,
        batch_size=1,
        im_size=(im_size, im_size),
        ncoils=ncoils,
        is_train=False,
    )

    psnr_list, nmse_list = [], []
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Evaluating {len(test_loader)} test slices...")
    with torch.no_grad():
        for i, (Ireal, smap) in enumerate(test_loader):
            Ireal = Ireal.to(device)
            smap = smap.to(device)

            output = model(Ireal, smap, is_train=False)
            recon = output["recon"]
            metrics = compute_metrics(recon, Ireal)

            psnr_list.append(metrics["psnr"])
            nmse_list.append(metrics["nmse"])

            if args.save_images and i < args.max_save_images:
                recon_mag = np.abs(recon.squeeze().cpu().numpy())
                targ_mag = np.abs(Ireal.squeeze().cpu().numpy())
                np.save(os.path.join(args.out_dir, f"recon_slice_{i:03d}.npy"), recon_mag)
                np.save(os.path.join(args.out_dir, f"target_slice_{i:03d}.npy"), targ_mag)

            if (i + 1) % 10 == 0 or (i + 1) == len(test_loader):
                print(f"Slice [{i+1}/{len(test_loader)}]: PSNR = {metrics['psnr']:.2f} dB, NMSE = {metrics['nmse']:.4f}")

    print("\n" + "=" * 50)
    print("FINAL EVALUATION METRICS:")
    print(f" Mean PSNR: {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f} dB")
    print(f" Mean NMSE: {np.mean(nmse_list):.4f} ± {np.std(nmse_list):.4f}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="BJORK Knee MRI Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--data_path", type=str, default="./knee_data", help="Path to knee test data")
    parser.add_argument("--im_size", type=int, default=None, help="Image size (default: from checkpoint)")
    parser.add_argument("--ncoils", type=int, default=None, help="Number of coils (default: from checkpoint)")
    parser.add_argument("--out_dir", type=str, default="./eval_results", help="Directory to save evaluation results")
    parser.add_argument("--save_images", action="store_true", help="Save reconstructed magnitude arrays")
    parser.add_argument("--max_save_images", type=int, default=10, help="Max test images to save")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (cuda:0 or cpu)")

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    main()
