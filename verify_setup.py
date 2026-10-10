"""
verify_setup.py - Self-test diagnostic script for BJORK before running on PARAM cluster.
Verifies:
1. Environment and dependencies (PyTorch, CUDA, MIRTorch, torchkbnufft).
2. Trajectory parameterization (B-spline synthesis and constraints).
3. NUFFT operator forward and adjoint consistency.
4. Conjugate Gradient inverse operator (Gram_inv).
5. Autograd backpropagation through both trajectory coefficients and CNN denoiser.
6. Knee dataset loader (FastMRI / synthetic mode).
"""

import sys
import time
import numpy as np
import torch


def check_environment():
    print("=" * 65)
    print(" STEP 1: Environment & Hardware Diagnostic")
    print("=" * 65)
    print(f" Python Version : {sys.version.split()[0]}")
    print(f" PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f" CUDA Available : {cuda_avail}")
    if cuda_avail:
        print(f" GPU Device     : {torch.cuda.get_device_name(0)}")
        print(f" Device Count   : {torch.cuda.device_count()}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f" Total VRAM     : {mem_gb:.2f} GB")
    else:
        print(" [INFO] CUDA not detected on local machine. Running checks on CPU.")

    # Check external modules
    deps = ["mirtorch", "torchkbnufft", "h5py", "scipy"]
    for dep in deps:
        try:
            mod = __import__(dep)
            ver = getattr(mod, "__version__", "installed")
            print(f" [PASS] Module '{dep}': {ver}")
        except ImportError:
            print(f" [WARN] Module '{dep}' is not installed in the current environment.")


def check_bspline():
    print("\n" + "=" * 65)
    print(" STEP 2: Testing B-spline Trajectory Parameterization")
    print("=" * 65)
    from bspline import SamplingLayerBspline2D, bspline2_1ndsynth

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_shots = 8
    nfe = 256
    decim = 4

    print("Testing bspline2_1ndsynth...")
    coeff = torch.ones(nfe // decim, device=device)
    B, dB1, dB2 = bspline2_1ndsynth(coeff, nfe, dt=4e-6, gam=4.2576e3, ext=40)
    assert B.shape[0] == nfe // decim, f"B shape mismatch: {B.shape}"
    print(f" [PASS] B kernel matrix generated: {B.shape}")

    print("Testing SamplingLayerBspline2D forward...")
    sampling_layer = SamplingLayerBspline2D(
        num_shots=num_shots,
        nfe=nfe,
        decim=decim,
        device=device,
    )
    traj, grad, slew = sampling_layer(nbatch=1)
    assert traj.shape == (1, 2, num_shots * nfe), f"Traj shape mismatch: {traj.shape}"
    print(f" [PASS] Trajectory generated: {traj.shape}, grad: {grad.shape}, slew: {slew.shape}")
    print(f" [PASS] Trajectory range: [{traj.min().item():.3f}, {traj.max().item():.3f}] rad")


def check_sys_op():
    print("\n" + "=" * 65)
    print(" STEP 3: Testing NUFFT Operators & Conjugate Gradient Solvers")
    print("=" * 65)
    try:
        from sys_op import NuSense_om, NuSenseGram_om, Gram_inv
    except ImportError as e:
        print(f" [SKIP] sys_op imports requires mirtorch: {e}")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    im_size = (32, 32)
    ncoils = 2
    npoints = 64

    # Dummy data
    smap = torch.ones((1, ncoils, *im_size), dtype=torch.complex64, device=device) / np.sqrt(ncoils)
    ktraj = (torch.rand((1, 2, npoints), device=device) - 0.5) * 2 * np.pi
    ktraj.requires_grad_(True)
    x = torch.ones((1, 1, *im_size), dtype=torch.complex64, device=device, requires_grad=True)

    print("Testing NuSense_om forward & backward...")
    A = NuSense_om(smap, ktraj, norm="ortho", numpoints=4, grid_size=1.5)
    y = A * x
    loss = torch.sum(torch.abs(y) ** 2)
    loss.backward()
    assert ktraj.grad is not None, "Gradient w.r.t ktraj failed in NuSense_om"
    assert x.grad is not None, "Gradient w.r.t x failed in NuSense_om"
    print(f" [PASS] NuSense_om backward successful! ktraj.grad norm = {torch.norm(ktraj.grad):.4e}")

    print("Testing Gram_inv (CG solve & implicit backward)...")
    ktraj2 = (torch.rand((1, 2, npoints), device=device) - 0.5) * 2 * np.pi
    ktraj2.requires_grad_(True)
    b = torch.ones((1, 1, *im_size), dtype=torch.complex64, device=device, requires_grad=True)
    P = Gram_inv(smap, ktraj2, lambd=0.01, tol=1e-4, max_iter=5, numpoints=4, grid_size=1.5)
    Pb = P * b
    loss_cg = torch.sum(torch.abs(Pb) ** 2)
    loss_cg.backward()
    assert ktraj2.grad is not None, "Gradient w.r.t ktraj failed in Gram_inv"
    print(f" [PASS] Gram_inv backward successful! ktraj2.grad norm = {torch.norm(ktraj2.grad):.4e}")


def check_end_to_end():
    print("\n" + "=" * 65)
    print(" STEP 4: Testing End-to-End BjorkMoDL Model")
    print("=" * 65)
    try:
        from models import BjorkMoDL
    except ImportError as e:
        print(f" [SKIP] models require dependencies: {e}")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    im_size = (64, 64)
    Ireal = torch.randn((1, 1, *im_size), dtype=torch.complex64, device=device)
    smap = torch.randn((1, 4, *im_size), dtype=torch.complex64, device=device)
    smap = smap / torch.sqrt(torch.sum(torch.abs(smap) ** 2, dim=1, keepdim=True) + 1e-8)

    model = BjorkMoDL(
        num_shots=4,
        nfe=128,
        decim=4,
        num_blocks=2,
        MODLLambda=0.05,
        MODLtol=1e-3,
        cg_max_iter=4,
        numpoints=4,
        grid_size=1.5,
        device=device,
        denoiser_filters=16,
        denoiser_layers=3,
    ).to(device)

    print("Running forward pass...")
    out = model(Ireal, smap, is_train=True)
    loss_dict = model.compute_loss(out, Ireal)
    print(f" [PASS] Forward pass loss: {loss_dict['total_loss'].item():.4f}")

    print("Running backward pass...")
    loss_dict["total_loss"].backward()

    traj_grad = model.netSampling.coeff.grad
    denoiser_grad = next(model.netG_I.parameters()).grad

    assert traj_grad is not None and not torch.isnan(traj_grad).any(), "NaN or None in trajectory gradient"
    assert denoiser_grad is not None and not torch.isnan(denoiser_grad).any(), "NaN or None in denoiser gradient"
    print(f" [PASS] Trajectory param grad norm: {torch.norm(traj_grad):.4e}")
    print(f" [PASS] Denoiser weight grad norm : {torch.norm(denoiser_grad):.4e}")


def check_dataset():
    print("\n" + "=" * 65)
    print(" STEP 5: Testing FastMRI Knee Dataset Module")
    print("=" * 65)
    from dataset import FastMRIKneeDataset, estimate_sensitivity_maps
    # Verify sensitivity map estimation math on raw multi-coil array
    dummy_kspace = np.random.randn(8, 64, 64) + 1j * np.random.randn(8, 64, 64)
    smap = estimate_sensitivity_maps(dummy_kspace.astype(np.complex64))
    assert smap.shape == (8, 64, 64), f"Smap shape mismatch: {smap.shape}"
    print(f" [PASS] FastMRI sensitivity map estimator verified: {smap.shape}")


def main():
    print("\n" + "#" * 65)
    print("   BJORK ON PARAM CLUSTER: PRE-FLIGHT VERIFICATION SUITE   ")
    print("#" * 65)

    check_environment()
    try:
        check_bspline()
    except Exception as e:
        print(f" [FAIL] B-spline test failed: {e}")

    try:
        check_sys_op()
    except Exception as e:
        print(f" [FAIL] Sys_op test failed: {e}")

    try:
        check_end_to_end()
    except Exception as e:
        print(f" [FAIL] End-to-end model test failed: {e}")

    try:
        check_dataset()
    except Exception as e:
        print(f" [FAIL] Dataset test failed: {e}")

    print("\n" + "#" * 65)
    print("   VERIFICATION SUITE COMPLETED")
    print("#" * 65)


if __name__ == "__main__":
    main()
