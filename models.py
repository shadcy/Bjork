"""
models.py - MoDL Reconstruction Network with BJORK Trajectory Parameterization.
Implements:
1. ResidualDenoiser2D (Complex-valued 2-channel CNN denoiser).
2. BjorkMoDL (End-to-end unrolled MoDL framework with learnable B-spline trajectory).
3. Hardware loss calculation for gradient and slew rate scanner constraints.
"""

import torch
import torch.nn as nn
from bspline import SamplingLayerBspline2D

try:
    from sys_op import NuSense_om, NuSenseGram_om, Gram_inv
    HAS_SYS_OP = True
except ImportError:
    NuSense_om = None
    NuSenseGram_om = None
    Gram_inv = None
    HAS_SYS_OP = False


class ResidualDenoiser2D(nn.Module):
    """
    Residual 2D CNN denoiser operating on 2 channels (Real, Imag).
    Implements: out = x + ConvNet(x).
    """

    def __init__(self, in_channels=2, num_filters=64, num_layers=5):
        super(ResidualDenoiser2D, self).__init__()
        layers = []
        # First layer
        layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=True))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        # Intermediate layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=True))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        # Final projection back to 2 channels
        layers.append(nn.Conv2d(num_filters, in_channels, kernel_size=3, padding=1, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)


class BjorkMoDL(nn.Module):
    """
    Joint Optimization of Reconstruction and Trajectory (BJORK) with MoDL backbone.

    Args:
        num_shots (int): Number of k-space trajectory shots.
        nfe (int): Number of frequency encoding points per shot.
        decim (int): B-spline decimation factor.
        num_blocks (int): Number of unrolled MoDL iterations.
        MODLLambda (float): MoDL data consistency weighting factor.
        MODLtol (float): CG solver tolerance.
        cg_max_iter (int): Maximum iterations for inner CG solver.
        noise_level (float): Simulated measurement noise level.
        grid_size (float): NUFFT oversampling factor.
        numpoints (int): NUFFT interpolation kernel size.
        g_max (float): Maximum physical gradient limit (Gauss/cm).
        s_max (float): Maximum physical slew rate limit (Gauss/cm/s).
        init_traj: Initial trajectory (.npy path, ndarray, or None).
        device: PyTorch device.
    """

    def __init__(
        self,
        num_shots=16,
        nfe=512,
        decim=4,
        num_blocks=4,
        MODLLambda=0.05,
        MODLtol=1e-5,
        cg_max_iter=8,
        noise_level=0.0,
        grid_size=2.0,
        numpoints=6,
        g_max=4.0,
        s_max=15000.0,
        init_traj=None,
        device=None,
        denoiser_filters=64,
        denoiser_layers=5,
    ):
        super(BjorkMoDL, self).__init__()
        self.num_blocks = num_blocks
        self.MODLLambda = MODLLambda
        self.MODLtol = MODLtol
        self.cg_max_iter = cg_max_iter
        self.noise_level = noise_level
        self.grid_size = grid_size
        self.numpoints = numpoints
        self.g_max = g_max
        self.s_max = s_max

        # 1. B-spline learnable sampling trajectory layer
        self.netSampling = SamplingLayerBspline2D(
            num_shots=num_shots,
            nfe=nfe,
            decim=decim,
            init_traj=init_traj,
            device=device,
        )

        # 2. Denoiser network
        self.netG_I = ResidualDenoiser2D(
            in_channels=2,
            num_filters=denoiser_filters,
            num_layers=denoiser_layers,
        )

    def forward(self, Ireal, smap, is_train=True):
        """
        Forward pass:
        Args:
            Ireal: Ground truth target image of shape [nbatch, 1, H, W] (complex64).
            smap: Multi-coil sensitivity maps of shape [nbatch, ncoils, H, W] (complex64).
            is_train: Boolean indicating training mode (simulates noise if True).

        Returns:
            dict containing:
                'recon': Reconstructed image [nbatch, 1, H, W] (complex64)
                'Iunder': Adjoint zero-filled image [nbatch, 1, H, W] (complex64)
                'ktraj': Generated trajectory [nbatch, 2, num_points]
                'grad': Trajectory gradients [nshot, num_kernels]
                'slew': Trajectory slew rates [nshot, num_kernels]
        """
        if not HAS_SYS_OP:
            raise ImportError(
                "NuSense_om and Gram_inv require MIRTorch. "
                "Please run in an environment with MIRTorch installed (e.g. on PARAM Rudra)."
            )
        nbatch = Ireal.shape[0]

        # Generate sampling trajectory and hardware waveforms
        ktraj, grad, slew = self.netSampling(nbatch)

        # Build forward NUFFT operator A
        A = NuSense_om(
            smap,
            ktraj,
            grid_size=self.grid_size,
            norm="ortho",
            numpoints=self.numpoints,
        )

        # Synthesize undersampled k-space measurements
        kunder = A * Ireal
        if is_train and self.noise_level > 0.0:
            noise = torch.randn_like(kunder) * torch.max(torch.abs(kunder)) * self.noise_level
            kunder = kunder + noise

        # Adjoint reconstruction: A^H * y
        Iunder = A.H * kunder

        # Initial QWLS reconstruction: P0 = (A'A + 0.001 I)^-1
        P0 = Gram_inv(
            smap,
            ktraj,
            lambd=0.001,
            tol=self.MODLtol,
            max_iter=min(12, self.cg_max_iter + 4),
            norm="ortho",
            numpoints=self.numpoints,
            grid_size=self.grid_size,
            alert=False,
        )

        # Unrolled data consistency operator: P1 = (A'A + lambda I)^-1
        P1 = Gram_inv(
            smap,
            ktraj,
            lambd=self.MODLLambda,
            tol=self.MODLtol,
            max_iter=self.cg_max_iter,
            norm="ortho",
            numpoints=self.numpoints,
            grid_size=self.grid_size,
            alert=False,
        )

        # Initialize estimation
        Iinit = P0 * Iunder
        Iiter = Iinit

        # Unrolled MoDL iterations
        for _ in range(self.num_blocks):
            # Complex -> 2-channel real: (B, 1, H, W) -> (B, 2, H, W)
            real_in = torch.view_as_real(Iiter.squeeze(1)).permute(0, 3, 1, 2).contiguous()
            # Denoising
            denoised_real = self.netG_I(real_in)
            # 2-channel real -> Complex: (B, 2, H, W) -> (B, 1, H, W)
            denoised_complex = torch.view_as_complex(
                denoised_real.permute(0, 2, 3, 1).contiguous()
            ).unsqueeze(1)
            # Data consistency step: P1 * (Iunder + lambda * denoised)
            Iiter = P1 * (Iunder + self.MODLLambda * denoised_complex)

        return {
            "recon": Iiter,
            "Iunder": Iunder,
            "ktraj": ktraj,
            "grad": grad,
            "slew": slew,
        }

    def compute_hardware_loss(self, grad, slew):
        """
        Compute gradient and slew rate constraint violations as penalty losses.
        Loss = mean(ReLU(|grad| - G_max)^2) + mean(ReLU(|slew| - S_max)^2).
        """
        loss_grad = torch.mean(torch.relu(torch.abs(grad) - self.g_max) ** 2)
        loss_slew = torch.mean(torch.relu(torch.abs(slew) - self.s_max) ** 2)
        return loss_grad, loss_slew

    def compute_loss(self, output, Ireal, lambda_grad=1e-3, lambda_slew=1e-4):
        """
        Compute total loss combining reconstruction loss and hardware penalties.
        """
        recon = output["recon"]
        grad = output["grad"]
        slew = output["slew"]

        # Reconstruction loss: L1 + 0.1 * L2
        diff = recon - Ireal
        l1_loss = torch.mean(torch.abs(diff))
        l2_loss = torch.mean(torch.abs(diff) ** 2)
        recon_loss = l1_loss + 0.1 * l2_loss

        # Hardware constraints
        loss_grad, loss_slew = self.compute_hardware_loss(grad, slew)
        total_loss = recon_loss + lambda_grad * loss_grad + lambda_slew * loss_slew

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "grad_loss": loss_grad,
            "slew_loss": loss_slew,
        }
