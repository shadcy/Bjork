import numpy as np
import torch
import torch.nn as nn
import time
import math
import scipy
import scipy.linalg


def bspline2_1ndsynth(coeff, nfe, dt, gam, ext):
    # Coeff: [Ncycle]: Contain coefficient. Default should be ones.
    # length: The length of time points
    num_kernels = coeff.shape[0]
    dev = coeff.device
    ti = torch.linspace(0, num_kernels, nfe + ext, device=dev, dtype=coeff.dtype)
    ti = torch.max(ti, -1.5 * torch.ones_like(ti))
    ti = torch.min(ti, (num_kernels + 0.5) * torch.ones_like(ti))
    ti = ti + 2
    n0 = torch.floor(ti - 0.5).to(dtype=torch.long)

    # Construct the quadratic b-spline kernel
    def b2f0(t):
        return 3/4 - torch.pow(t, 2)
    def b2f1(t):
        return torch.pow(torch.abs(t) - 1.5, 2) / 2

    B = torch.zeros(nfe + ext, num_kernels, device=dev, dtype=coeff.dtype)
    for i in range(coeff.shape[0]):
        coeff_i = torch.zeros(coeff.shape[0] + 5, dtype=coeff.dtype, device=dev)
        coeff_i[i + 2] = coeff[i]
        ft = coeff_i[0 + n0] * b2f1(ti - n0) + coeff_i[1 + n0] * b2f0(ti - (n0 + 1)) + coeff_i[2 + n0] * b2f1(ti - (n0 + 2))
        B[:, i] = ft

    dB1 = B[ext//2:nfe + ext//2 - 1, :] - B[ext//2 + 1:nfe + ext//2, :]
    dB2 = dB1[0:-1, :] - dB1[1:, :]
    (_, b1max) = torch.max(dB1, dim=0)
    (_, b1min) = torch.min(dB1, dim=0)
    (_, b2max) = torch.max(dB2, dim=0)
    (_, b2min) = torch.min(dB2, dim=0)
    b1ind = np.union1d(b1max.detach().cpu().numpy(), b1min.detach().cpu().numpy())
    b1ind = torch.as_tensor(b1ind, dtype=torch.long, device=dev)
    b2ind = np.union1d(b2max.detach().cpu().numpy(), b2min.detach().cpu().numpy())
    b2ind = torch.as_tensor(b2ind, dtype=torch.long, device=dev)
    dB1 = dB1[b1ind, :] / gam / dt
    dB2 = dB2[b2ind, :] / dt / gam / dt
    return B.permute(1, 0), dB1.permute(1, 0), dB2.permute(1, 0)


class SamplingLayerBspline2D(nn.Module):
    """
        The unit system are as follow:
            k: cycle / cm
            g: Gauss / cm
            s: Gauss / cm / s
            dt: s
            gamma: Hz / Gauss
            fov/res: cm
        Input:
            init_traj: [ndim, nacq], file path, ndarray, or None (defaults to radial)
            num_shots: number of shots
            nfe: num of points per acquisition
            decim: decimation rate
        Return:
            traj: [nbatch, ndim, num_points]
            grad: [nshot, num_kernels]
            slew: [nshot, num_kernels]
    """

    def __init__(self, num_shots, nfe, decim=4, gamma=4.2576e+03, dt=4e-6, res=0.1, init_traj=None, ndims=2, gpu_ids=[],
                 ext=40, device=None):
        super(SamplingLayerBspline2D, self).__init__()
        self.nfe = nfe
        self.num_kernels = nfe // decim
        self.ndims = ndims
        self.num_shots = num_shots
        self.kmax = 1 / res
        self.gamma = gamma
        self.dt = dt
        self.decim = decim
        self.ext = ext

        # Resolve device robustly
        if device is not None:
            self.device = torch.device(device)
        elif len(gpu_ids) > 0:
            gid = gpu_ids[0]
            self.device = torch.device(f"cuda:{gid}" if isinstance(gid, int) else gid)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.coeff = torch.ones(self.ndims, num_shots, self.num_kernels, device=self.device)

        if decim > 1:
            # zero-padding length
            self.B, self.dB1, self.dB2 = bspline2_1ndsynth(
                torch.ones(self.num_kernels, device=self.device), nfe, dt, gamma, self.ext
            )
            B_np = self.B.permute(1, 0).detach().cpu().numpy()

        # The shape of traj_ref should be [ndim, nshot, npoints]
        if isinstance(init_traj, str):
            traj_ref = np.load(init_traj)
        elif isinstance(init_traj, np.ndarray):
            traj_ref = init_traj
        elif isinstance(init_traj, torch.Tensor):
            traj_ref = init_traj.detach().cpu().numpy()
        elif init_traj is None:
            # Generate default radial trajectory in [-pi, pi]
            angles = np.linspace(0, np.pi, self.num_shots, endpoint=False)
            radii = np.linspace(-np.pi, np.pi, self.nfe)
            traj_ref = np.zeros((self.ndims, self.num_shots, self.nfe))
            for s in range(self.num_shots):
                traj_ref[0, s, :] = radii * np.cos(angles[s])
                traj_ref[1, s, :] = radii * np.sin(angles[s])
        else:
            raise ValueError(f"Unsupported init_traj format: {type(init_traj)}")

        if len(traj_ref.shape) == 2:
            traj_ref = np.reshape(traj_ref, (self.ndims, self.num_shots, self.nfe))

        traj_ref = traj_ref / np.pi * self.kmax / 2

        # No parameterization
        if decim == 1:
            self.coeff = torch.tensor(traj_ref, dtype=self.coeff.dtype, device=self.device)
        else:
            for ii in range(self.ndims):
                for jj in range(self.num_shots):
                    traj_ref_i = np.zeros(nfe + self.ext)
                    traj_ref_i[0:self.ext // 2] = traj_ref[ii, jj, 0]
                    traj_ref_i[nfe + self.ext // 2:nfe + self.ext] = traj_ref[ii, jj, nfe - 1]
                    traj_ref_i[self.ext // 2:nfe + self.ext // 2] = traj_ref[ii, jj, :]
                    lstsq_res = np.linalg.lstsq(B_np, traj_ref_i, rcond=None)[0]
                    self.coeff[ii, jj, :] = torch.tensor(lstsq_res, dtype=self.coeff.dtype, device=self.device)

        self.coeff = nn.Parameter(self.coeff)

    def forward(self, nbatch=1):
        # Extract the locations for maximum gradient and slew rate
        if self.decim == 1:
            traj = self.coeff * 1
            gradient = (traj[:, :, :-1] - traj[:, :, 1:]) / self.gamma / self.dt
            slew = (gradient[:, :, :-1] - gradient[:, :, 1:]) / self.dt
        else:
            traj = torch.matmul(self.coeff, self.B)[:, :, self.ext // 2:self.nfe + self.ext // 2]
            gradient = torch.matmul(self.coeff, self.dB1)
            slew = torch.matmul(self.coeff, self.dB2)

        traj = torch.reshape(traj, (self.ndims, self.num_shots * self.nfe)).unsqueeze(0)

        # transfer trajectory to om (normalization, between -pi to pi)
        traj = traj / self.kmax * 2 * np.pi

        if isinstance(nbatch, int) and nbatch > 1:
            traj = traj.repeat(nbatch, 1, 1)

        self.traj = traj
        self.gradient = gradient
        self.slew = slew
        return traj, gradient, slew

