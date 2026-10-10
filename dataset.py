"""
dataset.py - fastMRI Knee MRI Dataset Loader for BJORK.
Supports:
1. Raw FastMRI multicoil / singlecoil Knee dataset (.h5 files).
2. Preprocessed real Knee slice files (.npy / .npz).
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import h5py
except ImportError:
    h5py = None


def estimate_sensitivity_maps(kspace_slice, acs_size=(24, 24)):
    """
    Estimate multi-coil sensitivity maps from central k-space autocalibration signal (ACS).

    Args:
        kspace_slice (np.ndarray): Multi-coil kspace slice of shape (ncoils, H, W).
        acs_size (tuple): Central region size (acs_h, acs_w).

    Returns:
        smap (np.ndarray): Normalized sensitivity maps of shape (ncoils, H, W), complex64.
    """
    ncoils, h, w = kspace_slice.shape
    acs_h, acs_w = acs_size
    low_h = max(0, (h - acs_h) // 2)
    high_h = min(h, low_h + acs_h)
    low_w = max(0, (w - acs_w) // 2)
    high_w = min(w, low_w + acs_w)

    # Windowed low-resolution k-space
    acs_mask = np.zeros((h, w), dtype=np.float32)
    win_h = np.hanning(high_h - low_h)[:, None]
    win_w = np.hanning(high_w - low_w)[None, :]
    acs_mask[low_h:high_h, low_w:high_w] = win_h * win_w

    low_kspace = kspace_slice * acs_mask[None, :, :]
    # Centered 2D IFFT
    low_img = np.fft.ifftshift(
        np.fft.ifft2(np.fft.fftshift(low_kspace, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    )

    # Root sum-of-squares normalization
    rss = np.sqrt(np.sum(np.abs(low_img) ** 2, axis=0, keepdims=True)) + 1e-8
    smap = low_img / rss
    return smap.astype(np.complex64)


class FastMRIKneeDataset(Dataset):
    """
    PyTorch Dataset for fastMRI Knee dataset (.h5 files).
    """

    def __init__(self, data_dir, im_size=(256, 256), ncoils=8, is_train=True):
        super(FastMRIKneeDataset, self).__init__()
        self.data_dir = data_dir
        self.im_size = tuple(im_size)
        self.ncoils = ncoils
        self.is_train = is_train

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Knee dataset directory not found: {data_dir}")

        # Search for .h5 files
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
        if self.files:
            self.mode = "fastmri"
        else:
            # Check for preprocessed .npy / .npz files
            self.files = sorted(glob.glob(os.path.join(data_dir, "*.npy"))) + \
                         sorted(glob.glob(os.path.join(data_dir, "*.npz")))
            if self.files:
                self.mode = "preprocessed"
            else:
                raise FileNotFoundError(
                    f"No fastMRI .h5 or .npy/.npz files found in {data_dir}. "
                    f"Please provide the path to your fastMRI knee dataset."
                )

        # Index all individual 2D slices across the HDF5 volumes
        self.slice_index = []
        if self.mode == "fastmri":
            if h5py is None:
                raise ImportError(
                    "h5py is required to read FastMRI .h5 files. Please install it with 'pip install h5py'."
                )
            for fpath in self.files:
                try:
                    with h5py.File(fpath, "r") as hf:
                        if "kspace" in hf:
                            num_slices = hf["kspace"].shape[0]
                            for s in range(num_slices):
                                self.slice_index.append((fpath, s))
                except Exception as e:
                    print(f"Warning: Skipping unreadable file {fpath}: {e}")

            if not self.slice_index:
                raise ValueError(f"No valid slices found in fastMRI files under {data_dir}")

            self.length = len(self.slice_index)
            print(f"[FastMRIKneeDataset] Loaded {len(self.files)} knee volumes ({self.length} total 2D slices).")
        else:
            self.length = len(self.files)
            print(f"[FastMRIKneeDataset] Loaded {self.length} preprocessed knee slices.")

    def __len__(self):
        return self.length

    def _crop_or_pad(self, img, target_size):
        """Center crop or pad 2D array to target_size (H, W)."""
        th, tw = target_size
        h, w = img.shape[-2], img.shape[-1]

        # Crop / pad height
        if h > th:
            sh = (h - th) // 2
            img = img[..., sh:sh + th, :]
        elif h < th:
            pad_b = (th - h) // 2
            pad_a = th - h - pad_b
            pad_shape = list(img.shape)
            pad_shape[-2] = pad_b
            b = np.zeros(pad_shape, dtype=img.dtype)
            pad_shape[-2] = pad_a
            a = np.zeros(pad_shape, dtype=img.dtype)
            img = np.concatenate([b, img, a], axis=-2)

        # Crop / pad width
        if w > tw:
            sw = (w - tw) // 2
            img = img[..., sw:sw + tw]
        elif w < tw:
            pad_b = (tw - w) // 2
            pad_a = tw - w - pad_b
            pad_shape = list(img.shape)
            pad_shape[-1] = pad_b
            b = np.zeros(pad_shape, dtype=img.dtype)
            pad_shape[-1] = pad_a
            a = np.zeros(pad_shape, dtype=img.dtype)
            img = np.concatenate([b, img, a], axis=-1)

        return img

    def __getitem__(self, idx):
        if self.mode == "fastmri":
            fpath, slice_idx = self.slice_index[idx]
            with h5py.File(fpath, "r") as hf:
                kspace = hf["kspace"][slice_idx]  # shape: (ncoils, H, W) or (ncoils, H, W, 2)
                if kspace.ndim == 2:
                    kspace = kspace[None, :, :]
                elif kspace.ndim == 4:
                    # Real and imaginary components stored separately
                    kspace = kspace[..., 0] + 1j * kspace[..., 1]

                kspace_c = kspace.astype(np.complex64)
                imgs_coil = np.fft.ifftshift(
                    np.fft.ifft2(np.fft.fftshift(kspace_c, axes=(-2, -1)), axes=(-2, -1)),
                    axes=(-2, -1),
                )

                # Estimate sensitivity maps from ACS
                smap_c = estimate_sensitivity_maps(kspace_c)

                # Combine multi-coil images
                image_c = np.sum(imgs_coil * np.conj(smap_c), axis=0)

                # Normalize by 99th percentile
                norm_val = np.percentile(np.abs(image_c), 99) + 1e-8
                image_c = image_c / norm_val

                # Crop/pad to target im_size
                image_c = self._crop_or_pad(image_c, self.im_size)
                smap_c = self._crop_or_pad(smap_c, self.im_size)
        else:
            # Preprocessed .npy / .npz
            fpath = self.files[idx]
            data = np.load(fpath)
            if isinstance(data, np.ndarray):
                image_c = data
                smap_c = np.ones((self.ncoils, *self.im_size), dtype=np.complex64) / np.sqrt(self.ncoils)
            else:
                image_c = data["image"]
                smap_c = data["smap"]
            image_c = self._crop_or_pad(image_c, self.im_size)
            smap_c = self._crop_or_pad(smap_c, self.im_size)

        # Convert to torch complex tensors
        image_t = torch.from_numpy(image_c).unsqueeze(0).to(torch.complex64)
        smap_t = torch.from_numpy(smap_c).to(torch.complex64)

        return image_t, smap_t


def get_knee_dataloader(data_dir, batch_size=1, im_size=(256, 256), ncoils=8, is_train=True, num_workers=0):
    dataset = FastMRIKneeDataset(
        data_dir=data_dir,
        im_size=im_size,
        ncoils=ncoils,
        is_train=is_train,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        drop_last=is_train,
    )
    return loader
