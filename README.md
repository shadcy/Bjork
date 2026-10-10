# BJORK: B-spline Parameterized Joint Optimization of Reconstruction and K-space Trajectories for Accelerated 2D MRI

PyTorch implementation for joint optimization of 2D MRI sampling trajectories and deep unrolled reconstruction (MoDL) on high-performance computing clusters (e.g., **PARAM Cluster**) using the **fastMRI Knee dataset**.

---

## 📁 Repository Structure

| File | Description |
| :--- | :--- |
| [`bspline.py`](bspline.py) | Quadratic B-spline trajectory parameterization and hardware waveform generator ($G, S$). |
| [`sys_op.py`](sys_op.py) | Forward NUFFT operator ($A$), adjoint ($A^H$), Gram ($A^H A$), and CG inverse ($(A^H A + \lambda I)^{-1}$). |
| [`models.py`](models.py) | Unrolled MoDL reconstruction network and BJORK joint optimization model. |
| [`dataset.py`](dataset.py) | Knee MRI dataset loader (fastMRI `.h5` files, `.npy`/`.npz` slices, and synthetic phantoms). |
| [`traj_utils.py`](traj_utils.py) | Initial non-Cartesian trajectory generators (Radial, Golden-Angle, Archimedean Spiral). |
| [`demo.py`](demo.py) | Complete self-contained forward & backward gradient verification demo. |
| [`train.py`](train.py) | Full training pipeline with separate trajectory and reconstruction learning rates, and hardware constraints. |
| [`test.py`](test.py) | Evaluation script for trained models computing PSNR and NMSE metrics. |
| [`verify_setup.py`](verify_setup.py) | Automated pre-flight environment & numerical stability diagnostic suite. |
| [`submit_job.slurm`](submit_job.slurm) | Ready-to-run SLURM batch submission script for the PARAM cluster. |
| [`run_param.sh`](run_param.sh) | Helper launcher script for interactive or batch execution on the cluster. |
| [`requirements.txt`](requirements.txt) | Python dependencies. |
| [`environment.yml`](environment.yml) | Conda environment specification. |

---

## 🚀 Quick Start on PARAM Cluster

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/shadcy/Bjork.git
cd Bjork

# Option A: Using Conda (Recommended)
conda env create -f environment.yml
conda activate bjork

# Option B: Using Pip in an existing virtualenv / module
pip install -r requirements.txt
```

### 2. Pre-flight Verification

Before launching a long job, run the pre-flight verification script to confirm CUDA, MIRTorch, torchkbnufft, and autograd gradients:

```bash
python verify_setup.py
# or
bash run_param.sh verify
```

### 3. Quick Demo / Test Run

Run an end-to-end forward and backward pass on a simulated knee slice:

```bash
python demo.py
```

---

## 🏃 Training on FastMRI Knee Dataset

### Option A: Submit via SLURM Batch (PARAM Cluster)

Edit `submit_job.slurm` with your account/partition details, then submit:

```bash
sbatch submit_job.slurm /path/to/fastmri_knee/multicoil_train
```

### Option B: Interactive / Direct Execution

```bash
python train.py \
    --data_path /path/to/fastmri_knee/multicoil_train \
    --save_dir ./checkpoints/knee_run_01 \
    --epochs 50 \
    --batch_size 1 \
    --im_size 256 \
    --ncoils 8 \
    --num_shots 16 \
    --nfe 512 \
    --decim 4 \
    --num_blocks 4 \
    --modl_lambda 0.05 \
    --g_max 4.0 \
    --s_max 15000.0 \
    --device cuda:0
```

### Option C: Synthetic / Dummy Mode (No Data Download Required)

To test the entire training loop without downloading the multi-gigabyte fastMRI dataset:

```bash
python train.py --dummy_size 64 --epochs 5 --save_dir ./checkpoints/dummy_test --device cuda:0
```

---

## 📊 Evaluation & Testing

Evaluate a saved checkpoint on test knee slices:

```bash
python test.py \
    --checkpoint ./checkpoints/knee_run_01/checkpoint_best.pth \
    --data_path /path/to/fastmri_knee/multicoil_val \
    --out_dir ./eval_results \
    --save_images
```

---

## 🔬 Key Method Details

1. **Hardware Constraints:**
   - Maximum gradient strength $G_{\max} = 4.0\text{ G/cm}$.
   - Maximum slew rate $S_{\max} = 15000\text{ G/cm/s}$.
   - Penalties: $\mathcal{L}_{G} = \text{mean}(\text{ReLU}(|G| - G_{\max})^2)$, $\mathcal{L}_{S} = \text{mean}(\text{ReLU}(|S| - S_{\max})^2)$.

2. **Differentiable Operators:**
   - Uses `NuSense_om` and `Gram_inv` from `sys_op.py` with efficient NUFFT Jacobian approximation.
   - Backward gradients are propagated to both network denoiser weights and B-spline trajectory control points $\mathbf{c}$.

---

## References

```bibtex
@article{wang:22:bjork,
  author={Wang, Guanhua and Luo, Tianrui and Nielsen, Jon-Fredrik and Noll, Douglas C. and Fessler, Jeffrey A.},
  journal={IEEE Transactions on Medical Imaging}, 
  title={B-spline Parameterized Joint Optimization of Reconstruction and K-space Trajectories (BJORK) for Accelerated 2D MRI}, 
  year={2022},
  volume={41},
  number={9},
  pages={2318-2329},
  doi={10.1109/TMI.2022.3161875}
}

@article{wang:22:eaj,
  title={Efficient approximation of Jacobian matrices involving a non-uniform fast Fourier transform (NUFFT)},
  author={Wang, Guanhua and Fessler, Jeffrey A.},
  journal={IEEE Transactions on Computational Imaging},
  year={2023}
}
```
