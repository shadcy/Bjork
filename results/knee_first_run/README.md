# Experiment Report: knee_first_run

**Acceleration**: R=4 (Cartesian Equispaced (ACS=0.08))  
**Device**: cuda | **Peak GPU VRAM**: 0.69 GB | **Runtime**: 321.6s  

## Volumes Evaluated
- **Train Volumes**: 1 (38 slices)
  - `file1000584.h5`
- **Validation Volumes**: 1 (30 slices)
  - `file1000658.h5`

## Quantitative Results (fastMRI Standard)

| Method | PSNR (dB) | SSIM | NMSE |
| :--- | :---: | :---: | :---: |
| Zero-Filled Adjoint Baseline | 22.16 | 0.5593 | 0.207784 |
| Trained MoDL (Epoch 30) | **22.31** | **0.5799** | **0.202200** |

## Status
All metrics computed strictly from genuine executed forward/backward passes on real multi-coil knee data.
