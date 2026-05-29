# Noise-Aware Multi-inference Self-regulation for Self-supervised Denoising

*PyTorch implementation — PAKDD Camera Ready*

## Abstract

In real-world scenarios, the scarcity of clean visual signals poses a significant challenge for real-time computer vision applications. This problem is further compounded by complex noise distributions that lack ground truth labels and exhibit unpredictable instrumental characteristics. Adapting to these intricate distributions without clean supervision remains a formidable task. While contemporary self-supervised denoising methods attempt to address this, they often lack adaptability and suffer from inadequate reconstruction and spatial information loss.

To overcome these limitations, we propose a two-fold solution: (i) a novel data augmentation technique utilizing Tweedie's formula to broaden the model's robustness across diverse noise scenarios, and (ii) a Recursive Multi-Inference Strategy (RMIS) that minimizes denoising risk through a self-regulated process. Based on the assumption that noisy observations reside within an acceptable L2 radius of the latent ground truth, our approach effectively handles both signal-dependent and signal-independent noise, achieving competitive performance.

**Keywords:** Image Denoising · Self-supervised · Tweedie distribution

---

## Method Overview

Our framework consists of two stages:

**Stage 1 — Tweedie Data Augmentation:** Given a real noisy image `y`, two corrupted observations `y1` and `y2` are constructed using the Tweedie distribution (power index γ = 1.5), which covers a wide family of noise distributions including Gaussian, Poisson, and Gamma. A Fourier low-pass filter produces a pseudo-clean estimate `ŷ`, from which local variance and dispersion parameters are estimated patch-wise.

**Stage 2 — RMIS with Self-regulation:** A single shared denoiser is applied recursively across four stages. The total loss is:

```
L_total = L1 + L_smooth + L_id + L_rec
```

- `L1` — base MSE between the first prediction and `y2`
- `L_smooth` — piecewise smoothness prior via Fourier-smoothed target
- `L_id` — identity/Fejér monotonicity prior between consecutive predictions
- `L_rec` — Charbonnier reconstruction fidelity with whitening

---

## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- OpenCV
- scikit-image
- torchmetrics

Install all dependencies with:

```bash
pip install torch numpy opencv-python scikit-image torchmetrics
```

---

## Dataset

- **Training:** ImageNet
- **Synthetic noise testing:** BSD68, Kodak24, Set14
- **Real noise testing:** SIDD benchmark, CC, PolyU

---

## Usage

### 1. Prepare the dataloader

```bash
python dataloader.py
```

### 2. Train (synthetic noise)

```bash
python train_synthetic.py
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--batchSize` | 4 | Training batch size |
| `--epochs` | 5 | Number of training epochs |
| `--lr` | 1e-3 | Initial learning rate |
| `--noiseL` | 25 | Noise level σ |
| `--gamma` | 1.5 | Tweedie power index γ |
| `--patch_size` | 8 | Patch size for local variance estimation |
| `--mask_radius` | 0.5 | Low-pass Fourier mask radius |

Training for real noisy images will be released separately.

### 3. Test (synthetic noise)

```bash
python test_synthetic.py
```

---

## File Structure

```
multiinference_SSD/
├── augmentation.py        # Tweedie data augmentation (Stage 1)
├── train_synthetic.py     # Training script for synthetic noise
├── test_synthetic.py      # Evaluation script
├── dataloader.py          # DataLoader for ImageNet train/val
├── dataset.py             # Dataset preparation utilities
├── utils.py               # I/O and helper functions
└── models/
    └── DnCNN.py           # DnCNN backbone
```

---

## Results

### Synthetic Noise (AWGN) — PSNR (dB) / SSIM

| Method | BSD68 σ=25 | Kodak24 σ=25 | BSD68 σ=50 | Kodak24 σ=50 |
|---|---|---|---|---|
| BM3D | 28.56 / 0.801 | 31.87 / 0.868 | 25.62 / 0.687 | 27.02 / 0.813 |
| N2N (supervised) | 31.01 / 0.866 | 32.41 / 0.884 | 31.02 / 0.858 | 32.38 / 0.878 |
| B2UB | 30.89 / 0.875 | 32.27 / 0.880 | 30.82 / 0.859 | 31.31 / 0.862 |
| IDR | 31.23 / 0.883 | 32.22 / 0.891 | 30.98 / 0.862 | 31.68 / 0.844 |
| **Ours** | **37.70 / 0.931** | **37.00 / 0.944** | **32.32 / 0.861** | **32.94 / 0.872** |

### Real Noise — PSNR (dB) / SSIM on SIDD

| Method | SIDD Benchmark | SIDD Validation |
|---|---|---|
| APBSN | 34.79 / 0.913 | 34.83 / 0.916 |
| LGBPN | 35.91 / 0.922 | 36.02 / 0.930 |
| **Ours** | **35.12 / 0.931** | **36.07 / 0.932** |

---

## Acknowledgements

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2025-16072365). This research was also carried out with the support of the Jeju RISE Center, funded by the Ministry of Education and Jeju Special Self-Governing Province in 2025, as part of the "Regional Innovation System & Education (RISE): Global University 30" initiative.
