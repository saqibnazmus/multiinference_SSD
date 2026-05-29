# Noise-Aware Multi-inference Self-regulation for Self-supervised Denoising

> **📄 [PAKDD 2026](https://pakdd2026.org/)** 

---

## Abstract

In real-world scenarios, the scarcity of clean visual signals poses a significant challenge for real-time computer vision applications. This problem is further compounded by complex noise distributions that lack ground truth labels and exhibit unpredictable instrumental characteristics. Adapting to these intricate distributions without clean supervision remains a formidable task. While contemporary self-supervised denoising methods attempt to address this, they often lack adaptability and suffer from inadequate reconstruction and spatial information loss. To overcome these limitations, we propose a two-fold solution: (i) a novel data augmentation technique utilizing **Tweedie's formula** to broaden the model's robustness across diverse noise scenarios, and (ii) a **Recursive Multi-Inference Strategy (RMIS)** that minimizes denoising risk through a self-regulated process. Based on the assumption that noisy observations reside within an acceptable L2 radius of the latent ground truth, our approach effectively handles both signal-dependent and signal-independent noise, achieving competitive performance.


---

## Method Overview

Our framework consists of two stages:

**Stage 1 — Tweedie Data Augmentation** (`augmentation.py`)
Given a real noisy image `y`, two corrupted observations `y1` and `y2` are constructed using the Tweedie distribution (power index γ = 1.5), which covers a wide family of noise types including Gaussian, Poisson, and Gamma. A Fourier low-pass filter produces a pseudo-clean estimate `ŷ`, from which local variance and dispersion parameters are estimated patch-wise.

**Stage 2 — RMIS with Self-regulation** (`train_synthetic.py` / `train_real.py`)
A single shared denoiser is applied recursively across four stages. The total loss (Equation 8) is:

```
L_total = L1 + L_smooth + L_id + L_rec
```

| Term | Role |
|---|---|
| `L1` | Base MSE — maps first prediction toward alternate observation `y2` |
| `L_smooth` | Piecewise smoothness prior via Fourier-smoothed target |
| `L_id` | Fejér monotonicity — penalises oscillation between consecutive stages |
| `L_rec` | Charbonnier reconstruction fidelity with whitening — prevents over-smoothing |

---

## Repository Structure

```
multiinference_SSD/
├── augmentation.py       # Stage 1 — Tweedie data augmentation (Eq. 1–6)
├── train_synthetic.py    # Training on synthetic noise (Gaussian / Poisson)
├── train_real.py         # Training on real noise (SIDD Medium sRGB)
├── test_synthetic.py     # Evaluation on BSD68, Kodak24, Set14
├── test_real.py          # Evaluation on SIDD Validation + Benchmark
├── dataloader.py         # DataLoaders for ImageNet and SIDD
├── dataset.py            # Dataset preparation utilities
├── utils.py              # I/O and helper functions
├── README.md
└── models/
    └── DnCNN.py          # DnCNN backbone (17 layers)
```

---

## Reproducibility

### 1. Environment setup

```bash
git clone https://github.com/saqibnazmus/multiinference_SSD.git
cd multiinference_SSD

pip install torch torchvision numpy opencv-python scikit-image torchmetrics tqdm pillow
```

Tested with: Python 3.8 · PyTorch 1.13 · CUDA 11.7

---

### 2. Prepare datasets

#### Synthetic experiments
Download and organise the following into `./data/` and `./test/`:

| Split | Dataset | Path |
|---|---|---|
| Training | ImageNet | `./data/train/input/` and `./data/train/target/` |
| Validation | ImageNet val | `./data/validation/noisy/` and `./data/validation/clean/` |
| Test | BSD68 | `./test/bsd68/` (clean) and `./test/bsd68_noisy/` |
| Test | Kodak24 | `./test/kodak24/` and `./test/kodak24_noisy/` |
| Test | Set14 | `./test/set14/` and `./test/set14_noisy/` |

Then run the dataloader preparation:

```bash
python dataloader.py
```

#### Real noise experiments
Download [SIDD Medium Dataset](https://www.eecs.yorku.ca/~kamel/sidd/) (sRGB images) and organise as:

```
SIDD/
├── train/
│   └── noisy/        *.png   (sRGB noisy patches — no GT needed for training)
├── val/
│   ├── noisy/        *.png
│   └── gt/           *.png
└── benchmark/
    └── noisy/        *.png   (official benchmark — no GT)
```

---

### 3. Training

#### Synthetic noise (Gaussian / Poisson)

```bash
python train_synthetic.py \
    --batchSize 4 \
    --epochs 100 \
    --lr 1e-3 \
    --milestone 50 \
    --noiseL 25 \
    --gamma 1.5 \
    --patch_size 8 \
    --mask_radius 0.5 \
    --dataset_train_dir ./ImageNet/train \
    --dataset_val_dir   ./ImageNet/validation \
    --root_dir ./data \
    --outf ./weights
```

#### Real noise (SIDD Medium sRGB)

```bash
python train_real.py \
    --batchSize 4 \
    --epochs 100 \
    --lr 1e-3 \
    --milestone 50 \
    --patch_size 128 \
    --gamma 1.5 \
    --aug_patch_size 8 \
    --mask_radius 0.5 \
    --sidd_dir ./SIDD \
    --outf ./weights_real
```

Key training arguments:

| Argument | Default | Description |
|---|---|---|
| `--batchSize` | 4 | Training batch size |
| `--epochs` | 100 | Number of training epochs |
| `--lr` | 1e-3 | Initial learning rate |
| `--milestone` | 50 | Epoch at which LR decays by 10× |
| `--gamma` | 1.5 | Tweedie power index γ (paper: fixed at 1.5) |
| `--patch_size` | 8 | Patch size for local variance estimation |
| `--mask_radius` | 0.5 | Fourier low-pass mask cutoff radius |

---

### 4. Evaluation

#### Synthetic noise

```bash
python test_synthetic.py \
    --num_of_layers 17 \
    --logdir ./weights \
    --test_data bsd68 \
    --test_noiseL 25 \
    --root_dir ./test
```

#### Real noise — SIDD Validation + Benchmark

```bash
python test_real.py \
    --num_of_layers 17 \
    --logdir ./weights_real \
    --sidd_val_dir   ./SIDD/val \
    --sidd_bench_dir ./SIDD/benchmark \
    --save_dir ./results_real \
    --batch_size 4
```

Add `--save_images` to also save denoised outputs from the validation split.
The benchmark outputs in `./results_real/sidd_benchmark/` can be uploaded directly to the [SIDD online server](https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php) for official scores.

---

## Results

### Synthetic Noise (AWGN) — PSNR (dB) / SSIM

| Method | Supervision | BSD68 σ=25 | Kodak24 σ=25 | BSD68 σ=50 | Kodak24 σ=50 |
|---|---|---|---|---|---|
| BM3D | None | 28.56 / 0.801 | 31.87 / 0.868 | 25.62 / 0.687 | 27.02 / 0.813 |
| N2N | Weak | 31.01 / 0.866 | 32.41 / 0.884 | 31.02 / 0.858 | 32.38 / 0.878 |
| B2UB | Self | 30.89 / 0.875 | 32.27 / 0.880 | 30.82 / 0.859 | 31.31 / 0.862 |
| IDR | Self | 31.23 / 0.883 | 32.22 / 0.891 | 30.98 / 0.862 | 31.68 / 0.844 |
| **Ours** | **Self** | **37.70 / 0.931** | **37.00 / 0.944** | **32.32 / 0.861** | **32.94 / 0.872** |

### Real Noise — PSNR (dB) / SSIM on SIDD

| Method | SIDD Benchmark | SIDD Validation |
|---|---|---|
| APBSN | 34.79 / 0.913 | 34.83 / 0.916 |
| LGBPN | 35.91 / 0.922 | 36.02 / 0.930 |
| **Ours** | **35.12 / 0.931** | **36.07 / 0.932** |

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{saqib2026noise,
  title     = {Noise-Aware Multi-inference Self-regulation for Self-supervised Denoising},
  author    = {Saqib, Nazmus and Yoon, Yeochan and Jo, Jaechoon and Gil, Joon-Min},
  booktitle = {Proceedings of the Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)},
  year      = {2026}
}
```

---

## Acknowledgements

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2025-16072365). This research was also carried out with the support of the Jeju RISE Center, funded by the Ministry of Education and Jeju Special Self-Governing Province in 2025, as part of the "Regional Innovation System & Education (RISE): Global University 30" initiative.
