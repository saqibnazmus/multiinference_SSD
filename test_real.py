import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob
from tqdm import tqdm
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from models.DnCNN import DnCNN
from utils import writefile, create_folder

# ─────────────────────────────────────────────────────────────────────────────
# Datasets — SIDD Validation and SIDD Benchmark
#
# SIDD Validation  : noisy + GT pairs  → reports PSNR and SSIM
# SIDD Benchmark   : noisy only        → saves denoised outputs for submission
#
# Expected folder layout:
#
#   <sidd_val_dir>/
#       noisy/   *.png
#       gt/      *.png
#
#   <sidd_bench_dir>/
#       noisy/   *.png   (no GT available — official benchmark)
#
# ─────────────────────────────────────────────────────────────────────────────

class SIDD_Validation(Dataset):
    """SIDD Validation — noisy + GT pairs for quantitative evaluation."""

    def __init__(self, data_dir):
        super().__init__()
        self.noisy_paths = sorted(
            glob.glob(os.path.join(data_dir, "noisy", "*"))
        )
        self.gt_paths = sorted(
            glob.glob(os.path.join(data_dir, "gt", "*"))
        )
        assert len(self.noisy_paths) == len(self.gt_paths), (
            "Mismatch between noisy and gt image counts in SIDD Validation."
        )
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        noisy = self.transform(Image.open(self.noisy_paths[index]).convert("RGB"))
        gt    = self.transform(Image.open(self.gt_paths[index]).convert("RGB"))
        fname = os.path.basename(self.noisy_paths[index])
        return noisy, gt, fname

    def __len__(self):
        return len(self.noisy_paths)


class SIDD_Benchmark(Dataset):
    """SIDD Benchmark — noisy only, for saving denoised outputs."""

    def __init__(self, data_dir):
        super().__init__()
        self.noisy_paths = sorted(
            glob.glob(os.path.join(data_dir, "noisy", "*"))
        )
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        noisy = self.transform(Image.open(self.noisy_paths[index]).convert("RGB"))
        fname = os.path.basename(self.noisy_paths[index])
        return noisy, fname

    def __len__(self):
        return len(self.noisy_paths)


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--num_of_layers",  type=int, default=17,
                    help="Number of DnCNN layers")
parser.add_argument("--logdir",         type=str, default="./weights_real",
                    help="Directory containing model_real.pth")
parser.add_argument("--sidd_val_dir",   type=str, default="./SIDD/val",
                    help="Root of SIDD Validation (has noisy/ and gt/ sub-dirs)")
parser.add_argument("--sidd_bench_dir", type=str, default="./SIDD/benchmark",
                    help="Root of SIDD Benchmark (has noisy/ sub-dir, no GT)")
parser.add_argument("--save_dir",       type=str, default="./results_real",
                    help="Directory to save denoised benchmark outputs")
parser.add_argument("--batch_size",     type=int, default=4,
                    help="Batch size for inference")
parser.add_argument("--save_images",    action="store_true",
                    help="Save denoised images from SIDD Validation as well")
opt = parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helper — run the full 4-stage RMIS forward pass
# ─────────────────────────────────────────────────────────────────────────────

def rmis_inference(model, x):
    """
    Four-stage RMIS inference (Section 3.2).
    P1 → P2 → P3 → P4 with the same shared denoiser.
    Returns the final prediction P4.
    """
    with torch.no_grad():
        P1 = model(x)
        P2 = model(P1)
        P3 = model(P2)
        P4 = model(P3)
    return P4


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model.......\n")
    model = DnCNN(channels=3, num_of_layers=opt.num_of_layers)
    model.load_state_dict(
        torch.load(os.path.join(opt.logdir, "model_real.pth"), map_location="cpu")
    )
    model = model.cuda()
    model.eval()

    psnr_metric = PeakSignalNoiseRatio().cuda()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()

    # ── 1. SIDD Validation ────────────────────────────────────────────────────
    print("=" * 60)
    print("Evaluating on SIDD Validation...")
    print("=" * 60)

    val_save_dir = create_folder(opt.save_dir, "sidd_val") if opt.save_images else None

    val_dataset = SIDD_Validation(opt.sidd_val_dir)
    val_loader  = DataLoader(
        dataset=val_dataset,
        num_workers=4,
        batch_size=opt.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    psnr_val_total = 0.0
    ssim_val_total = 0.0

    for noisy, gt, fnames in tqdm(val_loader, desc="SIDD Validation"):
        noisy = noisy.cuda()
        gt    = gt.cuda()

        pred = rmis_inference(model, noisy)

        psnr_val_total += psnr_metric(pred, gt).item()
        ssim_val_total += ssim_metric(pred, gt).item()

        if opt.save_images:
            # Save each image in the batch
            pred_np = pred.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            for i, fname in enumerate(fnames):
                img_uint8 = (pred_np[i] * 255).astype(np.uint8)
                writefile(val_save_dir, fname, img_uint8[:, :, ::-1])  # RGB→BGR for cv2

    n_val = len(val_loader)
    print(
        f"\nSIDD Validation  —  "
        f"PSNR: {psnr_val_total / n_val:.4f} dB  |  "
        f"SSIM: {ssim_val_total / n_val:.4f}"
    )

    # ── 2. SIDD Benchmark ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Running on SIDD Benchmark (no GT — saving outputs)...")
    print("=" * 60)

    bench_save_dir = create_folder(opt.save_dir, "sidd_benchmark")

    bench_dataset = SIDD_Benchmark(opt.sidd_bench_dir)
    bench_loader  = DataLoader(
        dataset=bench_dataset,
        num_workers=4,
        batch_size=opt.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    for noisy, fnames in tqdm(bench_loader, desc="SIDD Benchmark"):
        noisy = noisy.cuda()
        pred  = rmis_inference(model, noisy)

        # Save denoised outputs for official submission
        pred_np = pred.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        for i, fname in enumerate(fnames):
            img_uint8 = (pred_np[i] * 255).astype(np.uint8)
            writefile(bench_save_dir, fname, img_uint8[:, :, ::-1])  # RGB→BGR for cv2

    print(f"\nBenchmark outputs saved to: {bench_save_dir}")
    print("Upload the contents of that folder to the SIDD online benchmark server.")
    print("\nDone.")
