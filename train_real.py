import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob
from datetime import datetime
from tqdm import tqdm
from torchmetrics import PeakSignalNoiseRatio

from models.DnCNN import DnCNN
from augmentation import tweedie_augment

# ─────────────────────────────────────────────────────────────────────────────
# Dataset — SIDD Medium sRGB
# Expected folder layout (matching the cvf-sid training procedure, Section 5.2):
#
#   <sidd_dir>/
#       train/
#           noisy/   *.png   (sRGB noisy patches)
#           gt/      *.png   (sRGB ground-truth patches, used for val PSNR only)
#       val/
#           noisy/   *.png
#           gt/      *.png
#
# ─────────────────────────────────────────────────────────────────────────────

class SIDD_Train(Dataset):
    """
    Training split of SIDD Medium sRGB.
    Returns (noisy_patch,) only — ground truth is never seen during training.
    The Tweedie augmentation in the training loop produces y1/y2 on the fly.
    """
    def __init__(self, data_dir, patch_size=128):
        super().__init__()
        self.noisy_paths = sorted(
            glob.glob(os.path.join(data_dir, "train", "noisy", "*"))
        )
        self.transform = transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        noisy = Image.open(self.noisy_paths[index]).convert("RGB")
        return self.transform(noisy)

    def __len__(self):
        return len(self.noisy_paths)


class SIDD_Val(Dataset):
    """
    Validation split of SIDD Medium sRGB.
    Returns (noisy, gt) pairs for PSNR/SSIM monitoring.
    """
    def __init__(self, data_dir, patch_size=256):
        super().__init__()
        self.noisy_paths = sorted(
            glob.glob(os.path.join(data_dir, "val", "noisy", "*"))
        )
        self.gt_paths = sorted(
            glob.glob(os.path.join(data_dir, "val", "gt", "*"))
        )
        assert len(self.noisy_paths) == len(self.gt_paths), (
            "Mismatch between noisy and gt counts in val split."
        )
        self.transform = transforms.Compose([
            transforms.CenterCrop(patch_size),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        noisy = self.transform(Image.open(self.noisy_paths[index]).convert("RGB"))
        gt    = self.transform(Image.open(self.gt_paths[index]).convert("RGB"))
        return noisy, gt

    def __len__(self):
        return len(self.noisy_paths)


# ─────────────────────────────────────────────────────────────────────────────
# Charbonnier loss (used in L_rec)
# ─────────────────────────────────────────────────────────────────────────────

def charbonnier_loss(diff, eps=1e-6):
    return torch.mean(torch.sqrt(diff ** 2 + eps))


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--batchSize",      type=int,   default=4,           help="Training batch size")
parser.add_argument("--num_of_layers",  type=int,   default=17,          help="Number of DnCNN layers")
parser.add_argument("--epochs",         type=int,   default=100,         help="Number of training epochs")
parser.add_argument("--lr",             type=float, default=1e-3,        help="Initial learning rate")
parser.add_argument("--milestone",      type=int,   default=50,          help="Epoch to decay LR by 10x")
parser.add_argument("--w_decay",        type=float, default=1e-8,        help="Adam weight decay")
parser.add_argument("--patch_size",     type=int,   default=128,         help="Training patch size")
parser.add_argument("--sidd_dir",       type=str,   default="./SIDD",    help="Root dir of SIDD Medium sRGB")
parser.add_argument("--outf",           type=str,   default="./weights_real", help="Path to save weights")
# Tweedie augmentation
parser.add_argument("--gamma",          type=float, default=1.5,         help="Tweedie power index γ")
parser.add_argument("--aug_patch_size", type=int,   default=8,           help="Patch size for local variance estimation")
parser.add_argument("--mask_radius",    type=float, default=0.5,         help="Fourier low-pass mask radius")
opt = parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs(opt.outf, exist_ok=True)

    # ── Dataloaders ───────────────────────────────────────────────────────────
    print("Loading SIDD Medium dataset......\n")
    dataset_train = SIDD_Train(opt.sidd_dir, patch_size=opt.patch_size)
    TrainingLoader = DataLoader(
        dataset=dataset_train,
        num_workers=8,
        batch_size=opt.batchSize,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    dataset_val = SIDD_Val(opt.sidd_dir)
    ValLoader = DataLoader(
        dataset=dataset_val,
        num_workers=4,
        batch_size=opt.batchSize,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # ── Model, loss, optimiser ────────────────────────────────────────────────
    model = DnCNN(channels=3, num_of_layers=opt.num_of_layers).cuda()
    criterion = nn.MSELoss(reduction='sum').cuda()
    psnr_metric = PeakSignalNoiseRatio().cuda()

    optimizer = optim.Adam(
        model.parameters(), lr=opt.lr, weight_decay=opt.w_decay
    )

    print("Batchsize={}, epochs={}".format(opt.batchSize, opt.epochs))
    now = datetime.now()
    print("Start training.....", now.strftime("%H:%M:%S"))

    for epoch in tqdm(range(opt.epochs), total=opt.epochs):

        # Learning-rate schedule
        current_lr = opt.lr if epoch < opt.milestone else opt.lr / 10
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        epoch_loss  = 0.0
        psnr_train  = 0.0

        for noisy in TrainingLoader:
            # noisy : (B, C, H, W) sRGB patch in [0, 1], no GT used
            y = noisy.cuda()

            # ── Stage 1: Tweedie augmentation ─────────────────────────────────
            y1, y2, y2_smooth = tweedie_augment(
                y,
                patch_size=opt.aug_patch_size,
                gamma=opt.gamma,
                mask_radius=opt.mask_radius,
            )
            # ──────────────────────────────────────────────────────────────────

            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            # ── Stage 2: RMIS four-stage unrolled update ──────────────────────
            P1 = model(y1)
            P2 = model(P1)
            P3 = model(P2)
            P4 = model(P3)

            N = y1.numel()

            loss_1      = criterion(P1, y2) / N                        # L1
            loss_smooth = criterion(P2, y2_smooth) / N                 # L_smooth
            loss_id     = criterion(P3, P4) / N                        # L_id

            sigma_y2 = y2.std(dim=[-2, -1], keepdim=True).clamp(min=1e-8)
            diff     = (1.0 / sigma_y2) * (P4 - y2)
            loss_rec = charbonnier_loss(diff)                           # L_rec

            loss = loss_1 + loss_smooth + loss_id + loss_rec           # Eq. 8
            # ──────────────────────────────────────────────────────────────────

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            psnr_train += psnr_metric(P4, y)   # vs. original noisy (no GT)

        psnr_avg_train = psnr_train / len(TrainingLoader)
        print(
            f"[Epoch {epoch + 1}/{opt.epochs}]  "
            f"loss: {epoch_loss / len(TrainingLoader):.6f}  "
            f"PSNR_train: {psnr_avg_train:.4f} dB"
        )

        # ── Validation (GT available) ─────────────────────────────────────────
        model.eval()
        psnr_val = 0.0
        for noisy_val, gt_val in ValLoader:
            noisy_val = noisy_val.cuda()
            gt_val    = gt_val.cuda()
            with torch.no_grad():
                out_val = model(noisy_val)
            psnr_val += psnr_metric(out_val, gt_val)
        psnr_avg_val = psnr_val / len(ValLoader)
        print("[Epoch %d] PSNR_val: %.4f dB" % (epoch + 1, psnr_avg_val))
        # ─────────────────────────────────────────────────────────────────────

        torch.save(
            model.state_dict(),
            os.path.join(opt.outf, "model_real.pth")
        )

    now = datetime.now()
    print("Total training time.....", now.strftime("%H:%M:%S"))
