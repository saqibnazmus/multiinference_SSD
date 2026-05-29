import os
import argparse
import numpy as np
import torch
import glob
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.DnCNN import DnCNN
import torch.fft
from datetime import datetime
from dataloader import DataLoader_ImageNet_train, DataLoader_ImageNet_val
from tqdm import tqdm
from torchmetrics import PeakSignalNoiseRatio
from dataset import prepare_training_dataset

# ── NEW: Tweedie data augmentation (Stage 1 of the paper) ─────────────────────
from augmentation import tweedie_augment
# ──────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--batchSize",       type=int,   default=4,      help="Training batch size")
parser.add_argument("--num_of_layers",   type=int,   default=17,     help="Number of total layers")
parser.add_argument("--epochs",          type=int,   default=5,      help="Number of training epochs")
parser.add_argument("--lr",              type=float, default=1e-3,   help="Initial learning rate")
parser.add_argument("--milestone",       type=int,   default=30,     help="Epoch to decay LR (must be < epochs)")
parser.add_argument("--w_decay",         type=float, default=1e-8)
parser.add_argument("--noiseL",          type=float, default=25,     help="Noise level (ignored in blind mode)")
parser.add_argument("--val_noiseL",      type=float, default=25,     help="Noise level for validation set")
parser.add_argument("--dataset_train_dir", type=str, default="./ImageNet/train")
parser.add_argument("--dataset_val_dir",   type=str, default="./ImageNet/validation")
parser.add_argument("--root_dir",          type=str, default="./data")
parser.add_argument("--outf",              type=str, default="./weights", help="Path for saved weights")
# ── NEW: Tweedie augmentation hyper-parameters ────────────────────────────────
parser.add_argument("--gamma",        type=float, default=1.5,  help="Tweedie power index γ (paper: 1.5)")
parser.add_argument("--patch_size",   type=int,   default=8,    help="Patch size for local variance estimation")
parser.add_argument("--mask_radius",  type=float, default=0.5,  help="Low-pass Fourier mask radius")
# ──────────────────────────────────────────────────────────────────────────────
opt = parser.parse_args()


def charbonnier_loss(diff, eps=1e-6):
    """
    Charbonnier penalty  ρ(x) = sqrt(x² + ε²).
    Used in L_rec to limit the influence of mis-modelled pixels.
    """
    return torch.mean(torch.sqrt(diff ** 2 + eps))


if __name__ == "__main__":

    print("Preparing training dataset..........\n")
    prepare_training_dataset(opt.dataset_train_dir, opt.dataset_val_dir, opt.root_dir)

    print("Loading dataset......\n")
    MODEL_PATH = opt.outf
    os.makedirs(MODEL_PATH, exist_ok=True)

    dataset_train = DataLoader_ImageNet_train(opt.root_dir)
    TrainingLoader = DataLoader(
        dataset=dataset_train,
        num_workers=8,
        batch_size=opt.batchSize,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    dataset_val = DataLoader_ImageNet_val(opt.root_dir)
    valLoader = DataLoader(
        dataset=dataset_val,
        num_workers=8,
        batch_size=opt.batchSize,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    psnr_metric = PeakSignalNoiseRatio().cuda()

    model = DnCNN(channels=3, num_of_layers=opt.num_of_layers)
    model = model.cuda()

    # MSE criterion used for L1, L_smooth, L_id  (sum reduction, then normalised manually)
    criterion = nn.MSELoss(reduction='sum')
    criterion.cuda()

    num_epoch = opt.epochs
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.w_decay)

    print("Batchsize={}, number of epoch={}".format(opt.batchSize, opt.epochs))
    now = datetime.now()
    print("Start training.....", now.strftime("%H:%M:%S"))

    for epoch in tqdm(range(num_epoch), total=num_epoch):

        # Learning-rate schedule
        current_lr = opt.lr if epoch < opt.milestone else opt.lr / 10
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        epoch_loss   = 0.0
        psnr_train   = 0.0

        for inp, tar in TrainingLoader:
            # inp  — noisy patch from dataloader  (B, C, H, W)  in [0, 1]
            # tar  — unused here; augmentation produces its own targets
            y = torch.FloatTensor(inp).cuda()

            # ── Stage 1: Tweedie data augmentation (Section 3.1) ──────────────
            #   y1        — first  corrupted observation  (Eq. 1)
            #   y2        — second corrupted observation  (Eq. 2)
            #   y2_smooth — F⁻¹(W ⊙ F(y2))               (target for L_smooth)
            y1, y2, y2_smooth = tweedie_augment(
                y,
                patch_size=opt.patch_size,
                gamma=opt.gamma,
                mask_radius=opt.mask_radius,
            )
            # ──────────────────────────────────────────────────────────────────

            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            # ── Stage 2: RMIS — four-stage unrolled update (Section 3.2) ──────
            #
            #   P1 = f_θ(y1)          initial prediction from y1
            #   P2 = f_θ(P1)          smoother variant of P1
            #   P3 = f_θ(P2)          third stage
            #   P4 = f_θ(P3)          final prediction
            #
            P1 = model(y1)
            P2 = model(P1)
            P3 = model(P2)
            P4 = model(P3)

            N = y1.numel()   # normalisation factor

            # L1 — base MSE loss (maps P1 toward y2, the alternate observation)
            loss_1 = criterion(P1, y2) / N

            # L_smooth — piecewise smoothness prior (Eq. 8)
            #   Forces P2 (smoother variant) toward the low-pass target y2_smooth,
            #   reducing high-frequency noise without over-fitting.
            loss_smooth = criterion(P2, y2_smooth) / N

            # L_id — identity / Fejér monotonicity prior (Eq. 8)
            #   Penalises the distance between consecutive intermediate predictions
            #   P3 and P4, enforcing convergence without oscillation.
            loss_id = criterion(P3, P4) / N

            # L_rec — reconstruction fidelity prior (Eq. 8)
            #   Mixed Poisson-Gamma data fidelity with Charbonnier penalty and
            #   whitening W = Σ_y2^{-1/2} (per-channel std of y2).
            #   Prevents over-smoothing by pulling the estimate toward measured evidence.
            sigma_y2 = y2.std(dim=[-2, -1], keepdim=True).clamp(min=1e-8)
            W_mat    = 1.0 / sigma_y2                          # (B, C, 1, 1)
            diff     = W_mat * (P4 - y2)                       # whitened residual
            loss_rec = charbonnier_loss(diff)

            # Total loss — Equation 8
            loss = loss_1 + loss_smooth + loss_id + loss_rec
            # ──────────────────────────────────────────────────────────────────

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            psnr_train += psnr_metric(P4, y)   # PSNR vs. original noisy input

        psnr_avg_train = psnr_train / len(TrainingLoader)
        print(
            f"[Epoch {epoch + 1}/{num_epoch}]  "
            f"loss: {epoch_loss / len(TrainingLoader):.6f}  "
            f"PSNR_train: {psnr_avg_train:.4f} dB"
        )

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        psnr_val = 0.0
        for i, (image, clean) in enumerate(valLoader):
            val_input = torch.FloatTensor(image).cuda()
            val_clean = torch.FloatTensor(clean).cuda()
            with torch.no_grad():
                out_val = model(val_input)
            psnr_val += psnr_metric(out_val, val_clean)
        psnr_avg = psnr_val / len(valLoader)
        print("[Epoch %d] PSNR_val: %.4f dB" % (epoch + 1, psnr_avg))
        # ─────────────────────────────────────────────────────────────────────

        torch.save(model.state_dict(), os.path.join(MODEL_PATH, "model.pth"))

    now = datetime.now()
    print("Total training time.....", now.strftime("%H:%M:%S"))
