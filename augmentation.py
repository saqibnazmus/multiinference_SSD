import torch
import torch.fft
import numpy as np


def fourier_smooth(y, mask_radius=0.5):
    """
    Compute ŷ = F⁻¹(W ⊙ Fy) — low-pass Fourier smoothing.

    Args:
        y           : (B, C, H, W) float tensor, values in [0, 1]
        mask_radius : fraction of the frequency radius to keep (low-pass cutoff)

    Returns:
        y_hat : (B, C, H, W) smoothed tensor
    """
    F = torch.fft.rfft2(y, norm='ortho')
    H, W = y.shape[-2], y.shape[-1]

    # Build low-pass mask W in the rfft2 frequency grid
    freq_h = torch.fft.fftfreq(H, device=y.device).reshape(-1, 1)      # (H, 1)
    freq_w = torch.fft.rfftfreq(W, device=y.device).reshape(1, -1)     # (1, W//2+1)
    radius = (freq_h ** 2 + freq_w ** 2).sqrt()                        # (H, W//2+1)
    W_mask = (radius <= mask_radius).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W//2+1)

    y_hat = torch.fft.irfft2(F * W_mask, s=(H, W), norm='ortho')
    return y_hat


def estimate_tweedie_params(y, y_hat, patch_size=8, gamma=1.5):
    """
    Estimate per-patch local mean µ and global dispersion φ from residuals.

    Residuals r_i = y_i - ŷ_i capture the noise remaining after Fourier
    smoothing.  For each non-overlapping spatial patch P:
        var_p  = Var(r_i ∈ P)
        mu_p   = mean(ŷ_i ∈ P)
        phi_p  = var_p / mu_p^gamma          (Tweedie variance relation)

    Global phi is the average of all phi_p values.

    Args:
        y          : (B, C, H, W) original noisy tensor
        y_hat      : (B, C, H, W) Fourier-smoothed pseudo-clean estimate
        patch_size : spatial size of non-overlapping patches
        gamma      : Tweedie power index (fixed at 1.5 per the paper)

    Returns:
        mu_map : (B, C, H, W) local mean map (µ)
        phi    : scalar tensor — global dispersion estimate
    """
    residuals = y - y_hat                          # pixel-wise noise residual
    B, C, H, W = residuals.shape

    phi_list = []
    mu_map   = torch.zeros_like(y_hat)

    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            r_patch = residuals[:, :, i:i + patch_size, j:j + patch_size]
            y_patch = y_hat[:, :, i:i + patch_size, j:j + patch_size]

            # Empirical variance of residuals within the patch
            var_p = r_patch.var(dim=[-2, -1], keepdim=True).clamp(min=1e-8)
            # Mean intensity of denoised image in the same patch
            mu_p  = y_patch.mean(dim=[-2, -1], keepdim=True).clamp(min=1e-8)

            mu_map[:, :, i:i + patch_size, j:j + patch_size] = mu_p
            phi_p = var_p / (mu_p ** gamma)
            phi_list.append(phi_p.mean())

    phi = torch.stack(phi_list).mean()
    return mu_map, phi


def tweedie_alpha(y_hat, mu, phi, gamma=1.5):
    """
    Saddle-point approximation of the Tweedie density (Equations 5 & 6).

    alpha = (2π φ ŷ)^{-1/2} · exp( -d(ŷ, µ) / (2φ) )

    where the unit deviance d(ŷ, µ) / (2φ) is given by Equation 6 with the
    Poisson exponential dispersion model (valid for γ ∈ (1, 2), here γ=1.5).

    Args:
        y_hat : (B, C, H, W) pseudo-clean estimate
        mu    : (B, C, H, W) local mean map
        phi   : scalar — global dispersion
        gamma : Tweedie power index (1.5)

    Returns:
        alpha : (B, C, H, W) perturbation random variable
    """
    eps   = 1e-8
    y_hat = y_hat.clamp(min=eps)
    mu    = mu.clamp(min=eps)

    # Unit deviance — Equation 6
    term1 = (y_hat ** (2 - gamma)) / ((1 - gamma) * (2 - gamma))
    term2 = (y_hat * mu ** (1 - gamma)) / (1 - gamma)
    term3 = (mu ** (2 - gamma)) / (2 - gamma)
    neg_deviance = 2.0 * (term1 - term2 + term3)   # = -d(ŷ,µ) / (2φ) factor

    # Prefactor (2π φ ŷ)^{-1/2}
    prefactor = (2.0 * np.pi * phi * y_hat).clamp(min=eps) ** (-0.5)

    alpha = prefactor * torch.exp(-neg_deviance / (2.0 * phi))
    return alpha


def cholesky_sqrt(y_hat):
    """
    Approximate √Σ_ŷ as a diagonal matrix using the per-channel std of ŷ.

    The paper uses the full Cholesky of ŷ to capture spatially correlated
    information; here we use a diagonal approximation (channel-wise std)
    which is efficient and captures the dominant scale.

    Args:
        y_hat : (B, C, H, W)

    Returns:
        std : (B, C, 1, 1) broadcastable std tensor
    """
    std = y_hat.std(dim=[-2, -1], keepdim=True).clamp(min=1e-8)
    return std


def tweedie_augment(y, patch_size=8, gamma=1.5, mask_radius=0.5):
    """
    Full Tweedie data augmentation (Section 3.1, Equations 1–6).

    Given a noisy image y, produces two corrupted observations y1 and y2
    whose added perturbations are known in distribution (Corollary 1), plus
    the Fourier-smoothed version of y2 needed by L_smooth in RMIS.

    Pipeline:
        1. ŷ  = F⁻¹(W ⊙ Fy)                — Fourier smooth
        2. µ, φ estimated from residuals r = y - ŷ per patch
        3. α  = Tweedie(ŷ; µ, φ, γ)         — saddle-point approximation
        4. √Σ = diag-approx Cholesky of ŷ
        5. y1 = y + √Σ · D^T · α            (D = I, Eq. 1)
        6. y2 = y − √Σ · D⁻¹  · α           (D = I, Eq. 2)
        7. y2_smooth = F⁻¹(W ⊙ F(y2))       — used by L_smooth

    Args:
        y           : (B, C, H, W) float tensor in [0, 1], on device
        patch_size  : patch size for local variance estimation
        gamma       : Tweedie power index (paper fixes at 1.5)
        mask_radius : low-pass Fourier cutoff

    Returns:
        y1        : (B, C, H, W) first corrupted observation
        y2        : (B, C, H, W) second corrupted observation
        y2_smooth : (B, C, H, W) Fourier-smoothed y2 (target for L_smooth)
    """
    # Step 1: pseudo-clean estimate via Fourier smoothing
    y_hat = fourier_smooth(y, mask_radius)

    # Step 2: patch-wise µ and global φ
    mu, phi = estimate_tweedie_params(y, y_hat, patch_size, gamma)

    # Step 3: Tweedie alpha (saddle-point approx)
    alpha = tweedie_alpha(y_hat, mu, phi, gamma)

    # Step 4: approximate Cholesky √Σ
    sqrt_sigma = cholesky_sqrt(y_hat)

    # Steps 5 & 6: perturbations δ1, δ2  (D = identity)
    delta1 =  sqrt_sigma * alpha   #  √Σ · D^T · α
    delta2 = -sqrt_sigma * alpha   # -√Σ · D⁻¹ · α

    y1 = (y + delta1).clamp(0.0, 1.0)
    y2 = (y + delta2).clamp(0.0, 1.0)

    # Step 7: smooth y2 for L_smooth target
    y2_smooth = fourier_smooth(y2, mask_radius)

    return y1, y2, y2_smooth
