# Copyright 2026 ddpwm
# Portions derived from the DDP-WM / dino_wm reference implementation; see third_party/README.md.
# SPDX-License-Identifier: MIT
import numpy as np
import torch
import torch.nn as nn


def create_objective_fn(alpha, base, mode="last"):
    """
    Loss calculated on the last pred frame.
    Args:
        alpha: int
        base: int. only used for objective_fn_all
    Returns:
        loss: tensor (B, )
    """
    metric = nn.MSELoss(reduction="none")

    def objective_fn_last(z_obs_pred, z_obs_tgt):
        """
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        loss_visual = metric(z_obs_pred["visual"][:, -1:], z_obs_tgt["visual"]).mean(
            dim=tuple(range(1, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(1, z_obs_pred["proprio"].ndim))
        )
        loss = loss_visual + alpha * loss_proprio
        return loss

    def objective_fn_all(z_obs_pred, z_obs_tgt):
        """
        Loss calculated on all pred frames.
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        coeffs = np.array([base**i for i in range(z_obs_pred["visual"].shape[1])], dtype=np.float32)
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)
        loss_visual = metric(z_obs_pred["visual"], z_obs_tgt["visual"]).mean(
            dim=tuple(range(2, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(2, z_obs_pred["proprio"].ndim))
        )
        loss_visual = (loss_visual * coeffs).mean(dim=1)
        loss_proprio = (loss_proprio * coeffs).mean(dim=1)
        loss = loss_visual + alpha * loss_proprio
        return loss

    if mode == "last":
        return objective_fn_last
    elif mode == "all":
        return objective_fn_all
    else:
        raise NotImplementedError


# =====================================================================
# Sparse MPC Cost Mask (DDP-WM Paper Equation 4)
# =====================================================================


def dilate_mask(mask, connectivity=4):
    """Dilate a binary mask on 14x14 grid.
    Args:
        mask: (B, 196) float, 0 or 1
        connectivity: 4=cross, 8=square, other=no dilation
    """
    import torch.nn.functional as F

    if connectivity not in [4, 8]:
        return mask
    b, n = mask.shape
    mask_2d = mask.view(b, 1, 14, 14)
    if connectivity == 4:
        kernel = torch.tensor(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=mask.dtype, device=mask.device
        ).view(1, 1, 3, 3)
    else:
        kernel = torch.ones(1, 1, 3, 3, dtype=mask.dtype, device=mask.device)
    dilated = F.conv2d(mask_2d, kernel, padding=1)
    return (dilated > 0).float().view(b, n)


def compute_patch_mask(obs_visual_0, obs_visual_g, threshold=0.1, dilation=4):
    """Compute binary task mask from pixel difference between observations.
    Args:
        obs_visual_0: (B, 1, C, H, W) or (B, C, H, W) current observation
        obs_visual_g: (B, 1, C, H, W) or (B, C, H, W) goal observation
        threshold: pixel difference threshold for patch activation
        dilation: mask dilation connectivity (4=cross, 8=square, 0=none)
    Returns:
        patch_mask: (B, 196) float tensor with 0/1 values
    """
    import torch.nn.functional as F

    v0 = obs_visual_0[:, 0] if obs_visual_0.dim() == 5 else obs_visual_0
    vg = obs_visual_g[:, 0] if obs_visual_g.dim() == 5 else obs_visual_g
    diff = (v0 - vg).abs().mean(dim=1)  # (B, H, W)
    pooled = F.adaptive_avg_pool2d(diff.unsqueeze(1), (14, 14))  # (B, 1, 14, 14)
    patch_active = (pooled >= threshold).float().view(diff.shape[0], -1)  # (B, 196)
    if dilation > 0:
        patch_active = dilate_mask(patch_active, connectivity=dilation)
    return patch_active


def masked_visual_loss(z_obs_pred, z_obs_tgt, patch_mask):
    """MSE on the last predicted frame, averaged over the active patches only (Eq. 4).

    Args:
        z_obs_pred: dict with 'visual' of shape (B, T, P, D); only the last frame is scored
        z_obs_tgt: dict with 'visual' of shape (B, 1, P, D)
        patch_mask: (B, P) float tensor in {0, 1}; the patches the cost is averaged over
    Returns:
        loss: (B,)
    """
    metric = nn.MSELoss(reduction="none")
    per_elem = metric(z_obs_pred["visual"][:, -1:], z_obs_tgt["visual"])  # (B, 1, P, D)
    per_patch = per_elem.mean(dim=-1)  # (B, 1, P)
    mask = patch_mask.to(device=per_patch.device, dtype=per_patch.dtype).unsqueeze(1)  # (B, 1, P)
    masked_sum = (per_patch * mask).sum(dim=-1)  # (B, 1)
    denominator = mask.sum(dim=-1).clamp(min=1.0)  # (B, 1)
    return (masked_sum / denominator).mean(dim=1)  # (B,)
