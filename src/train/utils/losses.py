# src/utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
import numpy as np
from scipy.ndimage import distance_transform_edt

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        prob_flat = prob.view(-1)
        target_flat = target.view(-1).float()
        intersection = (prob_flat * target_flat).sum()
        return 1 - ((2 * intersection + self.smooth) /
                    (prob_flat.sum() + target_flat.sum() + self.smooth))

class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) loss on the tumor ROI.
    Computes 1 - SSIM between prediction and target masks (in [0,1]).
    """
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, logits, target):
        # convert logits to probabilities in [0,1]
        prob = torch.sigmoid(logits)
        # compute SSIM; pytorch_msssim.ssim expects inputs in [0,1]
        l_ssim = ssim(
            prob, target.float(),
            data_range=1.0,
            size_average=self.size_average,
            win_size=self.window_size
        )
        return 1 - l_ssim

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, pos_weight=None, ssim_weight=0.0):
        """
        Combined loss: BCE + Dice + optional SSIM.

        alpha: weight for BCE term
        (1-alpha): weight for Dice term
        ssim_weight: additional multiplier for SSIM loss
        """
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.ssim_loss = SSIMLoss()
        self.ssim_weight = ssim_weight

    def forward(self, logits, target, return_components=True):
        l_bce  = self.bce(logits, target.float())
        l_dice = self.dice(logits, target)
        loss   = self.alpha * l_bce + (1 - self.alpha) * l_dice

        l_ssim = self.ssim_loss(logits, target)

        if self.ssim_weight > 0:
            loss = loss + self.ssim_weight * l_ssim

        if return_components:
            return loss, {"bce": l_bce, "dice": l_dice, "ssim": l_ssim}
        else:
            return loss

class BoundaryLoss(nn.Module):
    """
    L_boundary = mean( p * D_gt )
    where D_gt is the Euclidean distance transform of the GT mask's background.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, target):
        # logits: [B,1,D,H,W], target: [B,1,D,H,W] binary
        prob = torch.sigmoid(logits)

        # compute D_gt on CPU via scipy, then move to GPU
        dt_maps = []
        for b in range(target.shape[0]):
            gt = target[b,0].cpu().numpy().astype(np.uint8)
            # dist from each foreground voxel *to the nearest background*
            dt = distance_transform_edt(1 - gt)
            dt_maps.append(dt)
        D_gt = torch.from_numpy(np.stack(dt_maps)) \
                   .unsqueeze(1) \
                   .to(logits.device).float()  # [B,1,D,H,W]

        return torch.mean(prob * D_gt)

class HausdorffLoss(nn.Module):
    """
    Approximate symmetric Hausdorff:
      L_hd = max( mean( p * D_gt ),  mean( y * D_pred ) )
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)

        # 1) D_gt as above
        dt_maps = []
        for b in range(target.shape[0]):
            gt = target[b,0].cpu().numpy().astype(np.uint8)
            dt = distance_transform_edt(1 - gt)
            dt_maps.append(dt)
        D_gt = torch.from_numpy(np.stack(dt_maps)).unsqueeze(1).to(logits.device).float()

        # 2) D_pred: distance transform of the *predicted* binary mask
        #    threshold at 0.5 for surface extraction
        dt_maps_pred = []
        bin_pred = (prob.detach()>0.5).cpu().numpy().astype(np.uint8)
        for b in range(bin_pred.shape[0]):
            dt_maps_pred.append(distance_transform_edt(1 - bin_pred[b,0]))
        D_pred = torch.from_numpy(np.stack(dt_maps_pred)).unsqueeze(1).to(logits.device).float()

        # symmetric: 
        l1 = torch.mean(prob * D_gt)
        l2 = torch.mean(target * D_pred)
        return torch.max(l1, l2)

class FocalBCE(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, target):
        # use the logit‑safe variant under AMP
        # 1) compute per‑voxel BCE from logits directly
        bce = F.binary_cross_entropy_with_logits(
            logits,
            target.float(),
            reduction='none'
        )
        # 2) still need probabilities for focal weight
        prob = torch.sigmoid(logits)
        p_t = prob * target + (1 - prob) * (1 - target)
        weight = self.alpha * (1 - p_t) ** self.gamma
        return torch.mean(weight * bce)

class FocalBoundaryLoss(nn.Module):
    """
    1) Extract boundary from GT via 3×3×3 morphological gradient
    2) Apply focal‐BCE on predicted probabilities vs that boundary map
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.focal_bce = FocalBCE(gamma, alpha)
        # simple 3×3×3 structuring element
        self.kernel = torch.ones((1,1,3,3,3), dtype=torch.float32)

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        device = logits.device

        # compute GT boundary: dilate minus erode
        tgt = target.float()
        dil = F.conv3d(tgt, self.kernel.to(device), padding=1) > 0
        erd = F.conv3d(tgt, self.kernel.to(device), padding=1) == self.kernel.numel()
        boundary = (dil ^ erd).float()

        return self.focal_bce(logits, boundary)


def compute_l1_loss(pred_time, true_time, bg_weight):
    """
    Custom loss for tumor propagation:
      - Foreground voxels (true_time >= 0): L1 loss between prediction and ground truth.
      - Background voxels (true_time < 0): penalty if prediction > -1.

    Args:
        pred_time (Tensor): Model predictions, shape (B, ...).
        true_time (Tensor): Ground truth times, shape (B, ...).
        bg_weight (float): Weight for background penalty (default: 0.1).

    Returns:
        Tensor: scalar loss value.
    """
    # Identify foreground and background masks
    valid = (true_time >= 0)   # foreground
    neg   = ~valid             # background

    # Clamp predictions for loss (ignore values < -1, treat as -1)
    pred_for_loss = torch.maximum(pred_time, torch.tensor(-1.0, device=pred_time.device))

    # --- Foreground L1 ---
    l1_map = (pred_for_loss - true_time).abs()
    valid_count = valid.sum()
    l1 = (l1_map[valid].sum() / valid_count.clamp(min=1))
    l1 = torch.where(valid_count > 0, l1, torch.zeros_like(l1))

    # --- Background penalty ---
    if neg.any() and bg_weight > 0:
        # Penalize background predictions > -1
        neg_l1 = torch.relu(pred_for_loss[neg] + 1).mean()
    else:
        neg_l1 = torch.zeros_like(l1)

    # --- Final loss ---
    return l1 + bg_weight * neg_l1
