# propagate_and_score/src/utils/io.py

import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from typing import Tuple
import train.config as config

# ---------- I/O ----------
def load_nifti(path: str):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    return data, img.affine, img.header

def save_nifti(path: str, data: np.ndarray, affine, header=None):
    img = nib.Nifti1Image(data.astype(np.float32), affine, header=header)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(img, path)

# ---------- Preproc ----------
def zscore(vol: np.ndarray) -> np.ndarray:
    m, s = vol.mean(), vol.std()
    return (vol - m) / (s + 1e-8)

def robust_minmax(vol: np.ndarray, p1=1.0, p99=99.0) -> np.ndarray:
    lo, hi = np.percentile(vol, [p1, p99])
    if hi <= lo:
        return np.clip(vol, lo, hi)
    v = (vol - lo) / (hi - lo)
    return np.clip(v, 0.0, 1.0)

def apply_scale_mode(vol: np.ndarray, mode: str = "none") -> np.ndarray:
    if mode == "none" or mode is None:
        return vol
    if mode == "min-max":
        return robust_minmax(vol, 1, 99)
    if mode == "tanh":
        return (np.tanh(vol) * 0.5) + 0.5
    if mode == "sigmoid":
        return 1.0 / (1.0 + np.exp(-vol))
    raise ValueError(f"Unknown scale mode: {mode}")

def prep_modalities(bravo, flair, do_z, scale_mode):
    if do_z:
        bravo = zscore(bravo)
        flair = zscore(flair)
    bravo = apply_scale_mode(bravo, scale_mode)
    flair = apply_scale_mode(flair, scale_mode)
    return bravo.astype(np.float32), flair.astype(np.float32)

def binarize_mask(m: np.ndarray) -> np.ndarray:
    thr = getattr(config, "THRESHOLD", 0.01)
    return (m > thr).astype(np.uint8)

def binarize(vol: np.ndarray, thr: float) -> np.ndarray:
    return (vol > float(thr)).astype(np.uint8)

# ---------- Resize (no padding) ----------
def to_5d(arr: np.ndarray) -> torch.Tensor:
    # [D,H,W] -> [1,1,D,H,W]
    t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return t

def resize_to_shape(arr: np.ndarray, target_shape: Tuple[int,int,int], mode: str) -> np.ndarray:
    """mode in {'trilinear','nearest'}"""
    t = to_5d(arr)
    out = F.interpolate(t, size=target_shape, mode=mode, align_corners=False if mode=='trilinear' else None)
    return out[0,0].cpu().numpy()
