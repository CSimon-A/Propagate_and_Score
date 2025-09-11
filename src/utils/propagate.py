# propagate_and_score/src/utils/propagate.py

import numpy as np
import torch

@torch.inference_mode()
def t2e_predict_map(bravo: np.ndarray, flair: np.ndarray, mask_t0: np.ndarray,
                    model: torch.nn.Module, device='cpu', clamp_min: float = -1.0) -> np.ndarray:
    """
    Run the UNet once and return the raw per-voxel event-time map.
    Clamp only the lower bound: values < clamp_min are set to clamp_min (default -1).
    No sigmoid. No horizon masks.
    """
    x = np.stack([bravo, flair, mask_t0], axis=0)[None, ...]  # [1,3,D,H,W]
    xt = torch.from_numpy(x.astype(np.float32)).to(device)
    
    # Direct float output (no sigmoid)
    tmap = model(xt)[0, 0].detach().cpu().numpy()
    if clamp_min is not None:
        tmap = np.maximum(tmap, clamp_min)
    return tmap
