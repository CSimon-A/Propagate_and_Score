# propagate_and_score/src/utils/metric.py

import numpy as np
from scipy.ndimage import distance_transform_edt as edt

def growth_aware_score(
        R,
        T1,
        G,
        sigma_dist=10.0,
        sigma_vol=0.5,
        alpha=0.5,
        eps=1e-8
        ):
    """
    Parameters
    ----------
    R           : ndarray of bool
                  Binary resection mask (0 = not resected, 1 = resected).

    T1          : ndarray of bool
                  Binary ground-truth tumour mask at surgery time t1
                  (0 = non-tumour, 1 = tumour).
                  
    G           : ndarray of float in {0.0, 0.25, 0.5, 0.75, 1.0}
                  Discrete growth-probability map:
                  • 1.0 -> tumour already present at T1
                  • 0.75 -> appears by T2
                  • 0.50 -> appears by T3
                  • 0.25 -> appears by T4
                  • 0.00 -> no expected growth

    sigma_dist  : float
                  Controls how quickly the distance penalty decays (in the same
                  units as `dist`, e.g., mm or voxels).  
                  At dist = σ: ~63% of full penalty; at 2σ: ~86%; at 3σ: ~95%; 
                    at 0.5σ: ~40%.

    sigma_vol   : float
                  Controls how quickly the *relative over-resection volume*
                  penalty ramps up (unitless fraction of tumour volume).  
                  At over-resection = σ (i.e. σ × 100% of tumour volume): ~63%  
                  of full penalty; at 2σ: ~86%; at 3σ: ~95%; at 0.5σ: ~40%.

    alpha       : float in (0,1)
                  Trade-off between completeness (α) and tissue-sparing (1-α).

    eps         : float
                  Numerical stabiliser to avoid divide-by-zero.

    Returns
    -------
    C     : float
            Completeness / recall term          (higher = better).
    Pg    : float
            Growth-aware over-resection penalty (lower  = better).
    S     : float
            Composite score in [0,1]            (higher = better).
    wd    : ndarray
            Distance-based tissue weights.
    wg    : ndarray
            Distance-and-growth weighted penalty mask.
    over  : ndarray
            Voxels of over-resection.
    """

    # --- distance-based tissue weight ----------------------------------------
    # Compute Euclidean distance transform from non-tumour to nearest tumour voxel:
    dist = edt(1 - T1)
    # Convert distance into a smoothly increasing penalty weight (0 at tumour boundary, 1 far away):
    # TODO: sigma_dist should maybe be a parameter based on the tumour size/average diameter
    wd   = 1 - np.exp(-dist / sigma_dist)
    # Modulate by growth probability (1 -> T1, 0.75 -> T2, 0.5 -> T3, 0.25 -> T4): zero penalty where a voxel is sure to become tumour
    wg   = (1 - G) * wd

    # --- completeness (recall) "EOR" ------------------------------------------------
    # True positives: sum of resected voxels that are actually tumour
    tp = (R * T1).sum()
    # Normalize by total tumour volume for recall
    C = tp / (T1.sum() + eps)
    

    # --- growth-aware over-resection penalty -----------------------
    # Identify over-resection voxels (resection outside tumour mask)
    over = np.clip(R - T1, 0, None).astype(np.uint8)
    # Sum of distance-based weights over these over-resection voxels (+eps to avoid zero)
    over_wd_sum = (over * wd).sum() + eps
    # Average “wrongness”: how much of the over-resection is in low-growth / far-from-tumour areas
    avg_wrong = (over * wg).sum() / over_wd_sum           # ∈ [0,1]
    # Volume factor: penalize more if absolute over-resection volume is large
    # TODO: sigma_vol should maybe be a parameter based on the tumour size/average diameter
    over_sum = float(over.sum())
    tumor_sum = float(T1.sum()) 
    x = over_sum / (sigma_vol * (tumor_sum + eps))
    vol_factor = -np.expm1(-x)

    # Final growth-aware penalty: more if wrongness and volume are high
    Pg = avg_wrong * vol_factor

    # --- composite score -------------------------------------------------------
    # Combine recall (C) and the complement of penalty (1 - Pg)
    S = alpha * C + (1 - alpha) * (1 - Pg)

    # Return: completeness, penalty, composite score, plus some intermediate masks/weights
    return float(C), float(Pg), float(S), wd, wg, over

def normalize_growth(G, mode="linear", max_value=5.0, tau=2.0, r=None):
    """
    Maps the {-1, 0, 1, 2, ...} style G into [0,1] "growth-likelihood".
    -1 -> 0; 0 -> 1; larger k -> smaller value (monotone).
    mode:
      - "linear":  out = 1 - k/max_value (k>=0), clamped [0,1]
      - "exp":     out = exp(-k/tau)
      - "geom":    out = r**k  (default r=2/3)
    """
    G = np.asarray(G, dtype=float)
    out = np.zeros_like(G, dtype=float)

    pos = (G >= 0)

    if mode == "linear":
        M = float(max_value)
        out[pos] = np.clip(1.0 - (G[pos] / M), 0.0, 1.0)

    elif mode == "exp":
        # No need for a max; spacing controlled by tau.
        out[pos] = np.exp(-G[pos] / float(tau))

    elif mode == "geom":
        # Geometric decay with ratio r in (0,1), default ~2/3.
        rr = (2.0/3.0) if r is None else float(r)
        out[pos] = np.power(rr, G[pos])

    # negatives (e.g., -1) → 0 exactly
    out[G < 0] = 0.0
    return out
