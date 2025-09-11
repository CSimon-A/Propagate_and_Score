# propagate_and_score/src/utils/array_ops.py

import numpy as np

def ensure_same_shape(*arrays):
    shapes = [a.shape for a in arrays]
    if len(set(shapes)) != 1:
        raise ValueError(f"Input shapes differ: {shapes}. Please resample/crop upstream to identical shapes.")

def looks_binary(vol: np.ndarray, tol: float = 1e-6, min_frac: float = 0.995) -> bool:
    v = vol.ravel()
    near0 = np.abs(v - 0.0) <= tol
    near1 = np.abs(v - 1.0) <= tol
    return (near0 | near1).mean() >= min_frac