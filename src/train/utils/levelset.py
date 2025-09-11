# src/utils/levelset.py

from scipy.interpolate import interp1d
import numpy as np

def build_4d_field(float_masks: list[np.ndarray]) -> np.ndarray:
    """
    Stack your four float‐probability maps into one 4D array.
    
    Parameters
    ----------
    float_masks : list of 3D ndarrays, length=4
        [P1(x,y,z), P2(x,y,z), P3(x,y,z), P4(x,y,z)]
    
    Returns
    -------
    vol4d : ndarray, shape (X, Y, Z, 4)
    """
    return np.stack(float_masks, axis=-1)

def make_temporal_interpolator(
        vol4d: np.ndarray,
        times: tuple[float,...] = (1.,2.,3.,4.),
        kind: str = "linear"
    ) -> callable:
    """
    Build a function P_cont(t) that returns an interpolated 3D volume
    for any t between times[0] and times[-1].
    
    Parameters
    ----------
    vol4d : ndarray, shape (X,Y,Z,N)
       N volumes at times[].
    times : sequence of floats, length=N
       The acquisition times of each volume.
    kind : str
       Passed to scipy.interpolate.interp1d (e.g. "linear", "cubic").
    
    Returns
    -------
    P_cont : function t -> 3D ndarray (X,Y,Z)
    """
    # returns array shape (X,Y,Z) for any t in [times[0],…,times[-1]]
    interp = interp1d(times, vol4d, axis=-1, kind=kind, assume_sorted=True)
    return lambda t: interp(t)

def extract_mask(
        P_cont: callable,
        t: float
    ) -> np.ndarray:
    """
    Interpolate at time t and save the full probability volume
    """
    return P_cont(t).astype(np.float32)  # shape (X,Y,Z)
