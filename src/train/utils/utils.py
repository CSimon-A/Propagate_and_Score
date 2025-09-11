# src/utils/utils.py

import numpy as np
import nibabel as nib

def load_nifti(
        path: str,
        dtype: type = np.float32
    ) -> tuple[np.ndarray, np.ndarray, object]:
    """
    Load a NIfTI file and return its raw data array, affine transform, and header.

    Parameters
    ----------
    path : str
        Filesystem path to the NIfTI (.nii or .nii.gz) file to load.
    dtype : data-type, optional
        NumPy data type for the returned array. Default is np.float32.

    Returns
    -------
    data : ndarray
        3D (or 4D) image data as a NumPy array of type `dtype`.
    affine : ndarray
        4x4 affine transformation matrix associated with the image.
    header : nibabel header object
        Header metadata from the NIfTI file.
    """

    img = nib.load(path)
    data = img.get_fdata(dtype=dtype)
    return data, img.affine, img.header

def threshold(
        data: np.ndarray,
        thr: float = 0.01,
        binary: bool = True
    ) -> np.ndarray:
    """
    Apply a threshold to an array, either binarizing or zeroing below threshold.

    Parameters
    ----------
    data : ndarray
        Input array containing continuous values.
    thr : float, optional
        Threshold value. Voxels with value >= thr are kept or set to 1.
        Default is 0.01.
    binary : bool, optional
        If True, return a binary mask (0 or 1). If False, zero out values
        below thr but keep original values above thr. Default is True.

    Returns
    -------
    mask_or_data : ndarray
        If `binary` is True, returns an array of dtype float32 with 1 where
        data >= thr, else 0. If `binary` is False, returns an array of dtype
        float32 with original values for data >= thr, else 0.
    """

    if binary:
        return (data >= thr).astype(np.float32)
    else:
        return np.where(data >= thr, data, 0.0).astype(np.float32)

def save_nifti(
        data: np.ndarray,
        affine: np.ndarray,
        header: object,
        out_path: str
    ) -> None:
    """
    Save a NumPy array as a NIfTI file with given affine and header.

    Parameters
    ----------
    data : ndarray
        Image data array to save.
    affine : ndarray
        4x4 affine transformation matrix.
    header : nibabel header object
        Header metadata to embed in the NIfTI file.
    out_path : str
        Filesystem path where the new NIfTI file will be written.

    Returns
    -------
    None
    """

    img = nib.Nifti1Image(data, affine, header)
    nib.save(img, out_path)

def max_voxels(
        data: np.ndarray
    ) -> np.ndarray:
    """
    Create a mask selecting all voxels equal to the maximum value in the array.

    Parameters
    ----------
    data : ndarray
        Input array from which to select maximum-valued voxels.

    Returns
    -------
    mask : ndarray
        Binary mask (float32) with 1 where data equals its maximum, else 0.
    """

    return (data >= data.max()).astype(np.uint8)

def top_k_voxels(
        data: np.ndarray,
        k: int = 9
    ) -> np.ndarray:
    """
    Create a mask selecting the top k highest-valued voxels in the array.

    Parameters
    ----------
    data : ndarray
        Input array from which to select top values.
    k : int, optional
        Number of voxels to select. Default is 9.

    Returns
    -------
    mask : ndarray
        Binary mask (float32) with 1 for the top k voxels, else 0.

    Raises
    ------
    ValueError
        If the total number of voxels is less than k.
    """
    flat = data.ravel()
    nz_idx = np.flatnonzero(flat > 0)
    if nz_idx.size == 0:
        raise ValueError("No non-zero voxels found in `data`.")
    k = min(k, nz_idx.size)
    nz_vals = flat[nz_idx]
    # threshold among non-zeros
    cut = np.partition(nz_vals, -k)[-k:].min()
    mask = (data >= cut) & (data > 0)
    return mask.astype(np.uint8)


def top_Xperc_voxels(
        data: np.ndarray,
        frac: float = 0.01
    ) -> np.ndarray:
    """
    Keep the top `frac` of non-zero voxels (e.g., 0.01 = top 1%).

    Parameters
    ----------
    data : ndarray
        Input array containing intensity or probability values.
    frac : float
        Fraction of non-zero voxels to select.

    Returns
    -------
    mask : ndarray
        Binary mask (uint8) with 1 where data >= cutoff, else 0.

    Raises
    ------
    ValueError
        If no non-zero voxels are found in the input data.
    """
    flat = data.ravel()
    nz_mask = flat > 0
    vals = flat[nz_mask]
    if vals.size == 0:
        raise ValueError("No non-zero voxels found in `data`.")
    k = max(1, int(np.ceil(vals.size * float(frac))))
    # pick threshold by value among non-zeros
    cut = np.partition(vals, -k)[-k:].min()
    out = (data >= cut) & (data > 0)
    return out.astype(np.uint8)


def robust_minmax(arr, low_pct=1, high_pct=99):
    lo = np.percentile(arr, low_pct)
    hi = np.percentile(arr, high_pct)
    arr = np.clip(arr, lo, hi)
    return (arr - lo) / (hi - lo + 1e-8)
