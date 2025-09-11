# propagate_and_score/src/utils/mni_check.py

from typing import Sequence, Tuple
import numpy as np
import logging

from .geom import spatial_units_factor_to_mm, voxel_sizes_mm, center_world, try_codes
from .io_utils import load_nifti

def rot_mat_and_scales(affine: np.ndarray) -> Tuple[np.ndarray, Tuple[float,float,float]]:
    """Decompose affine[:3,:3] into rotation (orthonormal) and per-axis scales (in affine's units)."""
    M = affine[:3, :3].astype(float)
    sx = float(np.linalg.norm(M[:, 0])) or 1.0
    sy = float(np.linalg.norm(M[:, 1])) or 1.0
    sz = float(np.linalg.norm(M[:, 2])) or 1.0
    S_inv = np.diag([1.0/sx, 1.0/sy, 1.0/sz])
    R_approx = M @ S_inv
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt  # nearest orthonormal rotation
    return R, (sx, sy, sz)

def rot_delta_deg(R: np.ndarray, R_ref: np.ndarray) -> float:
    """Smallest rotation angle (deg) from R to R_ref."""
    dR = R @ R_ref.T
    t = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(t)))

def rel_scale_delta(scales: Tuple[float,float,float], ref_scales: Tuple[float,float,float]) -> float:
    rel = [abs(s - sr) / max(sr, 1e-8) for s, sr in zip(scales, ref_scales)]
    return float(max(rel))  # max relative delta across axes

def mni_check_one(logger: logging.Logger, name: str,
                   arr_shape: Sequence[int], affine: np.ndarray, header,
                   ref_shape: Sequence[int], ref_affine: np.ndarray, ref_header) -> bool:
    R, scales = rot_mat_and_scales(affine)
    Rr, scales_ref = rot_mat_and_scales(ref_affine)

    # Convert scales and centers to mm using header units
    f_test = spatial_units_factor_to_mm(header)
    f_ref  = spatial_units_factor_to_mm(ref_header)
    scales_mm     = (scales[0]*f_test,     scales[1]*f_test,     scales[2]*f_test)
    scales_ref_mm = (scales_ref[0]*f_ref,  scales_ref[1]*f_ref,  scales_ref[2]*f_ref)

    ang = rot_delta_deg(R, Rr)
    scl = rel_scale_delta(scales_mm, scales_ref_mm) * 100.0  # %
    c_world  = center_world(affine,     arr_shape) * f_test
    c_ref_w  = center_world(ref_affine, ref_shape) * f_ref
    c_off = float(np.linalg.norm(c_world - c_ref_w))

    exact = (tuple(arr_shape) == tuple(ref_shape)) and np.allclose(affine, ref_affine, atol=1e-4)
    codes = try_codes(header); codes_ref = try_codes(ref_header)

    logger.debug("%s — voxel sizes (mm): test=(%.3f, %.3f, %.3f) | ref=(%.3f, %.3f, %.3f)",
                 name, *scales_mm, *scales_ref_mm)
    logger.debug("%s — rotation delta: %.2f° | center offset: %.2f mm | voxel size delta: %.2f%% (max axis)",
                 name, ang, c_off, scl)
    if codes["qform_code"] is not None or codes["sform_code"] is not None:
        logger.debug("%s — qform_code=%s, sform_code=%s | ref qform_code=%s, sform_code=%s",
                     name, codes["qform_code"], codes["sform_code"],
                     codes_ref["qform_code"], codes_ref["sform_code"])

    pass_ang = ang <= 5.0
    pass_ctr = c_off <= 5.0
    pass_scl = (scl <= 5.0)
    mni_like = exact or (pass_ang and pass_ctr and pass_scl)

    reason = []
    if exact: reason.append("exact affine+shape")
    reason.append("rotation≤5°" if pass_ang else "rotation>5°")
    reason.append("center≤5mm"  if pass_ctr else "center>5mm")
    reason.append("voxelΔ≤5%"   if pass_scl else "voxelΔ>5%")

    (logger.info if mni_like else logger.error)("%s — MNI-like: %s (%s)",
                                                name, "YES" if mni_like else "NO", ", ".join(reason))
    return mni_like

def mni_check(logger: logging.Logger, ref_path: str, named_paths: Sequence[Tuple[str, str]]) -> bool:
    try:
        ref_arr, ref_aff, ref_hdr = load_nifti(ref_path)
    except Exception:
        logger.exception("Failed to load MNI reference: %s", ref_path)
        return False

    sx, sy, sz = voxel_sizes_mm(ref_aff, ref_hdr)
    logger.info("MNI reference loaded: %s | shape=%s | voxel sizes=(%.3f, %.3f, %.3f) mm",
                ref_path, tuple(ref_arr.shape), sx, sy, sz)

    all_ok = True
    for name, p in named_paths:
        try:
            arr, aff, hdr = load_nifti(p)
        except Exception:
            logger.exception("Failed to load %s (%s) for MNI check", name, p)
            all_ok = False
            continue
        ok = mni_check_one(logger, name, arr.shape, aff, hdr, ref_arr.shape, ref_aff, ref_hdr)
        all_ok = all_ok and ok
    return all_ok
