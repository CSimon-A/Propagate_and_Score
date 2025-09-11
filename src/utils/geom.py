# propagate_and_score/src/utils/geom.py

import numpy as np
from typing import Tuple, Sequence, Dict, Optional

def units_name(header) -> str:
    try:
        if hasattr(header, "get_xyzt_units"):
            u, _ = header.get_xyzt_units()
            if isinstance(u, bytes):
                u = u.decode("utf-8", "ignore")
            return (u or "mm")
    except Exception:
        pass
    return "mm"

def center_world(affine: np.ndarray, shape_zyx: Sequence[int]) -> np.ndarray:
    """
    Returns volume center in the affine's world units (x,y,z). Caller may convert to mm.
    """
    zc = (shape_zyx[0] - 1) / 2.0
    yc = (shape_zyx[1] - 1) / 2.0
    xc = (shape_zyx[2] - 1) / 2.0
    c = np.array([xc, yc, zc, 1.0], dtype=float)  # (x,y,z,1)
    return (affine @ c)[:3]  # (x,y,z) world-units

def try_codes(header) -> Dict[str, Optional[int]]:
    out = {"qform_code": None, "sform_code": None}
    try:
        q = int(header.get("qform_code")) if hasattr(header, "get") else None
        s = int(header.get("sform_code")) if hasattr(header, "get") else None
        out["qform_code"] = q
        out["sform_code"] = s
    except Exception:
        for k in ("qform_code", "sform_code"):
            try:
                out[k] = int(getattr(header, k))
            except Exception:
                pass
    return out

def spatial_units_factor_to_mm(header) -> float:
    """
    Returns the multiplier to convert the header's spatial units to millimeters.
    NIfTI xyzt_units can be 'm', 'mm', or 'micron' (um). If unavailable/unknown, assume mm.
    """
    unit = None
    try:
        if hasattr(header, "get_xyzt_units"):
            unit, _ = header.get_xyzt_units()
    except Exception:
        unit = None

    if isinstance(unit, bytes):
        unit = unit.decode("utf-8", "ignore")
    unit = (str(unit).lower() if unit is not None else "mm")

    if unit in ("mm", "millimeter", "millimetre"):
        return 1.0
    if unit in ("m", "meter", "metre", "meters", "metres"):
        return 1000.0
    if unit in ("micron", "micrometer", "micrometre", "um", "µm"):
        return 0.001
    # Unknown → assume mm
    return 1.0

def vox_to_mm(affine: np.ndarray, zyx: np.ndarray, header) -> np.ndarray:
    """
    zyx: [N,3] voxel coords (z,y,x)
    returns: [N,3] world coords (x,y,z) in **mm**
    """
    factor = spatial_units_factor_to_mm(header)
    xyzh = np.c_[zyx[:, 2], zyx[:, 1], zyx[:, 0], np.ones((zyx.shape[0], 1), dtype=zyx.dtype)]
    mm = (affine @ xyzh.T).T[:, :3] * factor
    return mm

def voxel_sizes_mm(affine: np.ndarray, header) -> Tuple[float, float, float]:
    """
    Returns voxel spacings along world axes (x,y,z), in **mm**.
    Assumes affine columns 0..2 encode world basis vectors for +x,+y,+z.
    """
    factor = spatial_units_factor_to_mm(header)
    return tuple(float(np.linalg.norm(affine[:3, i]) * factor) for i in range(3))

def principal_axis_angles_deg(obb_dirs_world: np.ndarray) -> np.ndarray:
    """
    obb_dirs_world: 3x3 matrix whose rows are world (x,y,z) unit vectors of principal axes (L,M,S).
    Returns: 3x3 angles (deg) between each principal axis and +X,+Y,+Z (absolute, 0–90°).
    """
    cosines = np.clip(np.abs(obb_dirs_world), -1.0, 1.0)
    return np.degrees(np.arccos(cosines))

def plane_angles_from_axis_angles(axis_angles_row: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert angles to axes (deg) → angles to planes (deg).
    axis_angles_row: length-3 array of angles of a vector to +X, +Y, +Z (deg).
    Returns: (sagittal, coronal, axial) plane angles in degrees.
    """
    ax_x, ax_y, ax_z = float(axis_angles_row[0]), float(axis_angles_row[1]), float(axis_angles_row[2])
    return (90.0 - ax_x, 90.0 - ax_y, 90.0 - ax_z)

def mask_stats(mask: np.ndarray, affine: np.ndarray, header) -> dict:
    """
    Computes:
      - Axis-aligned bbox sizes in voxels (Δz,Δy,Δx) and mm (Δz_mm,Δy_mm,Δx_mm)
      - Centroid in voxels and mm (reported as (z,y,x) for log familiarity)
      - Oriented (PCA) bbox: lengths (L,M,S) in mm, principal directions (world x,y,z unit vectors),
        angles to +X/+Y/+Z, and center in mm.
    """
    nz = np.argwhere(mask > 0)  # (z,y,x)
    voxels = int(nz.shape[0])
    if voxels == 0:
        empty_angles = np.array([np.nan, np.nan, np.nan], dtype=float)
        return {
            "voxels": 0,
            "bbox_vox": (0, 0, 0),
            "bbox_mm": (0.0, 0.0, 0.0),
            "centroid_vox": (np.nan, np.nan, np.nan),
            "centroid_mm": (np.nan, np.nan, np.nan),
            "obb_lengths_mm": (0.0, 0.0, 0.0),
            "obb_dirs_world": np.full((3, 3), np.nan, dtype=float),
            "obb_center_mm": (np.nan, np.nan, np.nan),
            "obb_angles_deg": np.vstack([empty_angles, empty_angles, empty_angles]),
        }

    # Axis-aligned bbox (voxel & mm)
    z0y0x0 = nz.min(axis=0)
    z1y1x1 = nz.max(axis=0) + 1  # exclusive
    size_vox = tuple((z1y1x1 - z0y0x0).tolist())  # (Δz, Δy, Δx)
    sx, sy, sz = voxel_sizes_mm(affine, header)   # spacings along world x,y,z in mm
    size_mm = (size_vox[0] * sz, size_vox[1] * sy, size_vox[2] * sx)

    # Centroid (voxel) and in mm (reported back as (z,y,x))
    centroid_vox = tuple((nz.mean(axis=0)).tolist())
    c_mm_xyz = vox_to_mm(affine, np.array([nz.mean(axis=0)]), header)[0]  # (x,y,z) mm
    centroid_mm = (float(c_mm_xyz[2]), float(c_mm_xyz[1]), float(c_mm_xyz[0]))

    # Oriented bbox (PCA) in world space (mm)
    pts_mm = vox_to_mm(affine, nz, header)   # [N,3] (x,y,z) mm
    center = pts_mm.mean(axis=0, keepdims=True)
    X = pts_mm - center
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    dirs = Vt  # (3,3); rows are principal directions (unit)
    extents = []
    for k in range(3):
        proj = X @ dirs[k]
        length = float(proj.max() - proj.min())
        extents.append((length, dirs[k]))
    extents.sort(key=lambda t: t[0], reverse=True)  # L >= M >= S

    obb_lengths = (extents[0][0], extents[1][0], extents[2][0])
    obb_dirs = np.vstack([extents[0][1], extents[1][1], extents[2][1]])  # rows (x,y,z)
    angles_deg = principal_axis_angles_deg(obb_dirs)  # rows to +X,+Y,+Z

    return {
        "voxels": voxels,
        "bbox_vox": size_vox,
        "bbox_mm": size_mm,
        "centroid_vox": centroid_vox,              # (z,y,x)
        "centroid_mm": centroid_mm,                # (z,y,x) mm
        "obb_lengths_mm": obb_lengths,             # (L,M,S) mm
        "obb_dirs_world": obb_dirs,                # 3x3 rows (x,y,z)
        "obb_center_mm": tuple(map(float, center.ravel())),  # (x,y,z) mm
        "obb_angles_deg": angles_deg,              # rows: angles to +X,+Y,+Z (deg)
    }

def centroid_distance_mm(mask_a: np.ndarray, aff_a: np.ndarray, hdr_a,
                         mask_b: np.ndarray, aff_b: np.ndarray, hdr_b) -> float:
    def _centroid_mm(m, aff, hdr):
        nz = np.argwhere(m > 0)
        if nz.size == 0:
            return np.array([np.nan, np.nan, np.nan])
        c_mm = vox_to_mm(aff, np.array([nz.mean(axis=0)]), hdr)[0]  # (x,y,z) mm
        return c_mm
    a = _centroid_mm(mask_a, aff_a, hdr_a)
    b = _centroid_mm(mask_b, aff_b, hdr_b)
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        return float("nan")
    return float(np.linalg.norm(a - b))

def uncrop_to_full_ds(cropped: np.ndarray,
                       ds_full_shape: tuple,
                       start: np.ndarray,
                       end: np.ndarray,
                       fill_value: float) -> np.ndarray:
    full = np.full(ds_full_shape, fill_value, dtype=cropped.dtype)
    z0, y0, x0 = map(int, start)
    z1, y1, x1 = map(int, end)

    # Clamp destination to valid bounds
    Z0, Y0, X0 = max(z0, 0), max(y0, 0), max(x0, 0)
    Z1, Y1, X1 = min(z1, ds_full_shape[0]), min(y1, ds_full_shape[1]), min(x1, ds_full_shape[2])

    # Corresponding region in the cropped source
    cz0, cy0, cx0 = Z0 - z0, Y0 - y0, X0 - x0
    cz1, cy1, cx1 = cz0 + (Z1 - Z0), cy0 + (Y1 - Y0), cx0 + (X1 - X0)

    # If there is no overlap, just return the padded canvas
    if (Z1 <= Z0) or (Y1 <= Y0) or (X1 <= X0):
        return full

    full[Z0:Z1, Y0:Y1, X0:X1] = cropped[cz0:cz1, cy0:cy1, cx0:cx1]
    return full

