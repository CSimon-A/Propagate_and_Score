# propagate_and_score/src/utils/qa.py

import numpy as np
import logging

from .io_utils import load_nifti
from .geom import voxel_sizes_mm, units_name, mask_stats, centroid_distance_mm, plane_angles_from_axis_angles
import train.config as config

def qa_masks(logger: logging.Logger, R_path: str, T1_path: str) -> None:
    R,  aff_R,  hdr_R  = load_nifti(R_path)
    T1, aff_T1, hdr_T1 = load_nifti(T1_path)

    if R.shape != T1.shape:
        logger.warning("R and T1 have different shapes: %s vs %s", R.shape, T1.shape)
    else:
        logger.info("Shapes: %s", R.shape)

    if not np.allclose(aff_R, aff_T1, atol=1e-4):
        logger.warning("Affines differ beyond tolerance (1e-4). Registration mismatch likely.")
    else:
        logger.info("Affines: consistent within tolerance.")

    # Log spatial units & voxel sizes in mm
    u_T1 = units_name(hdr_T1); u_R = units_name(hdr_R)
    vx_T1 = voxel_sizes_mm(aff_T1, hdr_T1); vx_R = voxel_sizes_mm(aff_R, hdr_R)
    logger.info("T1 spatial units: %s → voxel sizes (mm) = (%.3f, %.3f, %.3f)", u_T1, *vx_T1)
    logger.info("R  spatial units: %s → voxel sizes (mm) = (%.3f, %.3f, %.3f)", u_R,  *vx_R)

    thr = getattr(config, "THRESHOLD", 0.01)
    T1b = (T1 > thr).astype(np.uint8)
    Rb  = (R  > thr).astype(np.uint8)

    logger.info("Binarized tumor and resection files with threshold=%.4g", thr)

    # Binarity & emptiness
    for name, Vb in (("T1", T1b), ("R", Rb)):
        voxels = int(Vb.sum())
        is_bin = np.array_equal(Vb, Vb.astype(bool))
        logger.info("%s binarity: %s | nonzero voxels: %d", name, "OK" if is_bin else "NON-BINARY", voxels)
        if voxels == 0:
            logger.error("%s mask is empty after thresholding; downstream scoring will be meaningless.", name)

    # Volume difference as % (always WARN)
    v_T1 = int(T1b.sum()); v_R = int(Rb.sum())
    if v_T1 > 0:
        if v_R > v_T1:
            perc = (v_R - v_T1) / v_T1 * 100.0
            logger.warning("Voxel counts → |T1|=%d, |R|=%d → R is %.1f%% larger than T1", v_T1, v_R, perc)
        elif v_R < v_T1:
            perc = (v_T1 - v_R) / v_T1 * 100.0
            logger.warning("Voxel counts → |T1|=%d, |R|=%d → R is %.1f%% smaller than T1", v_T1, v_R, perc)
        else:
            logger.warning("Voxel counts → |T1|=%d, |R|=%d → volumes equal", v_T1, v_R)
    else:
        logger.warning("Voxel counts → |T1|=0, |R|=%d", v_R)

    # BBoxes, oriented axes & centroids
    st_T1 = mask_stats(T1b, aff_T1, hdr_T1)
    st_R  = mask_stats(Rb,  aff_R,  hdr_R)

    # Axis-aligned
    logger.info("T1 bbox (vox): %s | (mm): %.1f×%.1f×%.1f", st_T1["bbox_vox"], *st_T1["bbox_mm"])
    logger.info("R  bbox (vox): %s | (mm): %.1f×%.1f×%.1f", st_R["bbox_vox"],  *st_R["bbox_mm"])

    # Oriented (major axis) — vector + axis angles
    L_len_T1 = st_T1["obb_lengths_mm"][0]
    L_dir_T1 = st_T1["obb_dirs_world"][0]
    ang_T1   = st_T1["obb_angles_deg"][0]
    logger.info(
        "T1 principal axis: length=%.1f mm | dir_world(x,y,z)=(%.3f, %.3f, %.3f) | angles to axes: x=%.1f°, y=%.1f°, z=%.1f°",
        L_len_T1, L_dir_T1[0], L_dir_T1[1], L_dir_T1[2], ang_T1[0], ang_T1[1], ang_T1[2]
    )
    L_len_R  = st_R["obb_lengths_mm"][0]
    L_dir_R  = st_R["obb_dirs_world"][0]
    ang_R    = st_R["obb_angles_deg"][0]
    logger.info(
        "R  principal axis: length=%.1f mm | dir_world(x,y,z)=(%.3f, %.3f, %.3f) | angles to axes: x=%.1f°, y=%.1f°, z=%.1f°",
        L_len_R, L_dir_R[0], L_dir_R[1], L_dir_R[2], ang_R[0], ang_R[1], ang_R[2]
    )

    # Principal-axis angles to planes
    sag_T1, cor_T1, ax_T1 = plane_angles_from_axis_angles(ang_T1)
    sag_R,  cor_R,  ax_R  = plane_angles_from_axis_angles(ang_R)
    logger.info("T1 principal axis: angles to planes — sagittal=%.1f°, coronal=%.1f°, axial=%.1f°",
                sag_T1, cor_T1, ax_T1)
    logger.info("R  principal axis: angles to planes — sagittal=%.1f°, coronal=%.1f°, axial=%.1f°",
                sag_R, cor_R, ax_R)

    # Centroid distance (mm)
    d_mm = centroid_distance_mm(T1b, aff_T1, hdr_T1, Rb, aff_R, hdr_R)
    if np.isfinite(d_mm):
        logger.warning("Centroid distance: %.2f mm", d_mm)
    else:
        logger.warning("Centroid distance: nan mm (one of the masks is empty)")
