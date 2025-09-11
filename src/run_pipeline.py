#!/usr/bin/env python3
# src/run_pipeline.py

"""
run_pipeline.py — Orchestrates the two existing steps :
  1) predict.py  -> builds the T2E event map
  2) score.py    -> computes growth-aware score

It also performs input QA checks (logged):
  • mask binarity & emptiness (T1, R)
  • voxel counts (always logged as WARNING with % increase/decrease)
  • shapes & affine alignment
  • tumor bbox size along axes (in voxels & mm)
  • oriented principal axis (length + direction + angles to X/Y/Z, in mm/deg)
  • principal-axis angles to planes (sagittal / coronal / axial), in degrees
  • centroid distance (always logged as WARNING, in mm)
  • OPTIONAL: MNI conformance vs a reference (--mni_ref), with rotation/scale/center checks

Notes
-----
• All physical lengths are reported in millimeters (mm), derived from the affine and header units.
• It shells out via subprocess and forwards your chosen arguments verbatim.
• QA warnings for volume difference and centroid distance are always emitted.
"""

from __future__ import annotations

# stdlib
import argparse
import logging
import os
import sys
from typing import Sequence, Tuple

# third-party
import numpy as np

# local
from utils.array_ops import ensure_same_shape
from utils.geom import voxel_sizes_mm, units_name
from utils.io_utils import load_nifti
from utils.logger import setup_logging
from utils.mni_check import mni_check
from utils.qa import qa_masks
from utils.shell import run_cmd


def build_parser() -> argparse.ArgumentParser:
    import argparse

    epilog = r"""
Examples
--------
# 1) Run only prediction (produces a T2E map)
python src/run_pipeline.py predict \
  --bravo data/cas_1/t1_bravo_bet.nii.gz \
  --flair data/cas_1/t2_flair_bet.nii.gz \
  --mask  data/cas_1/tumor_1.nii.gz \
  --out_dir data/cas_1/output \
  --t2e_ckpt saved_models/v17/propagation_unet_best.pth \
  --mni_ref data/mni_masked.nii.gz

# 2) Run only scoring (uses an already-produced T2E map)
python src/run_pipeline.py score \
  --R data/cas_1/tumor_1_dil.nii.gz \
  --T1 data/cas_1/tumor_1.nii.gz \
  --G  data/cas_1/output/t2e_map.nii.gz \
  --out_dir data/cas_1/output --save_maps \
  --mni_ref data/mni_masked.nii.gz

# 3) Full pipeline (predict → score); by default score uses the predicted G
python src/run_pipeline.py both \
  --bravo data/cas_1/t1_bravo_bet.nii.gz \
  --flair data/cas_1/t2_flair_bet.nii.gz \
  --mask  data/cas_1/tumor_1.nii.gz \
  --pred_out_dir data/cas_1/output \
  --t2e_ckpt saved_models/v17/propagation_unet_best.pth \
  --R data/cas_1/tumor_1_dil.nii.gz \
  --T1 data/cas_1/tumor_1.nii.gz \
  --score_out_dir data/cas_1/output --save_maps \
  --mni_ref data/mni_masked.nii.gz

Notes
-----
• If --mni_ref is provided, inputs must be "MNI-like" (≤5° rotation, ≤5 mm center offset,
  ≤5% voxel-size delta) or the runner exits with code 3 before running predict/score.
• Axis/plane terminology assumes RAS world axes (x=Left↔Right, y=Posterior↔Anterior, z=Inferior↔Superior).
"""

    p = argparse.ArgumentParser(
        description="Orchestrate predict → score with QA checks and optional MNI conformance.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=epilog,
    )
    sub = p.add_subparsers(dest="mode", required=True, metavar="{predict,score,both}",
                           help="Choose which stage(s) to run.")

    # ---------------- Common (shared by all subcommands) ----------------
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--log_file", default=None,
        help=("Path to the runner log file. If omitted, defaults to:\n"
              "  predict: <out_dir>/runner.log\n"
              "  score:   <out_dir>/runner.log\n"
              "  both:    <score_out_dir>/runner.log\n"
              "Child scripts write their own logs separately.")
    )
    common.add_argument(
        "--log_level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"],
        help=("Console log level for the runner (default: INFO). "
              "DEBUG prints full diagnostics (affines, PCA angles, MNI deltas, etc.).")
    )
    common.add_argument(
        "--mni_ref", default=None,
        help=("Path to an MNI reference NIfTI. When provided, each input is compared to this reference.\n"
              "If any input is not MNI-like (rotation>5°, center offset>5 mm, voxel-size delta>5%),\n"
              "the runner logs an ERROR and exits with code 3 before running predict/score.\n"
              "Example: /Users/.../data/mni_masked.nii.gz")
    )

    # ---------------- QA convenience (for score/both) ----------------
    qa = argparse.ArgumentParser(add_help=False)
    qa.add_argument(
        "--check_only", action="store_true",
        help=("Run QA & optional MNI checks only, and do NOT call predict.py / score.py.\n"
              "Useful for preflight validation of masks, affines, principal axes, etc.")
    )

    # ---------------- predict ----------------
    sp_pred = sub.add_parser(
        "predict", parents=[common],
        help="Run predict.py only (build the T2E event map).",
        formatter_class=argparse.RawTextHelpFormatter
        )
    sp_pred.add_argument("--bravo", required=True,
                         help="Path to BRAVO/T1 structural image (3D NIfTI .nii or .nii.gz). Should align with --mask.")
    sp_pred.add_argument("--flair", required=True,
                         help="Path to T2-FLAIR image (3D NIfTI). Should align with --mask.")
    sp_pred.add_argument("--mask", required=True,
                         help=("Path to initial tumor mask at t0 (3D NIfTI). Soft or binary allowed; "
                               "runner logs binarity and emptiness; predict.py binarizes internally."))
    sp_pred.add_argument("--out_dir", required=True,
                         help="Directory to write T2E outputs (created if missing).")
    sp_pred.add_argument("--t2e_ckpt", required=True,
                         help="Path to the UNet checkpoint file (e.g., saved_models/v17/propagation_unet_best.pth).")
    # passthrough to predict.py
    sp_pred.add_argument("--device", default=None,
                         help="Torch device for inference, e.g. 'cuda:0' or 'cpu'. Defaults to CUDA if available.")
    sp_pred.add_argument("--expect_shape", type=int, nargs=3, metavar=("D","H","W"),
                         help=("Training box at FULL resolution BEFORE downsample (voxel counts). "
                               "Example: 192 192 192. Should be divisible by --down_factor."))
    sp_pred.add_argument("--down_factor", type=int,
                         help=("Integer downsample factor used during training (e.g., 2 means model grid ~96³ if expect_shape=192³). "
                               "Must evenly divide each dimension of --expect_shape."))
    sp_pred.add_argument("--no-crop", dest="crop", action="store_false",
                         help="Disable cropping to the training box after downsampling (default is to crop).")
    sp_pred.add_argument("--no_zscore", action="store_true",
                         help="Disable z-score normalization of BRAVO/FLAIR (predict.py enables it by default).")
    sp_pred.add_argument("--scale_mode", choices=["none","min-max","tanh","sigmoid"],
                         help=("Additional intensity scaling applied after (optional) z-score. "
                               "Default in predict.py is 'min-max'."))
    sp_pred.add_argument("--t2e_clamp_min", type=float,
                         help="Lower clamp bound for the T2E map in model grid (default in predict.py: -1.0).")
    sp_pred.add_argument("--pad_value", type=float,
                         help=("Value used outside the crop when undoing crop on the downsampled grid. "
                               "Default: uses --t2e_clamp_min."))
    sp_pred.add_argument("--t2e_out", default=None,
                         help="Filename for the saved T2E map inside --out_dir (default in predict.py: t2e_map.nii.gz).")
    sp_pred.add_argument("--model", default="unet3d",
                         choices=["unet3d_larger_skip", "unet3d"],
                         help=("Model architecture to use for T2E prediction (default: unet3d). "))

    # ---------------- score ----------------
    sp_score = sub.add_parser(
        "score", parents=[common, qa],
        help="Run score.py only (compute growth-aware score) with QA checks.",
        formatter_class=argparse.RawTextHelpFormatter
        )
    sp_score.add_argument("--R", required=True,
                          help=("Path to resection mask (3D NIfTI). Will be binarized inside score.py using --thr_R "
                                "(or config.THRESHOLD; default 0.01). Must be on the same grid as T1 and G."))
    sp_score.add_argument("--T1", required=True,
                          help=("Path to tumor mask at T1 (3D NIfTI). Will be binarized inside score.py using --thr_T1 "
                                "(or config.THRESHOLD). Must be on the same grid as R and G."))
    sp_score.add_argument("--G", required=True,
                          help=("Path to growth/event map (3D NIfTI, float). Usually the T2E map from predict.py. "
                                "Runner checks its shape/affine vs T1."))
    sp_score.add_argument("--out_dir", required=True,
                          help="Directory to write score outputs (JSON summary and optional debug maps).")

    # pass-through to score.py
    sp_score.add_argument("--summary_json",
                          help="Filename for JSON summary written in --out_dir (default in score.py: score_summary.json).")
    sp_score.add_argument("--thr_R", type=float,
                          help="Binarization threshold for R (overrides config.THRESHOLD; default if unset: 0.01).")
    sp_score.add_argument("--thr_T1", type=float,
                          help="Binarization threshold for T1 (overrides config.THRESHOLD; default if unset: 0.01).")
    sp_score.add_argument("--growth_mode", choices=["linear","exp","geom"],
                          help=("How to normalize the growth/event map:\n"
                                "  linear: clamp to [0, max_value]\n"
                                "  exp:    exp(-k / tau)\n"
                                "  geom:   geometric scheme with ratio r"))
    sp_score.add_argument("--growth_max_value", type=float,
                          help=("For --growth_mode=linear: sets the k at which the score hits 0. "
                                "Maps k>=0 to 1 - k/MAX (clamped to [0,1]); k<0 → 0. "
                                "Example: MAX=3 gives k=0→1, 1→0.667, 2→0.333, k≥3→0. "
                                "Ignored for 'exp' and 'geom' modes. (default in score.py: 5.0)."))
    sp_score.add_argument("--growth_tau", type=float,
                          help="For growth_mode=exp: tau parameter (default in score.py: 2.0).")
    sp_score.add_argument("--growth_r", type=float,
                          help="For growth_mode=geom: ratio r (default in score.py: 2/3 if not provided).")
    sp_score.add_argument("--sigma_dist", type=float,
                          help=("Distance scale for the distance-transform term (same units as EDT output; "
                                "typically voxel units of the input grid). Default in score.py: 10.0."))
    sp_score.add_argument("--sigma_vol", type=float,
                          help=("Over-resection volume scale as a FRACTION of tumor volume (default in score.py: 0.5)."))
    sp_score.add_argument("--alpha", type=float,
                          help=("Trade-off between recall and sparing (alpha in [0,1]); higher favors recall. "
                                "Default in score.py: 0.5."))
    sp_score.add_argument("--eps", type=float,
                          help="Numerical stability epsilon (default in score.py: 1e-8).")
    sp_score.add_argument("--save_maps", action="store_true",
                          help="Save debug maps (wd.nii.gz, wg.nii.gz, over_resection.nii.gz) into --out_dir.")
    sp_score.add_argument("--wd_name",
                          help="Filename for wd map (default in score.py: wd.nii.gz).")
    sp_score.add_argument("--wg_name",
                          help="Filename for wg map (default in score.py: wg.nii.gz).")
    sp_score.add_argument("--over_name",
                          help="Filename for over-resection mask (default in score.py: over_resection.nii.gz).")

    # ---------------- both ----------------
    sp_both = sub.add_parser(
        "both", parents=[common, qa],
        help="Run prediction then scoring (predict → score) with QA checks.",
        formatter_class=argparse.RawTextHelpFormatter
        )

    # predict args (same as in predict)
    sp_both.add_argument("--bravo", required=True,
                         help="Path to BRAVO/T1 structural image (3D NIfTI). Should align with --mask.")
    sp_both.add_argument("--flair", required=True,
                         help="Path to T2-FLAIR image (3D NIfTI). Should align with --mask.")
    sp_both.add_argument("--mask", required=True,
                         help="Path to initial tumor mask at t0 (3D NIfTI). Soft or binary; runner logs QA.")
    sp_both.add_argument("--pred_out_dir", required=True,
                         help="Directory where the T2E map will be written by predict.py.")
    sp_both.add_argument("--t2e_ckpt", required=True,
                         help="Path to the UNet checkpoint for predict.py.")
    sp_both.add_argument("--device", default=None,
                         help="Torch device, e.g. 'cuda:0' or 'cpu'.")
    sp_both.add_argument("--expect_shape", type=int, nargs=3, metavar=("D","H","W"),
                         help="Training box at full-res BEFORE downsample (e.g., 192 192 192).")
    sp_both.add_argument("--down_factor", type=int,
                         help="Training downsample factor (e.g., 2). Must divide each dim of --expect_shape.")
    sp_both.add_argument("--no-crop", dest="crop", action="store_false",
                         help="Disable crop to the training box (default is to crop).")
    sp_both.add_argument("--no_zscore", action="store_true",
                         help="Disable z-score on BRAVO/FLAIR.")
    sp_both.add_argument("--scale_mode", choices=["none","min-max","tanh","sigmoid"],
                         help="Extra intensity scaling; default in predict.py is 'min-max'.")
    sp_both.add_argument("--t2e_clamp_min", type=float,
                         help="Clamp lower bound for the T2E map (default in predict.py: -1.0).")
    sp_both.add_argument("--pad_value", type=float,
                         help="Fill value outside crop on ds grid; default uses --t2e_clamp_min.")
    sp_both.add_argument("--t2e_out", default="t2e_map.nii.gz",
                         help="Filename of the T2E map to be saved in --pred_out_dir (default: t2e_map.nii.gz).")
    sp_both.add_argument("--model", default="unet3d",
                         choices=["unet3d_larger_skip", "unet3d"],
                         help=("Model architecture to use for T2E prediction (default: unet3d). "))

    # score args (same as in score, but --G is optional override)
    sp_both.add_argument("--R", required=True,
                         help="Path to resection mask (3D NIfTI). Must align with T1 and predicted G.")
    sp_both.add_argument("--T1", required=False, default=None,
                         help=("Path to tumor mask at T1 (3D NIfTI). If omitted, the runner will "
                               "use --mask as T1 (logged). Must align with R and predicted G."))
    sp_both.add_argument("--score_out_dir", required=False, default=None,
                         help=("Directory to write score outputs. If omitted, falls back to "
                               "--pred_out_dir (logged)."))
    sp_both.add_argument("--G", default=None,
                         help=("Optional override for growth map path. If omitted, uses the file written by predict.py:\n"
                               "<pred_out_dir>/<t2e_out>."))
    sp_both.add_argument("--summary_json",
                         help="Filename for JSON summary written in --score_out_dir (default in score.py: score_summary.json).")
    sp_both.add_argument("--thr_R", type=float,
                         help="Binarization threshold for R (default if unset: config.THRESHOLD or 0.01).")
    sp_both.add_argument("--thr_T1", type=float,
                         help="Binarization threshold for T1 (default if unset: config.THRESHOLD or 0.01).")
    sp_both.add_argument("--growth_mode", choices=["linear","exp","geom"],
                         help="Growth normalization mode for score.py (linear/exp/geom).")
    sp_both.add_argument("--growth_max_value", type=float,
                         help=("For --growth_mode=linear: sets the k at which the score hits 0. "
                               "Maps k>=0 to 1 - k/MAX (clamped to [0,1]); k<0 → 0. "
                               "Example: MAX=3 gives k=0→1, 1→0.667, 2→0.333, k≥3→0. "
                               "Ignored for 'exp' and 'geom' modes. (default in score.py: 5.0)."))
    sp_both.add_argument("--growth_tau", type=float,
                         help="For exp mode: tau parameter (default 2.0).")
    sp_both.add_argument("--growth_r", type=float,
                         help="For geom mode: ratio r (default 2/3 if not provided).")
    sp_both.add_argument("--sigma_dist", type=float,
                         help="Distance scale for EDT term (voxel units). Default in score.py: 10.0.")
    sp_both.add_argument("--sigma_vol", type=float,
                         help="Over-resection volume scale as a fraction of tumor volume. Default: 0.5.")
    sp_both.add_argument("--alpha", type=float,
                         help="Trade-off between recall and sparing (0..1). Default: 0.5.")
    sp_both.add_argument("--eps", type=float,
                         help="Numerical epsilon. Default: 1e-8.")
    sp_both.add_argument("--save_maps", action="store_true",
                         help="Save wd/wg/over maps into --score_out_dir.")
    sp_both.add_argument("--wd_name",
                         help="Filename for wd map (default wd.nii.gz).")
    sp_both.add_argument("--wg_name",
                         help="Filename for wg map (default wg.nii.gz).")
    sp_both.add_argument("--over_name",
                         help="Filename for over-resection mask (default over_resection.nii.gz).")
    sp_both.add_argument("--Gprob_name",
                         help="Filename for Gprob map (default Gprob.nii.gz).")
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    # set up logging (choose sensible default log location per mode)
    default_log = None
    if getattr(args, "mode", None) == "predict":
        default_log = os.path.join(args.out_dir, "runner.log")
    elif args.mode == "score":
        default_log = os.path.join(args.out_dir, "runner.log")
    elif args.mode == "both":
        default_log = os.path.join((args.score_out_dir or args.pred_out_dir), "runner.log")

    setup_logging(args.log_file or default_log, getattr(args, "log_level", "INFO"))
    logger = logging.getLogger("runner")

    # Helper: perform MNI check for a list of (name, path) if --mni_ref is set
    def maybe_mni_check_or_die(pairs: Sequence[Tuple[str, str]]) -> None:
        if getattr(args, "mni_ref", None):
            ok = mni_check(logger, args.mni_ref, pairs)
            if not ok:
                logger.error("Aborting: one or more inputs are not MNI-like relative to the reference.")
                sys.exit(3)

    if args.mode == "predict":
        # Optional lightweight sanity: ensure modalities align with mask
        try:
            b, aff_b, hdr_b = load_nifti(args.bravo)
            f, aff_f, hdr_f = load_nifti(args.flair)
            m, aff_m, hdr_m = load_nifti(args.mask)
            ensure_same_shape(b, f, m)
            if not (np.allclose(aff_b, aff_m, atol=1e-4) and np.allclose(aff_f, aff_m, atol=1e-4)):
                logger.warning("Modalities and mask have mismatched affines (>1e-4). Inference could misalign.")
            # Units/voxel sizes
            logger.info("BRAVO units: %s → voxel sizes (mm) = (%.3f, %.3f, %.3f)",
                        units_name(hdr_b), *voxel_sizes_mm(aff_b, hdr_b))
            logger.info("FLAIR units: %s → voxel sizes (mm) = (%.3f, %.3f, %.3f)",
                        units_name(hdr_f), *voxel_sizes_mm(aff_f, hdr_f))
            logger.info("MASK  units: %s → voxel sizes (mm) = (%.3f, %.3f, %.3f)",
                        units_name(hdr_m), *voxel_sizes_mm(aff_m, hdr_m))
        except Exception:
            logger.exception("Pre-check failed (predict inputs)")

        # MNI check (if requested)
        maybe_mni_check_or_die([("BRAVO", args.bravo), ("FLAIR", args.flair), ("MASK", args.mask)])

        if getattr(args, "check_only", False):
            logger.info("check_only set — skipping predict.py invocation.")
            return 0

        cmd = [sys.executable, "src/predict.py",
               "--bravo", args.bravo,
               "--flair", args.flair,
               "--mask",  args.mask,
               "--out_dir", args.out_dir,
               "--t2e_ckpt", args.t2e_ckpt]
        # Propagate optional flags if provided
        if args.device:              cmd += ["--device", args.device]
        if args.expect_shape:        cmd += ["--expect_shape", *map(str, args.expect_shape)]
        if args.down_factor is not None: cmd += ["--down_factor", str(args.down_factor)]
        if getattr(args, "crop", True) is False: cmd += ["--no-crop"]
        if args.no_zscore:           cmd += ["--no_zscore"]
        if args.scale_mode:          cmd += ["--scale_mode", args.scale_mode]
        if args.t2e_clamp_min is not None: cmd += ["--t2e_clamp_min", str(args.t2e_clamp_min)]
        if args.pad_value is not None:    cmd += ["--pad_value", str(args.pad_value)]
        if args.t2e_out:             cmd += ["--t2e_out", args.t2e_out]

        return run_cmd(logger, cmd)

    elif args.mode == "score":
        # QA checks
        try:
            qa_masks(logger, args.R, args.T1)
            # shape/affine consistency with growth map
            G, aff_G, hdr_G = load_nifti(args.G)
            T1_vol, aff_T1, hdr_T1 = load_nifti(args.T1)
            if G.shape != T1_vol.shape:
                logger.warning("G and T1 shapes differ: %s vs %s", G.shape, T1_vol.shape)
            if not np.allclose(aff_G, aff_T1, atol=1e-4):
                logger.warning("G and T1 affines differ (>1e-4). Scoring may be misaligned.")
            # Units/voxel sizes
            logger.info("G units: %s → voxel sizes (mm) = (%.3f, %.3f, %.3f)",
                        units_name(hdr_G), *voxel_sizes_mm(aff_G, hdr_G))
        except Exception:
            logger.exception("QA checks failed")

        # MNI check (if requested)
        maybe_mni_check_or_die([("R", args.R), ("T1", args.T1), ("G", args.G)])

        if args.check_only:
            logger.info("check_only set — skipping score.py invocation.")
            return 0

        cmd = [sys.executable, "src/score.py",
               "--R", args.R,
               "--T1", args.T1,
               "--G",  args.G,
               "--out_dir", args.out_dir]
        # passthrough
        if args.summary_json:        cmd += ["--summary_json", args.summary_json]
        if args.thr_R is not None:   cmd += ["--thr_R", str(args.thr_R)]
        if args.thr_T1 is not None:  cmd += ["--thr_T1", str(args.thr_T1)]
        if args.growth_mode:         cmd += ["--growth_mode", args.growth_mode]
        if args.growth_max_value is not None: cmd += ["--growth_max_value", str(args.growth_max_value)]
        if args.growth_tau is not None: cmd += ["--growth_tau", str(args.growth_tau)]
        if args.growth_r is not None:   cmd += ["--growth_r", str(args.growth_r)]
        if args.sigma_dist is not None: cmd += ["--sigma_dist", str(args.sigma_dist)]
        if args.sigma_vol is not None:  cmd += ["--sigma_vol", str(args.sigma_vol)]
        if args.alpha is not None:      cmd += ["--alpha", str(args.alpha)]
        if args.eps is not None:        cmd += ["--eps", str(args.eps)]
        if args.save_maps:              cmd += ["--save_maps"]
        if args.wd_name:                cmd += ["--wd_name", args.wd_name]
        if args.wg_name:                cmd += ["--wg_name", args.wg_name]
        if args.over_name:              cmd += ["--over_name", args.over_name]

        return run_cmd(logger, cmd)

    elif args.mode == "both":
        # QA first (R vs T1)
        t1_path = args.T1 or args.mask
        if args.T1 is None:
            logger.info("No --T1 provided; using --mask as T1: %s", t1_path)
        try:
            qa_masks(logger, args.R, t1_path)
        except Exception:
            logger.exception("QA checks failed (R/T1)")

        # MNI check (if requested) on all inputs we know now
        maybe_mni_check_or_die([
            ("BRAVO", args.bravo), ("FLAIR", args.flair), ("MASK", args.mask),
            ("R", args.R), ("T1", t1_path)
        ])

        if args.check_only:
            logger.info("check_only set — skipping predict.py and score.py invocations.")
            return 0

        # 1) predict
        t2e_out_name = args.t2e_out or "t2e_map.nii.gz"
        pred_G_path = args.G or os.path.join(args.pred_out_dir, t2e_out_name)

        cmd_pred = [sys.executable, "src/predict.py",
                    "--bravo", args.bravo,
                    "--flair", args.flair,
                    "--mask",  args.mask,
                    "--out_dir", args.pred_out_dir,
                    "--t2e_ckpt", args.t2e_ckpt]
        if args.device:              cmd_pred += ["--device", args.device]
        if args.expect_shape:        cmd_pred += ["--expect_shape", *map(str, args.expect_shape)]
        if args.down_factor is not None: cmd_pred += ["--down_factor", str(args.down_factor)]
        if getattr(args, "crop", True) is False: cmd_pred += ["--no-crop"]
        if args.no_zscore:           cmd_pred += ["--no_zscore"]
        if args.scale_mode:          cmd_pred += ["--scale_mode", args.scale_mode]
        if args.t2e_clamp_min is not None: cmd_pred += ["--t2e_clamp_min", str(args.t2e_clamp_min)]
        if args.pad_value is not None:    cmd_pred += ["--pad_value", str(args.pad_value)]
        if args.t2e_out:             cmd_pred += ["--t2e_out", t2e_out_name]

        rc = run_cmd(logger, cmd_pred)
        if rc != 0:
            logger.error("predict.py exited with code %d — aborting.", rc)
            return rc

        # MNI check predicted G (if requested)
        maybe_mni_check_or_die([("G_pred", pred_G_path)])

        # Lightweight consistency check for G vs T1 before scoring
        try:
            G, aff_G, hdr_G = load_nifti(pred_G_path)
            T1, aff_T1, hdr_T1 = load_nifti(t1_path)
            if G.shape != T1.shape:
                logger.warning("Predicted G shape %s differs from T1 %s", G.shape, T1.shape)
            if not np.allclose(aff_G, aff_T1, atol=1e-4):
                logger.warning("Predicted G affine differs from T1 (>1e-4)")
            logger.info("G_pred units: %s → voxel sizes (mm) = (%.3f, %.3f, %.3f)",
                        units_name(hdr_G), *voxel_sizes_mm(aff_G, hdr_G))
        except Exception:
            logger.exception("Failed to read predicted G from %s", pred_G_path)

        score_out_dir = args.score_out_dir or args.pred_out_dir
        if args.score_out_dir is None:
            logger.info("No --score_out_dir provided; using --pred_out_dir as score output dir: %s", score_out_dir)

        # 2) score (use predicted G unless user provided --G override)
        G_path = args.G or pred_G_path
        cmd_score = [sys.executable, "src/score.py",
                     "--R", args.R,
                     "--T1", t1_path,
                     "--G",  G_path,
                     "--out_dir", score_out_dir]
        if args.summary_json:        cmd_score += ["--summary_json", args.summary_json]
        if args.thr_R is not None:   cmd_score += ["--thr_R", str(args.thr_R)]
        if args.thr_T1 is not None:  cmd_score += ["--thr_T1", str(args.thr_T1)]
        if args.growth_mode:         cmd_score += ["--growth_mode", args.growth_mode]
        if args.growth_max_value is not None: cmd_score += ["--growth_max_value", str(args.growth_max_value)]
        if args.growth_tau is not None: cmd_score += ["--growth_tau", str(args.growth_tau)]
        if args.growth_r is not None:   cmd_score += ["--growth_r", str(args.growth_r)]
        if args.sigma_dist is not None: cmd_score += ["--sigma_dist", str(args.sigma_dist)]
        if args.sigma_vol is not None:  cmd_score += ["--sigma_vol", str(args.sigma_vol)]
        if args.alpha is not None:      cmd_score += ["--alpha", str(args.alpha)]
        if args.eps is not None:        cmd_score += ["--eps", str(args.eps)]
        if args.save_maps:              cmd_score += ["--save_maps"]
        if args.wd_name:                cmd_score += ["--wd_name", args.wd_name]
        if args.wg_name:                cmd_score += ["--wg_name", args.wg_name]
        if args.over_name:              cmd_score += ["--over_name", args.over_name]
        if args.Gprob_name:             cmd_score += ["--Gprob_name", args.Gprob_name]

        return run_cmd(logger, cmd_score)

    else:
        parser.error("Unknown mode")
        return 2

if __name__ == "__main__":
    sys.exit(main())
