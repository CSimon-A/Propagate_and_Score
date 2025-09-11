#!/usr/bin/env python3
# src/score.py

# stdlib
import argparse
import json
import logging
import os
import sys

# third-party
import numpy as np

# local
from utils.array_ops import ensure_same_shape
from utils.io_utils import load_nifti, save_nifti, binarize
from utils.logger import setup_logging, log_stats
from utils.metric import normalize_growth, growth_aware_score
import train.config as config 


def parse_args():
    p = argparse.ArgumentParser(
        description="Score a resection against T1 and a growth map (event map)."
    )
    p.add_argument("--R", required=True, help="Resection mask NIfTI (will be binarized).")
    p.add_argument("--T1", required=True, help="Tumor mask at T1 NIfTI (will be binarized).")
    p.add_argument("--G", required=True, help="Growth/event map NIfTI.")

    p.add_argument("--out_dir", required=True, help="Output directory for logs/results.")
    p.add_argument("--summary_json", default="score_summary.json",
                   help="Filename for JSON summary (in out_dir).")

    # Binarization
    p.add_argument("--thr_R", type=float, default=None,
                   help="Binarization threshold for R (default: config.THRESHOLD or 0.01).")
    p.add_argument("--thr_T1", type=float, default=None,
                   help="Binarization threshold for T1 (default: config.THRESHOLD or 0.01).")

    # Growth normalization
    p.add_argument("--growth_mode", choices=["linear", "exp", "geom"],
                   default="linear", help="How to interpret/normalize the growth/event map.")
    p.add_argument("--growth_max_value", type=float, default=5.0,
                   help=("For --growth_mode=linear: sets the k at which the score hits 0. "
                         "Maps k>=0 to 1 - k/MAX (clamped to [0,1]); k<0 → 0. "
                         "Example: MAX=3 gives k=0→1, 1→0.667, 2→0.333, k≥3→0. "
                         "Ignored for 'exp' and 'geom' modes. (default 5.0)."))
    p.add_argument("--growth_tau", type=float, default=2.0,
                   help="For 'exp' mode: decay tau.")
    p.add_argument("--growth_r", type=float, default=None,
                   help="For 'geom' mode: ratio r (default 2/3).")

    # Metric hyperparams
    p.add_argument("--sigma_dist", type=float, default=10.0,
                   help="Distance scale (same units as the distance transform).")
    p.add_argument("--sigma_vol", type=float, default=0.5,
                   help="Over-resection volume scale as a fraction of tumor volume.")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Trade-off between recall (alpha) and sparing (1-alpha).")
    p.add_argument("--eps", type=float, default=1e-8)

    # Debug outputs
    p.add_argument("--save_maps", action="store_true",
                   help="Save wd, wg, and over as NIfTI maps.")
    p.add_argument("--wd_name", default="wd.nii.gz")
    p.add_argument("--wg_name", default="wg.nii.gz")
    p.add_argument("--over_name", default="over_resection.nii.gz")
    p.add_argument("--Gprob_name", default="Gprob.nii.gz")

    # Logging
    p.add_argument("--log_file", default=None,
                   help="Path to a log file. Defaults to <out_dir>/score.log")
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_file = args.log_file or os.path.join(args.out_dir, "score.log")
    setup_logging(log_file, args.log_level)
    logger = logging.getLogger(__name__)

    try:
        logger.info("== Load volumes ==")
        R, aff_R, hdr_R = load_nifti(args.R)
        T1, aff_T1, hdr_T1 = load_nifti(args.T1)
        G, aff_G, hdr_G = load_nifti(args.G)

        ensure_same_shape(R, T1, G)

        # pick a common affine/header for outputs (use T1's)
        affine, header = aff_T1, hdr_T1

        log_stats(logger, "R (raw)", R)
        log_stats(logger, "T1 (raw)", T1)
        log_stats(logger, "G (raw)", G)

        # binarize masks
        thr_default = getattr(config, "THRESHOLD", 0.01)
        thr_R = args.thr_R if args.thr_R is not None else thr_default
        thr_T1 = args.thr_T1 if args.thr_T1 is not None else thr_default

        Rb = binarize(R, thr_R)
        T1b = binarize(T1, thr_T1)

        logger.info("Binarized with thr_R=%.4g, thr_T1=%.4g", thr_R, thr_T1)
        log_stats(logger, "R (bin)",  Rb)
        log_stats(logger, "T1 (bin)", T1b)

        # growth normalization
        Gprob = normalize_growth(
            G, mode=args.growth_mode,
            max_value=args.growth_max_value,
            tau=args.growth_tau,
            r=args.growth_r
        )
        Gprob = np.clip(Gprob, 0.0, 1.0).astype(np.float32)

        out_path = os.path.join(args.out_dir, args.Gprob_name)
        save_nifti(out_path, Gprob, affine, header)

        log_stats(logger, "Gprob (after normalize)", Gprob)
        logger.info("Growth map normalized with mode=%s, max_value=%.4g, tau=%.4g, r=%.4g",
                    args.growth_mode, args.growth_max_value, args.growth_tau,
                    args.growth_r if args.growth_r is not None else 2/3)

        # compute metric
        logger.info("== Compute growth-aware score ==")
        C, Pg, S, wd, wg, over = growth_aware_score(
            R=Rb, T1=T1b, G=Gprob,
            sigma_dist=args.sigma_dist,
            sigma_vol=args.sigma_vol,
            alpha=args.alpha,
            eps=args.eps
            )
        logger.info("Scores: C=%.6f, Pg=%.6f, S=%.6f", C, Pg, S)

        # save summary JSON
        summary = {
            "C_recall": C,
            "Pg_penalty": Pg,
            "S_composite": S,
            "params": {
                "thr_R": thr_R, "thr_T1": thr_T1,
                "growth_mode": args.growth_mode,
                "growth_max_value": args.growth_max_value,
                "growth_tau": args.growth_tau,
                "growth_r": args.growth_r,
                "sigma_dist": args.sigma_dist,
                "sigma_vol": args.sigma_vol,
                "alpha": args.alpha,
                "eps": args.eps
                },
            "inputs": {
                "R": os.path.abspath(args.R),
                "T1": os.path.abspath(args.T1),
                "G": os.path.abspath(args.G),
                },
        }
        summ_path = os.path.join(args.out_dir, args.summary_json)
        with open(summ_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved summary: %s", summ_path)

        # optional debug maps
        if args.save_maps:
            logger.info("== Save debug maps ==")
            save_nifti(os.path.join(args.out_dir, args.wd_name), wd.astype(np.float32), affine, header)
            save_nifti(os.path.join(args.out_dir, args.wg_name), wg.astype(np.float32), affine, header)
            save_nifti(os.path.join(args.out_dir, args.over_name), over.astype(np.uint8),  affine, header)
            logger.info("Saved wd, wg, over maps.")

        logger.info("Done.")
        return 0

    except SystemExit as e:
        logger = logging.getLogger(__name__)
        if e.code not in (0, None):
            logger.error("SystemExit with code=%s", e.code)
        raise
    except Exception:
        logging.getLogger(__name__).exception("Fatal error during scoring")
        raise

if __name__ == "__main__":
    sys.exit(main())
