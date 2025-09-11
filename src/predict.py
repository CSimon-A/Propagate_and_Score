#!/usr/bin/env python3
# src/predict.py

# # stdlib
import argparse
import logging
import os
import sys

# third-party
import numpy as np
import torch
from scipy.ndimage import zoom

# local
from utils.array_ops import ensure_same_shape
from utils.geom import uncrop_to_full_ds
from utils.io_utils import (
    load_nifti,
    save_nifti,
    resize_to_shape,
    binarize_mask,
    prep_modalities,
)
from utils.logger import (
    setup_logging,
    log_stats,
    assert_finite_or_log,
    divisible_or_err,
)
from utils.model import load_unet
from utils.propagate import t2e_predict_map
from train.utils.cropping import crop_to_fixed_bbox
from train.utils.downsample import downsample as ds
import train.config as config


def parse_args():
    p = argparse.ArgumentParser(
        description='Build a T2E event map by mirroring training: '
                    'downsample -> (optional) crop -> forward -> upsample.'
    )
    p.add_argument('--bravo', required=True)
    p.add_argument('--flair', required=True)
    p.add_argument('--mask',  required=True, help='Initial tumor mask at t0 (will be binarized).')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')

    # Training box controls
    p.add_argument('--expect_shape', type=int, nargs=3, default=[192, 192, 192],
                   help='Training box at full-res BEFORE any downsampling (D H W).')
    p.add_argument('--down_factor', type=int, default=2,
                   help='If you trained with downsample=2, set --down_factor 2 (model input ~ 96^3).')

    # Cropping toggle (default ON; add --no-crop to disable)
    p.add_argument('--no-crop', dest='crop', action='store_false',
                   help='Disable cropping to the training box after downsampling (default: on).')

    # Preprocessing
    p.add_argument('--no_zscore', action='store_true', help='Disable z-score on BRAVO/FLAIR.')
    p.add_argument('--scale_mode', choices=['none', 'min-max', 'tanh', 'sigmoid'], default='min-max')

    # T2E
    p.add_argument('--t2e_ckpt', default=None, help='Path to T2E UNet checkpoint (required).')
    p.add_argument('--t2e_clamp_min', type=float, default=-1.0,
                   help='Clamp lower bound of the T2E map (default: -1).')
    p.add_argument('--pad_value', type=float, default=None,
                   help='Value to fill outside the crop when undoing crop on ds grid; '
                        'default: uses --t2e_clamp_min.')
    p.add_argument('--t2e_out', default='t2e_map.nii.gz',
                   help='Output filename for saved T2E map (in out_dir).')
    p.add_argument('--model', default='unet3d',
                   choices=['unet3d_larger_skip', 'unet3d'],
                   help='Model architecture to use for T2E prediction (default: unet3d).')

    # Logging
    p.add_argument('--log_file', default=None,
                   help='Path to a log file. Defaults to <out_dir>/predict.log')
    p.add_argument('--log_level', default='INFO',
                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                   help='Console log level (file always records DEBUG).')

    return p.parse_args()

def main():
    #TODO: Add a check for Python version if needed
    # e.g., if sys.version_info != (3, 10): raise RuntimeError("Python 3.10 required")
    # implement metric (fix if resection 1 mm over, still concidered 100% resection good)
    # work on preproc, if espace crush, get to mni
    # Check mask if outline or volume (if volume, mask or map?)
    # check all inputs for any weird things

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    log_file = args.log_file or os.path.join(args.out_dir, "predict.log")
    setup_logging(log_file, args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting T2E inference")
    logger.debug("Args: %s", vars(args))

    pad_value = args.t2e_clamp_min if args.pad_value is None else args.pad_value

    try:
        logger.info("== Load volumes ==")
        bravo, aff_b, hdr_b = load_nifti(args.bravo)
        flair, aff_f, hdr_f = load_nifti(args.flair)
        mask0,  aff_m, hdr_m = load_nifti(args.mask)
        ensure_same_shape(bravo, flair, mask0)
        affine, header = aff_m, hdr_m

        assert_finite_or_log(logger, "BRAVO", bravo); assert_finite_or_log(logger, "FLAIR", flair); assert_finite_or_log(logger, "MASK", mask0)
        log_stats(logger, "BRAVO (raw)", bravo)
        log_stats(logger, "FLAIR (raw)", flair)
        log_stats(logger, "MASK  (raw)", mask0)

        # Binarize mask (robust to soft masks)
        mask0 = binarize_mask(mask0)
        logger.info("Binarized tumor file with threshold=%.4g", getattr(config, "THRESHOLD", 0.01))
        tumor_voxels = int(mask0.sum())
        if tumor_voxels == 0:
            logger.warning("Input mask has 0 tumor voxels. Crop will center on image center.")
        else:
            logger.info("Mask tumor voxels: %d", tumor_voxels)

        logger.info("== Preprocess intensities (full-res) ==")
        bravo, flair = prep_modalities(bravo, flair, do_z=(not args.no_zscore), scale_mode=args.scale_mode)
        log_stats(logger, "BRAVO (preproc)", bravo)
        log_stats(logger, "FLAIR (preproc)", flair)

        logger.info("== Compute target sizes ==")
        expect_shape = tuple(int(x) for x in args.expect_shape)
        logger.info("expect_shape (full-res training box): %s", expect_shape)

        if args.down_factor > 1:
            divisible_or_err(logger, expect_shape, args.down_factor)
            target_ds = tuple(s // args.down_factor for s in expect_shape)
            logger.info("down_factor: %d → target_ds (model/crop grid): %s", args.down_factor, target_ds)

            logger.info("== Downsample to training grid (match training) ==")
            b_ds, f_ds = ds(bravo, flair, None, None, factor=args.down_factor, order=1)[:2]
            m_ds       = ds(mask0, None,  None, None,  factor=args.down_factor, order=0)[0]

            assert_finite_or_log(logger, "BRAVO_ds", b_ds); assert_finite_or_log(logger, "FLAIR_ds", f_ds); assert_finite_or_log(logger, "MASK_ds", m_ds)
            log_stats(logger, "BRAVO_ds", b_ds)
            log_stats(logger, "FLAIR_ds", f_ds)
            log_stats(logger, "MASK_ds", m_ds)

        else:
            target_ds = expect_shape
            logger.info("down_factor: %d (no downsampling) → target_ds = expect_shape = %s",
                args.down_factor, target_ds)
            logger.info("== No downsample; using full-res as model/crop grid ==")
            b_ds, f_ds, m_ds = bravo, flair, mask0

        # quick sanity: can we crop to target_ds on this grid?
        ds_full_shape = tuple(b_ds.shape)
        if any(t > s for t, s in zip(target_ds, ds_full_shape)):
            logger.error(
                "target_ds %s is larger than available grid %s. "
                "Check --expect_shape and --down_factor (and/or crop settings). "
                "Cropping may not behave as expected.",
                target_ds, ds_full_shape
                )

        # Defaults if no crop
        start = np.array([0, 0, 0], dtype=int)
        end   = np.array(ds_full_shape, dtype=int)

        if args.crop:
            logger.info("== Crop on the ds grid ==")
            (b_c, f_c, m_c, _), start_pad, end_pad, _, orig_start = crop_to_fixed_bbox(
                [b_ds, f_ds, m_ds, m_ds], target_ds, affine=None
                )
            # ensure ints for safety
            start_pad = np.asarray(start_pad, dtype=int)
            end_pad   = np.asarray(end_pad,   dtype=int)
            orig_start = np.asarray(orig_start, dtype=int)
            orig_end   = orig_start + np.array(target_ds, int)

            crop_size = tuple((end_pad - start_pad).astype(int))
            logger.info("Crop window (padded ds frame): start=%s, end=%s, size=%s",
                        start_pad.tolist(), end_pad.tolist(), crop_size)
            logger.info("Crop window (ORIGINAL ds frame): start=%s, end=%s",
                        orig_start.tolist(), orig_end.tolist())

            # warn if the crop window exceeds the ds grid bounds
            if (end > np.array(ds_full_shape)).any() or (start < 0).any():
                logger.warning("Crop window exceeds ds grid; will clip on uncrop. "
                            "ds_full_shape=%s, start=%s, end=%s",
                            ds_full_shape, start.tolist(), end.tolist())

            if not (tuple(b_c.shape) == target_ds and tuple(f_c.shape) == target_ds and tuple(m_c.shape) == target_ds):
                logger.error("Cropped shapes %s, %s, %s != target_ds %s. Continuing, but outputs may be misaligned.",
                            b_c.shape, f_c.shape, m_c.shape, target_ds)

            in_b, in_f, in_m = b_c.astype(np.float32), f_c.astype(np.float32), m_c.astype(np.float32)
        else:
            logger.info("== No crop selected ==")
            in_b, in_f, in_m = b_ds.astype(np.float32), f_ds.astype(np.float32), m_ds.astype(np.float32)


        logger.info("== Load model & run forward ==")
        if not args.t2e_ckpt:
            logger.error('--t2e_ckpt is required for T2E prediction.')
            raise SystemExit(2)
        logger.info("Loading checkpoint: %s", args.t2e_ckpt)
        model = load_unet(args.t2e_ckpt, args.model, device=args.device, in_ch=3, out_ch=1)

        logger.info("Predicting T2E map ...")
        tmap_ds = t2e_predict_map(in_b, in_f, in_m, model=model, device=args.device,
                                  clamp_min=args.t2e_clamp_min)
        assert_finite_or_log(logger, "T2E_map_ds", tmap_ds)
        log_stats(logger, "T2E_map_ds", tmap_ds)

        # === Undo crop on the ds grid BEFORE upsampling ===
        if args.crop:
            logger.info("== Undo crop on grid ==")
            tmap_ds_full = uncrop_to_full_ds(
                cropped=tmap_ds,
                ds_full_shape=ds_full_shape,
                start=orig_start,
                end=orig_end,
                fill_value=float(pad_value),
                )
            log_stats(logger, "T2E_map_ds_full (padded)", tmap_ds_full)
        else:
            tmap_ds_full = tmap_ds  # already full size

        if args.down_factor > 1:
            logger.info("== Upsample prediction back to full-res grid & save ==")
            tmap_save = resize_to_shape(tmap_ds_full, bravo.shape, mode='trilinear')
        else:
            logger.info("== No upsampling needed (full-res output) ==")
            tmap_save = tmap_ds_full

        # Final sanity: shape should match original full-res input
        if tmap_save.shape != bravo.shape:
            msg = (f"Upsampled map shape {tmap_save.shape} != input shape {bravo.shape}. "
                   f"Check --expect_shape and --down_factor.")
            logger.error(msg)
            raise RuntimeError(msg)

        out_path = os.path.join(args.out_dir, args.t2e_out)
        save_nifti(out_path, tmap_save, affine, header)
        log_stats(logger, "T2E_map_fullres (saved)", tmap_save)
        logger.info("Saved: %s", out_path)
        logger.info("Done.")
        return 0

    except SystemExit as e:
        # Already a controlled exit; ensure it’s visible in logs
        logger = logging.getLogger(__name__)
        if e.code not in (0, None):
            logger.error("SystemExit with code=%s", e.code)
        raise
    except Exception:
        logging.getLogger(__name__).exception("Fatal error during T2E inference")
        raise

if __name__ == '__main__':
    sys.exit(main())
