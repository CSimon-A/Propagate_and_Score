# propagate_and_score/src/utils/logging.py

import os, sys, logging, warnings
from typing import Optional
import numpy as np

def setup_logging(log_file: Optional[str], level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(lvl)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(lvl)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logging.captureWarnings(True)
    warnings.simplefilter("always")
    np.seterr(all="warn")

    def _excepthook(exctype, value, tb):
        if exctype not in (SystemExit, KeyboardInterrupt):
            logging.getLogger(__name__).exception("Uncaught exception", exc_info=(exctype, value, tb))
        sys.__excepthook__(exctype, value, tb)
    sys.excepthook = _excepthook

def log_stats(logger, name, vol):
    logger.info("  %s: shape=%s, dtype=%s, min=%.4g, max=%.4g, mean=%.4g",
                name, tuple(vol.shape), vol.dtype, np.nanmin(vol), np.nanmax(vol), np.nanmean(vol))

def assert_finite_or_log(logger, name, vol):
    if not np.isfinite(vol).all():
        logger.error("%s contains non-finite values (NaN/Inf); continuing but results may be invalid.", name)

def divisible_or_err(logger, expect_shape, down_factor):
    ok = all((d % down_factor) == 0 for d in expect_shape)
    if not ok:
        logger.error(
            "--expect_shape %s is not divisible by --down_factor %d. "
            "This may cause misalignment compared to training.",
            tuple(expect_shape), down_factor
        )
