# propagate_and_score/src/utils/subprocess.py

import logging
import shlex
import subprocess
from typing import Sequence

def run_cmd(logger: logging.Logger, args: Sequence[str]) -> int:
    cmd_str = " ".join(shlex.quote(a) for a in args)
    logger.info("→ %s", cmd_str)
    return subprocess.call(args)
