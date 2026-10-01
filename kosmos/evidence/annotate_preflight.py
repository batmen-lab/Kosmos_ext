"""Optionally expand an evidence config with annotation companions.

Kosmos and AutoEvidence stay SEPARATE here. This shells out to the
`autoevidence annotate` CLI exactly as an evidence config's own `server:` lines
shell out to `autoevidence-serve` -- a process boundary, not a library import.
Neither package imports the other's internals through this path, so they remain
independent: if AutoEvidence is absent, or the step finds nothing, or it fails,
the run simply proceeds on the ORIGINAL config. The input file is never mutated.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _find_cli(evidence_config: Path) -> Optional[str]:
    """Locate the `autoevidence` CLI without importing it.

    Order: an explicit override; then PATH; then next to the `autoevidence-serve`
    binary the config already names -- because the config itself declares which
    AutoEvidence install to use, and `autoevidence` sits beside `-serve`.
    """
    env = os.environ.get("KOSMOS_AUTOEVIDENCE_CLI")
    if env and Path(env).exists():
        return env
    found = shutil.which("autoevidence")
    if found:
        return found
    # The autoevidence CLI that pairs with this Kosmos install sits in the SAME
    # virtualenv bin as the running interpreter. This is the reliable locator for
    # a gateway-free config, which carries no `autoevidence-serve` path to derive
    # one from, and works even when the venv bin is not on PATH (running
    # `venv/bin/kosmos` directly does not add it).
    import sys
    sibling = Path(sys.executable).with_name("autoevidence")
    if sibling.exists():
        return str(sibling)
    try:
        text = Path(evidence_config).read_text()
        m = re.search(r"([\w./~+-]*)autoevidence-serve", text)  # path chars only; no quotes
        if m:
            candidate = m.group(1) + "autoevidence"
            if Path(candidate).exists():
                return candidate
    except Exception:
        pass
    return None


def maybe_expand(
    evidence_config,
    annotation_build: Optional[str] = None,
    timeout: int = 180,
) -> Path:
    """Return an annotation-expanded evidence config, or the original unchanged.

    Never raises, never mutates the input (writes a temp copy). Disable entirely
    with KOSMOS_AUTO_ANNOTATE=0.
    """
    cfg = Path(evidence_config)
    if os.environ.get("KOSMOS_AUTO_ANNOTATE", "1") == "0":
        return cfg
    if not cfg.is_file():
        return cfg
    cli = _find_cli(cfg)
    if not cli:
        logger.info("auto-annotate: autoevidence CLI not found; using original config")
        return cfg

    out = Path(tempfile.mkdtemp(prefix="kosmos-ev-")) / f"{cfg.stem}.expanded.yaml"
    cmd = [cli, "annotate", "--evidence-config", str(cfg), "--out", str(out)]
    if annotation_build:
        cmd += ["--build", annotation_build]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as e:  # subprocess missing, timeout, etc.
        logger.warning("auto-annotate: %s; using original config", e)
        return cfg
    if result.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        msg = (result.stderr or result.stdout or "").strip()[:140]
        logger.info("auto-annotate: no expansion (%s); using original config", msg)
        return cfg
    logger.info("auto-annotate: expanded evidence config -> %s", out)
    return out
