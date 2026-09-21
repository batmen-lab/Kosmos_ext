"""``file://`` -- data already on this machine, copied into the staging root.

Copied rather than referenced in place, so every fetch result has the same
shape: one directory, one manifest, a set of artifacts. A dataset that is
merely pointed at cannot be re-checked later, and a path outside the root can
change under a run that already recorded it.

The download cap is deliberately *not* the one applied here: nothing is being
downloaded, and a 3 GB local `.h5ad` is a normal thing to stage. `local_max_bytes`
is the bound that matters -- it stops a `file:///dev/sda`-shaped mistake from
filling the disk.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..config import DataFetcherConfig
from ..errors import SourceError
from ..models import FileRecord
from ..references import Reference
from ..store import sha256_file, stage_dir
from .base import Resolved, name_for, register


class LocalSource:
    scheme = "file"

    def fetch(self, ref: Reference, config: DataFetcherConfig) -> Resolved:
        source = Path(ref.locator)
        if not source.exists():
            raise SourceError(f"{source} does not exist")
        if source.is_dir():
            raise SourceError(
                f"{source} is a directory; file:// takes one file. Archive it "
                f"first, or fetch the specific file you need."
            )
        size = source.stat().st_size
        if size > config.local_max_bytes:
            raise SourceError(
                f"{source} is {size:,} bytes, over the "
                f"{config.local_max_bytes:,}-byte copy cap; a file this size is "
                f"usually a mistake (a device node, a disk image)"
            )
        directory = stage_dir(config, self.scheme, name_for(ref))
        target = directory / source.name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        return Resolved(
            scheme=self.scheme,
            locator=ref.locator,
            directory=directory,
            files=[
                FileRecord(
                    path=target.name,
                    bytes=size,
                    sha256=sha256_file(target),
                    source_url=source.as_uri(),
                )
            ],
            notes=[f"copied from {source}"],
        )


register(LocalSource())
