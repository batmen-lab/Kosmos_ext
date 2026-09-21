"""``https://`` -- any reachable file, streamed to the staging directory.

The escape hatch: a repository this package has no adapter for is still usable
when a human has the direct URL. The name on disk comes from the last path
segment, so the file is recognisable months later.
"""

from __future__ import annotations

from urllib.parse import unquote, urlparse

from ..config import DataFetcherConfig
from ..errors import ReferenceError
from ..references import Reference
from ..store import stage_dir
from .base import Resolved, name_for, register
from .net import download


class HttpSource:
    scheme = "http"

    def fetch(self, ref: Reference, config: DataFetcherConfig) -> Resolved:
        parsed = urlparse(f"{ref.scheme}://{ref.locator}")
        filename = unquote(parsed.path.rsplit("/", 1)[-1]) or "download.bin"
        if ref.selector:
            # A selector on a URL is a mistake worth naming: the URL already
            # names one resource.
            raise ReferenceError(
                f"{ref}: a plain URL takes no #selector; the path already names "
                f"the file"
            )
        url = f"{ref.scheme}://{ref.locator}"
        # http and https share one directory: "where did this come from" is one
        # question, and splitting them would make `list --scheme http` lie.
        directory = stage_dir(config, "http", name_for(ref))
        record = download(url, directory / filename, config)
        return Resolved(
            scheme=self.scheme,
            locator=ref.locator,
            directory=directory,
            files=[record],
            notes=[f"direct download from {url}"],
        )


register(HttpSource())


class HttpsSource(HttpSource):
    scheme = "https"


register(HttpsSource())
