"""HTTP helpers shared by the connectors. Standard library only.

Two properties, both learned from AutoEvidence's GEO connector:

  * read with a **cap**, not with `Content-Length`, because a chunked response
    may not send one; and
  * report the *remedy* in the error, not just the status code -- an HTTP 404
    from a GEO path usually means "that accession has no file of that name",
    which is a different fact from "GEO is down".
"""

from __future__ import annotations

import ssl
import urllib.error
import urllib.request
from pathlib import Path

from ..config import DataFetcherConfig
from ..errors import SourceError
from ..models import FileRecord
from ..store import sha256_file


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001 - fall back to the system store
        return ssl.create_default_context()


def _request(
    url: str, config: DataFetcherConfig, method: str = "GET"
) -> urllib.request.Request:
    return urllib.request.Request(
        url, headers={"User-Agent": config.user_agent}, method=method
    )


def _open(url: str, config: DataFetcherConfig, method: str = "GET"):
    """The single place a request is issued.

    One call site, so a test can replace it and assert the exact URL a
    connector builds without a socket or a server.
    """
    return urllib.request.urlopen(
        _request(url, config, method=method),
        timeout=config.timeout_s,
        context=_ssl_context(),
    )


def head_size(url: str, config: DataFetcherConfig) -> int | None:
    """How many bytes `url` is, from the server's own headers, or None.

    A size is what lets a run decide *not* to download something: two files of
    the same series differ by 5x, and only the header says so before the bytes
    move. Servers that refuse HEAD are not a problem -- the answer is None.
    """
    _check_offline(config, url)
    try:
        with _open(url, config, "HEAD") as response:
            length = response.headers.get("Content-Length")
    except Exception:  # noqa: BLE001 - a size is an optimisation, not a fact
        return None
    try:
        return int(length) if length else None
    except (TypeError, ValueError):
        return None


def _check_offline(config: DataFetcherConfig, url: str) -> None:
    if config.offline:
        raise SourceError(
            f"offline mode is on, so {url} was not requested. Re-run without "
            f"KOSMOS_DATAFETCHER_OFFLINE / --offline to reach the network."
        )


def get_bytes(url: str, config: DataFetcherConfig, *, limit: int | None = None) -> bytes:
    """Fetch a small resource (an index listing) with the same bound."""
    blob = get_prefix(url, config, limit=limit)
    cap = config.max_bytes if limit is None else min(config.max_bytes, limit)
    if len(blob) > cap:
        raise SourceError(
            f"{url} is larger than the {cap:,}-byte cap; raise "
            f"--max-bytes or fetch a smaller file"
        )
    return blob


def get_prefix(url: str, config: DataFetcherConfig, *, limit: int | None = None) -> bytes:
    """The first `limit` bytes of a resource, whether or not there are more.

    A description -- a series summary, a dataset card -- is worth having from a
    resource that is far larger than the description: read the prefix and stop,
    rather than refusing to describe it.
    """
    _check_offline(config, url)
    cap = config.max_bytes if limit is None else min(config.max_bytes, limit)
    try:
        with _open(url, config) as response:
            blob = response.read(cap + 1)
    except urllib.error.HTTPError as e:
        raise SourceError(f"{url} returned HTTP {e.code}") from e
    except Exception as e:  # noqa: BLE001 - offline DNS, TLS, timeouts
        raise SourceError(f"could not fetch {url}: {e}") from e
    return blob


def download(url: str, dest: Path, config: DataFetcherConfig) -> FileRecord:
    """Stream a URL to `dest`, bounded by the size cap, hashed as it lands."""
    _check_offline(config, url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    written = 0
    try:
        with _open(url, config) as response, partial.open("wb") as handle:
            first = True
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                if first:
                    first = False
                    page = looks_like_a_web_page(block, _content_type(response))
                    if page:
                        raise SourceError(
                            f"{url} answered with a web page ({page}), not a data "
                            f"file. A moved or renamed dataset path usually still "
                            f"returns a 200 with the site's own 'not found' page, "
                            f"so check the path -- or, if the dataset really is "
                            f"published as an HTML table, convert it to CSV first."
                        )
                written += len(block)
                if written > config.max_bytes:
                    raise SourceError(
                        f"{url} exceeds the {config.max_bytes:,}-byte cap; raise "
                        f"--max-bytes or fetch a smaller file"
                    )
                handle.write(block)
    except urllib.error.HTTPError as e:
        partial.unlink(missing_ok=True)
        raise SourceError(f"{url} returned HTTP {e.code}") from e
    except SourceError:
        partial.unlink(missing_ok=True)
        raise
    except Exception as e:  # noqa: BLE001
        partial.unlink(missing_ok=True)
        raise SourceError(f"could not download {url}: {e}") from e
    partial.replace(dest)
    return FileRecord(
        path=dest.name, bytes=written, sha256=sha256_file(dest), source_url=url
    )


def _content_type(response) -> str:
    headers = getattr(response, "headers", None)
    if headers is None:
        return ""
    try:
        return str(headers.get("Content-Type") or "")
    except AttributeError:  # a fake response in a test
        return ""


#: The first bytes of an HTML document, as a browser's server would send them.
_PAGE_STARTS = (b"<!doctype html", b"<html", b"<head")


def looks_like_a_web_page(block: bytes, content_type: str = "") -> str:
    """The page's title when these bytes are a web page, else "".

    A dataset URL that has moved usually still answers `200 OK` with the site's
    "page not found" document -- a *soft* 404. The file is then a valid HTML
    page, the download succeeds, and the failure only surfaces much later as
    "this is not a table", which reads like a problem with the data rather than
    with the URL. Catching it here means the refusal is about the identifier,
    and the retrieval step can correct it.
    """
    if "text/html" in (content_type or "").lower():
        return _page_title(block) or "an HTML document"
    head = block[:2048].lstrip().lower()
    if not any(head.startswith(marker) for marker in _PAGE_STARTS):
        return ""
    return _page_title(block) or "an HTML document"


def _page_title(block: bytes) -> str:
    import re

    match = re.search(rb"<title[^>]*>(.*?)</title>", block[:8192], re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    title = " ".join(match.group(1).decode("utf-8", "replace").split())
    return title[:120]
