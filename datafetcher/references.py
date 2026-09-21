"""Reference strings: ``scheme://locator#selector`` with an optional ``@revision``.

Opaque URLs are not accepted as a scheme. A reference is parsed into its parts
and then *validated against the scheme's own grammar* before anything touches
the network, because the locator is the one string here that ends up in a path
on disk and in a URL -- it is not merely displayed.

    geo://GSE55296                      a GEO series
    geo://GSE55296#suppl/counts.txt.gz  one supplementary file from it
    hf://owner/name                     a HuggingFace dataset repository
    hf://owner/name@<commit>#file.csv   one file, pinned to a revision
    https://example.org/data.csv        any reachable file
    file:///abs/path/data.csv           a file already on this machine
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from .errors import ReferenceError

#: Control characters, backslashes and NULs are never legitimate here.
_UNSAFE = re.compile(r"[\x00-\x1f\\]")

SUPPORTED_SCHEMES = ("geo", "hf", "tu", "http", "https", "file")


@dataclass(frozen=True)
class Reference:
    scheme: str
    locator: str
    selector: str | None = None
    revision: str | None = None

    def __str__(self) -> str:
        if self.scheme in {"http", "https"}:
            text = f"{self.scheme}://{self.locator}"
        elif self.scheme == "file":
            text = f"file://{self.locator}"
        else:
            text = f"{self.scheme}://{self.locator}"
        if self.revision:
            text = f"{text}@{self.revision}"
        if self.selector:
            text = f"{text}#{self.selector}"
        return text

    @property
    def safe_locator(self) -> str:
        """The locator as a single path component, safe to put on disk.

        Raises rather than sanitising: a rewritten accession is an accession
        this package invented for a dataset that does not have it.
        """
        segments = self.locator.split("/")
        if (
            _UNSAFE.search(self.locator)
            or self.locator in {"", ".", ".."}
            or any(segment in {"", ".", ".."} for segment in segments)
        ):
            raise ReferenceError(
                f"reference locator {self.locator!r} is not usable as a "
                f"directory name; it contains a path separator or traversal"
            )
        return self.locator.replace("/", "__").replace(":", "_")


def parse_reference(reference: str) -> Reference:
    """Parse and validate a reference string."""
    text = (reference or "").strip()
    if not text:
        raise ReferenceError("empty reference")
    if "://" not in text:
        raise ReferenceError(
            f"{text!r} has no scheme. Use one of: "
            f"{', '.join(s + '://' for s in SUPPORTED_SCHEMES)}"
        )
    scheme, rest = text.split("://", 1)
    scheme = scheme.lower()
    if scheme not in SUPPORTED_SCHEMES:
        raise ReferenceError(
            f"unsupported scheme {scheme!r}; this package knows "
            f"{', '.join(SUPPORTED_SCHEMES)}"
        )

    selector: str | None = None
    if "#" in rest:
        rest, selector = rest.split("#", 1)
        selector = selector.strip() or None
        if selector and _UNSAFE.search(selector.replace("/", "").replace("\\", "")):
            raise ReferenceError(f"selector {selector!r} contains control characters")
        if selector and (".." in selector.split("/")):
            raise ReferenceError(f"selector {selector!r} contains a path traversal")

    revision: str | None = None
    # Only schemes with a revision concept take `@`. A URL may legitimately
    # contain one (`user@host`), and `geo://GSE1@r2` has to reach the GEO
    # connector to be refused with its own explanation of why.
    if scheme in {"hf", "geo"} and "@" in rest:
        rest, revision = rest.split("@", 1)
        revision = revision.strip() or None

    locator = rest.strip()
    if not locator:
        raise ReferenceError(f"{scheme}:// reference names nothing")
    if scheme in {"http", "https"}:
        parsed = urlparse(f"{scheme}://{locator}")
        if not parsed.netloc:
            raise ReferenceError(f"{text!r} has no host")
    if scheme == "file":
        if not locator.startswith("/"):
            raise ReferenceError(
                f"file:// expects an absolute path, got {locator!r}"
            )
    return Reference(scheme=scheme, locator=locator, selector=selector, revision=revision)
