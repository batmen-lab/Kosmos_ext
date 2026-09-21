"""One connector per reference scheme, resolved through a registry.

Borrowed from AutoEvidence's `sources/`: a scheme is registered by its own
module, so adding one is a new file and one `register(...)` call rather than an
edit to a dispatcher. Unlike upstream there is no credential layer and no
classification step -- every source here is public by construction.
"""

from __future__ import annotations

# Import for the side effect of registering each scheme.
from . import geo as _geo  # noqa: E402,F401
from . import hf as _hf  # noqa: E402,F401
from . import http_source as _http  # noqa: E402,F401
from . import local as _local  # noqa: E402,F401
from . import tooluniverse as _tooluniverse  # noqa: E402,F401
from .base import SOURCES, Resolved, Source, name_for, register, source_for

__all__ = ["SOURCES", "Resolved", "Source", "name_for", "register", "source_for"]
