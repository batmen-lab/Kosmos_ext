"""Exception types, one per layer, so a caller can tell what failed.

A search that found nothing and a search whose payload was unrecognised are
different facts. Conflating them is how "the repository has no such dataset"
gets reported for an upstream change, so the adapters raise and the caller
decides -- they do not silently return an empty list.
"""


class DataFetcherError(Exception):
    """Base class for everything this package raises."""


class ReferenceError(DataFetcherError):
    """A reference string is malformed or names an unsupported scheme."""


class SearchError(DataFetcherError):
    """A repository search failed, or its response was not recognised."""


class SourceError(DataFetcherError):
    """A reference could not be resolved to bytes."""


class FetchError(DataFetcherError):
    """Staging the resolved bytes failed."""
