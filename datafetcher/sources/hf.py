"""``hf://`` -- a HuggingFace dataset repository, pinned to a commit.

Two things this connector does that the Hub's own CLI does not do by default,
both because a fetch result has to be checkable afterwards:

  * it resolves the requested revision to a **commit sha** and records it, so
    "which revision of this repository" is a fact in the manifest rather than a
    branch name that moved; and
  * it reads the repository's file listing **before** downloading, so the size
    cap is enforced against the whole selection instead of discovered halfway
    through a multi-gigabyte snapshot.

`huggingface_hub` is imported inside the method: this package has no required
network dependency, and a deployment that only ever fetches GEO should not need
one installed.
"""

from __future__ import annotations

import re

from ..config import DataFetcherConfig
from ..errors import SourceError
from ..models import FileRecord
from ..references import Reference
from ..store import sha256_file, stage_dir
from .base import Resolved, name_for, register

#: The Hub's own repository-id grammar, and nothing outside it: the id becomes
#: both a directory name and an API path here.
_REPO_ID = re.compile(r"^[\w.-]+/[\w.-]+$")
#: Suffixes that hold rows rather than documentation or weights.
TABLE_SUFFIXES = (".csv", ".tsv", ".parquet", ".arrow", ".jsonl", ".h5ad", ".mtx")


def looks_like_data(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(TABLE_SUFFIXES) or any(
        part in lowered for part in ("_matrix_", "_barcodes_", "_genes_")
    )


def _listing(sizes: dict[str, int], most: int = 8) -> str:
    """`a.parquet (1.2 MB), b.parquet (3.4 MB), ...` for an error message."""
    ordered = sorted(sizes.items(), key=lambda pair: pair[1])
    shown = ", ".join(f"{name} ({size:,} bytes)" for name, size in ordered[:most])
    if len(ordered) > most:
        shown += f", ... ({len(ordered) - most} more)"
    return shown or "listed as empty"


def _flatten(value) -> str:
    """A card field (`["biology"]`, `"biology"`, None) as one short string."""
    if not value:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in list(value)[:8])
    return str(value)


def _bytes(size: int) -> str:
    for unit, scale in (("GB", 1024**3), ("MB", 1024**2), ("kB", 1024)):
        if size >= scale:
            return f"{size / scale:.1f} {unit}"
    return f"{size} bytes"


def choose_bounded(files: dict[str, int], cap: int) -> tuple[list[str], str]:
    """Which files of a repository to take when the caller named none.

    Taking the whole repository is how two unrelated multi-gigabyte datasets
    arrived: the size cap was checked against the total, so a 1.5 GB repository
    passed it and every file in it came down. A bounded read wants one table --
    the smallest one that is data rather than documentation -- and the listing
    already says which that is. A file that says `train` is preferred over a
    smaller `test`: the split that carries the rows to learn from is the one a
    training run is asking for.
    """
    if not files:
        return [], "the repository lists no files"
    data = {name: size for name, size in files.items() if looks_like_data(name)}
    pool = data or files
    training = {name: size for name, size in pool.items() if "train" in name.lower()}
    pick_from = training or pool
    smallest = min(pick_from, key=lambda name: pick_from[name])
    if smallest in pool and pool[smallest] > cap:
        return [], (
            f"its smallest data file {smallest} is {pool[smallest]:,} bytes, over "
            f"the {cap:,}-byte cap"
        )
    return [smallest], (
        f"no file was named, so the smallest data file was taken "
        f"({smallest}, {pick_from[smallest]:,} bytes, out of {len(files)} file(s)); "
        f"name another as hf://<repo>#<path>"
    )


def repo_id_of(locator: str) -> str:
    """The repository id, accepting the prefix people copy from a Hub URL.

    A Hub page reads `huggingface.co/datasets/owner/name`, and a model that has
    seen one writes `hf://datasets/owner/name`. That is a URL fragment, not a
    different repository.
    """
    text = locator.strip().strip("/")
    for prefix in ("datasets/", "dataset/"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break
    return text


def _hub_error(repo_id: str, error: Exception) -> str:
    """Say what an unauthenticated Hub failure usually means.

    The Hub answers a missing repository, a private one and a gated one the
    same way when no token is presented ("see the authentication docs"), so the
    raw text sends a reader looking for a credentials problem in the wrong
    place -- often the repository simply is not called that.
    """
    text = str(error)
    if any(word in text.lower() for word in ("401", "unauthorized", "authentication", "token")):
        return (
            f"could not read {repo_id} from the Hub: {text}. An unauthenticated "
            f"failure looks the same for a repository that does not exist, one "
            f"that is private, and one that is gated: check the name, and set "
            f"HF_TOKEN (or run `huggingface-cli login`) if you have access"
        )
    return f"could not read {repo_id} from the Hub: {text}"


class HuggingFaceSource:
    scheme = "hf"

    def listing(self, ref: Reference, config: DataFetcherConfig) -> list[dict]:
        """The repository's files, without downloading any of them.

        A repository usually holds several tables (`train.csv`, `test.csv`),
        and after the labeled one is known those siblings are the most likely
        source of evidence: same columns, different rows.
        """
        repo_id = repo_id_of(ref.locator)
        if not _REPO_ID.match(repo_id) or ".." in repo_id:
            raise SourceError(
                f"hf:// expects a repository id like hf://owner/name; got "
                f"{ref.locator!r}"
            )
        try:
            from huggingface_hub import HfApi
        except ImportError as e:  # pragma: no cover - depends on the install
            raise SourceError("hf:// needs huggingface_hub: pip install huggingface_hub") from e
        try:
            info = HfApi().dataset_info(repo_id, revision=ref.revision, files_metadata=True)
        except Exception as e:  # noqa: BLE001 - the Hub raises its own types
            raise SourceError(_hub_error(repo_id, e)) from e
        return [
            {
                "path": sibling.rfilename,
                "bytes": sibling.size or 0,
                "size": _bytes(sibling.size or 0),
                # The exact string to name this one file, so a caller that has
                # read the listing does not have to assemble it itself.
                "reference": f"hf://{repo_id}#{sibling.rfilename}",
            }
            for sibling in (getattr(info, "siblings", None) or [])
        ]

    def fetch(self, ref: Reference, config: DataFetcherConfig) -> Resolved:
        repo_id = repo_id_of(ref.locator)
        if not _REPO_ID.match(repo_id) or ".." in repo_id:
            raise SourceError(
                f"hf:// expects a repository id like hf://owner/name; got "
                f"{ref.locator!r}"
            )
        if config.offline:
            raise SourceError(
                "offline mode is on, so the Hub was not contacted. A cached "
                "snapshot is read through huggingface_hub's own cache instead "
                "of this connector."
            )
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as e:  # pragma: no cover - depends on the install
            raise SourceError(
                "hf:// needs huggingface_hub: pip install huggingface_hub"
            ) from e

        api = HfApi()
        try:
            info = api.dataset_info(
                repo_id, revision=ref.revision, files_metadata=True
            )
        except Exception as e:  # noqa: BLE001 - the Hub raises its own types
            raise SourceError(_hub_error(repo_id, e)) from e

        revision = getattr(info, "sha", None)
        siblings = list(getattr(info, "siblings", None) or [])
        sizes = {s.rfilename: (s.size or 0) for s in siblings}
        if ref.selector:
            if ref.selector not in sizes:
                raise SourceError(
                    f"{repo_id} has no file {ref.selector!r} at revision "
                    f"{revision}. It has {len(sizes)} files; list them with "
                    f"`hf://{repo_id}` and no selector."
                )
            wanted = [ref.selector]
            notes: list[str] = []
        else:
            # Nothing was named: take one table, not the repository. The files'
            # sizes are known before the transfer, which is the whole point of
            # reading the listing first.
            wanted, why = choose_bounded(sizes, config.max_bytes)
            if not wanted:
                raise SourceError(
                    f"nothing was downloaded from {repo_id}: {why}. Its files are "
                    f"{_listing(sizes)}"
                )
            notes = [why]

        total = sum(sizes.get(name, 0) for name in wanted)
        if total > config.max_bytes:
            raise SourceError(
                f"the selection from {repo_id} is {total:,} bytes, over the "
                f"{config.max_bytes:,}-byte cap; name one file as "
                f"hf://{repo_id}#<path/in/repo> or raise --max-bytes"
            )

        directory = stage_dir(config, self.scheme, name_for(ref))
        records: list[FileRecord] = []
        for name in wanted:
            try:
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=name,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=directory,
                )
            except Exception as e:  # noqa: BLE001
                raise SourceError(
                    f"could not download {repo_id}/{name}: {_hub_error(repo_id, e)}"
                ) from e
            records.append(
                FileRecord(
                    path=name,
                    bytes=sizes.get(name, 0),
                    sha256=sha256_file(path),
                    source_url=f"https://huggingface.co/datasets/{repo_id}/blob/{revision}/{name}",
                )
            )
        return Resolved(
            scheme=self.scheme,
            locator=repo_id,
            directory=directory,
            files=records,
            revision=revision,
            notes=[
                f"{len(records)} file(s) from {repo_id} pinned to {revision}",
                "the Hub reports public for this repository; no credential was used",
                *notes,
            ],
        )

    def about(self, ref: Reference, config: DataFetcherConfig, *, limit: int = 1200) -> str:
        """What the dataset card says it is, without downloading the dataset.

        The card is what turns "the model named a repository" into "the run read
        what the repository is before spending its bandwidth".
        """
        repo_id = repo_id_of(ref.locator)
        if not _REPO_ID.match(repo_id) or ".." in repo_id:
            return ""
        try:
            from huggingface_hub import HfApi
        except ImportError:  # pragma: no cover - depends on the install
            return ""
        try:
            info = HfApi().dataset_info(repo_id, revision=ref.revision)
        except Exception:  # noqa: BLE001 - a missing card is not an error
            return ""
        card = getattr(info, "cardData", None) or {}
        parts = [
            str(card.get("pretty_name") or ""),
            str(card.get("description") or card.get("summary") or ""),
            _flatten(card.get("task_categories")),
            _flatten(card.get("tags")),
        ]
        return " | ".join(part for part in parts if part)[:limit]


register(HuggingFaceSource())
