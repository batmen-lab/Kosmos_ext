"""``tu://`` -- a ToolUniverse download tool, run in ToolUniverse's interpreter.

    tu://download_file#{"url": "https://example.org/data.csv"}
    tu://GEO_get_dataset_details#{"accession": "GSE194122"}

The tool name is the locator and the arguments are the selector, so the existing
reference grammar carries a tool call without new syntax. Only the download
allowlist in `datafetcher.tooluniverse` can be reached, and the worker re-checks
it before importing anything.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import DataFetcherConfig
from ..errors import SourceError
from ..models import FileRecord
from ..references import Reference
from ..store import sha256_file, stage_dir
from .base import Resolved, name_for, register


class ToolUniverseSource:
    scheme = "tu"

    def fetch(self, ref: Reference, config: DataFetcherConfig) -> Resolved:
        from ..tooluniverse import (
            ToolUniverseDownloader,
            ToolUniverseUnavailable,
            parse_tool_reference,
        )

        try:
            tool, arguments = parse_tool_reference(str(ref))
        except ValueError as e:
            raise SourceError(str(e)) from e

        try:
            downloader = ToolUniverseDownloader(
                python=getattr(config, "tooluniverse_python", None),
                timeout_s=getattr(config, "tooluniverse_timeout_s", 900.0),
            )
        except ToolUniverseUnavailable as e:
            raise SourceError(str(e)) from e

        directory = stage_dir(config, self.scheme, name_for(ref))
        try:
            payload = downloader.run(tool, arguments, directory)
        except Exception as e:  # noqa: BLE001 - a failed tool call is a fetch failure
            raise SourceError(f"ToolUniverse tool {tool!r} failed: {e}") from e

        records: list[FileRecord] = []
        for entry in payload.get("files", []):
            path = Path(entry["path"])
            target = directory / path.name
            if path != target:
                path.replace(target)
            records.append(
                FileRecord(
                    path=target.name,
                    bytes=target.stat().st_size,
                    sha256=sha256_file(target),
                    source_url=ref.__str__(),
                )
            )
        notes = [f"ToolUniverse tool {tool!r}: {downloader.detail}"]
        if not records:
            notes.append(
                "the tool returned no file, so no artifact was staged; its result "
                "is recorded in the manifest"
            )
        resolved = Resolved(
            scheme=self.scheme,
            locator=tool,
            directory=directory,
            files=records,
            notes=notes,
        )
        if not records:
            # Keep the tool's answer even when it produced no file (a listing
            # tool, say): it is the evidence that the call did something.
            (directory / "tool_result.json").write_text(
                json.dumps(payload.get("result"), indent=2, default=str),
                encoding="utf-8",
            )
        return resolved


register(ToolUniverseSource())
