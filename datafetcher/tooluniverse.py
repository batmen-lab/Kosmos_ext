"""ToolUniverse as the mechanical download layer.

Kosmos's own client asks ToolUniverse questions; Kosmos's own connectors are not
extensible enough to be the only way to get bytes. So downloads go through
ToolUniverse's tools, executed in ToolUniverse's interpreter by
`sources/_tu_worker.py`, and this module is the client: is it available, what can
it do, and run one tool call and bring the files back into the staging directory.

Two properties are deliberate:

  * **The allowlist is download-only.** ToolUniverse contains code-execution and
    arbitrary-request tools; none of them are reachable from here, and the worker
    re-checks the list itself before importing anything.
  * **Availability is reported, never assumed.** No interpreter configured (or
    the wrong one) is a message telling the operator what to install, not a run
    that silently downloads nothing.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: Kept in step with `sources/_tu_worker.py`; the worker's copy is the one that
#: actually gates execution.
ALLOWED_TOOLS = (
    # Repository search: read-only metadata queries. Round one of retrieval is
    # mechanical, so it may ask a repository what it has, but it never invents an
    # accession.
    "OmicsDI_search_datasets",
    "HuggingFace_search_datasets",
    "GEO_search_rnaseq_datasets",
    "GEO_search_methylation_datasets",
    "GEO_search_chipseq_datasets",
    "GEO_search_atacseq_datasets",
    "Zenodo_search_records",
    # Downloads.
    "download_file",
    "download_binary_file",
    "GEO_get_dataset_details",
    "geo_list_supplementary_files",
    "geo_get_dataset_info",
    "FourDN_get_download_url",
    "ENCODE_get_file",
    "ENCODE_list_files",
    "GDC_list_files",
    "NCBI_SRA_get_download_urls",
    "BioModels_download_model",
    "BiGG_download_model",
    "G2P_download_panel",
    "Dryad_get_dataset_files",
    "MGnify_list_analysis_downloads",
    "BioImageArchive_list_study_files",
    "HuBMAP_get_dataset_provenance",
)

ENV_PYTHON = "KOSMOS_TOOLUNIVERSE_PYTHON"
WORKER = Path(__file__).parent / "sources" / "_tu_worker.py"
#: Searches write nothing; a scratch directory keeps the worker's contract
#: (it always receives an `out_dir`) without polluting the staging root.
_SCRATCH = Path(tempfile.gettempdir()) / "kosmos-tu-search"

#: The subset of `ALLOWED_TOOLS` that is a read-only query.
SEARCH_TOOLS = {
    "OmicsDI_search_datasets",
    "HuggingFace_search_datasets",
    "GEO_search_rnaseq_datasets",
    "GEO_search_methylation_datasets",
    "GEO_search_chipseq_datasets",
    "GEO_search_atacseq_datasets",
    "Zenodo_search_records",
}


class ToolUniverseUnavailable(RuntimeError):
    """No usable ToolUniverse interpreter was configured."""


def interpreter(explicit: str | None = None) -> str | None:
    """The ToolUniverse interpreter: explicit argument, then the environment."""
    candidate = explicit or os.environ.get(ENV_PYTHON)
    return candidate or None


def available(explicit: str | None = None) -> tuple[bool, str]:
    """Whether downloads can run, and a sentence explaining the answer."""
    python = interpreter(explicit)
    if not python:
        return False, (
            f"no ToolUniverse interpreter configured; set {ENV_PYTHON} to one with "
            f"tooluniverse installed (e.g. python -m venv ~/venvs/tooluniverse && "
            f"~/venvs/tooluniverse/bin/pip install tooluniverse==1.4.1)"
        )
    path = Path(python)
    if not path.exists() and shutil.which(python) is None:
        return False, f"the configured ToolUniverse interpreter does not exist: {python}"
    probe = subprocess.run(
        [python, "-c", "import tooluniverse; print(tooluniverse.__version__)"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if probe.returncode != 0:
        return False, (
            f"{python} cannot import tooluniverse: "
            f"{(probe.stderr or probe.stdout).strip().splitlines()[-1:] or ['']}"
        )
    return True, f"{python} has tooluniverse {probe.stdout.strip()}"


class ToolUniverseDownloader:
    """Run allowlisted ToolUniverse tools through one long-lived worker.

    The worker stays up for the life of the fetcher: loading the tool registry
    costs about two seconds and 2,600 tools are registered, which used to be
    paid again for every search and every download. A worker that dies is
    restarted on the next call rather than taking the run with it.
    """

    def __init__(self, python: str | None = None, timeout_s: float = 900.0):
        ok, message = available(python)
        if not ok:
            raise ToolUniverseUnavailable(message)
        self.python = interpreter(python)
        self.timeout_s = timeout_s
        self.detail = message
        #: Filled in by `catalog()`: which ToolUniverse answered.
        self.source = ""
        self.version = ""
        self._process: subprocess.Popen | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()

    def catalog(self) -> list[dict[str, Any]]:
        """The allowlisted tools this interpreter actually has."""
        payload = self._call({"tool": "__catalog__"})
        # Which ToolUniverse answered: this repository ships a copy, and the
        # interpreter may have another installed.
        self.source = str(payload.get("tooluniverse") or "")
        self.version = str(payload.get("version") or "")
        return list(payload.get("catalog") or [])

    def run(
        self, tool: str, arguments: dict[str, Any], out_dir: str | Path
    ) -> dict[str, Any]:
        if tool not in ALLOWED_TOOLS:
            raise ValueError(
                f"{tool!r} is not on the download allowlist ({', '.join(ALLOWED_TOOLS[:4])}, …)"
            )
        return self._call(
            {"tool": tool, "arguments": arguments, "out_dir": str(out_dir)}
        )

    def search(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run a read-only repository search tool and return its payload.

        Separate from `run` only in intent: a search produces no files, and the
        caller is going to adapt its answer into download references.
        """
        if tool not in SEARCH_TOOLS:
            raise ValueError(
                f"{tool!r} is not one of the search tools round one may call: "
                f"{', '.join(sorted(SEARCH_TOOLS))}"
            )
        return self._call({"tool": tool, "arguments": arguments, "out_dir": str(_SCRATCH)})

    def _call(self, request: dict[str, Any]) -> dict[str, Any]:
        """One request to the warm worker, one response back."""
        with self._lock:
            process = self._ensure_process()
            try:
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, ValueError) as e:
                self._stop()
                raise RuntimeError(f"ToolUniverse worker is not accepting input: {e}") from e
            try:
                line = self._lines.get(timeout=self.timeout_s)
            except queue.Empty:
                self._stop()
                raise RuntimeError(
                    f"ToolUniverse call {request.get('tool')!r} timed out after "
                    f"{self.timeout_s:.0f}s"
                ) from None
        if line is None:
            self._stop()
            raise RuntimeError("ToolUniverse worker exited before answering")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"ToolUniverse worker returned no JSON: {line[:200]!r}"
            ) from e
        if not payload.get("ok", False):
            raise RuntimeError(f"ToolUniverse tool failed: {payload.get('error')}")
        return payload

    # -- the worker process --------------------------------------------------

    def _ensure_process(self) -> subprocess.Popen:
        if self._process is not None and self._process.poll() is None:
            return self._process
        self._stop()
        self._process = subprocess.Popen(
            [self.python, str(WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._read_lines, args=(self._process,), daemon=True).start()
        return self._process

    def _read_lines(self, process: subprocess.Popen) -> None:
        """Feed responses to the caller; a closed pipe means the worker is gone."""
        for line in process.stdout:
            self._lines.put(line.strip())
        self._lines.put(None)

    def _stop(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        try:
            process.terminate()
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            try:
                process.kill()
            except OSError:
                pass

    def close(self) -> None:
        with self._lock:
            self._stop()

    def __enter__(self) -> ToolUniverseDownloader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def parse_tool_reference(reference: str) -> tuple[str, dict[str, Any]]:
    """Split `tu://tool#json` back into a tool name and its arguments."""
    if not reference.startswith("tu://"):
        raise ValueError(f"{reference!r} is not a tu:// reference")
    body = reference[len("tu://"):]
    tool, _, selector = body.partition("#")
    tool = tool.strip()
    if not tool:
        raise ValueError(f"{reference!r} names no tool")
    if tool not in ALLOWED_TOOLS:
        raise ValueError(
            f"{tool!r} is not on the download allowlist; allowed: "
            f"{', '.join(ALLOWED_TOOLS)}"
        )
    if not selector:
        return tool, {}
    try:
        arguments = json.loads(selector)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{reference!r} has a selector that is not JSON: {e}"
        ) from e
    if not isinstance(arguments, dict):
        raise ValueError(f"{reference!r}: the selector must be a JSON object of arguments")
    return tool, arguments


def format_catalog(catalog: Sequence[dict[str, Any]]) -> str:
    return "\n".join(f"- {entry['name']}: {entry['description']}" for entry in catalog)
