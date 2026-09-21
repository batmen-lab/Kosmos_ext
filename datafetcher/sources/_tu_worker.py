"""The ToolUniverse worker. Runs in ToolUniverse's own interpreter.

Started as a subprocess by `datafetcher.tooluniverse`. It reads one JSON request
per line and writes one JSON response per line, and it **stays up between
requests**: loading the tool registry takes about two seconds and registers
2,600 tools, and that used to be paid again for every search and every download
in a run. Two reasons it is a separate process rather than an import:

  * `tooluniverse==1.4.1` needs numpy>=2.2 and mcp<2.0, while Kosmos runs numpy
    1.26 and mcp 2.x -- importing it in the main process breaks Kosmos, not just
    this module; and
  * a 2,600-tool library that can execute code does not belong in the process
    that also holds a training run.

**Only the allowlist below can be executed.** It contains download and
file-listing tools and nothing else; `download_file` is called with an explicit
`output_path` inside the staging directory, so the output location is ours and
not a tool's default. A name outside the list is refused before ToolUniverse is
loaded at all.
"""

from __future__ import annotations

import json
import sys
from hashlib import sha256
from pathlib import Path

# Running as a script puts this directory first on sys.path, and a module named
# `http.py` in it shadowed the standard library's `http` for every import below
# (tooluniverse -> requests -> urllib3 -> http.client). Nothing here needs its
# neighbours, so the directory is removed before anything else is imported.
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [entry for entry in sys.path if str(Path(entry or ".").resolve()) != _HERE]

#: The ToolUniverse source that ships with this repository, if it is there.
#: Importing it beats whichever version happens to be installed in the
#: interpreter: retrieval behaviour is then pinned to the copy under review
#: rather than to an environment someone upgraded.
_LOCAL_SOURCE = Path(__file__).resolve().parents[2] / "ToolUniverse" / "src"
if _LOCAL_SOURCE.is_dir():
    sys.path.insert(0, str(_LOCAL_SOURCE))

#: The line protocol owns stdout: one JSON response per line, nothing else.
#: ToolUniverse prints progress ("Number of tools after load tools: 2716") as it
#: loads, and a caller reading responses would take that for an answer.
_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr

#: Kept in step with `datafetcher.tooluniverse.ALLOWED_TOOLS`, which the client
#: also checks before spawning this process. Two copies on purpose: the check
#: that holds is the one inside the process that can execute a tool.
ALLOWED_TOOLS = (
    # read-only repository search (round one is mechanical and must not invent
    # a dataset that does not exist)
    "OmicsDI_search_datasets",
    "HuggingFace_search_datasets",
    "GEO_search_rnaseq_datasets",
    "GEO_search_methylation_datasets",
    "GEO_search_chipseq_datasets",
    "GEO_search_atacseq_datasets",
    "Zenodo_search_records",
    # download
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


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths_in(value, out_dir: Path) -> list[Path]:
    """Every existing file the tool's result points at."""
    found: list[Path] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and ("path" in key or "file" in key):
                candidate = Path(item)
                if not candidate.is_absolute():
                    candidate = out_dir / candidate
                if candidate.exists() and candidate.is_file():
                    found.append(candidate)
            else:
                found.extend(_paths_in(item, out_dir))
    elif isinstance(value, list):
        for item in value:
            found.extend(_paths_in(item, out_dir))
    return list(dict.fromkeys(found))


#: Parameter names tools use for "write the file here". Only one that the tool
#: actually declares is passed: ToolUniverse validates arguments strictly
#: (`additionalProperties: false`), so an invented key turns a good download into
#: a validation error.
_DESTINATION_KEYS = ("output_path", "output_dir", "save_path", "dest", "destination", "path")


def _destination_parameters(universe, tool: str) -> dict[str, str]:
    spec = (universe.all_tool_dict or {}).get(tool) or {}
    parameter = spec.get("parameter") or {}
    properties = parameter.get("properties") if isinstance(parameter, dict) else None
    if not isinstance(properties, dict):
        return {}
    return {key: key for key in _DESTINATION_KEYS if key in properties}


def _tool_failed(result) -> str | None:
    """ToolUniverse returns errors instead of raising them."""
    if isinstance(result, dict):
        status = str(result.get("status", "")).lower()
        if status in {"error", "failed", "failure"}:
            return str(result.get("error") or result.get("message") or "unknown tool error")
    return None


def _respond(payload: dict) -> None:
    """One line of JSON to the caller, on the stream the line protocol owns."""
    _REAL_STDOUT.write(json.dumps(payload) + "\n")
    _REAL_STDOUT.flush()


def handle(request: dict) -> int:
    """One request, one response. Exits the process only on a fatal error."""
    tool = str(request.get("tool", ""))
    arguments = dict(request.get("arguments") or {})
    out_dir = Path(request.get("out_dir") or ".").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if tool == "__catalog__":
        universe = _universe()
        import tooluniverse

        catalog = []
        for name in ALLOWED_TOOLS:
            spec = (universe.all_tool_dict or {}).get(name)
            if spec:
                catalog.append(
                    {
                        "name": name,
                        "description": str(spec.get("description", ""))[:300],
                        # The model chooses datasets now, so it has to be told
                        # which of these look things up and which fetch them.
                        "kind": (
                            "search"
                            if "search" in name.lower() or name.lower().startswith("list_")
                            else "download"
                        ),
                    }
                )
        # Which copy answered is provenance: this repository ships one, and the
        # interpreter may have another installed.
        _respond(
            {
                "ok": True,
                "catalog": catalog,
                "tooluniverse": str(getattr(tooluniverse, "__file__", "")),
                "version": str(getattr(tooluniverse, "__version__", "")),
            }
        )
        return 0

    if tool not in ALLOWED_TOOLS:
        _respond(
            {
                "ok": False,
                "error": (
                    f"{tool!r} is not on the download allowlist; this worker "
                    f"executes download tools only"
                ),
            }
        )
        return 0

    before = {p for p in out_dir.rglob("*") if p.is_file()}
    try:
        universe = _universe()
        # Give a tool that writes to a path of its choosing a destination inside
        # the staging directory, so the artifact is ours to hash and keep -- but
        # only a key the tool declares.
        for key in _destination_parameters(universe, tool):
            arguments.setdefault(key, str(out_dir / f"{tool}-output"))
        result = universe.run_one_function({"name": tool, "arguments": arguments})
    except Exception as e:  # noqa: BLE001 - reported as JSON, not a traceback
        _respond({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return 0

    failure = _tool_failed(result)
    if failure:
        _respond({"ok": False, "tool": tool, "error": failure[:600]})
        return 0

    after = {p for p in out_dir.rglob("*") if p.is_file()}
    files = _paths_in(result, out_dir)
    for path in sorted(after - before):
        if path not in files:
            files.append(path)
    payload = {
        "ok": True,
        "tool": tool,
        "result": result if isinstance(result, (dict, list, str, int, float, bool)) else str(result),
        "files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in files
        ],
    }
    _respond(payload)
    return 0


_UNIVERSE = None


def _universe():
    """The warm tool registry: loaded once per worker process, not per call."""
    global _UNIVERSE
    if _UNIVERSE is None:
        import tooluniverse

        universe = tooluniverse.ToolUniverse()
        universe.load_tools()
        _UNIVERSE = universe
    return _UNIVERSE


def main() -> int:
    """One JSON request per line, one JSON response per line, until EOF.

    A caller that writes a single request and closes stdin (the old contract)
    reads exactly like a one-shot worker, because the loop ends at EOF.
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            _respond({"ok": False, "error": f"request was not JSON: {e}"})
            continue
        handle(request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
