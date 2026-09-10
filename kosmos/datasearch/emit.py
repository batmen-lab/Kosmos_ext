"""Turn one dataset candidate into an `evidence.yaml` a human reads first.

Pure by construction: everything here is a string built from arguments, and the
only I/O is `write_emit` putting two files where the caller asked. No network,
no gateway, no `tooluniverse`, no import of `kosmos.agents` or
`kosmos.execution` -- so this module is testable without a search server and
without an LLM, which is the only reason its output can be asserted on.

**Why a file and not a call.** `find-data` could hand the candidate straight to
the research director. It does not, and the reason is the one line this module
spends most of its comment budget on: `server:`. That line is a command this
machine will execute, naming a source AutoEvidence will fetch and classify.
Deciding to admit a dataset is the whole of admission control here, and a
decision nobody reads is not a decision. So the output is a document, the
document says where the candidate came from and what is unverified about it,
and running it is a second act by a person.

**What is deliberately left wrong.** `subject_key` is emitted as
`REPLACE_ME_OR_DELETE` rather than omitted. In `evidence/federation.py` an
absent `subject_key` DECLARES that the dataset is not keyed on subjects, which
in turn permits co-mounting it with another dataset. Silence there is a
permission, not an abstention -- and this module cannot know which is true of a
dataset it has never seen. A placeholder is a question the operator must answer;
an omission would be an answer this module invented.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any, Optional

# The file `find-data` writes beside the config, holding the whole verified
# capsule. `run.py` looks for it next to a `--evidence-config` and, when it
# names that config, says in the run banner that the dataset was FOUND rather
# than supplied, and where it came from. That is the provenance channel: the
# YAML comments below are for the person, this file is for the run.
PROVENANCE_FILENAME = "found_datasets.json"
CONFIG_FILENAME = "evidence.yaml"

# Emitted rather than omitted; see the module docstring.
SUBJECT_KEY_PLACEHOLDER = "REPLACE_ME_OR_DELETE"


class EmitError(RuntimeError):
    """The candidate cannot be turned into a runnable source.

    Raised, not warned: a half-written `evidence.yaml` naming a reference no
    connector resolves would fail inside `load_sources` much later, with a
    message about YAML rather than about the candidate that was picked.
    """


def _wrap(text: str, *, width: int, indent: str = "") -> list[str]:
    """Soft-wrap one sentence into comment lines. Never splits a word.

    `textwrap` would do this, and is not used for one reason: the strings that
    reach here contain backtick-quoted config keys such as
    `huggingface.allow_private: true`, and an operator copying one out of a
    comment must get it back intact. `textwrap` collapses runs of whitespace
    and can break after a colon, both of which quietly rewrite the thing the
    reader is meant to copy. This only ever inserts newlines between existing
    spaces.

    A word longer than `width` gets its own over-long line rather than being
    cut: an unbreakable token here is a URL or a config key, and a wrapped one
    is worse than a ragged margin.
    """
    lines: list[str] = []
    current = indent
    for word in text.split():
        if current.strip() and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = indent + word
        else:
            current = f"{current} {word}" if current.strip() else indent + word
    if current.strip():
        lines.append(current)
    return lines


def _yaml_str(value: str) -> str:
    """A double-quoted YAML scalar with the two characters that break it escaped.

    Hand-rolled rather than `yaml.dump`ed because the whole point of this file
    is its comment block, and a dumper would drop every comment. The values that
    reach here are a command line the operator supplied and an accession from a
    repository, so backslash and double-quote are the entire escaping surface of
    a double-quoted YAML scalar; a literal newline in either would be a bug
    upstream, and is refused below rather than emitted.
    """
    if "\n" in value or "\r" in value:
        raise EmitError(
            f"refusing to emit a value containing a newline into evidence.yaml: "
            f"{value!r}"
        )
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


# Fallback when this machine's memory cannot be read. Generous enough for the
# summary-statistics tables a found dataset usually is, small enough to be a
# plausible ceiling on a modest machine.
_FALLBACK_BUDGET_BYTES = 3 * 1024**3

# What fraction of free memory a found dataset may occupy. The gateway's own
# default is a quarter, which is the right posture for a server serving many
# callers; this is a single run on the operator's own machine, told to fetch
# one dataset, so it can be more willing.
_BUDGET_FRACTION = 0.6


def _open_data_budget() -> int:
    """A memory ceiling this machine can actually honour, for the emitted config.

    Emitted because leaving it out produced a config that could not run. The
    gateway defaults to a quarter of available memory, and a 3.8M-row table of
    summary statistics wants 1.7 GiB against a 807.9 MiB default -- so the
    first `kosmos run` over a freshly found dataset refused before generating
    anything, with a message about a capacity limit on data it had already
    confirmed was public.

    This is a CAPACITY setting, never a disclosure one: it bounds what the
    agent's process is asked to hold, and every gate still decides what may
    leave. Raising it cannot widen what is released.
    """
    try:
        import psutil

        available = int(psutil.virtual_memory().available)
    except Exception:
        return _FALLBACK_BUDGET_BYTES
    if available <= 0:
        return _FALLBACK_BUDGET_BYTES
    return max(int(available * _BUDGET_FRACTION), _FALLBACK_BUDGET_BYTES)


def _serve_command(
    *,
    serve_cmd: str,
    reference: str,
    dataset_id: str,
    signing_key: Optional[str],
    staging_dir: str,
) -> str:
    """The stdio command line that will serve this candidate.

    A COMMAND, never a URL, and the reason is not style. AutoEvidence refuses
    `--staging-dir` over http, and `evidence/client.py` refuses a capsule naming
    a server-local staged file when it reached the gateway over http -- because
    that absolute path either does not exist on this machine or, worse, resolves
    to an unrelated local file of the same name. A found public dataset is
    released whole, through exactly that staged-file path. So an http URL here
    would produce a config that parses, connects, and then denies at the last
    step; a command line is the only shape that works end to end.

    `--key` is emitted only when one is actually known. AutoEvidence's own
    default is to search `$AUTOEVIDENCE_SIGNING_KEY` and its standard location,
    so writing a guessed path would replace a working default with a file that
    does not exist -- a refusal manufactured out of helpfulness.
    """
    # Shell-quoted, because the consumer splits this line with `shlex.split`
    # before spawning it. An unquoted path containing a space silently becomes
    # two arguments: `--staging-dir /Users/.../data/myocardial` followed by a
    # stray `fibrosis`, so the gateway stages into the wrong directory or
    # refuses on an unknown argument. `--source` is quoted for the same reason
    # and one more: an `hf://...#file.parquet` selector carries a `#`, which
    # some shells treat as a comment.
    key_fragment = f"--key {shlex.quote(signing_key)} " if signing_key else ""
    # `serve_cmd` needs the same treatment and did not get it. It was
    # interpolated raw, so an absolute path containing a SPACE -- ordinary on a
    # Mac, e.g. `/Omics/Kosmos Batmen Lab/Kosmos_ext/.venv/bin/...` -- splits at
    # that space and the consumer tries to execute a DIRECTORY, dying with
    # `PermissionError: [Errno 13]` on a path that is only the first half of the
    # real one. The failure names a path that exists, so it reads like a
    # permissions problem rather than a quoting one.
    #
    # Quoted only when it names an existing FILE: an operator may legitimately
    # set a multi-token command (`python -m autoevidence.server`), and quoting
    # that would collapse it into one nonsense argv[0].
    serve = shlex.quote(serve_cmd) if os.path.isfile(serve_cmd) else serve_cmd
    return (
        f"{serve} --source {shlex.quote(reference)} "
        f"--dataset {shlex.quote(dataset_id)} "
        f"{key_fragment}--staging-dir {shlex.quote(staging_dir)} "
        f"--open-data-budget-bytes {_open_data_budget()}"
    )


def evidence_yaml_for(
    candidate: dict[str, Any],
    *,
    dataset_id: str,
    serve_cmd: str,
    signing_key: Optional[str],
    staging_dir: str,
    intent: str,
    repository: str,
    tool: str,
    endpoint: str,
    capsule_sig: str,
    companion_reference: Optional[str] = None,
) -> str:
    """One candidate as an `evidence.yaml` body, header comments and all.

    `candidate` is one `DatasetCandidate` dict out of a signed candidate
    capsule. It must carry a `reference` -- a scheme AutoEvidence's `sources/`
    layer already routes. A candidate without one is a LEAD, not a source: a GEO
    or OmicsDI accession names something real that no connector in this system
    can fetch, and manufacturing an `hf://` for it would produce a config that
    fails at fetch time with a message about HuggingFace. Refused here instead.

    Exactly one source is emitted. `find-data` returns many candidates and a
    person picks one; writing several would make every one of them `primary`
    ambiguous and would quietly turn "I found some options" into "run over all
    of these", which is a different and much larger decision.
    """
    reference = candidate.get("reference")
    if not reference or not isinstance(reference, str):
        raise EmitError(
            f"candidate {candidate.get('accession', '?')!r} in {repository!r} "
            f"carries no `reference`, so no AutoEvidence connector can fetch it. "
            f"It is a lead, not a source: open its landing page, get the data by "
            f"hand, and run with --data-path, or add a connector for its scheme."
        )
    accession = candidate.get("accession") or "?"
    landing = candidate.get("landing_url") or "(none)"
    reported_public = candidate.get("reported_public")

    # The gateway's own note about whether its connectors can fetch this at
    # all, when there is one. It belongs in THIS document above every other
    # caveat here, because this header's whole job is to be read before the
    # `server:` line is run, and this is the one line that says the line will
    # not work. Without it a gated candidate emits a config identical in every
    # visible respect to a working one, whose only difference surfaces as a
    # credential error from `sources/hf.py` several steps later -- at which
    # point the reader has no way to know it was knowable here.
    access_lines: list[str] = []
    note = candidate.get("access_note")
    if isinstance(note, str) and note.strip():
        access_lines = [
            "",
            "THIS GATEWAY SAYS IT CANNOT FETCH THIS AS CONFIGURED:",
            *_wrap(note.strip(), width=69, indent="  "),
            "  The `server:` line below is still written exactly as it would be",
            "  for any other candidate -- it is not disabled and not edited. It",
            "  will fail at fetch time unless a steward changes the access this",
            "  note describes. Verify with the resolve command below first.",
        ]

    header = "\n".join(
        f"# {line}" for line in [
            "-" * 73,
            "GENERATED by `kosmos find-data`. READ THE `server:` LINE BEFORE RUNNING THIS.",
            "That line is a command this machine will execute, naming a source",
            "AutoEvidence will fetch, pin and classify. Nothing has fetched it yet.",
            "",
            f"intent sent : {intent!r}",
            f"repository  : {repository}   tool: {tool}",
            f"endpoint    : {endpoint or '(not reported)'}",
            f"candidate   : {accession}   (an UNVERIFIED third-party claim)",
            f"landing     : {landing}",
            f"repo says   : reported_public={reported_public!r} -- the repository's own",
            "              flag, NOT a classification this gateway made. AutoEvidence",
            "              decides public vs private itself, when it fetches.",
            f"capsule sig : {capsule_sig[:16]}",
            *access_lines,
            "",
            "Check what this points at before you run it:",
            f"  autoevidence sources  --resolve {reference}",
            f"  autoevidence snapshot --source  {reference}",
            "",
            f"subject_key is written with a placeholder, not omitted. OMITTING IT",
            "DECLARES this dataset is not keyed on subjects, which PERMITS",
            "co-mounting it with another dataset (federation.mountable_together).",
            "Silence here is a permission, not an abstention. Delete the line only",
            "if you know this dataset is feature-keyed (gene, variant, cell line).",
            "-" * 73,
        ]
    )

    body = "\n".join([
        "version: 1",
        "sources:",
        f"  - name: {_yaml_str(dataset_id)}",
        "    server: " + _yaml_str(_serve_command(
            serve_cmd=serve_cmd,
            reference=reference,
            dataset_id=dataset_id,
            signing_key=signing_key,
            staging_dir=staging_dir,
        )),
        f"    dataset: {_yaml_str(dataset_id)}",
        # `unknown` is the value that assumes the most, and it is the only
        # honest one here: nobody has told this run whether these subjects
        # overlap with anything else, and a search result is not a steward.
        "    subject_overlap: unknown",
        f"    subject_key: {_yaml_str(SUBJECT_KEY_PLACEHOLDER)}",
        "    primary: true",
        "",
    ])
    if companion_reference:
        # The labels that say which sample is which group, published in the
        # same file as the values and separated because they are a different
        # schema. Emitted as its own source (a capsule holds one table) and
        # tied back with `companion_of`, which is what lets the join-key gate
        # allow the two halves into one experiment.
        companion_id = f"{dataset_id}_phenotype"
        body += "\n".join([
            f"  - name: {_yaml_str(companion_id)}",
            "    server: " + _yaml_str(_serve_command(
                serve_cmd=serve_cmd,
                reference=companion_reference,
                dataset_id=companion_id,
                signing_key=signing_key,
                staging_dir=staging_dir,
            )),
            f"    dataset: {_yaml_str(companion_id)}",
            "    subject_overlap: unknown",
            f"    subject_key: {_yaml_str(SUBJECT_KEY_PLACEHOLDER)}",
            f"    companion_of: {_yaml_str(dataset_id)}",
            "",
        ])
    return header + "\n" + body


def dataset_id_for(candidate: dict[str, Any]) -> str:
    """A filesystem-safe source name derived from the accession.

    `name` keys the staging directory and the container-side filename, so it has
    to survive a path join. `owner/dataset.v2` becomes `owner_dataset_v2`.
    """
    accession = str(candidate.get("accession") or "found_dataset")
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in accession)
    return safe.strip("_") or "found_dataset"


def write_emit(
    out_dir: Path,
    capsule: dict[str, Any],
    chosen_index: int,
    *,
    serve_cmd: str,
    signing_key: Optional[str] = None,
    staging_dir: Optional[str] = None,
    dataset_id: Optional[str] = None,
    reference_override: Optional[str] = None,
    companion_reference: Optional[str] = None,
) -> tuple[Path, Path]:
    """Write `found_datasets.json` and `evidence.yaml`; return both paths.

    The WHOLE capsule is written, not just the chosen candidate, and it is
    written whether or not the chosen one is routable. It is the record of what
    left this machine (the intent, verbatim) and what came back (every
    candidate, with its unverified-ness attached), and a record that only keeps
    the option somebody took is not a record of the decision.

    Note what this does not do: it does not run anything, and it does not hand
    the config to a director. See the module docstring.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    body = capsule.get("capsule", capsule)
    candidates = body.get("candidates") or []
    if not candidates:
        raise EmitError("the capsule holds no candidates; nothing to emit")
    if not 0 <= chosen_index < len(candidates):
        raise EmitError(
            f"chosen candidate {chosen_index} is out of range "
            f"(the capsule holds {len(candidates)})"
        )
    chosen = candidates[chosen_index]

    # A connector can refuse a candidate while OFFERING narrower references in
    # its place -- a GEO series assayed on two platforms publishes one matrix
    # per platform and will not pick between them. Whoever picked one picked
    # something the capsule does not literally hold, so the config carries the
    # narrowed reference and the record carries BOTH: the capsule stays
    # byte-for-byte what was signed, and the narrowing is stated beside it
    # rather than folded into it, where it would look like the search returned
    # something it never returned.
    narrowing: Optional[dict[str, str]] = None
    if reference_override and reference_override != chosen.get("reference"):
        narrowing = {
            "capsule_reference": str(chosen.get("reference") or ""),
            "emitted_reference": str(reference_override),
        }
        chosen = {**chosen, "reference": reference_override}

    queries = body.get("queries") or []
    query = next(
        (q for q in queries if q.get("repository") == chosen.get("repository")),
        queries[0] if queries else {},
    )

    resolved_id = dataset_id or dataset_id_for(chosen)
    resolved_staging = staging_dir or str((out_dir / "staging").resolve())

    yaml_text = evidence_yaml_for(
        chosen,
        dataset_id=resolved_id,
        serve_cmd=serve_cmd,
        signing_key=signing_key,
        staging_dir=resolved_staging,
        intent=str(body.get("intent") or ""),
        repository=str(chosen.get("repository") or query.get("repository") or "?"),
        tool=str(query.get("tool") or "?"),
        endpoint=str(query.get("endpoint") or ""),
        capsule_sig=str(capsule.get("signature_hex") or ""),
        companion_reference=companion_reference,
    )

    config_path = out_dir / CONFIG_FILENAME
    provenance_path = out_dir / PROVENANCE_FILENAME
    config_path.write_text(yaml_text)
    provenance_path.write_text(json.dumps(
        {
            # Named so `run.py` can confirm this record belongs to the config it
            # was handed, rather than to a different one that happens to sit in
            # the same directory from an earlier search.
            "evidence_config": str(config_path.resolve()),
            "chosen_index": chosen_index,
            "chosen": chosen,
            "reference_narrowed": narrowing,
            "companion_reference": companion_reference,
            "signed_capsule": capsule,
        },
        indent=2,
        default=str,
    ))
    return config_path, provenance_path


def read_provenance(evidence_config: Path) -> Optional[dict[str, Any]]:
    """The `found_datasets.json` beside this config, if it describes THIS config.

    Returns None for every ordinary hand-written config, which is the common
    case and must stay silent. Every failure here -- unreadable file, bad JSON,
    a record naming a different config -- is None as well: provenance is a thing
    a run SAYS, and a run must not fail because it could not say something.
    """
    try:
        config = Path(evidence_config).resolve()
        record_path = config.parent / PROVENANCE_FILENAME
        if not record_path.exists():
            return None
        record = json.loads(record_path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    if record.get("evidence_config") != str(config):
        return None
    return record


def provenance_summary(record: dict[str, Any]) -> dict[str, Any]:
    """The few fields worth showing and worth carrying into a run's config.

    Flattened here rather than at the call site so the run banner, the flat
    config the agents see, and anything a report renders later all say the same
    thing about where the data came from.
    """
    chosen = record.get("chosen") or {}
    body = (record.get("signed_capsule") or {}).get("capsule", {})
    queries = body.get("queries") or []
    query = next(
        (q for q in queries if q.get("repository") == chosen.get("repository")),
        queries[0] if queries else {},
    )
    return {
        "origin": "found",
        "intent": body.get("intent"),
        "repository": chosen.get("repository"),
        "tool": query.get("tool"),
        "endpoint": query.get("endpoint"),
        "accession": chosen.get("accession"),
        "reference": chosen.get("reference"),
        "landing_url": chosen.get("landing_url"),
        "reported_public": chosen.get("reported_public"),
        # Carried for the same reason `reported_public` is, and it is the more
        # actionable of the two: `reported_public` is the repository's claim
        # about the data, while this is the gateway's statement about whether
        # its own connectors can fetch it. A run whose banner reports where a
        # dataset came from, and omits the one sentence saying it will not
        # arrive, is a banner that explains the wrong failure.
        "access_note": chosen.get("access_note"),
        "verified": False,
        "generated_at": body.get("generated_at"),
    }
