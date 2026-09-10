"""Find candidate data for a question that arrived without any.

The command a run is sent to when it has a question and nothing to run it on.
It asks an AutoEvidence discovery server for candidate datasets in public
repositories, prints them, and -- on `--emit` -- writes an `evidence.yaml`
naming one, plus the record of where it came from.

**It never chains into a run.** That is the design decision this whole command
exists to hold, so it is stated here rather than buried: the `server:` line in
the emitted config is a command this machine will execute, naming a source
AutoEvidence will fetch and classify, and reading that line is the entirety of
admission control on found data. Auto-chaining would make it a line nobody
reads. The exit of this command is a config and two verification commands; the
next act is a person's.

**Nothing here imports `tooluniverse`, and nothing here searches anything.** The
search runs inside the discovery server, in an interpreter the steward named in
`finder.yaml` -- not this one, which pins numpy<2 and mcp 2.x against
ToolUniverse's numpy>=2.2 and mcp<2.0. This module's imports are stdlib, typer,
rich and two kosmos modules, deliberately: `cli/main.py` registers commands in a
try/except that silently DROPS one whose import fails, so a heavy import here
would make a misconfigured environment show no command and no reason. The
command must always exist, and fail at call time with a remedy.
"""

from __future__ import annotations

import json
import os
import pathlib
from pathlib import Path
from typing import Any, List, Optional

import typer
from rich.markup import escape
from rich.panel import Panel

from kosmos.cli.utils import (
    console,
    create_table,
    format_size,
    print_error,
    print_info,
    print_success,
    print_warning,
    truncate_text,
)

FINDER_SERVER_ENV_VAR = "KOSMOS_FINDER_SERVER"
SERVE_CMD_ENV_VAR = "KOSMOS_AUTOEVIDENCE_SERVE"
SIGNING_KEY_ENV_VAR = "AUTOEVIDENCE_SIGNING_KEY"
_SERVE_BIN = "autoevidence-serve"


def _default_serve_cmd() -> str:
    """An `autoevidence-serve` that will actually execute, absolute if need be.

    A bare name only works when the venv holding it is on the spawned process's
    PATH, and it usually is not: Kosmos is normally invoked by its absolute
    venv path, so the child inherits a PATH with no venv in it and the emitted
    config dies at `[Errno 2] No such file or directory: 'autoevidence-serve'`
    -- after search, after emit, at the first run, having found the data
    correctly.

    Looked for on PATH first (an operator's own install wins), then beside the
    running interpreter, which is where a venv console script lives.
    """
    import shutil
    import sys

    found = shutil.which(_SERVE_BIN)
    if found:
        return found
    sibling = pathlib.Path(sys.executable).parent / _SERVE_BIN
    if sibling.exists():
        return str(sibling)
    # Neither: emit the bare name and let the failure name the missing binary
    # rather than a path this machine never had.
    return _SERVE_BIN


DEFAULT_SERVE_CMD = _default_serve_cmd()

# Printed whenever no finder server is configured. It is long on purpose: the
# reason ToolUniverse cannot live in this interpreter is not guessable, and an
# operator who is told only "no finder server configured" will try to
# `pip install tooluniverse` into this venv, which lands a second libomp beside
# torch's and breaks a working environment to fix a missing feature.
NO_FINDER_SERVER = f"""No finder server configured, so there is nothing to search with.

Dataset search runs inside an AutoEvidence discovery server, which runs
ToolUniverse in its own interpreter -- not this one. ToolUniverse requires
numpy>=2.2 and pins mcp<2.0; this environment pins numpy<2.0 and is on mcp 2.x.
Installing it here would break the environment, not enable the feature.

The path below is a suggestion, not a required location -- put the venv
anywhere you can write. `python3` is spelled out because on many machines
`python` is an interactive shell alias with no executable behind it. The
version is not advice: AutoEvidence's tool allowlist is pinned against it and
the discovery server refuses to start against any other.

    python3 -m venv ~/tooluniverse-venv            # needs Python >= 3.10
    ~/tooluniverse-venv/bin/pip install 'tooluniverse==1.4.1'
    # then write finder.yaml -- see AutoEvidence/finder.yaml.example
    export {FINDER_SERVER_ENV_VAR}="/abs/bin/autoevidence-serve \\
        --finder-config /abs/finder.yaml --key /abs/keys/capsule_signing.key"

Or pass the command directly with --finder-server."""


def _candidates_of(capsule: dict[str, Any]) -> list[dict[str, Any]]:
    return list((capsule.get("capsule", capsule)).get("candidates") or [])


def _looks_like_a_sentence(intent: str) -> bool:
    """Is this a research question rather than a set of search words?

    Crude on purpose -- word count and terminal punctuation, nothing cleverer.
    It only decides which of two true explanations to lead with, so being wrong
    costs a slightly misordered paragraph, and anything smarter would be a
    second thing to keep correct for no gain.
    """
    return intent.strip().endswith("?") or len(intent.split()) > 6


def _no_candidates_advice(intent: str, retry_flag: Optional[str]) -> str:
    """Why an empty result is usually the query, said plainly.

    This message earns its length. An empty result is the single most
    misleading thing this feature can produce: on its face it is
    indistinguishable from "no such data exists anywhere", and a reader who
    takes it that way has been told something false about the world by a tool
    that only meant "no dataset name contained these words".

    And that IS what happened, nearly every time, because the two ends do not
    mean the same thing by a query. `find_datasets` sends the intent VERBATIM
    -- deliberately, so that what left the machine is exactly what the ledger
    and the signed capsule say left it -- while HuggingFace's `search`
    parameter is close to a substring match over dataset names and topics. A
    whole research question therefore matches nothing at all. Verified on this
    machine against the live Hub: "single-cell" returns candidates,
    "single-cell heterogeneity" returns zero.

    So the honest report is not "try different words". It is: the search ran,
    the repository answered, and a question is not a query. Nothing here
    rewrites the intent to fix that -- an intent this code invented would be
    text leaving the machine that no human wrote and no ledger row could
    honestly attribute, which is a far larger decision than a bad search
    deserves. The operator narrows it; the ledger keeps saying exactly what
    was sent.
    """
    lines = [
        "The search ran and the repository answered. Nothing there matched "
        "these words -- which is NOT the same as no such data existing.",
        "",
        f"Sent verbatim: {intent!r}",
        "",
    ]
    if _looks_like_a_sentence(intent):
        lines += [
            "That reads as a research question, and a repository search is not "
            "a research assistant: the repository matches your text against "
            "dataset names and topics, close to a substring match. A whole "
            "question almost never matches anything.",
            "",
            "Search with the few words a dataset would be NAMED after:",
            "    single-cell CRISPR screen     not   Which genes drive ...?",
            "    adult census income           not   What predicts income?",
            "",
        ]
    else:
        lines += [
            "Repository search matches dataset names and topics, so every extra "
            "word narrows it hard. Try fewer and broader ones.",
            "",
        ]
    if retry_flag:
        lines.append(
            f"Give those words to this run with {retry_flag}. The research "
            f"question itself stays exactly as you wrote it."
        )
    else:
        lines.append("Re-run with different words, or name another --repository.")
    return "\n".join(lines)


def _print_capsule(capsule: dict[str, Any]) -> None:
    """The candidate table, the outbound queries, and what is NOT known.

    Every field printed here came from a repository's API, which is to say from
    strangers, so it is `escape`d before it reaches a Rich console: a summary
    containing `[red]` or a fake panel border would otherwise render as console
    output this command appears to have written. The capsule's own limitations
    say the same thing in prose -- titles and summaries are data, not
    instructions -- and this is that sentence enforced one layer down.
    """
    body = capsule.get("capsule", capsule)
    candidates = _candidates_of(capsule)

    for q in body.get("queries") or []:
        console.print(
            f"[muted]sent[/muted] {escape(str(q.get('query_sent', '')))!r} "
            f"[muted]->[/muted] {escape(str(q.get('repository', '?')))} "
            f"[muted]({escape(str(q.get('tool', '?')))} at "
            f"{escape(str(q.get('endpoint') or 'unreported'))})[/muted]"
        )
    console.print()

    table = create_table(
        title=f"Candidate datasets ({len(candidates)})",
        columns=["#", "Repository", "Accession", "Reference", "Public?", "Size", "What it says"],
    )
    # Candidates the gateway itself said it cannot fetch, gathered while the
    # table is built and printed underneath it. A footnote rather than an
    # eighth column: the note is a sentence, and this table is already as wide
    # as a terminal. What matters is that it appears at all. A candidate whose
    # reference will not resolve on this deployment, listed beside four that
    # will with nothing to tell them apart, is the row a reader picks and then
    # cannot explain -- and `--choose` takes a number off exactly this table.
    barred: list[tuple[int, str]] = []
    for i, c in enumerate(candidates):
        size = c.get("reported_size_bytes")
        public = c.get("reported_public")
        if c.get("access_note"):
            barred.append((i, str(c["access_note"])))
        table.add_row(
            str(i),
            escape(str(c.get("repository") or "?")),
            escape(str(c.get("accession") or "?")),
            # A candidate with no reference is a LEAD: a real accession in a
            # repository no connector in this system can fetch. Shown as `--`
            # rather than dropped, because it is still the answer to the
            # question that was asked -- and shown rather than invented,
            # because manufacturing a reference for a scheme nothing serves
            # would produce a config that fails at fetch time.
            escape(str(c.get("reference") or "--")),
            # `reported_public` is one boolean covering two different Hub
            # flags: a repository is not public if it is `private` OR if it is
            # `gated`. The old "reported private" cell named the first of those
            # for both, which asserts a specific barrier the capsule does not
            # distinguish -- and the gated case is the common one. So the cell
            # says only what the boolean says, and the note beneath the table
            # says which barrier it actually is.
            {True: "reported public", False: "NOT public *", None: "?"}.get(public, "?"),
            format_size(size) if isinstance(size, int) else "?",
            # SUMMARY first, then title. The other order was written when
            # `title` looked like a name and was in fact always the accession:
            # the AutoEvidence adapter read a `title` key the Hub's search
            # listing has never sent, so `title` was truthy on every candidate,
            # the summary fallback was unreachable, and this column repeated the
            # two columns to its left on every row while the real one-line
            # descriptions sat unshown in the capsule. This is the column a
            # reader uses to choose which of ten leads to follow; it must say
            # what the candidate SAYS, not what it is called.
            #
            # Whitespace collapsed because a Hub description is often a README
            # fragment with newlines and tabs in it, which turns one table row
            # into six. The capsule keeps the string exactly as it arrived; this
            # is a rendering of it.
            escape(truncate_text(
                " ".join(str(c.get("summary") or c.get("title") or "").split()), 60
            )),
        )
    console.print(table)
    console.print()

    # The gateway's own sentence about its own connectors, so it is NOT escaped
    # as a stranger's text would be -- but it is still printed as a quoted
    # attributed line rather than as this command's own prose, because the
    # reader should be able to see whose statement it is. It is bounded at 400
    # characters by the capsule schema, so it cannot become a wall.
    for index, note in barred:
        console.print(
            f"[warning]* #{index} cannot be fetched by the gateway that found "
            f"it.[/warning] It said: {escape(note)}"
        )
    if barred:
        console.print(
            "[muted]  Not a refusal of the dataset -- the landing page above "
            "is still real, and a steward who configures access can route it. "
            "It means this reference will not resolve here as it stands, so "
            "`--choose` on that row emits a config that fails at fetch "
            "time.[/muted]"
        )
        console.print()

    console.print("[warning]None of this is verified.[/warning]")
    for line in body.get("limitations") or []:
        console.print(f"  [muted]-[/muted] {escape(str(line))}")
    console.print()


def _emit(
    *,
    out_dir: Path,
    capsule: dict[str, Any],
    choose: int,
    serve_cmd: str,
    signing_key: Optional[str],
    staging_dir: Optional[str],
    reference_override: Optional[str] = None,
    companion_reference: Optional[str] = None,
) -> Optional[Path]:
    """Write the config for one candidate. Returns its path, or None on refusal."""
    from kosmos.datasearch.emit import EmitError, write_emit

    try:
        config_path, provenance_path = write_emit(
            out_dir,
            capsule,
            choose,
            serve_cmd=serve_cmd,
            signing_key=signing_key,
            staging_dir=staging_dir,
            reference_override=reference_override,
            companion_reference=companion_reference,
        )
    except EmitError as e:
        print_error(str(e), title="Nothing emitted")
        return None

    chosen = _candidates_of(capsule)[choose]
    reference = reference_override or chosen.get("reference")
    print_success(
        f"Wrote {config_path}\n"
        f"      {provenance_path}  (what was searched for, and everything found)",
        title="Config written -- not run",
    )
    console.print()

    # Said here as well as in the file's own header, because the two are read
    # by different people at different moments: the header is for whoever opens
    # the config before running it, and this is for whoever is looking at the
    # terminal right now and is about to copy the resolve command below. The
    # config is still written -- refusing to emit would be this command
    # overruling a choice the operator made off a table that showed them the
    # barrier, and a steward CAN configure access. It is written and it says so.
    access_note = chosen.get("access_note")
    if access_note:
        print_warning(
            f"{escape(str(access_note))}\n\n"
            f"The config was written unchanged. The resolve command below is "
            f"how you confirm this from the connector rather than from the "
            f"note -- expect it to refuse.",
            title=f"Candidate #{choose} will not fetch on this deployment",
        )
        console.print()

    console.print("[bold]Read the `server:` line, then check what it points at:[/bold]")
    console.print(f"    [code]autoevidence sources  --resolve {escape(str(reference))}[/code]")
    console.print(f"    [code]autoevidence snapshot --source  {escape(str(reference))}[/code]")
    console.print()
    console.print("[bold]Then, if you are satisfied it is the right data:[/bold]")
    console.print(f"    [code]kosmos run \"<your question>\" --evidence-config {config_path}[/code]")
    console.print()
    print_info(
        "This command did not run anything and did not fetch anything. "
        "`subject_key` in the config is a placeholder you must answer: leaving "
        "it out would DECLARE the dataset is not keyed on subjects, which "
        "permits co-mounting it with another.",
        title="Two things still yours to decide",
    )
    return config_path


# Words that carry no dataset in them. A repository index matches names and
# topics, so a whole question matches nothing -- "Which transcripts are most
# consistently altered in fibrotic myocardium?" returns zero while
# "transcripts myocardium fibrotic" returns eleven hits and four fetchable GEO
# series. This list is what separates the two.
_QUESTION_WORDS = frozenset("""
which what how why who when where do does did are is was were has have had can could
the a an and or but of in on for to with by from at as any some most more than that
this these those between across within their its it we you our
carry carries information point points shared possible consistently altered predict
predicts explain explains beyond already best separate separates distinguish differ
differs change changes track tracks associate associates identify propose mechanism
mechanisms measures measure much many independent
""".split())


def search_terms_from_question(question: str, *, keep: int = 3) -> str:
    """The few words a dataset would be NAMED after, taken from a question.

    Longest-first, because domain nouns are long and connectives are short:
    "myocardium", "preeclampsia" and "hippocampal" survive where "altered" and
    "consistently" are dropped by the stop list. Crude, and deliberately so --
    it is a starting query, not an interpretation of the question, and the
    caller can always pass exact terms with --find-data-intent when it misses.
    """
    import re

    words = re.findall(r"[A-Za-z][A-Za-z-]+", question.lower())
    kept = [w for w in words if w not in _QUESTION_WORDS and len(w) > 3]
    kept.sort(key=len, reverse=True)
    seen: list[str] = []
    for word in kept:
        if word not in seen:
            seen.append(word)
        if len(seen) == keep:
            break
    return " ".join(seen)


def _relevance_terms(question: str) -> list[str]:
    """The content words of a question, longest first, for matching a title."""
    import re

    words = re.findall(r"[A-Za-z][A-Za-z-]+", question.lower())
    kept: list[str] = []
    for word in words:
        if word in _QUESTION_WORDS or len(word) <= 3 or word in kept:
            continue
        kept.append(word)
    kept.sort(key=len, reverse=True)
    return kept


def _candidate_text(candidate: dict) -> str:
    """The words a candidate describes ITSELF with, lowercased."""
    return " ".join(
        str(candidate.get(field) or "").lower()
        for field in ("title", "summary", "accession")
    )


def _rank_by_relevance(
    candidates: list[dict], terms: list[str]
) -> list[tuple[int, dict, list[str]]]:
    """Candidates ordered by how much of the question they mention, best first.

    ORDERING, not filtering, because the useful signal here is comparative. A
    threshold has to answer "how much overlap is enough", and on real results
    there is no such number: searching "transcripts myocardium fibrotic"
    returned 111 candidates of which roughly half mention each term
    individually, so every cutoff either admits an atrial-fibrillation study
    (GSE2240 matched `transcripts` and `myocardium`, missed `fibrotic`, and was
    analysed for four minutes) or rejects datasets that simply word their
    titles differently. Ranking needs no such number: the candidate mentioning
    all three outranks the one mentioning two, and if it turns out to be
    unfetchable the next one is tried anyway.

    Candidates mentioning NOTHING the question asked about are still dropped by
    the caller -- that judgement needs no threshold either, since zero overlap
    is not a near miss.

    Ties keep their original order, so a repository's own ranking survives
    wherever this has nothing to add.
    """
    scored: list[tuple[int, dict, list[str]]] = []
    for position, candidate in enumerate(candidates):
        text = _candidate_text(candidate)
        scored.append((position, candidate, [t for t in terms if t[:5] in text]))
    scored.sort(key=lambda row: (-len(row[2]), row[0]))
    return scored


def auto_emit_config(
    *,
    question: str,
    out_dir: Path,
    finder_server: Optional[str] = None,
    intent: Optional[str] = None,
    limit: int = 50,
    serve_cmd: Optional[str] = None,
    signing_key: Optional[str] = None,
) -> Optional[Path]:
    """Search, take the first FETCHABLE candidate, write its config. Or None.

    The unattended half of `find-data`, and it makes one decision the attended
    command refuses to make: `subject_key`. There is nobody to ask, so it is
    written fail-closed -- the dataset is DECLARED subject-keyed, which is the
    answer that forbids co-mounting it with another. Wrong in the harmless
    direction: a feature-keyed table marked subject-keyed loses the ability to
    be joined with a second dataset, while the reverse silently permits a join
    across people.

    Returns the config path, or None when nothing fetchable was found -- the
    caller reports that; this function does not exit the process.
    """
    server = finder_server or os.environ.get(FINDER_SERVER_ENV_VAR)
    if not server:
        print_error(NO_FINDER_SERVER, title="Cannot search")
        return None

    query = (intent or search_terms_from_question(question)).strip()
    if not query:
        print_error(
            f"Could not derive search terms from the question. Pass them with "
            f"--find-data-intent.",
            title="Cannot search",
        )
        return None

    console.print(
        f"[muted]No dataset given. Searching public repositories for "
        f"[bold]{escape(query)}[/bold] (from the question).[/muted]"
    )

    from kosmos.evidence.client import search_datasets

    result = search_datasets(server, query, repositories=None, limit=limit)
    if not result.get("ok"):
        print_error(
            "\n".join(f"  - {d}" for d in result.get("denials", [])),
            title="Dataset search failed",
        )
        return None

    capsule = result["capsule"]
    candidates = _candidates_of(capsule)
    routable = [c for c in candidates if c.get("reference")]
    if not routable:
        print_warning(
            f"Searched for '{query}' and found {len(candidates)} candidate(s), "
            f"but none is fetchable: their repositories have no connector in "
            f"AutoEvidence. Run `kosmos find-data \"{query}\"` to see them, "
            f"obtain one by hand, and pass it with --data-path.",
            title="Nothing fetchable found",
        )
        return None

    # The first candidate that actually RESOLVES, not merely the first with a
    # reference. `geo://GSE319771` is fetchable and holds no table -- a
    # sequencing series publishing only RAW.tar -- so taking routable[0]
    # blindly handed the run a source the gateway then refused, after the
    # search, after the emit, at federation time. Probing is one small fetch
    # per candidate (a series matrix is a few KB) against a run that costs
    # minutes.
    chosen = None
    chosen_index = -1
    chosen_matched: list[str] = []
    reference_override: Optional[str] = None
    skipped: list[str] = []
    # Enumerated over ALL candidates, not over `routable`, because what the
    # emit needs is the position in the capsule and a filtered copy cannot give
    # it: an earlier version searched for the chosen dict afterwards with
    # `.index()`, which raised the moment a narrowed reference made it a dict
    # the list never held.
    terms = _relevance_terms(question)
    irrelevant: list[str] = []
    for position, candidate, matched in _rank_by_relevance(candidates, terms):
        reference = candidate.get("reference")
        if not reference:
            continue
        # Checked BEFORE the probe: it is a string comparison against text the
        # search already returned, so an off-topic candidate costs nothing to
        # skip, while probing one costs a fetch.
        if terms and not matched:
            irrelevant.append(
                f"{candidate.get('accession')}: {str(candidate.get('title') or '')[:70]}"
            )
            continue
        from autoevidence.execute.data_adapter import resolve_source

        override = None
        try:
            resolve_source(reference)
        except Exception as e:
            # A refusal that offers ALTERNATIVES is a choice the connector
            # declined to make, not an absence -- a GEO series assayed on two
            # platforms publishes two matrices and will not pick between them.
            # A human picks; here there is nobody, so take the first and say so.
            # The alternative is discarding a usable dataset because it was
            # measured twice.
            alternatives = list(getattr(e, "alternatives", ()) or ())
            for alternative in alternatives:
                try:
                    resolve_source(alternative)
                except Exception:
                    continue
                override = alternative
                break
            if override is None:
                skipped.append(f"{candidate.get('accession')}: {str(e)[:90]}")
                continue
            console.print(
                f"[muted]{escape(str(candidate.get('accession')))} offered "
                f"{len(alternatives)} references and this run took the first "
                f"that resolved: {escape(override)}[/muted]"
            )
        chosen = candidate
        chosen_index = position
        chosen_matched = matched
        reference_override = override
        break

    if chosen is None:
        if irrelevant and not skipped:
            print_warning(
                f"Searched for '{query}' and found {len(routable)} fetchable "
                f"candidate(s), but none of them is ABOUT what was asked -- no "
                f"title or summary mentions {', '.join(terms[:4])}:\n  - "
                + "\n  - ".join(irrelevant[:4])
                + f"\n\nThe repositories matched on their own index, not on the "
                f"question. Try different words with --find-data-intent, or pass "
                f"a dataset you trust with --data-path.",
                title="Found, but none on topic",
            )
            return None
        print_warning(
            f"Searched for '{query}' and found {len(routable)} fetchable "
            f"candidate(s), but none holds a readable table:\n"
            + "\n".join(f"  - {s}" for s in skipped[:4]),
            title="Found, but none usable",
        )
        return None

    # A GEO series matrix publishes its per-sample labels in the same file, and
    # the connector splits them out. Without them an experiment has values and
    # no way to form a contrast -- so look for the companion, and say plainly
    # when there is none rather than discovering it inside the container.
    companion_reference = None
    chosen_reference = reference_override or chosen.get("reference")
    if str(chosen_reference or "").startswith("geo://"):
        from autoevidence.execute.data_adapter import resolve_source

        base = str(chosen_reference).split("#", 1)
        matrix = base[1][len("matrix/"):] if len(base) > 1 and base[1].startswith("matrix/") else ""
        probe = f"{base[0]}#pheno/{matrix}" if matrix else f"{base[0]}#pheno"
        try:
            resolve_source(probe)
            companion_reference = probe
            console.print(
                f"[muted]Per-sample labels published with this series; "
                f"mounting them alongside as a phenotype table.[/muted]"
            )
        except Exception as e:
            console.print(
                f"[muted]No per-sample labels available for this series "
                f"({str(e)[:80]}). An experiment will have values but no "
                f"published grouping.[/muted]"
            )

    if skipped:
        console.print(
            f"[muted]Skipped {len(skipped)} candidate(s) that resolved to no "
            f"readable table.[/muted]"
        )
    if irrelevant:
        console.print(
            f"[muted]Skipped {len(irrelevant)} candidate(s) whose own title and "
            f"summary mention none of: {', '.join(terms[:4])}. First few:\n  - "
            + "\n  - ".join(escape(x) for x in irrelevant[:3])
            + "[/muted]"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = _emit(
        out_dir=out_dir,
        capsule=capsule,
        choose=chosen_index,
        serve_cmd=serve_cmd or os.environ.get(SERVE_CMD_ENV_VAR) or DEFAULT_SERVE_CMD,
        signing_key=signing_key,
        staging_dir=None,
        reference_override=reference_override,
        companion_reference=companion_reference,
    )
    if config_path is None:
        return None

    _write_fail_closed_subject_key(config_path)
    console.print(
        f"[muted]Chose {escape(str(chosen.get('accession')))} from "
        f"{escape(str(chosen.get('repository')))} "
        f"({len(routable)} of {len(candidates)} candidates were fetchable). "
        # Which of the question's words this dataset actually mentions, and
        # which it does not. It is the difference between a dataset about the
        # subject asked about and one that merely shares its tissue, and it is
        # the operator's to judge -- so it is said rather than implied by the
        # fact that something was chosen.
        + (
            f"Mentions {', '.join(chosen_matched)}"
            + (
                f" but NOT {', '.join(t for t in terms if t not in chosen_matched)}"
                if len(chosen_matched) < len(terms) else ""
            )
            + ". "
            if terms else ""
        )
        + f"Config: {escape(str(config_path))}[/muted]"
    )
    console.print()
    return config_path


def _write_fail_closed_subject_key(config_path: Path) -> None:
    """Answer the placeholder the unattended path cannot ask about."""
    text = config_path.read_text()
    placeholder = '    subject_key: "REPLACE_ME_OR_DELETE"'
    if placeholder not in text:
        return
    config_path.write_text(
        text.replace(
            placeholder,
            "    # Written by an UNATTENDED search, which had nobody to ask. Declared\n"
            "    # subject-keyed because that is the fail-closed answer: it forbids\n"
            "    # co-mounting this dataset with another -- except a source it was\n"
            "    # PUBLISHED WITH, which says so via companion_of. If the table is\n"
            "    # keyed on a feature (gene, variant, cell line), delete this line --\n"
            "    # that deletion is the claim, and it is yours to make.\n"
            '    subject_key: "__unverified__"',
        )
    )


def search_and_report(
    question: str,
    *,
    finder_server: Optional[str] = None,
    repositories: Optional[List[str]] = None,
    limit: int = 10,
    emit: Optional[Path] = None,
    choose: int = 0,
    serve_cmd: Optional[str] = None,
    signing_key: Optional[str] = None,
    staging_dir: Optional[str] = None,
    as_json: bool = False,
    research_question: Optional[str] = None,
    retry_flag: Optional[str] = None,
) -> int:
    """The whole command as a function, returning a process exit code.

    Factored out of the typer command so `kosmos run --find-data` can reach the
    identical path rather than growing a second, drifting copy of it. A run that
    searches and a search that runs alone must produce the same config, the same
    provenance record and the same refusals, or the config a run writes is not
    the config the docs describe.

    `question` is the SEARCH INTENT -- the text that leaves this machine, sent
    verbatim and recorded verbatim. `research_question` is the separate thing a
    run is actually asking, shown alongside when the two differ so the panel
    cannot imply the whole question was used as the query. `retry_flag` names
    the flag that narrows the intent in whichever context this was called from,
    since "re-run with different words" is wrong advice inside a `kosmos run`.
    """
    server = finder_server or os.environ.get(FINDER_SERVER_ENV_VAR)
    if not server or not server.strip():
        print_error(NO_FINDER_SERVER, title="Dataset search is not configured")
        return 1

    from kosmos.evidence.client import search_datasets

    # When a run supplied its own search words, both are shown. Printing only
    # the question would say the question was the query; printing only the
    # words would hide what this search is in aid of. Both are true and the
    # difference between them is the thing an operator most needs to see.
    asked = (
        f"**Research question:** {escape(research_question)}\n"
        if research_question and research_question.strip() != question.strip()
        else ""
    )
    console.print()
    console.print(
        Panel(
            f"[cyan]Searching public repositories for data that could answer this.[/cyan]\n\n"
            f"{asked}"
            f"**Searching for:** {escape(question)}\n"
            f"**Repositories:** {', '.join(repositories) if repositories else 'all the server offers'}\n\n"
            f"[muted]The words above are sent verbatim, off this machine, to "
            f"the repositories the server's steward configured, and are "
            f"recorded verbatim in that server's audit ledger. Nothing here "
            f"rewrites them.[/muted]",
            title="[bright_blue]Dataset search[/bright_blue]",
            border_style="bright_blue",
        )
    )
    console.print()

    result = search_datasets(
        server, question, repositories=repositories or None, limit=limit
    )
    if not result.get("ok"):
        print_error(
            "No candidates.\n\n" + "\n".join(f"  - {d}" for d in result.get("denials", [])),
            title="Search returned nothing",
        )
        return 1

    capsule = result["capsule"]
    candidates = _candidates_of(capsule)

    if as_json:
        # The WHOLE signed capsule, not a rendering of it. A caller asking for
        # JSON is asking for the thing that was signed; handing back a prettier
        # subset would leave them holding something no signature covers.
        console.print_json(json.dumps(capsule, default=str))
    else:
        _print_capsule(capsule)

    if not candidates:
        print_warning(
            _no_candidates_advice(question, retry_flag),
            title="No candidates",
        )
        return 1

    routable = [i for i, c in enumerate(candidates) if c.get("reference")]
    if not routable:
        print_warning(
            "Every candidate is a LEAD, not a source: their repositories have no "
            "connector in AutoEvidence, so nothing here can be fetched "
            "automatically. Open a landing page above, obtain the data by hand, "
            "and run with --data-path.",
            title="Found, but not fetchable",
        )
        return 1

    if emit is None:
        print_info(
            f"{len(routable)} of {len(candidates)} candidate(s) are fetchable. "
            f"Re-run with `--emit <dir>` (and `--choose <#>`) to write an "
            f"evidence.yaml for one of them.",
            title="Nothing written",
        )
        return 0

    if choose not in routable:
        print_error(
            f"Candidate #{choose} has no reference, so no connector can fetch it. "
            f"Fetchable candidates: {', '.join('#' + str(i) for i in routable)}.",
            title="Nothing emitted",
        )
        return 1

    written = _emit(
        out_dir=emit,
        capsule=capsule,
        choose=choose,
        serve_cmd=serve_cmd or os.environ.get(SERVE_CMD_ENV_VAR) or DEFAULT_SERVE_CMD,
        signing_key=signing_key or os.environ.get(SIGNING_KEY_ENV_VAR),
        staging_dir=staging_dir,
    )
    return 0 if written is not None else 1


def find_data(
    question: str = typer.Argument(
        ...,
        help="What you want data for, in plain words. This is a SEARCH QUERY, "
             "not a research question: it is sent verbatim, and repositories "
             "match it against dataset names and topics. `single-cell CRISPR "
             "screen` finds things; `Which genes drive heterogeneity?` finds "
             "nothing.",
    ),
    finder_server: Optional[str] = typer.Option(
        None, "--finder-server", "-F",
        help=f"Shell command that spawns an AutoEvidence discovery server started "
             f"with --finder-config. Defaults to ${FINDER_SERVER_ENV_VAR}. An "
             f"https:// URL is refused: search is served over stdio only.",
    ),
    repository: Optional[List[str]] = typer.Option(
        None, "--repository", "-r",
        help="Repository label to search, repeatable. These are labels the "
             "server's steward fixed; an unknown one is refused with the known "
             "ones listed. Naming none searches all of them.",
    ),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum candidates per repository"),
    emit: Optional[Path] = typer.Option(
        None, "--emit", "-e",
        help="Directory to write evidence.yaml and found_datasets.json into. "
             "Without this, nothing is written and nothing is chosen.",
    ),
    choose: int = typer.Option(
        0, "--choose", "-c",
        help="Which candidate (by the # column) to write the config for.",
    ),
    serve_cmd: Optional[str] = typer.Option(
        None, "--serve-cmd",
        help=f"The autoevidence-serve command the emitted config will invoke. "
             f"Defaults to ${SERVE_CMD_ENV_VAR}, else `{DEFAULT_SERVE_CMD}`.",
    ),
    signing_key: Optional[str] = typer.Option(
        None, "--key",
        help=f"Capsule signing key for the emitted server line. Defaults to "
             f"${SIGNING_KEY_ENV_VAR}; omitted from the command line entirely "
             f"when neither is set, so AutoEvidence's own default applies.",
    ),
    staging_dir: Optional[str] = typer.Option(
        None, "--staging-dir",
        help="Where the emitted server should stage a full release. Defaults to "
             "a `staging/` directory beside the config.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the signed capsule instead of a table"),
):
    """
    Find candidate datasets for a question you have no data for.

    Prints POINTERS -- an accession, a reference, a size -- and never a column,
    a row or a count. It is not a quieter way to see a dataset's schema: that is
    `describe_dataset`, which is gated, and this cannot reach it.

    Nothing found here has been fetched, hashed or classified. A candidate is a
    third party's unverified claim; `--emit` turns one into an `evidence.yaml`
    for you to read, and running it is a separate command you type yourself.

    Examples:

        # See what is out there
        kosmos find-data "single-cell CRISPR perturbation screens in K562"

        # Write a config for candidate #2, then read it before running
        kosmos find-data "adult census income" --emit ./found --choose 2
        kosmos run "Which factors predict income?" --evidence-config ./found/evidence.yaml
    """
    code = search_and_report(
        question,
        finder_server=finder_server,
        repositories=list(repository) if repository else None,
        limit=limit,
        emit=emit,
        choose=choose,
        serve_cmd=serve_cmd,
        signing_key=signing_key,
        staging_dir=staging_dir,
        as_json=as_json,
        # Here the intent IS the argument, so narrowing means editing it.
        retry_flag=None,
    )
    if code != 0:
        raise typer.Exit(code)
