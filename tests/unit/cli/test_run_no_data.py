"""`kosmos run` with a question and no data: refuse, and say where data comes from.

The behaviour change these tests pin down is deliberate and is the point of the
feature: a run with a question and no data source used to proceed, plan,
generate code, and then fail inside every experiment's data-loading preamble.
This refuses instead.

Half of this file is therefore about what did NOT change. Every existing way of
supplying data -- `--data-path`, `--evidence-server`, `--evidence-config`, and
the data-driven no-question mode -- must reach the run exactly as before, and
the interactive wizard must still be the answer to "no question and no data".
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from kosmos.cli.main import app


runner = CliRunner()


@pytest.fixture(autouse=True)
def unmuted_console():
    """Belt and braces against a process-wide console mute.

    `cli/main.py` now assigns `console.quiet = quiet` unconditionally, so a
    `--quiet` invocation no longer latches the module-level Rich console for
    the rest of the process -- which it did when these tests were written, and
    which made assertions here read an empty stdout for reasons that had
    nothing to do with the code under test.

    Kept anyway, because the failure it guards against is silent and
    cross-file: any future code that sets `quiet` one-way would make this
    file's assertions fail with "expected text not in ''", which points at the
    wrong module entirely. Cheap insurance against an expensive debugging
    session.
    """
    from kosmos.cli.utils import console

    console.quiet = False
    yield
    console.quiet = False


REFUSAL = "No data source, so there is nothing to run this question against"


@pytest.fixture
def csv(tmp_path) -> Path:
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n3,4\n")
    return path


@pytest.fixture
def stub_run(monkeypatch):
    """Let a run get past validation without doing any research.

    Only the two seams a run crosses after the checks under test: the director
    and the async progress loop. Everything before them -- which is all of the
    validation this file is about -- runs for real.
    """
    import kosmos.agents.registry as registry
    import kosmos.agents.research_director as rd
    import kosmos.cli.commands.run as run_mod

    monkeypatch.setattr(rd, "ResearchDirectorAgent", MagicMock())
    monkeypatch.setattr(registry, "get_registry", MagicMock())

    async def _no_research(*a, **k):
        return {"hypotheses": [], "experiments": [], "metrics": {}}

    monkeypatch.setattr(run_mod, "run_with_progress_async", _no_research)


# --- the new refusal -------------------------------------------------------

def test_a_question_with_no_data_refuses_and_names_find_data():
    result = runner.invoke(app, ["run", "Does SOD2 predict fibrosis?"])

    assert result.exit_code == 1
    assert REFUSAL in result.stdout
    assert "kosmos find-data" in result.stdout
    assert "--find-data" in result.stdout


def test_find_data_on_run_searches_and_stops(monkeypatch, tmp_path):
    """`--find-data` turns the refusal into an answer, and still does not run.

    It reaches the same code path `kosmos find-data` does -- asserted here by
    checking the artefacts, since a second copy of the emitter would be a second
    place for the `server:` line to drift.
    """
    import kosmos.agents.research_director as rd

    class _Boom:  # pragma: no cover -- must never be constructed
        def __init__(self, *a, **k):
            raise AssertionError("--find-data started a research run")

    monkeypatch.setattr(rd, "ResearchDirectorAgent", _Boom)
    monkeypatch.setenv(
        "KOSMOS_FINDER_SERVER",
        "autoevidence-serve --finder-config /x/finder.yaml",
    )
    monkeypatch.setattr(
        "kosmos.evidence.client.search_datasets",
        lambda *a, **k: {"ok": True, "capsule": {
            "capsule": {
                "intent": "x",
                "queries": [{
                    "repository": "huggingface",
                    "tool": "HuggingFace_search_datasets",
                    "endpoint": "https://huggingface.co/api/datasets",
                    "query_sent": "x",
                }],
                "candidates": [{
                    "repository": "huggingface",
                    "accession": "owner/name",
                    "reference": "hf://owner/name",
                    "reported_public": True,
                    "verified": False,
                }],
                "note": "",
                "limitations": [],
            },
            "signature_hex": "ab" * 32,
            "public_key_hex": "cd" * 32,
            "algorithm": "ed25519",
        }},
    )

    result = runner.invoke(app, [
        "run", "Q?", "--find-data",
        "--find-data-out", str(tmp_path / "found"),
    ])

    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "found" / "evidence.yaml").exists()
    assert (tmp_path / "found" / "found_datasets.json").exists()


def test_the_search_words_can_differ_from_the_research_question(monkeypatch, tmp_path):
    """`--find-data-intent` is what makes `--find-data` useful rather than merely honest.

    A repository matches your text against dataset NAMES and topics -- close to
    a substring match -- and `find_datasets` sends the intent verbatim by
    design, so the ledger records exactly what left the machine. Those two facts
    together mean a research question, which is the only text a run has,
    reliably matches nothing: verified against the live Hub, "single-cell"
    returns candidates and "Which genes drive single-cell transcriptional
    heterogeneity?" returns zero.

    So the question and the query are different arguments. What must NOT happen
    is this code deriving one from the other: that text is written verbatim into
    the gateway's ledger as what this principal searched for, and a phrase
    Kosmos invented would attribute words to an operator who never wrote them.
    """
    sent = []
    monkeypatch.setenv("KOSMOS_FINDER_SERVER", "autoevidence-serve --finder-config /x")
    monkeypatch.setattr(
        "kosmos.evidence.client.search_datasets",
        lambda server, intent, **k: sent.append(intent)
        or {"ok": False, "denials": ["none"]},
    )

    result = runner.invoke(app, [
        "run", "Which genes drive single-cell transcriptional heterogeneity?",
        "--find-data", "--find-data-intent", "single-cell",
        "--find-data-out", str(tmp_path / "found"),
    ])

    assert sent == ["single-cell"], "the run searched for something else"
    # Both are shown: printing only the question would claim the question was
    # the query; printing only the words would hide what the search is in aid of.
    assert "single-cell" in result.stdout
    assert "Research question:" in result.stdout


def test_without_search_words_the_question_is_used_and_nothing_rewrites_it(
    monkeypatch, tmp_path
):
    """The default is the honest one rather than the effective one.

    Falling back to the question keeps the promise that what leaves the machine
    is text a human wrote. It will usually match nothing, and the empty result
    says so in its own words rather than being quietly improved upon here.
    """
    sent = []
    monkeypatch.setenv("KOSMOS_FINDER_SERVER", "autoevidence-serve --finder-config /x")
    monkeypatch.setattr(
        "kosmos.evidence.client.search_datasets",
        lambda server, intent, **k: sent.append(intent)
        or {"ok": False, "denials": ["none"]},
    )

    runner.invoke(app, [
        "run", "Does SOD2 predict fibrosis?", "--find-data",
        "--find-data-out", str(tmp_path / "found"),
    ])

    assert sent == ["Does SOD2 predict fibrosis?"]


def test_search_words_without_find_data_are_refused_not_ignored(csv):
    """Silently dropping the one argument whose whole purpose is to leave the machine.

    An operator who believed they had narrowed their search would otherwise get
    a run that never searched at all, with no line of output saying so.
    """
    result = runner.invoke(app, [
        "run", "Q?", "--data-path", str(csv), "--find-data-intent", "single-cell",
    ])

    assert result.exit_code == 1
    assert "--find-data-intent" in result.stdout


def test_an_empty_result_explains_itself_rather_than_blaming_the_world(
    monkeypatch, tmp_path
):
    """"Found nothing" must never read as "no such data exists".

    It is the most misleading thing this feature can produce, and the usual
    cause is that a question was used where a query belongs. The message has to
    say which of the two happened, and name the flag that fixes it.
    """
    monkeypatch.setenv("KOSMOS_FINDER_SERVER", "autoevidence-serve --finder-config /x")
    monkeypatch.setattr(
        "kosmos.evidence.client.search_datasets",
        lambda *a, **k: {"ok": True, "capsule": {
            "capsule": {
                "intent": "Which genes drive single-cell transcriptional heterogeneity?",
                "queries": [], "candidates": [], "note": "found 0 candidates",
                "limitations": [],
            },
            "signature_hex": "ab" * 32, "public_key_hex": "cd" * 32,
            "algorithm": "ed25519",
        }},
    )

    result = runner.invoke(app, [
        "run", "Which genes drive single-cell transcriptional heterogeneity?",
        "--find-data", "--find-data-out", str(tmp_path / "found"),
    ])

    out = result.stdout
    assert "NOT the same as no such data existing" in out
    # It names the knob that actually helps, in the context it was called from.
    assert "--find-data-intent" in out


def test_find_data_alongside_a_data_source_is_refused(csv):
    result = runner.invoke(app, ["run", "Q?", "--find-data", "--data-path", str(csv)])

    assert result.exit_code == 1
    # Rich hard-wraps the panel, so match a phrase short enough to survive it.
    assert "--find-data searches for data" in result.stdout


# --- everything that did not change ---------------------------------------

def test_a_question_with_data_path_is_untouched(csv, stub_run):
    result = runner.invoke(app, ["run", "Q?", "--data-path", str(csv)])

    assert REFUSAL not in result.stdout


def test_a_question_with_evidence_config_is_untouched(tmp_path, stub_run):
    config = tmp_path / "evidence.yaml"
    config.write_text(
        "version: 1\nsources:\n  - name: d\n    server: autoevidence-serve\n"
    )

    result = runner.invoke(app, ["run", "Q?", "--evidence-config", str(config)])

    assert REFUSAL not in result.stdout


def test_a_question_with_evidence_server_is_untouched(stub_run):
    result = runner.invoke(app, [
        "run", "Q?", "--evidence-server", "autoevidence-serve --policy p.yaml",
    ])

    assert REFUSAL not in result.stdout


def test_data_driven_no_question_mode_is_untouched(csv, stub_run):
    """A data source with no question still auto-generates its own question."""
    result = runner.invoke(app, ["run", "--data-path", str(csv)])

    assert REFUSAL not in result.stdout
    assert "formulate its own research question" in result.stdout


def test_no_question_and_no_data_is_still_interactive_mode(monkeypatch):
    """The wizard keeps that slot; the new refusal must not steal it.

    Aborting a session the operator just filled in, to tell them to run a
    different command, would be worse than the blind run it prevents.
    """
    import kosmos.cli.commands.run as run_mod

    entered = []
    monkeypatch.setattr(
        run_mod, "run_interactive_mode", lambda: entered.append(True) or None
    )

    result = runner.invoke(app, ["run"])

    assert entered == [True]
    assert REFUSAL not in result.stdout
    assert result.exit_code == 0


# --- provenance ------------------------------------------------------------

def test_a_found_config_makes_the_run_say_so(tmp_path, stub_run):
    """The run must be able to say the dataset was found, not supplied."""
    from kosmos.datasearch.emit import write_emit

    config_path, _ = write_emit(
        tmp_path / "found",
        {
            "capsule": {
                "intent": "single-cell CRISPR screens in K562",
                "queries": [{
                    "repository": "huggingface",
                    "tool": "HuggingFace_search_datasets",
                    "endpoint": "https://huggingface.co/api/datasets",
                    "query_sent": "single-cell CRISPR screens in K562",
                }],
                "candidates": [{
                    "repository": "huggingface",
                    "accession": "owner/k562-screen",
                    "reference": "hf://owner/k562-screen",
                    "landing_url": "https://huggingface.co/datasets/owner/k562-screen",
                    "reported_public": True,
                    "verified": False,
                }],
                "note": "",
                "limitations": [],
            },
            "signature_hex": "ab" * 32,
            "public_key_hex": "cd" * 32,
            "algorithm": "ed25519",
        },
        0,
        serve_cmd="autoevidence-serve",
    )

    result = runner.invoke(app, ["run", "Q?", "--evidence-config", str(config_path)])

    assert "found by a search" in result.stdout
    assert "owner/k562-screen" in result.stdout
    assert "single-cell CRISPR screens in K562" in result.stdout


def test_a_hand_written_config_says_nothing_about_provenance(tmp_path, stub_run):
    """The common case stays silent: no record, no panel, no failure."""
    config = tmp_path / "evidence.yaml"
    config.write_text(
        "version: 1\nsources:\n  - name: d\n    server: autoevidence-serve\n"
    )

    result = runner.invoke(app, ["run", "Q?", "--evidence-config", str(config)])

    assert "found by a search" not in result.stdout


# --- --find-data is decided before the wizard, never discarded --------------

def _searching(monkeypatch) -> list:
    """Record what `search_and_report` was asked, and answer without a network."""
    import kosmos.cli.commands.find_data as fd

    calls: list = []

    def _fake(question, **kwargs):
        calls.append({"intent": question, **kwargs})
        return 0

    monkeypatch.setattr(fd, "search_and_report", _fake)
    return calls


def test_find_data_with_no_question_searches_instead_of_entering_the_wizard(
    monkeypatch, tmp_path
):
    """The silent-discard bug, pinned.

    `--find-data` used to be handled inside the no-data branch, which is
    guarded by `not from_interactive` -- and `from_interactive` is set by
    `interactive or (not question and not has_data_source)`. So this exact
    command line dropped into the wizard, discarded BOTH flags with no line of
    output saying so, and then started a full research run with no data source
    at all: the plan, the hypotheses and the generated code paid for, and every
    experiment dying in its data-loading preamble. That is the blind run the
    whole feature exists to prevent, reached through the feature's own flag.
    """
    import kosmos.cli.commands.run as run_mod

    entered = []
    monkeypatch.setattr(
        run_mod, "run_interactive_mode", lambda: entered.append(True) or None
    )
    calls = _searching(monkeypatch)

    result = runner.invoke(app, [
        "run", "--find-data", "--find-data-intent", "single-cell",
        "--find-data-out", str(tmp_path / "found"),
    ])

    assert entered == [], "the wizard ran and the search flags were discarded"
    assert [c["intent"] for c in calls] == ["single-cell"]
    assert result.exit_code == 0


def test_find_data_with_interactive_is_refused_not_silently_dropped(monkeypatch):
    """--find-data never starts a run, so the wizard's answers would be discarded."""
    import kosmos.cli.commands.run as run_mod

    entered = []
    monkeypatch.setattr(
        run_mod, "run_interactive_mode", lambda: entered.append(True) or None
    )
    calls = _searching(monkeypatch)

    result = runner.invoke(app, ["run", "--interactive", "--find-data",
                                 "--find-data-intent", "single-cell"])

    assert result.exit_code == 1
    assert entered == [] and calls == []
    assert "--find-data" in result.stdout


def test_find_data_with_nothing_to_search_for_is_refused(monkeypatch):
    """No question and no intent: there is no query, and none is invented."""
    import kosmos.cli.commands.run as run_mod

    monkeypatch.setattr(run_mod, "run_interactive_mode", lambda: None)
    calls = _searching(monkeypatch)

    result = runner.invoke(app, ["run", "--find-data"])

    assert result.exit_code == 1
    assert calls == [], "an empty query left the machine"
    assert "nothing to search for" in result.stdout


# --- the found-data banner --------------------------------------------------

def _found_config(tmp_path, candidate: dict, intent: str = "gene expression"):
    from kosmos.datasearch.emit import write_emit

    config_path, _ = write_emit(
        tmp_path / "found",
        {
            "capsule": {
                "intent": intent,
                "queries": [{
                    "repository": "huggingface",
                    "tool": "HuggingFace_search_datasets",
                    "endpoint": "https://huggingface.co/api/datasets",
                    "query_sent": intent,
                }],
                "candidates": [candidate],
                "note": "",
                "limitations": [],
            },
            "signature_hex": "ab" * 32,
            "public_key_hex": "cd" * 32,
            "algorithm": "ed25519",
        },
        0,
        serve_cmd="autoevidence-serve",
    )
    return config_path


def test_the_banner_reports_a_barrier_the_gateway_already_knew_about(
    tmp_path, stub_run
):
    """`provenance_summary` carries `access_note` so this panel can show it.

    It did not show it. A config emitted for a gated candidate produced a
    confident provenance panel naming the repository, the endpoint and the
    reference, and then the run died at fetch with "HF_TOKEN is empty" --
    the failure the field was added to pre-empt, at the one moment it was added
    for. `find-data` warns when the config is WRITTEN; a config is meant to be
    read and run later, by a different person, which is this moment.
    """
    config = _found_config(tmp_path, {
        "repository": "huggingface",
        "accession": "wanglab/promoter_rbs_gene_expression",
        "reference": "hf://wanglab/promoter_rbs_gene_expression",
        "landing_url": "https://huggingface.co/datasets/wanglab/promoter_rbs_gene_expression",
        "reported_public": False,
        "access_note": "HuggingFace reports this repository as gated. This "
                       "gateway's hf connector refuses private or gated "
                       "repositories unless the steward set allow_private.",
        "verified": False,
    })

    result = runner.invoke(app, ["run", "Q?", "--evidence-config", str(config)])

    assert "found by a search" in result.stdout
    assert "cannot fetch" in result.stdout
    assert "gated" in result.stdout


def test_the_banner_does_not_let_a_repository_write_its_own_lines(tmp_path, stub_run):
    """Third-party strings, escaped -- as `find_data._print_capsule` already does.

    Two failures, and the second is not cosmetic. An accession carrying
    `[/muted][bold green]VERIFIED PUBLIC[/bold green]` renders as this panel's
    own prose, inside the one panel whose job is to say the dataset is
    unverified. And an unmatched closing tag raises `MarkupError` out of
    `console.print` -- which sits outside the try/except that guards reading the
    provenance -- killing `kosmos run` with a traceback before the run starts.
    """
    config = _found_config(tmp_path, {
        "repository": "huggingface",
        "accession": "owner/[/muted][bold green]VERIFIED PUBLIC[/bold green][nope]",
        "reference": "hf://owner/name",
        "landing_url": "https://huggingface.co/datasets/owner/name",
        "reported_public": None,
        "verified": False,
    })

    result = runner.invoke(app, ["run", "Q?", "--evidence-config", str(config)])

    assert result.exception is None, result.exception
    assert "found by a search" in result.stdout
    # The markup arrived as text, not as styling this command appears to have
    # written.
    assert "[bold green]" in result.stdout
