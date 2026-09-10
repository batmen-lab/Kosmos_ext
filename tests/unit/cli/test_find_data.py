"""`kosmos find-data`: what it refuses, and what it writes.

Nothing here reaches a network, a gateway or ToolUniverse. The two things worth
testing are both refusals -- an unconfigured search must not silently do
nothing, and an unverified capsule must not be printed as though it were
findings -- plus the one artefact this feature actually produces, which is an
`evidence.yaml` that has to survive `federation.load_sources` unchanged.
"""

from __future__ import annotations

import asyncio
import json

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



# --- fixtures --------------------------------------------------------------

CANDIDATE = {
    "repository": "huggingface",
    "accession": "scikit-learn/adult-census-income",
    "reference": "hf://scikit-learn/adult-census-income",
    "title": "Adult census income",
    "summary": "Predict whether income exceeds $50K/yr from census data.",
    "landing_url": "https://huggingface.co/datasets/scikit-learn/adult-census-income",
    "reported_size_bytes": 4_000_000,
    "reported_public": True,
    "verified": False,
}

LEAD_ONLY = dict(CANDIDATE, accession="GSE12345", reference=None, repository="geo")

CAPSULE_BODY = {
    "intent": "adult census income",
    "queries": [{
        "repository": "huggingface",
        "tool": "HuggingFace_search_datasets",
        "endpoint": "https://huggingface.co/api/datasets",
        "query_sent": "adult census income",
    }],
    "candidates": [CANDIDATE],
    "note": "1 candidate",
    "limitations": ["Candidates are a third party's unverified claims."],
    "generated_at": "2026-09-01T00:00:00+00:00",
}

SIGNED = {
    "capsule": CAPSULE_BODY,
    "signature_hex": "00" * 32,
    "public_key_hex": "11" * 32,
    "algorithm": "ed25519",
}


class _FakeSession:
    """The two coroutines `_search_converse` uses, and a record of the call."""

    def __init__(self, tools, response):
        self._tools = tools
        self._response = response
        self.calls = []

    async def list_tools(self):
        return self._tools

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return self._response


# --- refusals --------------------------------------------------------------

def test_no_finder_server_configured_refuses_with_the_setup_recipe(monkeypatch):
    """No search configured is a refusal that names the fix, not an empty result.

    Also asserts nothing was spawned: the whole failure has to happen before any
    subprocess, because the reason ToolUniverse is not in this interpreter is
    exactly that trying to put it there breaks the interpreter.
    """
    monkeypatch.delenv("KOSMOS_FINDER_SERVER", raising=False)
    called = []
    monkeypatch.setattr(
        "kosmos.evidence.client.search_datasets",
        lambda *a, **k: called.append(a) or {"ok": False, "denials": []},
    )

    result = runner.invoke(app, ["find-data", "anything"])

    assert result.exit_code == 1
    assert not called
    out = result.stdout
    assert "tooluniverse-venv" in out
    assert "KOSMOS_FINDER_SERVER" in out
    # The recipe has to be one an operator can paste and have work, which is a
    # stronger claim than "a recipe is printed" and is what was actually wrong
    # with it: it named `python` (an interactive shell alias on this machine,
    # with no executable behind it) and `/opt/...` (unwritable without root).
    # Both are pinned here so neither comes back.
    assert "python3 -m venv" in out, "the recipe must name a command that exists"
    assert "/opt/" not in out, "the recipe must not require root to follow"
    # The version is load-bearing rather than illustrative: AutoEvidence's
    # `backend.assert_ready` refuses to start against any other release, so a
    # recipe that installs an unpinned tooluniverse produces a server that
    # never comes up, with an error about a pin this text never mentioned.
    assert "tooluniverse==1.4.1" in out


def test_an_http_finder_server_is_refused_without_opening_a_socket(monkeypatch):
    """Search is stdio-only, and the client says so rather than trying it."""
    from kosmos.evidence import client

    def _explode(*a, **k):  # pragma: no cover -- must never run
        raise AssertionError("a session was opened for an http finder server")

    monkeypatch.setattr(client, "_stdio_session", _explode)

    out = client.search_datasets("https://gateway.example/mcp", "anything")

    assert out["ok"] is False
    assert "stdio" in out["denials"][0]


def test_an_unsigned_capsule_is_refused_not_printed(monkeypatch):
    """A capsule whose signature does not verify is refused.

    Not a warning: everything inside a candidate capsule is text a stranger
    wrote, and the signature is the only thing saying which gateway stands
    behind having relayed it, with the intent it sent recorded alongside.
    """
    from kosmos.evidence import client

    monkeypatch.setattr(client, "_verify_candidate_capsule", lambda signed: False)
    session = _FakeSession(["find_datasets"], {"kind": "candidates", "signed": SIGNED})

    out = asyncio.run(client._search_converse(
        session, intent="x", repositories=None, limit=10
    ))

    assert out["ok"] is False
    assert "did not verify" in out["denials"][0]


def test_a_server_without_the_search_tool_names_the_missing_flag(monkeypatch):
    from kosmos.evidence import client

    session = _FakeSession(["list_datasets", "describe_dataset"], {})

    out = asyncio.run(client._search_converse(
        session, intent="x", repositories=None, limit=10
    ))

    assert out["ok"] is False
    assert "--finder-config" in out["denials"][0]
    assert not session.calls


def test_the_client_names_no_tool_endpoint_or_credential(monkeypatch):
    """What leaves this process is an intent, some labels and a limit. Nothing else.

    The gateway's promise is that a caller cannot execute a tool of its
    choosing through it; this is the client half of that, asserted rather than
    assumed, so a later convenience argument has to edit a test that says why.
    """
    from kosmos.evidence import client

    monkeypatch.setattr(client, "_verify_candidate_capsule", lambda signed: True)
    session = _FakeSession(["find_datasets"], {"kind": "candidates", "signed": SIGNED})

    asyncio.run(client._search_converse(
        session, intent="x", repositories=["huggingface"], limit=5
    ))

    name, args = session.calls[0]
    assert name == "find_datasets"
    assert set(args) == {"intent", "repositories", "limit"}


# --- what it writes --------------------------------------------------------

def test_emitted_config_loads_through_federation(tmp_path):
    """The artefact has to be a config `load_sources` accepts, unchanged.

    `_parse_entries` refuses unknown keys, so this is the assertion that keeps
    the emitter honest about the seven keys it may use.
    """
    from kosmos.datasearch.emit import write_emit
    from kosmos.evidence.federation import load_sources

    config_path, provenance_path = write_emit(
        tmp_path, SIGNED, 0, serve_cmd="/abs/bin/autoevidence-serve",
        signing_key="/abs/keys/capsule_signing.key",
    )

    sources = load_sources(config_path)

    assert len(sources) == 1
    assert sources[0].primary is True
    assert sources[0].server.startswith("/abs/bin/autoevidence-serve")
    assert provenance_path.exists()


def test_emitted_config_always_writes_subject_key(tmp_path):
    """Omitting subject_key would GRANT co-mounting by silence.

    In `federation.py` an absent subject_key declares the dataset is not keyed
    on subjects, which permits opening it alongside another. A generator that
    left it out would be answering, on a dataset it has never seen, a question
    only the operator can answer.
    """
    from kosmos.datasearch.emit import SUBJECT_KEY_PLACEHOLDER, write_emit

    config_path, _ = write_emit(
        tmp_path, SIGNED, 0, serve_cmd="autoevidence-serve",
    )

    text = config_path.read_text()
    assert "subject_key:" in text
    assert SUBJECT_KEY_PLACEHOLDER in text


def test_emitted_server_is_a_command_line_never_a_url(tmp_path):
    """A URL would produce a config that connects and then denies the release.

    AutoEvidence refuses --staging-dir over http, and the client refuses a
    capsule naming a server-local staged file it reached over http. A found
    public dataset is released whole, through exactly that path.
    """
    from kosmos.datasearch.emit import write_emit
    from kosmos.evidence.federation import load_sources

    config_path, _ = write_emit(
        tmp_path, SIGNED, 0, serve_cmd="autoevidence-serve",
    )

    server = load_sources(config_path)[0].server
    assert not server.lower().startswith(("http://", "https://"))
    assert "--staging-dir" in server


def test_a_candidate_with_no_reference_is_a_lead_and_is_refused(tmp_path):
    """A GEO accession names something real that no connector here can fetch."""
    from kosmos.datasearch.emit import EmitError, write_emit

    capsule = {**SIGNED, "capsule": {**CAPSULE_BODY, "candidates": [LEAD_ONLY]}}

    with pytest.raises(EmitError) as e:
        write_emit(tmp_path, capsule, 0, serve_cmd="autoevidence-serve")

    assert "lead" in str(e.value)


def test_the_whole_capsule_is_recorded_not_just_the_chosen_candidate(tmp_path):
    """The record is of the decision, not of the option somebody took."""
    from kosmos.datasearch.emit import write_emit

    capsule = {
        **SIGNED,
        "capsule": {**CAPSULE_BODY, "candidates": [CANDIDATE, LEAD_ONLY]},
    }

    _, provenance_path = write_emit(
        tmp_path, capsule, 0, serve_cmd="autoevidence-serve",
    )

    record = json.loads(provenance_path.read_text())
    assert len(record["signed_capsule"]["capsule"]["candidates"]) == 2
    assert record["chosen"]["accession"] == CANDIDATE["accession"]
    assert record["signed_capsule"]["capsule"]["intent"] == "adult census income"


def test_provenance_is_only_claimed_for_the_config_it_names(tmp_path):
    """A record beside a DIFFERENT config must not be attached to this one."""
    from kosmos.datasearch.emit import read_provenance, write_emit

    config_path, _ = write_emit(tmp_path, SIGNED, 0, serve_cmd="autoevidence-serve")

    assert read_provenance(config_path) is not None

    other = tmp_path / "evidence-other.yaml"
    other.write_text("version: 1\n")
    assert read_provenance(other) is None
    assert read_provenance(tmp_path / "nowhere" / "evidence.yaml") is None


# --- the command as a whole ------------------------------------------------

def test_find_data_prints_candidates_and_never_runs_research(monkeypatch, tmp_path):
    """It suggests data. It does not start a run, and cannot.

    Asserted by importing the director module and replacing its class with
    something that fails loudly: `find-data` is a search-and-write command, and
    the moment it can start a run the emitted `server:` line stops being read.
    """
    import kosmos.agents.research_director as rd

    class _Boom:  # pragma: no cover -- must never be constructed
        def __init__(self, *a, **k):
            raise AssertionError("find-data started a research run")

    monkeypatch.setattr(rd, "ResearchDirectorAgent", _Boom)
    monkeypatch.setattr(
        "kosmos.evidence.client.search_datasets",
        lambda *a, **k: {"ok": True, "capsule": SIGNED},
    )

    result = runner.invoke(app, [
        "find-data", "adult census income",
        "--finder-server", "autoevidence-serve --finder-config /x/finder.yaml",
        "--emit", str(tmp_path / "found"),
    ])

    assert result.exit_code == 0, result.stdout
    assert "scikit-learn/adult-census-income" in result.stdout
    assert (tmp_path / "found" / "evidence.yaml").exists()
    assert (tmp_path / "found" / "found_datasets.json").exists()
    # It hands over the commands rather than performing them.
    assert "autoevidence sources" in result.stdout
    assert "--evidence-config" in result.stdout


def test_find_data_writes_nothing_without_emit(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "kosmos.evidence.client.search_datasets",
        lambda *a, **k: {"ok": True, "capsule": SIGNED},
    )

    result = runner.invoke(app, [
        "find-data", "adult census income", "--finder-server", "cmd",
    ])

    assert result.exit_code == 0
    assert not list(tmp_path.iterdir())
    assert "Nothing written" in result.stdout


def test_find_data_refuses_when_every_candidate_is_a_lead(monkeypatch):
    capsule = {**SIGNED, "capsule": {**CAPSULE_BODY, "candidates": [LEAD_ONLY]}}
    monkeypatch.setattr(
        "kosmos.evidence.client.search_datasets",
        lambda *a, **k: {"ok": True, "capsule": capsule},
    )

    result = runner.invoke(app, ["find-data", "x", "--finder-server", "cmd"])

    assert result.exit_code == 1
    assert "not fetchable" in result.stdout.lower()


def test_signature_verification_uses_autoevidences_own_verifier():
    """The verifier is AutoEvidence's, not a reimplementation here.

    Skipped where AutoEvidence is not installed -- which is itself the point of
    the ImportError branch in `_verify_candidate_capsule`: it refuses, rather
    than treating "could not check" as "checked".
    """
    pytest.importorskip("autoevidence")
    from kosmos.evidence.client import _verify_candidate_capsule

    # A capsule signed with all-zero bytes is not signed by anybody.
    assert _verify_candidate_capsule(SIGNED) is False


# --- a candidate this deployment cannot actually fetch -----------------------
#
# The capsule carries `access_note`: the gateway's own sentence saying its
# connectors will refuse this reference (a gated or private HuggingFace repo,
# which `sources/hf.py::_check_access` declines without `allow_private` AND a
# token). It is the only field on a candidate written by the gateway rather
# than transcribed from a repository, and it is the difference between a lead
# worth following and one that cannot work here.
#
# It used to reach the provenance JSON and stop there -- so the table, the
# terminal and the emitted `evidence.yaml` all presented a gated candidate as
# indistinguishable from a working one, and the difference surfaced much later
# as a credential error from a connector, at a point where nobody could tell it
# had been knowable at search time. These pin all three surfaces.

GATED = dict(
    CANDIDATE,
    accession="wanglab/promoter_rbs_gene_expression",
    reference="hf://wanglab/promoter_rbs_gene_expression",
    reported_public=False,
    access_note=(
        "HuggingFace reports this repository as gated. This gateway's hf "
        "connector refuses private or gated repositories unless the steward "
        "set `huggingface.allow_private: true` in the sources config and "
        "configured a token with access, so this reference will very likely "
        "not resolve here as it stands."
    ),
)

GATED_SIGNED = {**SIGNED, "capsule": {**CAPSULE_BODY, "candidates": [GATED]}}


def test_the_emitted_config_says_the_gateway_cannot_fetch_this(tmp_path):
    """The barrier is in the document the operator is told to read first.

    The `server:` line is still written, and written unchanged: refusing to
    emit would overrule a choice the operator made from a table that showed
    them the barrier, and a steward CAN configure the access. What must not
    happen is emitting it silently.
    """
    from kosmos.datasearch.emit import write_emit

    config_path, _ = write_emit(
        tmp_path, GATED_SIGNED, 0, serve_cmd="autoevidence-serve"
    )
    text = config_path.read_text()

    assert "CANNOT FETCH" in text
    assert "gated" in text
    # The config is still a real, complete config -- not a disabled one.
    assert "hf://wanglab/promoter_rbs_gene_expression" in text
    assert "server:" in text
    # The note is one 300-character sentence upstream; here it must be wrapped
    # into comment lines rather than emitted as a single line that runs off the
    # edge of an editor. Scoped to the block this test is about -- other header
    # lines interpolate an accession of unbounded length and are not `_wrap`'s
    # business.
    # `[1:]` drops the tail of the line the split landed inside.
    block = text.split("CANNOT FETCH")[1].split("Check what this points at")[0]
    lines = block.splitlines()[1:]
    assert len(lines) > 3, "the note was not wrapped"
    for line in lines:
        assert line.startswith("#"), f"escaped the comment block: {line!r}"
        assert len(line) <= 80, line
    # The config key an operator has to copy survives wrapping intact. This is
    # why `_wrap` exists instead of `textwrap`, which may break after a colon.
    assert "`huggingface.allow_private: true`" in text


def test_a_fetchable_candidate_gets_no_access_warning(tmp_path):
    """The warning is absent when there is nothing to warn about.

    Guards the direction that would be easy to get wrong later: a header that
    always carried the section, empty, would train an operator to skip it.
    """
    from kosmos.datasearch.emit import write_emit

    config_path, _ = write_emit(tmp_path, SIGNED, 0, serve_cmd="autoevidence-serve")

    assert "CANNOT FETCH" not in config_path.read_text()


def test_the_provenance_summary_carries_the_access_note(tmp_path):
    """A run's banner reports where data came from; it must also report this.

    A banner that explains the origin of a dataset and omits the one sentence
    saying it will not arrive explains the wrong failure.
    """
    from kosmos.datasearch.emit import (
        provenance_summary,
        read_provenance,
        write_emit,
    )

    config_path, _ = write_emit(
        tmp_path, GATED_SIGNED, 0, serve_cmd="autoevidence-serve"
    )
    summary = provenance_summary(read_provenance(config_path))

    assert "gated" in summary["access_note"]
    assert summary["reported_public"] is False


def test_the_candidate_table_marks_and_explains_an_unfetchable_row(monkeypatch):
    """The table is what `--choose` takes a number off, so it has to say so."""
    monkeypatch.setattr(
        "kosmos.evidence.client.search_datasets",
        lambda *a, **k: {"ok": True, "capsule": GATED_SIGNED},
    )

    result = runner.invoke(app, ["find-data", "x", "--finder-server", "cmd"])

    out = result.stdout
    # The cell no longer claims "private": `reported_public` is False for a
    # gated repo too, and the capsule does not distinguish the two.
    assert "private" not in out.split("It said:")[0]
    assert "cannot be fetched" in out
    assert "gated" in out
