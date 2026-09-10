"""M7: the Kosmos-side evidence client (pure materialization + request derivation).

These cover the deterministic core -- turning a menu into a seed request and a
capsule into a table. The MCP transport (`fetch_capsule`) is exercised against a
live gateway separately; here we prove the capsule -> CSV path a released capsule
takes into the rest of the pipeline.
"""

from __future__ import annotations

import csv

import pytest

from kosmos.evidence.client import (
    EvidenceDenied,
    capsule_to_rows,
    default_request,
    materialize,
)


def test_default_request_prefers_a_no_variable_template():
    menu = {
        "templates": [
            {"template_id": "assoc", "evidence_type": "association",
             "required_roles": ["outcome", "predictor"]},
            {"template_id": "qc", "evidence_type": "quality", "required_roles": []},
        ],
        "variables": [],
    }
    req = default_request(menu)
    assert req["template_id"] == "qc" and req["variables"] == []


def test_default_request_fills_required_roles_from_variables():
    menu = {
        "templates": [{"template_id": "assoc", "evidence_type": "association",
                       "required_roles": ["outcome", "predictor"]}],
        "variables": [
            {"name": "resp", "allowed_roles": ["outcome"]},
            {"name": "score", "allowed_roles": ["predictor"]},
        ],
    }
    req = default_request(menu)
    assert req["template_id"] == "assoc"
    assert {(v["name"], v["role"]) for v in req["variables"]} == {
        ("resp", "outcome"), ("score", "predictor")
    }


def test_default_request_raises_when_no_template_can_be_satisfied():
    menu = {"templates": [{"template_id": "assoc", "evidence_type": "association",
                           "required_roles": ["outcome"]}],
            "variables": [{"name": "x", "allowed_roles": ["group"]}]}
    with pytest.raises(EvidenceDenied):
        default_request(menu)


def test_capsule_to_rows_prefers_cohort_table_then_groups_then_estimates():
    cohort = {"capsule": {"cohort_table": [{"cell": {"age": "40-59"}, "n_band": "10-49"}],
                          "groups": [], "estimates": []}}
    assert capsule_to_rows(cohort) == [{"age": "40-59", "n_band": "10-49"}]

    groups = {"capsule": {"cohort_table": [], "estimates": [],
                          "groups": [{"group": "a", "n_band": "50-199", "outcome": None,
                                      "mean": None, "mean_band": None, "sd": None,
                                      "median": None, "median_band": None}]}}
    assert capsule_to_rows(groups)[0]["group"] == "a"

    est = {"capsule": {"cohort_table": [], "groups": [],
                       "estimates": [{"term": "slope", "estimate": 0.5, "std_error": 0.1,
                                      "p_value": 0.01, "ci_low": 0.3, "ci_high": 0.7}]}}
    assert capsule_to_rows(est)[0]["term"] == "slope"


def test_capsule_to_rows_accepts_bare_or_signed_capsule():
    bare = {"cohort_table": [], "groups": [{"group": "g", "n_band": "10-49"}], "estimates": []}
    signed = {"capsule": bare}
    assert capsule_to_rows(bare) == capsule_to_rows(signed)


def test_materialize_writes_banded_csv_without_exact_counts(tmp_path):
    signed = {"capsule": {"cohort_table": [], "estimates": [],
                          "groups": [{"group": "control", "n_band": "250-999", "outcome": None,
                                      "mean": None, "mean_band": None, "sd": None,
                                      "median": None, "median_band": None}]}}
    out = materialize(signed, tmp_path / "e.csv")
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["group"] == "control" and rows[0]["n_band"] == "250-999"
    assert "n" not in rows[0]  # a banded n_band, never an exact n


# --- transport: stdio vs http ----------------------------------------------
#
# The pure tests above are the contract with the rest of the pipeline. These are
# the contract with the gateway: which transport a value selects, that an
# authenticated gateway is never reached without a credential, that the
# credential never comes back out, and that ONE body of conversation logic
# serves both transports. They use a stub session rather than a live gateway --
# what is being proved here is this client's behaviour, not the gateway's.

import asyncio
import inspect
from contextlib import asynccontextmanager
from types import SimpleNamespace

from kosmos.evidence import client
from kosmos.evidence.client import (
    API_KEY_ENV_VAR,
    _converse,
    _normalise_url,
    _resolve_api_key,
    _transport_for,
    fetch_source,
)

_SENTINEL_KEY = "ae_TEST-KEY-THAT-MUST-NOT-ESCAPE"


class _StubSession:
    """A ClientSession-shaped stand-in: names some tools, answers some calls.

    `responses` maps a tool name to the payload it returns, or to an exception
    it raises, so a test can make the gateway refuse mid-conversation.
    """

    def __init__(self, tools, responses):
        self._tools = list(tools)
        self._responses = dict(responses)
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self):
        return SimpleNamespace(tools=[SimpleNamespace(name=n) for n in self._tools])

    async def call_tool(self, name, args=None):
        self.calls.append((name, args or {}))
        payload = self._responses[name]
        if isinstance(payload, BaseException):
            raise payload
        return SimpleNamespace(structured_content=payload)


def _evidence_stub(final):
    """A stub evidence gateway whose one template returns `final`."""
    menu = {
        "templates": [{"template_id": "qc", "evidence_type": "quality",
                       "required_roles": []}],
        "variables": [],
    }
    return _StubSession(
        ["describe_evidence", "ask_evidence", "poll_evidence"],
        {"describe_evidence": {"menu": menu}, "ask_evidence": final},
    )


def _run_converse(session, transport):
    return asyncio.run(_converse(
        session, dataset=None, request=None, poll_interval=0.0,
        max_polls=1, transport=transport,
    ))


def test_a_url_is_detected_as_http_and_a_command_as_stdio():
    for url in ("https://h/mcp", "http://h:8931/mcp", "HTTPS://H/mcp"):
        assert _transport_for(url) == "http", url
    for cmd in ("autoevidence-serve --policy p", "/usr/bin/autoevidence-serve",
                'sh -c "autoevidence-serve"'):
        assert _transport_for(cmd) == "stdio", cmd


def test_a_bare_origin_gets_the_default_mount_path():
    assert _normalise_url("https://h") == "https://h/mcp"
    assert _normalise_url("https://h/") == "https://h/mcp"
    # A steward who mounted the gateway somewhere else meant it.
    assert _normalise_url("https://h/custom") == "https://h/custom"


def test_an_http_gateway_without_a_key_is_denied_before_connecting(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)

    def _explode(*a, **k):  # pragma: no cover -- must never run
        raise AssertionError("opened a connection without a credential")

    monkeypatch.setattr(client, "_http_session", _explode)
    monkeypatch.setattr(client, "_stdio_session", _explode)
    with pytest.raises(EvidenceDenied) as e:
        fetch_source("https://gateway.example.org/mcp")
    assert "--evidence-key" in str(e.value) and "mint-key" in str(e.value)


def test_the_key_comes_from_the_environment_when_the_flag_is_absent(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, _SENTINEL_KEY)
    assert _resolve_api_key() == _SENTINEL_KEY
    # An explicit key wins, and neither an empty flag nor an empty variable
    # counts as a credential.
    assert _resolve_api_key("ae_other") == "ae_other"
    monkeypatch.setenv(API_KEY_ENV_VAR, "   ")
    assert _resolve_api_key() is None


def test_the_key_never_appears_in_a_returned_result(monkeypatch):
    """The credential reaches the opener and nothing else."""
    seen: dict[str, str] = {}
    stub = _evidence_stub({"kind": "capsule", "signed": {"capsule": {"n_band": "50-199"}}})

    @asynccontextmanager
    async def _fake_http(url, api_key):
        seen["key"] = api_key
        yield stub

    monkeypatch.setattr(client, "_http_session", _fake_http)
    result = fetch_source("https://gateway.example.org/mcp", api_key=_SENTINEL_KEY)
    assert seen["key"] == _SENTINEL_KEY          # it did reach the transport
    assert _SENTINEL_KEY not in repr(result)     # and stopped there
    assert result["kind"] == "evidence"


def test_one_conversation_body_serves_both_transports():
    """The transport chooses a socket, not an answer."""
    final = {"kind": "capsule", "signed": {"capsule": {"n_band": "50-199"}}}
    over_stdio = _run_converse(_evidence_stub(final), "stdio")
    over_http = _run_converse(_evidence_stub(final), "http")
    assert over_stdio == over_http

    # ...and the logic was MOVED, not copied: one call site each, so a poll loop
    # or a denial rule cannot drift between two transports.
    src = inspect.getsource(client)
    assert src.count('call_tool("describe_evidence"') == 1
    assert src.count('call_tool("ask_evidence"') == 1
    assert src.count('"poll_evidence"') == 1  # wrapped over lines; count the name


def test_an_mcp_error_becomes_a_denial_not_an_exception_group():
    from mcp.shared.exceptions import MCPError

    refusal = MCPError(-32001, "no access role for this credential")
    session = _StubSession(["describe_evidence", "ask_evidence"],
                           {"describe_evidence": refusal})
    result = _run_converse(session, "http")
    assert result["ok"] is False
    assert "no access role" in result["denials"][0]


def test_an_open_data_capsule_over_http_is_refused():
    """A staged file the gateway named is on the GATEWAY's disk."""
    staged = {"kind": "open_data", "signed": {"capsule": {"staged_path": "/srv/ae/x.csv"}}}
    over_http = _run_converse(_evidence_stub(staged), "http")
    assert over_http["ok"] is False
    assert "stdio" in over_http["denials"][0]
    # The same capsule over stdio is exactly what the bulk-release path wants.
    over_stdio = _run_converse(_evidence_stub(staged), "stdio")
    assert over_stdio["kind"] == "open_data"


def test_stdio_behaviour_is_unchanged():
    """The stdio opener still forwards the environment and strips HF offline."""
    src = inspect.getsource(client._stdio_session)
    assert "dict(os.environ)" in src
    for flag in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        assert flag in src
    # A command line is still shell-split, never treated as a URL.
    assert _transport_for("autoevidence-serve --policy p --source s") == "stdio"


def test_an_exported_api_key_never_blocks_a_non_evidence_run():
    """Regression for the envvar= footgun on --evidence-key.

    The help text tells users to prefer $AUTOEVIDENCE_API_KEY over the flag. If
    the typer Option declared envvar="AUTOEVIDENCE_API_KEY", typer would fill
    the parameter from an exported key and the mutual-exclusion check
    (`evidence_key and not evidence_server`) would abort every plain
    --data-path run with an error naming a flag the user never passed. So the
    Option must NOT declare the envvar; the client reads the environment itself
    as its per-call fallback, and run.py consults os.environ directly for the
    early https fail-closed check.
    """
    import kosmos.cli.commands.run as run_mod

    opt = inspect.signature(run_mod.run_research).parameters["evidence_key"].default
    assert getattr(opt, "envvar", None) is None, (
        "--evidence-key must not declare envvar=: an exported "
        "$AUTOEVIDENCE_API_KEY would trip the mutual-exclusion error on runs "
        "that use no evidence server at all"
    )
    # The early fail-closed check for an https gateway must still see the
    # exported key -- via os.environ, not via typer's fill.
    src = inspect.getsource(run_mod.run_research)
    assert 'os.environ.get("AUTOEVIDENCE_API_KEY")' in src
