"""Talk to an AutoEvidence gateway over MCP and stage the capsule as a CSV.

Kosmos never sees the private store: the gateway returns an Evidence Capsule of
approved, banded statistics, and this module renders it to one row per released
group / estimate / cohort-cell -- a table the rest of the pipeline reads exactly
as it would a `--data-path` CSV.

The deterministic parts (`default_request`, `capsule_to_rows`, `materialize`) are
pure and unit-tested. `fetch_capsule` is the MCP transport around them.

That transport speaks two dialects, chosen by the shape of `--evidence-server`.
A shell command spawns the gateway on stdio -- the original arrangement, in
which the agent launches the very component that is protecting data from it, on
its own machine, as its own uid. An `https://` URL instead reaches a gateway a
data steward already runs on infrastructure this process does not control, and
presents an API key the steward issued. Only the second makes the gateway's
disclosure controls load-bearing rather than courteous, so it is the one to
prefer; the first stays the default because it needs no deployment. What is
asked for does not change between them -- see `_converse`.
"""

from __future__ import annotations

import csv
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:  # pragma: no cover -- typing only; `mcp` stays a lazy import
    from mcp.client.session import ClientSession


class EvidenceDenied(Exception):
    """The gateway refused the request (a Denial, not a capsule)."""


# --- pure: choose a request, and turn a capsule into rows ------------------

def _fill_required(tmpl: dict[str, Any], variables: list[dict[str, Any]]) -> list[dict[str, str]] | None:
    """Fill a template's required roles with DISTINCT eligible variables, or None.

    Distinct-per-role matters: linear_association needs outcome and predictor to
    be different columns, or the gateway rejects it for regressing a variable on
    itself. Returns None when a required role has no unused eligible variable.
    """
    chosen: list[dict[str, str]] = []
    used: set[str] = set()
    for role in tmpl.get("required_roles", []):
        var = next(
            (
                v for v in variables
                if role in v.get("allowed_roles", []) and v["name"] not in used
            ),
            None,
        )
        if var is None:
            return None
        used.add(var["name"])
        chosen.append({"name": var["name"], "role": role})
    return chosen


def candidate_requests(menu: dict[str, Any]) -> list[dict[str, Any]]:
    """Every menu template that can be filled, as ready-to-send requests.

    Ordered simplest-first (fewest required roles), so a no-variable summary or a
    counts template is tried before a regression that may not be estimable on a
    small cohort. `fetch_capsule` tries them in turn and takes the first that
    releases, so one un-estimable template no longer fails the whole fetch.
    """
    templates = menu.get("templates", [])
    variables = menu.get("variables", [])
    if not templates:
        raise EvidenceDenied("evidence menu exposes no templates to request")
    reqs: list[dict[str, Any]] = []
    for tmpl in sorted(templates, key=lambda t: len(t.get("required_roles", []))):
        if tmpl.get("evidence_type") == "records":
            req = _extract_request(tmpl, variables)  # raw passthrough columns
        else:
            chosen = _fill_required(tmpl, variables)
            req = None if chosen is None else {
                "evidence_type": tmpl["evidence_type"],
                "template_id": tmpl["template_id"],
                "variables": chosen,
                "filters": {},
            }
        if req is not None:
            reqs.append(req)
    if not reqs:
        raise EvidenceDenied("no menu template can be satisfied by the exposed variables")
    return reqs


def _extract_request(tmpl: dict[str, Any], variables: list[dict[str, Any]]) -> dict[str, Any] | None:
    """A cohort_extract request over EVERY raw-releasable (passthrough) column.

    The gateway releases these columns' exact values row-by-row, so this is what
    an AI needs to run its own MR/coloc. Each column is named with a menu role it
    allows that the extract template accepts, avoiding `covariate` (which is
    capped by max_covariates). Returns None if the menu exposes no raw column.
    """
    raw = [v for v in variables if v.get("raw_releasable")]
    if not raw:
        return None
    accepted = set(tmpl.get("optional_roles", [])) | set(tmpl.get("required_roles", []))
    picks: list[dict[str, str]] = []
    for v in raw:
        roles = [r for r in v.get("allowed_roles", []) if not accepted or r in accepted]
        role = next((r for r in roles if r != "covariate"), roles[0] if roles else None)
        if role is None:
            return None
        picks.append({"name": v["name"], "role": role})
    return {
        "evidence_type": tmpl["evidence_type"],
        "template_id": tmpl["template_id"],
        "variables": picks,
        "filters": {},
    }


def default_request(menu: dict[str, Any]) -> dict[str, Any]:
    """The single best seed request: the simplest fillable template.

    A template needing no variables (a composition/quality summary) sorts first,
    else the first whose required roles fill from the menu. `fetch_capsule` uses
    the full ordered `candidate_requests` list; this is kept for callers/tests
    that want just the seed.
    """
    return candidate_requests(menu)[0]


def _capsule_body(capsule: dict[str, Any]) -> dict[str, Any]:
    """Accept a SignedCapsule ({capsule: ...}) or a bare EvidenceCapsule."""
    return capsule.get("capsule", capsule)


# The column names THIS RENDERER owns, as opposed to names that come from the
# dataset. Everything below is emitted for every released capsule of the
# matching shape, so any two gated datasets carry them in common -- which makes
# them useless as join keys and actively dangerous to offer as one: a consumer
# told `n_band` is a shared column can merge two unrelated datasets on the
# coincidence that both had a cohort in the same size band.
#
# Kept HERE, beside `capsule_to_rows`, because this is the function that emits
# them: a reader changing the renderer sees the list in the same screen. It was
# previously restated in `agents/research_director.py`, a hand-copy that would
# have gone stale silently the first time a column was added here.
# `tests/test_evidence_federation.py::test_rendered_columns_matches_the_renderer`
# derives this set by RUNNING the renderer and fails if the two drift apart.
#
# `_row` is the one entry the renderer does not itself add: it arrives inside a
# cohort_table `cell`, injected by the gateway's cohort_extract template as a
# uniqueness ordinal. It is listed because from a consumer's side it is exactly
# the same kind of thing -- a column that is present because of how the release
# was constructed, not because the dataset has such a feature.
RENDERED_COLUMNS = frozenset({
    "_row", "n_band", "classification",
    "group", "outcome", "mean", "mean_band", "sd", "median", "median_band",
    "term", "estimate", "std_error", "p_value", "ci_low", "ci_high",
})


def capsule_to_rows(capsule: dict[str, Any]) -> list[dict[str, Any]]:
    """Render a released capsule to a list of row dicts (banded, never raw)."""
    cap = _capsule_body(capsule)
    if cap.get("cohort_table"):
        rows = []
        for r in cap["cohort_table"]:
            row = dict(r.get("cell", {}))
            row["n_band"] = r.get("n_band")
            rows.append(row)
        return rows
    if cap.get("groups"):
        return [
            {
                "group": g.get("group"), "n_band": g.get("n_band"),
                "outcome": g.get("outcome"), "mean": g.get("mean"),
                "mean_band": g.get("mean_band"), "sd": g.get("sd"),
                "median": g.get("median"), "median_band": g.get("median_band"),
            }
            for g in cap["groups"]
        ]
    if cap.get("estimates"):
        return [
            {
                "term": e.get("term"), "estimate": e.get("estimate"),
                "std_error": e.get("std_error"), "p_value": e.get("p_value"),
                "ci_low": e.get("ci_low"), "ci_high": e.get("ci_high"),
            }
            for e in cap["estimates"]
        ]
    # A capsule with only scalars/n_band: a single summary row.
    row: dict[str, Any] = {"n_band": cap.get("n_band"), "classification": cap.get("classification")}
    row.update(cap.get("scalars", {}))
    return [row]


def materialize(capsule: dict[str, Any], out_path: str | Path) -> Path:
    """Write the capsule's rows to a CSV the sandbox can read. Returns the path.

    An OPEN-DATA capsule is different in kind and handled first: it describes a
    public dataset the gateway already staged in full, so there is nothing to
    render and nothing to bound. Copying it to `out_path` rather than returning
    the staged path in place keeps one rule for callers -- the data is always at
    the path this returns -- and leaves the gateway's staging directory as the
    gateway's to manage.
    """
    # Callers pass either the signed wrapper or the capsule body; normalise
    # before looking for anything, or an open-data release silently falls
    # through to the banded-statistics renderer below and the agent receives a
    # two-line file instead of the dataset.
    body = capsule.get("capsule", capsule)

    if "staged_path" in body:
        import shutil

        staged = Path(body["staged_path"])
        if not staged.exists():
            raise EvidenceDenied(
                f"the open-data capsule names {staged}, which does not exist. "
                f"The gateway's staging directory may have been cleared."
            )
        out = Path(out_path)
        shutil.copyfile(staged, out)
        return out

    rows = capsule_to_rows(capsule)
    columns: list[str] = []
    for r in rows:
        for k in r:
            if k not in columns:
                columns.append(k)
    out = Path(out_path)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return out


# --- MCP transport ---------------------------------------------------------
#
# Two transports, one conversation. The transport decides WHO RUNS THE GATEWAY:
# on stdio the agent spawns it as a subprocess, as its own uid, on its own
# machine -- an arrangement in which every disclosure control the gateway
# applies is a courtesy the agent could step around. Over HTTP the gateway is
# already running on the data steward's infrastructure and this process reaches
# it through one authenticated endpoint, which is what the design always
# assumed. That is the whole of the difference, and it is a difference about
# deployment, not about what is asked for: the questions, the poll loop, the
# denial aggregation and the open-data branch are written ONCE, in `_converse`,
# and both transports hand it the same `ClientSession`. A second copy per
# transport would be a second place for those to drift.

API_KEY_ENV_VAR = "AUTOEVIDENCE_API_KEY"
DEFAULT_MOUNT_PATH = "/mcp"

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

_HTTP_STAGED_FILE_DENIAL = (
    "the gateway returned a full-release capsule naming a server-local staged "
    "file, which this client cannot read over HTTP. Serve full releases over stdio."
)


def _transport_for(evidence_server: str) -> str:
    """'http' if the value is a URL, else 'stdio' (a shell command line).

    The rule is deliberately one-way and not a heuristic: a value beginning with
    `http://` or `https://` is ALWAYS a URL and is never shell-split; anything
    else is a command line, as it has always been. There is no third case and no
    sniffing, because both ambiguous outcomes fail badly -- POSTing to
    `autoevidence-serve --policy ...` fails late and confusingly, and
    shell-splitting a URL would EXECUTE the first token of a value that came out
    of a config file.

    Failure mode for a genuinely ambiguous value (a command whose first token is
    itself a URL): unsupported, and it resolves as http. Wrap such a command in
    `sh -c` if it ever exists.
    """
    return "http" if _URL_RE.match(evidence_server.strip()) else "stdio"


def _normalise_url(url: str) -> str:
    """Append the default `/mcp` mount path when the URL names only an origin.

    The gateway mounts its endpoint at `/mcp` (`autoevidence-serve --path`), so a
    bare origin 404s -- and a 404 during `initialize` surfaces as an opaque
    transport error rather than "you left the path off". Guessing here is safe
    because it only ever ADDS the documented default to a URL that named no path
    at all; a URL that names one is left exactly as given, because a steward who
    mounted the gateway somewhere else meant it.
    """
    url = url.strip()
    parts = urlsplit(url)
    if parts.path not in ("", "/"):
        return url
    fixed = urlunsplit(
        (parts.scheme, parts.netloc, DEFAULT_MOUNT_PATH, parts.query, parts.fragment)
    )
    print(
        f"[evidence] {_redact_url(url)} names no path; using the gateway's "
        f"default mount path {DEFAULT_MOUNT_PATH!r}",
        file=sys.stderr,
    )
    return fixed


def _redact_url(url: str) -> str:
    """The endpoint with its query string and fragment removed.

    Used in every message that names the endpoint. A gateway URL should carry no
    query, but "should" is not a control: a URL that ever carried a token (a
    presigned link, a `?key=` a well-meaning operator added) must not be echoed
    into a log, an exception or a run report, and the cheapest way to guarantee
    that is to never print the query at all.
    """
    parts = urlsplit(url)
    if not parts.query and not parts.fragment:
        return url
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", "")) + "?<redacted>"


def _resolve_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """The API key this client presents, from the flag or $AUTOEVIDENCE_API_KEY.

    Kosmos is the CALLER here, presenting a credential a data steward issued to
    it. That is legitimate, and it is not the thing AutoEvidence's design
    forbids: the forbidden move is a caller NAMING its own access role, and this
    key names nothing -- it is an opaque string the gateway looks a role up from,
    server-side, in a table Kosmos can neither read nor write. Kosmos cannot
    widen its own surface by choosing a different value; it can only fail to
    authenticate.

    The key is never logged, never printed, never written into a capsule, a
    report, a run config or an audit row. It lives in this process's memory and
    in exactly one Authorization header. `run.py` deliberately passes it through
    the ENVIRONMENT rather than through `flat_config`, because flat_config is
    handed to every agent and dumped into run artifacts.
    """
    for candidate in (explicit, os.environ.get(API_KEY_ENV_VAR)):
        if candidate and candidate.strip():
            return candidate.strip()
    return None


async def _converse(
    session: "ClientSession",
    *,
    dataset: Optional[str],
    request: Optional[dict[str, Any]],
    poll_interval: float,
    max_polls: int,
    transport: str,
) -> dict[str, Any]:
    """Everything this client does once a session is open, for BOTH transports.

    The transport differs only in how the two streams were obtained; the
    conversation is identical, so it lives here once. `transport` is used for
    exactly one decision (the staged-file check below) and for error text --
    never to change what is asked for, which would make a run's evidence depend
    on how it happened to reach the gateway.

    Every MCPError is caught HERE, inside the session, and turned into a denial.
    anyio re-raises an exception crossing a task-group boundary as a
    BaseExceptionGroup, so an `except MCPError` placed outside the session block
    never fires -- and a gateway's 401 ("this credential maps to no access role")
    would reach the operator as an opaque ExceptionGroup instead of the sentence
    the gateway wrote for exactly this moment.
    """
    import asyncio

    from mcp.shared.exceptions import MCPError

    try:
        listed = await session.list_tools()
        tools = {getattr(t, "name", t) for t in getattr(listed, "tools", listed)}

        # Discovery server: no policy -> schema only.
        if "describe_evidence" not in tools and "describe_dataset" in tools:
            name = dataset
            if not name and "list_datasets" in tools:
                # Nothing was named, so ask. A server started on one dataset
                # knows which one; making the caller repeat it is friction, and
                # getting it wrong is a denial.
                listed = _tool_json(
                    await session.call_tool("list_datasets", {})
                ).get("datasets", [])
                if len(listed) == 1:
                    name = listed[0]
                elif not listed:
                    return {"ok": False, "denials": [
                        "the discovery server serves no datasets"
                    ]}
                else:
                    return {"ok": False, "denials": [
                        f"the server serves {len(listed)} datasets "
                        f"({', '.join(listed)}); name one with "
                        f"--evidence-dataset"
                    ]}
            if not name:
                return {"ok": False, "denials": [
                    "no dataset named and the server cannot list its own"
                ]}
            res = _tool_json(await session.call_tool(
                "describe_dataset", {"dataset": name}
            ))
            if res.get("kind") == "open_data":
                # A public dataset released whole. Nothing to render: the
                # gateway already wrote the file and the capsule names it, so
                # `materialize` just points at it -- when the file is on THIS
                # machine, which over HTTP it is not.
                if transport == "http":
                    return {"ok": False, "denials": [_HTTP_STAGED_FILE_DENIAL]}
                return {"kind": "open_data", "signed": res["signed"]}
            if res.get("kind") == "schema":
                return {"kind": "schema", "signed": res["signed"]}
            reason = (res.get("denial") or {}).get("message", f"kind={res.get('kind')}")
            return {"ok": False, "denials": [f"describe_dataset: {reason}"]}

        if "describe_evidence" not in tools:
            return {"ok": False, "denials": [
                "server exposes neither describe_evidence nor describe_dataset"
            ]}

        # Evidence gateway: try each menu template until one releases.
        # Note the absence of `dataset` here: an evidence gateway is bound to
        # exactly one policy, so the connection already fixes which dataset is
        # being asked about. The menu names it.
        menu = _tool_json(await session.call_tool("describe_evidence", {}))["menu"]
        candidates = [request] if request is not None else candidate_requests(menu)
        denials: list[str] = []
        for req in candidates:
            res = _tool_json(await session.call_tool("ask_evidence", req))
            polls = 0
            while res.get("kind") == "pending" and polls < max_polls:
                await asyncio.sleep(poll_interval)
                polls += 1
                res = _tool_json(await session.call_tool(
                    "poll_evidence", {"review_id": res["pending"]["review_id"]}
                ))
            if res.get("kind") == "open_data":
                # A POLICY-GRANTED BULK RELEASE from an evidence gateway: the
                # steward granted this role the real rows, staged to a file the
                # capsule names. Same tag the public full-release path uses, so
                # `materialize` already knows how to stage it -- but it arrives
                # here, on the evidence path, and without this branch it was
                # being discarded as "no capsule" and the run fell through to
                # templates that could not answer.
                if transport == "http":
                    # The capsule names a SERVER-LOCAL absolute path. Over HTTP
                    # the server is another machine: that path either does not
                    # exist here or, worse, resolves to an unrelated local file
                    # of the same name, which `materialize` would copy in as the
                    # dataset without a word. The gateway refuses --staging-dir
                    # on the http transport for the same reason; this is the
                    # client half of one rule.
                    return {"ok": False, "denials": [_HTTP_STAGED_FILE_DENIAL]}
                return {"kind": "open_data", "signed": res["signed"], "menu": menu}
            if res.get("kind") == "capsule":
                # Carry the menu back too: it names the REAL variables (with
                # descriptions/levels), which grounds hypotheses far better than
                # the capsule's rendered columns.
                return {"kind": "evidence", "signed": res["signed"], "menu": menu}
            reason = (res.get("denial") or {}).get(
                "message", f"no capsule (kind={res.get('kind')})"
            )
            denials.append(f"{req.get('template_id', '?')}: {reason}")
        # No template released. Return the reasons rather than raising here: an
        # EvidenceDenied raised inside the anyio task group comes back wrapped in
        # an opaque ExceptionGroup, so `fetch_source` raises it outside the async
        # context, where its message survives.
        return {"ok": False, "denials": denials}
    except MCPError as e:
        # The gateway's JSON-RPC-shaped 401/403 arrives here with ITS code and
        # message intact (-32001 is "this credential maps to no access role").
        # Returned as a denial rather than re-raised, so it takes the same path
        # home as every other refusal and is reported as one.
        return {"ok": False, "denials": [f"gateway refused the connection: {e}"]}


@asynccontextmanager
async def _stdio_session(evidence_server: str) -> AsyncIterator["ClientSession"]:
    """Spawn the gateway as a subprocess and hand back its session.

    The original arrangement, unchanged: the agent launches the very component
    that is protecting data from it. It survives because it needs no deployment
    -- a steward handing over a policy and a command line has a working gateway
    -- and because the controls still hold against an agent that plays by them.
    """
    import shlex

    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    parts = shlex.split(evidence_server)
    # Forward this process's environment to the gateway subprocess. Without it,
    # mcp's stdio transport launches the server with a stripped default
    # environment (PATH and little else), so a sources config using
    # `token_env: HF_TOKEN` reads an EMPTY token and every private/gated source
    # is refused for lack of a credential the operator did in fact supply. The
    # gateway is a trusted, operator-launched component -- here Kosmos is the
    # operator launching it -- so passing the environment through is the correct
    # scope, not a leak across a trust boundary.
    gateway_env = dict(os.environ)
    # ...but NOT the HuggingFace offline flags. The main Kosmos process may run
    # offline to serve its cached embedding model without a Hub round trip, but
    # the gateway's entire job is to fetch the dataset FRESH from the source and
    # pin it to a revision -- forcing it offline would either fail outright (a
    # moving ref like @main cannot resolve offline) or serve a stale cache,
    # which is no better than a local file. The two HuggingFace consumers want
    # opposite modes; only the model-loading side gets to be offline.
    for _offline in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        gateway_env.pop(_offline, None)
    params = StdioServerParameters(
        command=parts[0], args=parts[1:], env=gateway_env
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@asynccontextmanager
async def _http_session(url: str, api_key: str) -> AsyncIterator["ClientSession"]:
    """Connect to a gateway a data steward already runs.

    The Authorization header goes on the HTTP CLIENT, not on a transport
    argument and never on a tool call: it is a property of this connection, and
    keeping it in one place is what makes "the key appears in exactly one string
    in this package" a checkable claim rather than a hope.

    Note the SDK vendors `httpx2`, not `httpx`: build the client with
    `create_mcp_http_client` and never hand `streamable_http_client` a plain
    `httpx.AsyncClient`, which is a different type it will not accept. The
    timeout is left at the SDK's own default (30s connect/write, 300s read)
    rather than pinned to a flat value -- the read leg holds a long-lived SSE
    stream open across the whole poll loop, and a flat timeout shorter than
    `poll_interval * max_polls` would kill the connection precisely when a
    request is being held for steward review, which is the one case the poll
    loop exists for.
    """
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

    headers = {"Authorization": f"Bearer {api_key}"}
    async with create_mcp_http_client(headers=headers) as http_client:
        # terminate_on_close (the default) sends the session DELETE on the way
        # out, so a gateway serving many callers reclaims the session rather
        # than holding it until it times out.
        async with streamable_http_client(url, http_client=http_client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


def fetch_source(
    evidence_server: str,
    dataset: Optional[str] = None,
    request: Optional[dict[str, Any]] = None,
    poll_interval: float = 2.0,
    max_polls: int = 60,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """Connect to an AutoEvidence server and return its evidence, data OR schema.

    `evidence_server` is either a SHELL COMMAND that spawns a gateway on stdio --
    the original arrangement, where the agent launches the very component that is
    protecting data from it -- or an `https://` URL for a gateway a data steward
    already runs on their own infrastructure. The second is the one the design
    always wanted; the first stays the default because it needs no deployment.
    A URL needs an API key (`api_key=` or $AUTOEVIDENCE_API_KEY): the steward
    mints it, and the gateway looks the ACCESS ROLE up from it server-side, so
    holding one lets this client authenticate and nothing more.

    One entry point for both server kinds, told apart by the tools they expose:
      - an EVIDENCE gateway (describe_evidence / ask_evidence, a policy governs
        the dataset) -> {"kind": "evidence", "signed": <evidence capsule>}
      - a DISCOVERY server (describe_dataset, no policy):
          * public dataset  -> {"kind": "open_data", "signed": <open-data capsule>}
            The full dataset, staged by the gateway; the capsule names the file.
          * private/unknown -> {"kind": "schema", "signed": <schema capsule>}

    Raises EvidenceDenied on a denial or an empty result.
    """
    import asyncio

    transport = _transport_for(evidence_server)
    url = ""
    key: Optional[str] = None
    if transport == "http":
        url = _normalise_url(evidence_server)
        key = _resolve_api_key(api_key)
        if key is None:
            # Fail here, before a socket is opened. A keyless connection to an
            # authenticated gateway is a 401 the caller has to decode; saying so
            # up front, with the command that fixes it, costs one branch.
            raise EvidenceDenied(
                f"an AutoEvidence gateway at {_redact_url(url)} needs an API key "
                f"and none was given. Pass --evidence-key, or set "
                f"${API_KEY_ENV_VAR}. The steward mints one with "
                f"`autoevidence mint-key --principal <you> --role <role>`."
            )
    endpoint = _redact_url(url) if transport == "http" else evidence_server

    async def _run() -> dict[str, Any]:
        opener = (
            _http_session(url, key) if transport == "http"
            else _stdio_session(evidence_server)
        )
        async with opener as session:
            return await _converse(
                session,
                dataset=dataset,
                request=request,
                poll_interval=poll_interval,
                max_polls=max_polls,
                transport=transport,
            )

    try:
        result = asyncio.run(_run())
    except EvidenceDenied:
        raise
    except BaseException as e:  # noqa: BLE001 -- unwrap anyio's ExceptionGroup
        denied = _first_denied(e)
        if denied is not None:
            raise denied
        # A gateway that refuses the HANDSHAKE (a bad, revoked or expired key)
        # fails inside `_http_session`, before `_converse` can catch it, and
        # anyio hands it back wrapped. Unwrap it here so the operator reads the
        # gateway's own sentence instead of "unhandled errors in a TaskGroup".
        refused = _first_mcp_error(e)
        if refused is not None:
            raise EvidenceDenied(
                f"the AutoEvidence gateway at {endpoint} refused the "
                f"connection: {refused}"
            ) from e
        if transport == "http":
            # Everything else that fails on the way to a REMOTE gateway -- a
            # wrong port, a host that is not listening, a TLS failure -- is a
            # deployment mistake somebody has to fix, and anyio hands it over as
            # "unhandled errors in a TaskGroup (1 sub-exception)", which names
            # neither the host nor the problem. Name both. `from e` keeps the
            # original traceback, so nothing is hidden, only introduced. Left
            # deliberately untouched for stdio, whose failure modes have their
            # own established handling.
            leaf = _sole_leaf(e)
            if leaf is not None:
                raise EvidenceDenied(
                    f"could not reach the AutoEvidence gateway at {endpoint}: "
                    f"{type(leaf).__name__}: {leaf}"
                ) from e
        raise
    if result.get("ok") is False:
        raise EvidenceDenied(
            "AutoEvidence server returned nothing usable -- "
            + "; ".join(result["denials"])
        )
    return result


def fetch_capsule(
    evidence_server: str,
    dataset: str,
    request: Optional[dict[str, Any]] = None,
    poll_interval: float = 2.0,
    max_polls: int = 60,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """Fetch an evidence capsule (rows). Thin wrapper over `fetch_source`.

    Kept for callers that specifically want released statistics; raises if the
    server is a discovery (schema-only) server rather than an evidence gateway.
    """
    result = fetch_source(
        evidence_server, dataset, request, poll_interval, max_polls, api_key
    )
    if result["kind"] != "evidence":
        raise EvidenceDenied(
            f"expected an evidence capsule but the server returned {result['kind']!r}"
        )
    return result["signed"]


# --- dataset search --------------------------------------------------------
#
# The one call in this module that asks a gateway for something OTHER than a
# dataset it holds: candidate datasets in public repositories, for a run that
# was given a question and nothing to run it on.
#
# It is deliberately kept apart from `fetch_source` rather than folded in as a
# fourth `kind`. `fetch_source` answers "what may I have from this dataset"; the
# whole of its shape -- a menu, a request, a poll loop, a staged file -- is
# about a dataset that already exists on the far side. A search answers "does a
# dataset exist at all", returns POINTERS and never bytes, and reaches a server
# that may hold no dataset whatsoever. Folding them together would put a branch
# in `_converse` that changes what is asked for, which is the one thing that
# function's docstring says it will not do.

_SEARCH_TOOL = "find_datasets"

_HTTP_SEARCH_DENIAL = (
    "dataset search is served over stdio only. AutoEvidence refuses "
    "--finder-config on its http transport, because the control that bounds an "
    "outbound free-text search -- refusing to search once the session has seen "
    "a private schema -- is per-process, and over http one process serves many "
    "callers. Point --finder-server at a command line that spawns the gateway."
)


def _verify_candidate_capsule(signed: dict[str, Any]) -> bool:
    """True if this candidate capsule carries a signature AutoEvidence made.

    Verified rather than trusted, and a failure is a refusal rather than a
    warning, because the signature is the only thing separating a capsule from
    prose. Everything inside one is third-party text -- repository titles and
    summaries, written by strangers -- and the capsule is what says which
    gateway stands behind having merely RELAYED it, with the intent it sent and
    the endpoint it went to recorded alongside. Unsigned, the same JSON is a
    list of dataset names an intermediary could have chosen.

    `autoevidence` is imported lazily, and its absence is a refusal rather than
    a degradation to trust: the signature is over the pydantic model's own
    canonical JSON, so there is no checking it without the model, and "could not
    check" must not reach the caller wearing the same face as "checked, fine".
    Kosmos imports `autoevidence` nowhere else, and no run that never searches
    is made to import it now.
    """
    from autoevidence.release.signing import verify_candidate_capsule
    from autoevidence.schema.candidates import SignedCandidateCapsule

    return bool(verify_candidate_capsule(SignedCandidateCapsule(**signed)))


async def _search_converse(
    session: "ClientSession",
    *,
    intent: str,
    repositories: Optional[list[str]],
    limit: int,
) -> dict[str, Any]:
    """Ask one discovery server for candidates. Every failure is a denial.

    Note what is NOT sent: no tool name, no endpoint, no URL, no credential.
    Those are the operator's, fixed server-side in `finder.yaml`, and a client
    that could name them would be doing arbitrary tool execution through an
    interface that promises none. All this client chooses is what it is looking
    for, which repositories -- out of a set somebody else fixed -- to look in,
    and how many answers to take.
    """
    from mcp.shared.exceptions import MCPError

    try:
        listed = await session.list_tools()
        tools = {getattr(t, "name", t) for t in getattr(listed, "tools", listed)}
        if _SEARCH_TOOL not in tools:
            return {"ok": False, "denials": [
                "this gateway exposes no dataset search, so it was started "
                "without --finder-config. Ask its operator to start it with "
                "`--finder-config <finder.yaml>` (see AutoEvidence's "
                "finder.yaml.example), or point --finder-server at one that has."
            ]}
        args: dict[str, Any] = {"intent": intent, "limit": limit}
        if repositories:
            args["repositories"] = list(repositories)
        res = _tool_json(await session.call_tool(_SEARCH_TOOL, args))
    except MCPError as e:
        return {"ok": False, "denials": [f"gateway refused the search: {e}"]}

    if res.get("kind") != "candidates":
        reason = (res.get("denial") or {}).get(
            "message", f"no candidates (kind={res.get('kind')})"
        )
        return {"ok": False, "denials": [f"{_SEARCH_TOOL}: {reason}"]}

    signed = res.get("signed")
    if not isinstance(signed, dict):
        return {"ok": False, "denials": [
            f"{_SEARCH_TOOL} returned no signed capsule"
        ]}
    try:
        ok = _verify_candidate_capsule(signed)
    except ImportError as e:
        return {"ok": False, "denials": [
            f"cannot verify the candidate capsule's signature: {e}. Install "
            f"AutoEvidence into this environment (`pip install -e AutoEvidence`); "
            f"an unverifiable capsule is refused rather than trusted."
        ]}
    except Exception as e:  # noqa: BLE001 -- a malformed capsule is a refusal
        return {"ok": False, "denials": [
            f"the candidate capsule did not parse as one: {type(e).__name__}: {e}"
        ]}
    if not ok:
        return {"ok": False, "denials": [
            "the candidate capsule's signature did not verify. Refusing it: "
            "unsigned, its contents are third-party text no gateway stands behind."
        ]}
    return {"ok": True, "signed": signed}


def search_datasets(
    finder_server: str,
    intent: str,
    *,
    repositories: Optional[list[str]] = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Ask an AutoEvidence discovery server for candidate datasets.

    `finder_server` is a SHELL COMMAND that spawns a gateway on stdio. An
    `https://` URL is refused here rather than attempted -- see
    `_HTTP_SEARCH_DENIAL`; this is the client half of a rule the gateway
    enforces on its own side, and refusing before the socket is opened means the
    operator reads that reason instead of a missing-tool error.

    Returns `{"ok": True, "capsule": <SignedCandidateCapsule as a dict>}` or
    `{"ok": False, "denials": [str, ...]}`. Raises nothing: a transport failure,
    a refusal, a missing tool and a bad signature are all the same kind of answer
    to the caller -- "no candidates, and here is why" -- and a command whose job
    is to suggest data has nothing useful to do with a traceback.

    What comes back are POINTERS. Nothing has been fetched, hashed, classified or
    admitted; `reported_public` inside a candidate is the repository's own flag,
    not a classification. A candidate becomes a source only when somebody writes
    it into an `evidence.yaml` (see `kosmos.datasearch.emit`) and AutoEvidence
    classifies it for itself.
    """
    import asyncio

    if _transport_for(finder_server) == "http":
        return {"ok": False, "denials": [_HTTP_SEARCH_DENIAL]}

    intent = (intent or "").strip()
    if not intent:
        return {"ok": False, "denials": ["no intent given; nothing to search for"]}

    async def _run() -> dict[str, Any]:
        async with _stdio_session(finder_server) as session:
            return await _search_converse(
                session, intent=intent, repositories=repositories, limit=limit
            )

    try:
        result = asyncio.run(_run())
    except BaseException as e:  # noqa: BLE001 -- unwrap anyio's ExceptionGroup
        # The same unwrapping the fetch path does, for the same reason: a
        # gateway that dies during the handshake -- a bad --finder-config, a
        # missing ToolUniverse interpreter, both of which AutoEvidence turns into
        # a startup refusal on stderr -- comes back as "unhandled errors in a
        # TaskGroup", which names nothing an operator can act on.
        refused = _first_mcp_error(e)
        leaf = refused or _sole_leaf(e) or e
        head = finder_server.split()[0] if finder_server.split() else "?"
        return {"ok": False, "denials": [
            f"could not reach the dataset-search gateway ({head}): "
            f"{type(leaf).__name__}: {leaf}"
        ]}
    if result.get("ok") is not True:
        return result
    return {"ok": True, "capsule": result["signed"]}


def _first_denied(exc: BaseException) -> "EvidenceDenied | None":
    """Recover an EvidenceDenied wrapped inside an anyio ExceptionGroup."""
    if isinstance(exc, EvidenceDenied):
        return exc
    for sub in getattr(exc, "exceptions", ()) or ():
        found = _first_denied(sub)
        if found is not None:
            return found
    return None


def _first_mcp_error(exc: BaseException) -> Optional[BaseException]:
    """Recover an MCPError wrapped inside an anyio ExceptionGroup, or None.

    Separate from `_first_denied` because it is a different claim: that one
    recovers a refusal THIS module raised, this one recovers one the GATEWAY
    sent -- a JSON-RPC error whose code and message the gateway chose so the
    operator would read something better than a transport failure.
    """
    try:
        from mcp.shared.exceptions import MCPError
    except ImportError:  # pragma: no cover -- mcp absent; nothing to unwrap
        return None
    if isinstance(exc, MCPError):
        return exc
    for sub in getattr(exc, "exceptions", ()) or ():
        found = _first_mcp_error(sub)
        if found is not None:
            return found
    return None


def _sole_leaf(exc: BaseException) -> Optional[BaseException]:
    """The one underlying exception of a chain of single-child ExceptionGroups.

    Only when every group on the way down holds exactly one child: then there is
    no ambiguity about which failure the group is reporting, and unwrapping loses
    nothing. A group holding several is a genuinely compound failure and is left
    alone rather than having one of its members promoted to speak for the rest.
    """
    subs = getattr(exc, "exceptions", None)
    if not subs or len(subs) != 1:
        return None
    return _sole_leaf(subs[0]) or subs[0]


def _tool_json(result: Any) -> dict[str, Any]:
    """Extract the JSON payload a tool returned, across MCP result shapes."""
    import json

    # A structured-content result carries the dict directly.
    for attr in ("structured_content", "structuredContent", "data"):
        val = getattr(result, attr, None)
        if isinstance(val, dict):
            return val
    # Otherwise the first text content block holds JSON.
    content = getattr(result, "content", None)
    if content:
        text = getattr(content[0], "text", None)
        if text:
            return json.loads(text)
    if isinstance(result, dict):
        return result
    raise EvidenceDenied("could not parse the gateway tool response")
