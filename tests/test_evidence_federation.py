"""Phase 1a: N gateways in one run, all of them described to hypothesis generation.

The capability this adds is GROUNDING, not combination. What these tests pin
down, in order of how badly each would hurt if it regressed:

  1. Every source reaches the hypothesis prompt. A source that vanishes silently
     is the worst outcome here, because the run still succeeds and the report
     still reads as though it saw everything.
  2. Each source is described with the regime it ACTUALLY got. A bulk release
     announced as "banded aggregates, never raw rows" tells the model the
     opposite of the truth about data it is holding in full.
  3. No experiment gains a second dataset. `data_path` stays one file.
  4. A single source produces byte-identical grounding to today.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kosmos.evidence.federation import (
    EvidenceConfigError,
    EvidenceSource,
    MaterializedSource,
    load_sources,
    materialize_sources,
)


# --- config loading --------------------------------------------------------

def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "evidence.yaml"
    p.write_text(body)
    return p


def test_a_source_list_loads_with_names_and_a_default_primary(tmp_path):
    cfg = _write(tmp_path, """
version: 1
sources:
  - name: screen_k562
    server: "autoevidence-serve --policy a.yaml --source a.duckdb"
    dataset: k562
  - name: screen_rpe1
    server: "autoevidence-serve --policy b.yaml --source b.duckdb"
    dataset: rpe1
""")
    sources = load_sources(cfg)
    assert [s.name for s in sources] == ["screen_k562", "screen_rpe1"]
    # Unstated primary defaults to the first, the least surprising reading.
    assert sources[0].primary and not sources[1].primary
    # Unstated overlap is `unknown`, which assumes the most.
    assert {s.subject_overlap for s in sources} == {"unknown"}


def test_a_declared_primary_wins_and_two_are_refused(tmp_path):
    cfg = _write(tmp_path, """
sources:
  - {name: a, server: "x", dataset: a}
  - {name: b, server: "y", dataset: b, primary: true}
""")
    assert load_sources(cfg)[1].primary

    both = _write(tmp_path, """
sources:
  - {name: a, server: "x", primary: true}
  - {name: b, server: "y", primary: true}
""")
    with pytest.raises(EvidenceConfigError, match="primary"):
        load_sources(both)


def test_duplicate_names_are_refused(tmp_path):
    """Names key the staging directory, so a duplicate overwrites data."""
    cfg = _write(tmp_path, """
sources:
  - {name: cohort, server: "x"}
  - {name: cohort, server: "y"}
""")
    with pytest.raises(EvidenceConfigError, match="duplicate source name"):
        load_sources(cfg)


def test_a_misspelled_key_is_refused_not_ignored(tmp_path):
    """A silently-ignored `key_env` typo means an unauthenticated call."""
    cfg = _write(tmp_path, """
sources:
  - {name: a, server: "x", keyenv: MY_KEY}
""")
    with pytest.raises(EvidenceConfigError, match="unknown key"):
        load_sources(cfg)


def test_a_missing_credential_fails_at_load_not_mid_run(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_UNSET_KEY", raising=False)
    cfg = _write(tmp_path, """
sources:
  - {name: a, server: "https://gw/mcp", key_env: SOME_UNSET_KEY}
""")
    with pytest.raises(EvidenceConfigError, match="SOME_UNSET_KEY"):
        load_sources(cfg)


def test_an_invalid_overlap_value_is_refused(tmp_path):
    cfg = _write(tmp_path, """
sources:
  - {name: a, server: "x", subject_overlap: probably}
""")
    with pytest.raises(EvidenceConfigError, match="subject_overlap"):
        load_sources(cfg)


def test_the_key_never_appears_in_the_redacted_view(tmp_path, monkeypatch):
    monkeypatch.setenv("AE_TEST_KEY", "ae_supersecret")
    cfg = _write(tmp_path, """
sources:
  - {name: a, server: "https://gw/mcp", key_env: AE_TEST_KEY}
""")
    source = load_sources(cfg)[0]
    assert source.api_key == "ae_supersecret"
    blob = json.dumps(source.redacted())
    assert "ae_supersecret" not in blob
    assert '"authenticated": true' in blob.lower()


# --- materialising ---------------------------------------------------------

def _fake_fetch(results: dict):
    def fetch(server, dataset=None, api_key=None):
        if isinstance(results[server], Exception):
            raise results[server]
        return results[server]
    return fetch


def _fake_stage(rows: str = "a,b\n1,2\n"):
    def stage(signed, out_path):
        Path(out_path).write_text(rows)
        return Path(out_path)
    return stage


def test_each_source_is_staged_into_its_own_directory(tmp_path):
    sources = [
        EvidenceSource(name="one", server="s1", dataset="d1", primary=True),
        EvidenceSource(name="two", server="s2", dataset="d2"),
    ]
    fetch = _fake_fetch({
        "s1": {"kind": "evidence", "signed": {"capsule": {"dataset": "d1"}}, "menu": {}},
        "s2": {"kind": "evidence", "signed": {"capsule": {"dataset": "d2"}}, "menu": {}},
    })
    mats = materialize_sources(sources, tmp_path, fetch=fetch, stage=_fake_stage())

    assert [m.source.name for m in mats] == ["one", "two"]
    assert all(m.has_rows for m in mats)
    # Distinct files, so two stewards both naming their dataset `cohort` cannot
    # overwrite each other.
    assert len({m.path for m in mats}) == 2
    assert mats[0].path == tmp_path / "one" / "one.csv"


def test_a_failed_source_is_carried_not_dropped(tmp_path):
    """A run that lost a source must be able to say so."""
    sources = [
        EvidenceSource(name="ok", server="s1", primary=True),
        EvidenceSource(name="dead", server="s2"),
    ]
    fetch = _fake_fetch({
        "s1": {"kind": "evidence", "signed": {"capsule": {"dataset": "d1"}}},
        "s2": RuntimeError("gateway refused: unknown_dataset"),
    })
    mats = materialize_sources(sources, tmp_path, fetch=fetch, stage=_fake_stage())

    assert len(mats) == 2
    dead = mats[1]
    assert dead.kind == "failed" and not dead.has_rows
    assert "unknown_dataset" in dead.error
    assert "UNAVAILABLE" in dead.describe_release()


def test_the_per_source_key_is_the_one_that_travels(tmp_path):
    """Two stewards, two keys: source two must not authenticate with source one's."""
    seen: list[tuple[str, str | None]] = []

    def fetch(server, dataset=None, api_key=None):
        seen.append((server, api_key))
        return {"kind": "evidence", "signed": {"capsule": {"dataset": "d"}}}

    sources = [
        EvidenceSource(name="a", server="s1", api_key="key-a", primary=True),
        EvidenceSource(name="b", server="s2", api_key="key-b"),
    ]
    materialize_sources(sources, tmp_path, fetch=fetch, stage=_fake_stage())
    assert seen == [("s1", "key-a"), ("s2", "key-b")]


# --- the release regime, which the model is told -------------------------

def test_a_bulk_release_is_not_described_as_banded():
    """The defect this replaces: a full release announced as bands.

    A bulk capsule carries the same menu as a gated one, so the description
    cannot be inferred from the menu's presence -- it comes from the capsule kind.
    """
    bulk = MaterializedSource(
        source=EvidenceSource(name="b", server="s"), dataset_id="d",
        kind="open_data", path=Path("/tmp/x.csv"),
        body={"n_rows": 32561, "n_columns": 15},
    )
    text = bulk.describe_release()
    assert "IN FULL" in text and "exact values" in text
    assert "banded" not in text.lower()
    assert "32,561 rows" in text

    gated = MaterializedSource(
        source=EvidenceSource(name="g", server="s"), dataset_id="d",
        kind="evidence", path=Path("/tmp/y.csv"),
    )
    assert "banded aggregates" in gated.describe_release()
    assert "never raw rows" in gated.describe_release()

    schema = MaterializedSource(
        source=EvidenceSource(name="s", server="s"), dataset_id="d", kind="schema",
    )
    assert "no rows" in schema.describe_release().lower()
    assert not schema.has_rows


# --- grounding: the whole point of the phase ------------------------------

class _Director:
    """The grounding methods under test, without constructing a full agent."""

    from kosmos.agents.research_director import ResearchDirectorAgent as _RD

    _CONTEXT_SAMPLE_ROWS = _RD._CONTEXT_SAMPLE_ROWS
    _CAPSULE_RENDER_COLUMNS = _RD._CAPSULE_RENDER_COLUMNS
    _federated_data_context = _RD._federated_data_context
    _describe_source = _RD._describe_source
    _summarise_file = _RD._summarise_file
    _menu_to_context = _RD._menu_to_context
    _combination_instructions = _RD._combination_instructions
    _shared_columns = _RD._shared_columns
    _mountable_datasets = _RD._mountable_datasets
    # `_combination_instructions` now tells hypothesis generation which datasets
    # can actually be related, so the double needs the method that measures it.
    _merge_compatibility = _RD._merge_compatibility
    _id_shape = _RD._id_shape
    evidence_dataset = "primary"

    def __init__(self, datasets):
        self.datasets = datasets


def _mat(name, dataset, kind="evidence", menu=None, path=None, overlap="unknown"):
    return MaterializedSource(
        source=EvidenceSource(name=name, server="s", dataset=dataset,
                              subject_overlap=overlap),
        dataset_id=dataset, kind=kind, path=path, menu=menu, body={},
    )


_MENU = {
    "variables": [
        {"name": "perturbation", "dtype": "categorical",
         "description": "Target gene.", "allowed_roles": ["group"]},
    ],
    "templates": [{"template_id": "group_counts"}],
}


def test_every_source_reaches_the_prompt(tmp_path):
    """The regression that would be invisible: a source silently missing."""
    f = tmp_path / "d.csv"
    f.write_text("gene,effect\nTP53,0.4\nMYC,0.2\n")
    d = _Director([
        _mat("with_menu", "k562", menu=_MENU, path=f),
        _mat("no_menu", "rpe1", path=f),          # discovery path returns no menu
        _mat("schema_only", "private", kind="schema"),
    ])
    d.datasets[2].body = {"columns": [{"name": "age", "dtype": "numeric"}]}

    ctx = d._federated_data_context()
    assert "3 datasets AT ONCE" in ctx
    for dataset in ("k562", "rpe1", "private"):
        assert f"Dataset '{dataset}'" in ctx, f"{dataset} vanished from grounding"
    # The menu-less source fell back to its file, not to nothing.
    assert "gene" in ctx and "effect" in ctx
    # The schema-only source contributed its columns.
    assert "age" in ctx


def test_the_prompt_requires_a_combined_hypothesis_when_mounting_is_allowed(tmp_path):
    """Being handed N datasets is only worth anything if a claim needs several.

    Left permissive the model writes one hypothesis per dataset, each answerable
    alone -- so the run learns nothing N separate runs would not have. The
    instruction therefore REQUIRES a combined claim, and names the two forms a
    combined claim can take.
    """
    a = tmp_path / "a.csv"; a.write_text("gene,effect\nTP53,0.4\n")
    b = tmp_path / "b.csv"; b.write_text("gene,beta\nTP53,0.2\n")
    d = _Director([_mat("one", "d1", path=a), _mat("two", "d2", path=b)])
    ctx = d._federated_data_context()

    assert "COMBINING THEM -- REQUIRED" in ctx
    assert "cannot be tested from any single dataset alone" in ctx
    assert "per-feature JOIN" in ctx
    assert "ESTIMATE-LEVEL relation" in ctx
    # Subject linkage stays forbidden however the claim is framed.
    assert "Never link rows that represent individual SUBJECTS" in ctx
    # The stale Phase 1a claim must be gone: experiments CAN now open several.
    assert "each experiment reads exactly one dataset" not in ctx


def test_a_genuine_shared_feature_column_is_offered_as_a_join_key(tmp_path):
    a = tmp_path / "a.csv"; a.write_text("gene,effect\nTP53,0.4\n")
    b = tmp_path / "b.csv"; b.write_text("gene,beta\nTP53,0.2\n")
    d = _Director([_mat("one", "d1", path=a), _mat("two", "d2", path=b)])
    ctx = d._federated_data_context()
    assert "candidate join keys): gene" in ctx


def test_capsule_render_columns_are_never_offered_as_join_keys(tmp_path):
    """`n_band` is shared by construction, so a join on it is meaningless.

    Every banded capsule renders the same vocabulary -- n_band, mean_band,
    group. Offering those as candidate keys invites merging two unrelated
    datasets on the coincidence that both had a cohort in the same size band.
    """
    a = tmp_path / "a.csv"; a.write_text("group,n_band,mean,mean_band\n1,5000+,2,2\n")
    b = tmp_path / "b.csv"; b.write_text("group,n_band,mean,mean_band\n1,5000+,3,3\n")
    d = _Director([_mat("one", "d1", path=a), _mat("two", "d2", path=b)])

    assert d._shared_columns(d.datasets) == set()
    ctx = d._federated_data_context()
    assert "candidate join keys" not in ctx
    assert "share NO column names" in ctx


def test_unknown_overlap_warns_against_assuming_independence(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("a,b\n1,2\n")
    d = _Director([
        _mat("one", "d1", path=f, overlap="unknown"),
        _mat("two", "d2", path=f, overlap="unknown"),
    ])
    assert "NOT statistically independent" in d._federated_data_context()

    disjoint = _Director([
        _mat("one", "d1", path=f, overlap="disjoint"),
        _mat("two", "d2", path=f, overlap="disjoint"),
    ])
    assert "NOT statistically independent" not in disjoint._federated_data_context()


def test_a_failed_source_is_named_so_the_model_does_not_reason_about_it(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("a,b\n1,2\n")
    d = _Director([
        _mat("alive", "d1", path=f),
        _mat("alive2", "d2", path=f),
        MaterializedSource(source=EvidenceSource(name="gone", server="s"),
                           dataset_id="d3", kind="failed", error="401"),
    ])
    ctx = d._federated_data_context()
    assert "Unavailable in this run" in ctx and "gone" in ctx


def test_one_source_produces_no_multi_dataset_scaffolding(tmp_path):
    """A single federated source must read exactly like today's single-source run."""
    f = tmp_path / "d.csv"
    f.write_text("a,b\n1,2\n")
    d = _Director([_mat("only", "d1", menu=_MENU, path=f)])
    ctx = d._federated_data_context()
    assert "AT ONCE" not in ctx
    assert "### Dataset" not in ctx
    assert "perturbation" in ctx


# --- the renderer/constant binding ------------------------------------------

def _columns_the_renderer_actually_emits() -> set:
    """Derive the renderer's own vocabulary by RUNNING it on each capsule shape.

    Each shape is fed a capsule carrying no dataset content, so every key that
    comes back was put there by `capsule_to_rows` itself rather than by the
    data. That is the definition `RENDERED_COLUMNS` is supposed to encode, and
    computing it here is what stops the constant from being a hand-copy.
    """
    from kosmos.evidence.client import capsule_to_rows

    shapes = [
        {"groups": [{}]},                        # group-summary release
        {"estimates": [{}]},                     # regression release
        {},                                      # scalars/summary fallback
        {"cohort_table": [{"cell": {"_row": 0}}]},  # row-level extract
    ]
    emitted: set = set()
    for shape in shapes:
        for row in capsule_to_rows({"capsule": shape}):
            emitted |= set(row)
    return emitted


def test_rendered_columns_matches_the_renderer():
    """The constant must equal what the renderer emits -- no hand-copy drift.

    If someone adds a column to `capsule_to_rows` and forgets the constant, that
    column starts looking like a feature two datasets genuinely share, and gets
    offered to the model as a candidate join key. The failure is silent and
    produces a meaningless merge, so it is pinned here rather than trusted.
    """
    from kosmos.evidence.client import RENDERED_COLUMNS

    emitted = _columns_the_renderer_actually_emits()

    missing = emitted - RENDERED_COLUMNS
    assert not missing, (
        f"capsule_to_rows emits {sorted(missing)}, which RENDERED_COLUMNS does "
        f"not list. Add them, or they will be offered as join keys."
    )
    stale = RENDERED_COLUMNS - emitted
    assert not stale, (
        f"RENDERED_COLUMNS lists {sorted(stale)}, which the renderer no longer "
        f"emits. Remove them, or a real dataset column of that name will be "
        f"silently excluded from candidate join keys."
    )


def test_the_director_reads_the_constant_from_the_renderer():
    """The director must not carry its own copy of the vocabulary."""
    from kosmos.agents.research_director import ResearchDirectorAgent
    from kosmos.evidence.client import RENDERED_COLUMNS

    d = ResearchDirectorAgent.__new__(ResearchDirectorAgent)
    assert d._CAPSULE_RENDER_COLUMNS is RENDERED_COLUMNS


# --- the steward's dataset-level account must reach hypothesis generation ----

def test_the_menu_description_is_rendered(tmp_path):
    """It was fetched and dropped, so the one warning that mattered never landed.

    The t1_gwas policy says colocalisation cannot be computed from a
    significant-hits-only release. Hypothesis generation never saw it, proposed
    colocalisation anyway, tested 15 loci and found PP.H4 > 0.8 in none.
    """
    d = _Director([])
    ctx = d._menu_to_context(
        {
            "description": "Significant associations only; COLOCALISATION CANNOT BE COMPUTED.",
            "variables": [{"name": "beta", "dtype": "numeric", "description": "Effect on T1."}],
            "templates": [],
        },
        dataset_id="t1_gwas",
    )

    assert "from the data steward" in ctx
    assert "COLOCALISATION CANNOT BE COMPUTED" in ctx
    assert "beta" in ctx, "variables must still render"


def test_menu_notes_are_rendered(tmp_path):
    d = _Director([])
    ctx = d._menu_to_context(
        {
            "description": "A dataset.",
            "notes": ["Derived from the parquet conversion branch, which can lag main."],
            "variables": [],
            "templates": [],
        },
        dataset_id="x",
    )

    assert "Note: Derived from the parquet conversion branch" in ctx


def test_a_menu_without_a_description_renders_as_before(tmp_path):
    d = _Director([])
    ctx = d._menu_to_context(
        {"variables": [{"name": "beta", "dtype": "numeric", "description": "Effect."}],
         "templates": []},
        dataset_id="x",
    )

    assert "from the data steward" not in ctx
    assert "Variables:" in ctx and "beta" in ctx
