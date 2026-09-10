"""Phase 1b: one experiment reading several datasets, gated on the join key.

Phase 1a let hypothesis generation see N datasets at once. This phase lets a
single experiment OPEN more than one of them -- which is a real widening, so it
is gated. The gate is the join key, not the dataset count:

  * feature-keyed sources (gene, variant, cell line) may be mounted together;
    joining them relates features and re-identifies nobody
  * two subject-keyed sources may not, unless a steward declared their subjects
    `disjoint` -- two individual-level tables in one container can be joined on
    their subject key whatever the code was asked to do

What these tests protect, in order of how badly each would hurt:

  1. The gate itself. A wrong `allowed` here is a disclosure decision.
  2. The codegen bypass. Without it N files mount and nothing opens them, and
     the experiment silently analyses the primary while reporting on all.
  3. Path rewriting into the container, including the prefix and collision
     cases that only appear once there is more than one file.
  4. Single-dataset runs unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kosmos.evidence.federation import (
    EvidenceSource,
    MaterializedSource,
    mountable_together,
)


def _mat(name, *, subject_key=None, overlap="unknown", path="/tmp/x.csv", kind="evidence"):
    return MaterializedSource(
        source=EvidenceSource(
            name=name, server="s", dataset=name,
            subject_overlap=overlap, subject_key=subject_key,
        ),
        dataset_id=name,
        kind=kind,
        path=Path(path) if path else None,
    )


# --- the gate ---------------------------------------------------------------

def test_feature_keyed_sources_may_be_mounted_together():
    """Two screens joined on target gene link features, not people."""
    decision = mountable_together([
        _mat("screen_k562", path="/tmp/k562.csv"),
        _mat("screen_rpe1", path="/tmp/rpe1.csv"),
    ])
    assert decision.allowed
    assert set(decision.names) == {"screen_k562", "screen_rpe1"}


def test_two_subject_keyed_sources_are_refused_when_overlap_is_unknown():
    """The default must refuse: a run never told cannot claim nothing links."""
    decision = mountable_together([
        _mat("clinic_a", subject_key="patient_id", overlap="unknown", path="/tmp/a.csv"),
        _mat("clinic_b", subject_key="patient_id", overlap="unknown", path="/tmp/b.csv"),
    ])
    assert not decision.allowed
    assert "subject-keyed" in decision.reason
    assert decision.names == ()


def test_two_subject_keyed_sources_are_refused_when_overlap_is_shared():
    decision = mountable_together([
        _mat("clinic_a", subject_key="patient_id", overlap="shared", path="/tmp/a.csv"),
        _mat("clinic_b", subject_key="patient_id", overlap="shared", path="/tmp/b.csv"),
    ])
    assert not decision.allowed


def test_subject_keyed_sources_declared_disjoint_may_be_mounted():
    """Declared-disjoint subjects cannot be linked, so there is nothing to gate."""
    decision = mountable_together([
        _mat("cohort_a", subject_key="patient_id", overlap="disjoint", path="/tmp/a.csv"),
        _mat("cohort_b", subject_key="patient_id", overlap="disjoint", path="/tmp/b.csv"),
    ])
    assert decision.allowed


def test_one_subject_keyed_source_among_feature_keyed_ones_is_allowed():
    """Linkage needs two subject-keyed tables; one cannot be joined to nobody."""
    decision = mountable_together([
        _mat("clinic", subject_key="patient_id", path="/tmp/a.csv"),
        _mat("gtex", path="/tmp/b.csv"),
    ])
    assert decision.allowed


def test_a_single_source_is_never_a_multi_mount():
    decision = mountable_together([_mat("only", path="/tmp/a.csv")])
    assert not decision.allowed
    assert "only one source" in decision.reason


def test_schema_only_sources_do_not_count_toward_mounting():
    """A source with no rows contributes no file, so it cannot be co-mounted."""
    decision = mountable_together([
        _mat("with_rows", path="/tmp/a.csv"),
        _mat("schema", kind="schema", path=None),
    ])
    assert not decision.allowed


# --- the codegen bypass -----------------------------------------------------

class _StubLLM:
    """Records the prompt it was asked to answer."""

    # The default has to pass `_validate_substance`: code that opens no data
    # is now rejected and falls back, which is the point of that gate.
    def __init__(
        self,
        code="import pandas as pd\ndf = pd.read_csv(data_path)\nresults = {'n': len(df)}\n",
    ):
        self.code = code
        self.prompts: list[str] = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return f"```python\n{self.code}```"


def _generator(datasets, llm=None):
    from kosmos.execution.code_generator import ExperimentCodeGenerator

    gen = ExperimentCodeGenerator.__new__(ExperimentCodeGenerator)
    gen.use_templates = True
    gen.use_llm = True
    gen.llm_enhance_templates = False
    gen.llm_client = llm or _StubLLM()
    gen.datasets = datasets
    gen.dataset_context = None
    gen.templates = []
    return gen


def _protocol():
    from kosmos.models.experiment import (
        ExperimentProtocol,
        ExperimentType,
        ProtocolStep,
        ResourceRequirements,
    )

    return ExperimentProtocol(
        name="cross-screen comparison",
        hypothesis_id="hyp_1",
        experiment_type=ExperimentType.DATA_ANALYSIS,
        domain="biology",
        description="Compare effect sizes between two perturbation screens.",
        objective="Test whether per-gene effects reproduce across cell lines.",
        steps=[
            ProtocolStep(
                step_number=1,
                title="Compare per-gene effects",
                description="Correlate per-gene effect sizes between screens.",
                action="Join both screens on target gene and correlate.",
            )
        ],
        variables={},
        resource_requirements=ResourceRequirements(),
        statistical_tests=[],
    )


def test_multi_dataset_generation_bypasses_the_template_matcher():
    """The defect this exists to prevent: N files mounted, none opened.

    Every shipped template loads one table from `data_path`, and the catch-all
    template matches any DATA_ANALYSIS protocol -- so without a bypass ahead of
    matching, the LLM path is unreachable and the extra datasets are ignored.
    """
    from kosmos.execution.code_generator import ExperimentCodeGenerator

    llm = _StubLLM()
    gen = _generator({"a": "/tmp/a.csv", "b": "/tmp/b.csv"}, llm)

    called = {"matched": False}

    def _never(protocol):
        called["matched"] = True
        return None

    gen._match_template = _never
    code = ExperimentCodeGenerator.generate(gen, _protocol())

    assert llm.prompts, "the LLM was never asked: the bypass did not fire"
    assert called["matched"] is False, "template matching ran before the bypass"
    assert "results" in code


def test_the_multi_dataset_prompt_names_every_dataset_and_bars_subject_linkage():
    llm = _StubLLM()
    gen = _generator({"k562": "/tmp/k.csv", "rpe1": "/tmp/r.csv"}, llm)
    gen._match_template = lambda protocol: None
    from kosmos.execution.code_generator import ExperimentCodeGenerator

    ExperimentCodeGenerator.generate(gen, _protocol())

    prompt = llm.prompts[0]
    assert "2 DATASETS" in prompt
    assert "'k562'" in prompt and "'rpe1'" in prompt
    assert "FEATURE identifier" in prompt
    assert "Do NOT link" in prompt
    # The single-table instruction must be REPLACED, not appended: telling the
    # model to use only data_path contradicts handing it two datasets.
    assert "ONLY the real dataset at" not in prompt


def test_a_single_dataset_still_takes_the_template_path():
    """One dataset must not change behaviour at all."""
    from kosmos.execution.code_generator import ExperimentCodeGenerator

    llm = _StubLLM()
    gen = _generator({"only": "/tmp/a.csv"}, llm)
    marker = "# from template\nresults = {}\n"

    class _T:
        name = "stub"

        def generate(self, protocol):
            return marker

    gen._match_template = lambda protocol: _T()
    code = ExperimentCodeGenerator.generate(gen, _protocol())

    assert code == marker
    assert not llm.prompts, "single-dataset run reached the LLM path"


def test_bypass_failure_falls_back_to_the_template_path():
    """A generation failure must narrow the analysis, never fail the experiment."""
    from kosmos.execution.code_generator import ExperimentCodeGenerator

    class _Boom:
        def generate(self, prompt):
            raise RuntimeError("provider down")

    gen = _generator({"a": "/tmp/a.csv", "b": "/tmp/b.csv"}, _Boom())
    fallback = "# fallback\nresults = {}\n"

    class _T:
        name = "stub"

        def generate(self, protocol):
            return fallback

    gen._match_template = lambda protocol: _T()
    assert ExperimentCodeGenerator.generate(gen, _protocol()) == fallback


# --- mounting into the sandbox ---------------------------------------------

def test_execute_with_data_injects_datasets_as_a_code_literal(tmp_path):
    """As a literal, not a local_var -- or container rewriting cannot reach it."""
    from kosmos.execution.executor import CodeExecutor

    seen = {}

    class _Exec(CodeExecutor):
        def execute(self, code, local_vars=None, retry_on_error=False, **kw):
            seen["code"] = code
            seen["local_vars"] = local_vars
            return "sentinel"

    ex = _Exec.__new__(_Exec)
    out = CodeExecutor.execute_with_data(
        ex, "pass", "/host/a.csv",
        data_files={"a": "/host/a.csv", "b": "/host/b.csv"},
    ) if False else _Exec.execute_with_data(
        ex, "pass", "/host/a.csv",
        data_files={"a": "/host/a.csv", "b": "/host/b.csv"},
    )

    assert out == "sentinel"
    # The mapping is in the code text, where path rewriting can see it.
    assert "datasets = {" in seen["code"]
    assert "/host/b.csv" in seen["code"]
    # And the plumbing key rides in local_vars for the sandbox to mount from.
    assert seen["local_vars"]["__data_files__"] == {"a": "/host/a.csv", "b": "/host/b.csv"}


def test_execute_with_data_is_unchanged_without_data_files():
    from kosmos.execution.executor import CodeExecutor

    seen = {}

    class _Exec(CodeExecutor):
        def execute(self, code, local_vars=None, retry_on_error=False, **kw):
            seen["code"] = code
            seen["local_vars"] = local_vars
            return None

    ex = _Exec.__new__(_Exec)
    _Exec.execute_with_data(ex, "pass", "/host/a.csv")

    assert "datasets = {" not in seen["code"]
    assert seen["local_vars"] == {"data_path": "/host/a.csv"}


def test_longer_host_paths_are_rewritten_first(tmp_path):
    """A path that is a prefix of another must not corrupt it.

    `/run/a.csv` is a prefix of `/run/a.csv.bak`; replacing the short one first
    turns the long one into `/workspace/data/a.csv.bak` built from a
    half-rewritten string that names no real file.
    """
    host_paths = ["/run/a.csv", "/run/a.csv.bak"]
    code = "x = '/run/a.csv'\ny = '/run/a.csv.bak'\n"
    import os

    for host_path in sorted(host_paths, key=len, reverse=True):
        code = code.replace(host_path, f"/workspace/data/{os.path.basename(host_path)}")

    assert "x = '/workspace/data/a.csv'" in code
    assert "y = '/workspace/data/a.csv.bak'" in code
    assert "/workspace/data/a.csv.bak.bak" not in code


def test_a_basename_collision_is_refused_not_silently_overwritten():
    """Two datasets landing on one container path would swap under the analysis."""
    from kosmos.execution.executor import CodeExecutor

    ex = CodeExecutor.__new__(CodeExecutor)
    ex.use_sandbox = True
    ex.sandbox = None

    with pytest.raises(ValueError, match="share the filename"):
        CodeExecutor._execute_in_sandbox(
            ex, "pass",
            {
                "data_path": "/run/one/data.csv",
                "__data_files__": {
                    "one": "/run/one/data.csv",
                    "two": "/run/two/data.csv",
                },
            },
        )


# --- the steward's column descriptions must reach CODE generation -----------

class TestColumnDescriptionsReachCodegen:
    """A join written from column names alone can be silently wrong.

    Observed: the model merged cis-pQTL against the T1 GWAS on `GENPOS`, and the
    container raised "No overlapping variants after merging cis-pQTL and T1
    GWAS". The datasets DO overlap -- `GENPOS` is GRCh38 while the T1 `varId` is
    GRCh37, and the policy says so on both columns. Hypothesis generation saw
    those sentences; code generation was handed names, dtypes and ranges only.
    """

    def _director(self, materialised):
        from kosmos.agents.research_director import ResearchDirectorAgent

        d = ResearchDirectorAgent.__new__(ResearchDirectorAgent)
        d.datasets = materialised
        d._CONTEXT_SAMPLE_ROWS = 100
        return d

    def _source(self, name, variables):
        mat = _mat(name, path=f"/tmp/{name}.csv")
        mat.menu = {"variables": variables}
        return mat

    def test_descriptions_are_collected_per_dataset(self):
        d = self._director([
            self._source("cis_pqtl", [
                {"name": "ID", "description": "chr:pos:ref:alt on GRCh37 -- the join key to the T1 GWAS."},
                {"name": "GENPOS", "description": "Position, GRCh38. NOT the position inside ID."},
                {"name": "BETA", "description": ""},
            ]),
            self._source("t1_gwas", [
                {"name": "varId", "description": "Variant id (GRCh37) -- join key to cis-pQTL ID."},
            ]),
        ])

        described = d._column_descriptions()

        assert "GRCh37" in described["cis_pqtl"]["ID"]
        assert "GRCh38" in described["cis_pqtl"]["GENPOS"]
        assert "GRCh37" in described["t1_gwas"]["varId"]
        # An empty description is not a description.
        assert "BETA" not in described["cis_pqtl"]

    def test_a_source_without_a_menu_contributes_nothing(self):
        d = self._director([_mat("plain", path="/tmp/plain.csv")])
        assert d._column_descriptions() == {}

    def test_the_description_is_rendered_beside_the_column(self, tmp_path):
        csv = tmp_path / "cis_pqtl.csv"
        csv.write_text("ID,GENPOS,BETA\n1:100:A:G,200,0.5\n")
        d = self._director([])

        summary = d._summarise_file(
            str(csv),
            descriptions={"ID": "chr:pos:ref:alt on GRCh37 -- the join key to the T1 GWAS."},
        )

        assert "ID: object -- chr:pos:ref:alt on GRCh37" in summary
        assert "GENPOS: int64" in summary and "GENPOS: int64 --" not in summary

    def test_the_summary_is_unchanged_without_descriptions(self, tmp_path):
        csv = tmp_path / "d.csv"
        csv.write_text("a,b\n1,2\n")
        d = self._director([])

        assert " -- " not in d._summarise_file(str(csv))

    def test_the_context_tells_the_model_to_pick_join_keys_from_descriptions(self, tmp_path):
        csv = tmp_path / "cis_pqtl.csv"
        csv.write_text("ID,GENPOS\n1:100:A:G,200\n")
        d = self._director([
            self._source("cis_pqtl", [
                {"name": "ID", "description": "GRCh37 -- the join key to the T1 GWAS."},
            ]),
        ])

        ctx = d._staged_file_context({"cis_pqtl": str(csv)})

        assert "choose join keys from the descriptions" in ctx
        assert "different genome builds" in ctx
        assert "the join key to the T1 GWAS" in ctx


def test_the_prompt_forbids_hand_written_entity_lists():
    """A curated list written from memory can exclude the whole answer.

    Observed: the model restricted the MR to 23 hand-written ECM gene names.
    Four of them existed in the released pQTL data, and NONE of those four had a
    variant shared with the T1 GWAS -- so the result frame was empty and the
    script died 200 lines later with `KeyError: 'gene'`. Unrestricted, 171
    proteins have a shared variant, including every one the analysis was
    looking for.
    """
    from kosmos.execution.code_generator import ExperimentCodeGenerator

    gen = ExperimentCodeGenerator.__new__(ExperimentCodeGenerator)
    gen.dataset_context = None
    text = gen._data_access_instructions({"a": "/a.csv", "b": "/b.csv"})

    assert "DO NOT FILTER TO A HAND-WRITTEN LIST" in text
    assert "computable FROM the data" in text
    assert "CHECK EVERY FILTER AND JOIN" in text


def test_the_single_dataset_prompt_is_untouched():
    """One dataset keeps the byte-identical instructions it always had."""
    from kosmos.execution.code_generator import ExperimentCodeGenerator

    gen = ExperimentCodeGenerator.__new__(ExperimentCodeGenerator)
    gen.dataset_context = None
    text = gen._data_access_instructions({"only": "/a.csv"})

    assert "DO NOT FILTER TO A HAND-WRITTEN LIST" not in text
    assert "ONLY the real dataset at" in text


class TestMergeCompatibility:
    """The conclusion, not the evidence.

    The staged-file summary already showed the model example values from every
    column -- `17:40602553:TA:T:imp:v1` alongside `chr11_47691640_G_A_b38` --
    and it merged them anyway, twice. What it never received was the finding
    those examples support: that the two identifier spaces cannot meet.
    """

    def _director(self):
        from kosmos.agents.research_director import ResearchDirectorAgent

        d = ResearchDirectorAgent.__new__(ResearchDirectorAgent)
        d.datasets = []
        d._CONTEXT_SAMPLE_ROWS = 100
        return d

    def _files(self, tmp_path, **frames):
        import pandas as pd

        mount = {}
        for name, data in frames.items():
            path = tmp_path / f"{name}.csv"
            pd.DataFrame(data).to_csv(path, index=False)
            mount[name] = str(path)
        return mount

    def test_the_real_four_dataset_verdicts(self, tmp_path):
        mount = self._files(
            tmp_path,
            cis_pqtl={"ID": ["17:40602553:TA:T:imp:v1", "6:160112638:A:G:imp:v1",
                             "1:150264714:T:C:imp:v1", "2:11:A:G:imp:v1"]},
            t1_gwas={"varId": ["6:134379869:A:T", "6:160071652:T:G",
                               "1:150264714:T:C", "2:11:A:G"]},
            heart_eqtl={"variant_id": ["chr11_47691640_G_A_b38", "chr3_160977645_C_T_b38",
                                       "chr1_814773_T_C_b38", "chr2_11_A_G_b38"]},
        )

        text = self._director()._merge_compatibility(mount)

        # The pQTL id carries `:imp:v1` the T1 id does not -- truncation, not equality.
        assert "MERGES AFTER TRUNCATION: cis_pqtl.ID joins t1_gwas.varId" in text
        assert "first 4 ':'-separated fields" in text
        # GRCh37 colon vs GRCh38 underscore: no merge, however cleaned.
        assert "CANNOT MERGE: cis_pqtl.ID" in text
        assert "heart_eqtl.variant_id" in text
        assert "ZERO rows however the columns are cleaned" in text

    def test_identical_shapes_merge_directly(self, tmp_path):
        mount = self._files(
            tmp_path,
            a={"key": ["1:100:A:G", "2:200:C:T", "3:300:G:A"]},
            b={"varId": ["1:100:A:G", "9:900:T:C", "4:400:A:A"]},
        )

        assert "MERGES DIRECTLY: a.key == b.varId" in self._director()._merge_compatibility(mount)

    def test_the_verdict_does_not_depend_on_which_rows_are_sampled(self, tmp_path):
        """Two sorted files opening on different chromosomes still merge.

        Comparing sampled VALUES would report "cannot merge" here, which is the
        false negative this is built to avoid: wrongly telling a run its
        datasets cannot be related is worse than saying nothing.
        """
        mount = self._files(
            tmp_path,
            a={"key": ["1:100:A:G", "1:200:C:T", "1:300:G:A"]},
            b={"varId": ["22:100:A:G", "22:200:C:T", "22:300:G:A"]},
        )

        text = self._director()._merge_compatibility(mount)

        assert "MERGES DIRECTLY" in text
        assert "CANNOT MERGE" not in text

    def test_columns_within_one_dataset_are_not_compared(self, tmp_path):
        mount = self._files(
            tmp_path,
            a={"key": ["1:100:A:G", "2:200:C:T"], "other": ["x_1_y", "x_2_y"]},
            b={"varId": ["1:100:A:G", "9:900:T:C"]},
        )

        text = self._director()._merge_compatibility(mount)
        assert "a.key" in text and "a.other vs" not in text

    def test_plain_columns_are_not_treated_as_identifiers(self, tmp_path):
        mount = self._files(
            tmp_path,
            a={"beta": [0.1, 0.2, 0.3], "gene": ["SOD2", "CILP", "ECM1"]},
            b={"slope": [0.4, 0.5, 0.6]},
        )

        assert self._director()._merge_compatibility(mount) is None

    def test_a_single_dataset_yields_nothing(self, tmp_path):
        mount = self._files(tmp_path, only={"key": ["1:100:A:G", "2:200:C:T"]})
        assert self._director()._merge_compatibility(mount) is None

    def test_the_block_reaches_the_code_generation_context(self, tmp_path):
        mount = self._files(
            tmp_path,
            cis_pqtl={"ID": ["1:100:A:G:imp:v1", "2:200:C:T:imp:v1"]},
            heart_eqtl={"variant_id": ["chr1_100_A_G_b38", "chr2_200_C_T_b38"]},
        )

        ctx = self._director()._staged_file_context(mount)

        assert "MERGE COMPATIBILITY OF IDENTIFIER COLUMNS" in ctx
        assert "CANNOT MERGE" in ctx


class TestLargeFileGuidance:
    """Exit 137 is a memory cap, and it arrives with no traceback at all.

    Observed: `Container exited with code 137 (no stderr produced)` on a run
    whose designed analysis was correct -- a per-protein Wald-ratio MR with FDR
    correction and Steiger filtering. The staged cis-pQTL CSV is 548 MB and the
    container's limit was 2 GB, so pandas was killed mid-read.
    """

    def _director(self):
        from kosmos.agents.research_director import ResearchDirectorAgent

        d = ResearchDirectorAgent.__new__(ResearchDirectorAgent)
        d.datasets = []
        d._CONTEXT_SAMPLE_ROWS = 100
        return d

    def test_a_large_file_is_flagged_with_its_size(self, tmp_path):
        import pandas as pd

        big = tmp_path / "cis_pqtl.csv"
        pd.DataFrame({"ID": ["1:2:A:G"] * 4000, "pad": ["x" * 30000] * 4000}).to_csv(big, index=False)
        assert big.stat().st_size > 100_000_000

        summary = self._director()._summarise_file(str(big))

        assert "MB ON DISK" in summary
        assert "usecols" in summary

    def test_a_small_file_carries_no_such_warning(self, tmp_path):
        import pandas as pd

        small = tmp_path / "t1_gwas.csv"
        pd.DataFrame({"varId": ["1:2:A:G"], "beta": [0.1]}).to_csv(small, index=False)

        summary = self._director()._summarise_file(str(small))

        assert "MB ON DISK" not in summary
        assert "Shape: 1 rows" in summary


class TestSandboxMemoryCeiling:
    """The 2 GB default cost a correct analysis, so there is no default now.

    Unset means no explicit cap: the container may use whatever the Docker VM
    has, which is the upper bound available on the machine.
    """

    def test_unset_means_no_cap(self, monkeypatch):
        from kosmos.execution.sandbox import _default_memory_limit

        monkeypatch.delenv("KOSMOS_SANDBOX_MEMORY", raising=False)
        assert _default_memory_limit() is None

    def test_an_explicit_value_is_honoured(self, monkeypatch):
        from kosmos.execution.sandbox import _default_memory_limit

        monkeypatch.setenv("KOSMOS_SANDBOX_MEMORY", "6g")
        assert _default_memory_limit() == "6g"

    def test_an_empty_value_is_treated_as_unset(self, monkeypatch):
        from kosmos.execution.sandbox import _default_memory_limit

        monkeypatch.setenv("KOSMOS_SANDBOX_MEMORY", "")
        assert _default_memory_limit() is None

    def test_no_cap_omits_the_key_rather_than_sending_none(self):
        """`mem_limit=None` is not the same as not passing it; Docker rejects it."""
        from kosmos.execution.sandbox import DockerSandbox

        s = DockerSandbox.__new__(DockerSandbox)
        s.memory_limit = None
        assert {**({"mem_limit": s.memory_limit} if s.memory_limit else {})} == {}

        s.memory_limit = "6g"
        assert {**({"mem_limit": s.memory_limit} if s.memory_limit else {})} == {"mem_limit": "6g"}


def test_the_context_states_the_staged_file_format(tmp_path):
    """GWAS data is usually TSV, so the delimiter has to be stated, not assumed.

    Observed: `pd.read_csv(datasets['cis_pqtl'], usecols=cis_cols, sep='\\t')`
    against a comma-separated file. Every row parses into one column and pandas
    reports "Usecols do not match columns" listing all eleven real columns as
    missing -- which reads like the wrong file rather than the wrong delimiter.
    """
    import pandas as pd

    from kosmos.agents.research_director import ResearchDirectorAgent

    d = ResearchDirectorAgent.__new__(ResearchDirectorAgent)
    d.datasets = []
    d._CONTEXT_SAMPLE_ROWS = 100

    csv = tmp_path / "cis_pqtl.csv"
    pd.DataFrame({"ID": ["1:100:A:G"], "BETA": [0.1]}).to_csv(csv, index=False)

    ctx = d._staged_file_context({"cis_pqtl": str(csv)})

    assert "COMMA-separated with a header row" in ctx
    assert "do NOT pass sep=" in ctx


def test_the_context_warns_against_recalled_column_names(tmp_path):
    """Recognising the dataset is what broke the run.

    The model knew the Wisconsin breast-cancer data from scikit-learn and wrote
    `mean radius` / `worst radius` from memory; this copy names them
    `radius_mean` / `radius_worst`. Not one of the six columns it asked for
    existed, and its own column check raised
    `ValueError: Missing feature columns: [...]`.
    """
    import pandas as pd

    from kosmos.agents.research_director import ResearchDirectorAgent

    d = ResearchDirectorAgent.__new__(ResearchDirectorAgent)
    d.datasets = []
    d._CONTEXT_SAMPLE_ROWS = 100

    csv = tmp_path / "bc.csv"
    pd.DataFrame({"id": [1], "diagnosis": ["M"], "radius_mean": [17.9]}).to_csv(csv, index=False)

    ctx = d._staged_file_context({"bc": str(csv)})

    assert "IF YOU RECOGNISE THIS DATASET" in ctx
    assert "do NOT write the column names you remember" in ctx
    assert "radius_mean" in ctx, "the real columns must still be listed"


def test_the_context_spells_out_the_string_column_failure(tmp_path):
    """`could not convert string to float: 'B'` -- the most repeated failure.

    The dtypes were listed and ignored twice in one run: the label column
    `diagnosis` (object, values M/B) went into a scaler both times. Listing a
    dtype is not the same as stating what happens when you ignore it.
    """
    import pandas as pd

    from kosmos.agents.research_director import ResearchDirectorAgent

    d = ResearchDirectorAgent.__new__(ResearchDirectorAgent)
    d.datasets = []
    d._CONTEXT_SAMPLE_ROWS = 100

    csv = tmp_path / "bc.csv"
    pd.DataFrame({"id": [1, 2], "diagnosis": ["M", "B"], "radius_mean": [17.9, 12.1]}).to_csv(
        csv, index=False
    )

    ctx = d._staged_file_context({"bc": str(csv)})

    assert "COLUMN TYPES ARE LISTED ABOVE AND THEY BIND" in ctx
    assert "could not convert string to float" in ctx
    assert "select_dtypes('number')" in ctx
    # The dtypes themselves must still be there for the rule to refer to.
    assert "diagnosis: object" in ctx


class TestMissingnessIsStated:
    """A name and a dtype cannot say that a column is empty.

    `Unnamed_32` -- the trailing-comma artifact in the Wisconsin CSV -- is
    569/569 null and looked like a 33rd feature. The run died on its own
    "Dataset contains missing values" check, having been given no way to know
    which column was the problem.
    """

    def _summary(self, tmp_path, frame):
        import pandas as pd

        from kosmos.agents.research_director import ResearchDirectorAgent

        d = ResearchDirectorAgent.__new__(ResearchDirectorAgent)
        d.datasets = []
        d._CONTEXT_SAMPLE_ROWS = 1000
        csv = tmp_path / "d.csv"
        pd.DataFrame(frame).to_csv(csv, index=False)
        return d._summarise_file(str(csv))

    def test_an_all_null_column_is_flagged_as_empty(self, tmp_path):
        summary = self._summary(tmp_path, {
            "radius_mean": [17.9, 12.1, 19.7],
            "Unnamed_32": [None, None, None],
        })

        assert "Unnamed_32" in summary
        assert "EMPTY: every sampled value is missing -- drop this column" in summary

    def test_partial_missingness_is_counted(self, tmp_path):
        summary = self._summary(tmp_path, {
            "radius_mean": [17.9, None, 19.7],
            "texture_mean": [10.4, 11.2, 9.9],
        })

        assert "[1/3 sampled values missing]" in summary
        # A complete column carries no note at all.
        assert "texture_mean: float64\n" in summary + "\n"

    def test_a_complete_frame_gains_no_missingness_notes(self, tmp_path):
        summary = self._summary(tmp_path, {"a": [1, 2], "b": [3.0, 4.0]})

        assert "missing" not in summary
        assert "EMPTY" not in summary


class TestEveryColumnIsNamed:
    """A column the model cannot see is one it will invent.

    The listing stopped at 40 names and said "... and 111 more columns". The
    diabetes-readmission table one-hot encodes age as `age_70`, `age_0_10`,
    `age_20_50` and so on, all past the 40th column -- so the model wrote
    `df['age']` from its memory of the UCI dataset and the container raised
    `KeyError: 'age'`. Names are cheap; the per-column statistics are what cost,
    and those stay capped.
    """

    def _summary(self, tmp_path, n_cols, max_cols=40):
        import pandas as pd

        from kosmos.agents.research_director import ResearchDirectorAgent

        d = ResearchDirectorAgent.__new__(ResearchDirectorAgent)
        d.datasets = []
        d._CONTEXT_SAMPLE_ROWS = 100
        frame = {f"col_{i:03d}": [float(i), float(i + 1)] for i in range(n_cols - 1)}
        frame["age_20_50"] = [1.0, 0.0]  # the column that was hidden
        csv = tmp_path / "wide.csv"
        pd.DataFrame(frame).to_csv(csv, index=False)
        return d._summarise_file(str(csv), max_cols=max_cols)

    def test_a_column_past_the_cap_is_still_named(self, tmp_path):
        summary = self._summary(tmp_path, n_cols=151)

        assert "age_20_50" in summary, "the 151st column was not named"
        names = [l for l in summary.splitlines()
                 if l.startswith("  - ") and "mean=" not in l
                 and "takes values" not in l]
        assert len(names) == 151

    def test_the_note_says_only_the_statistics_are_capped(self, tmp_path):
        summary = self._summary(tmp_path, n_cols=151)

        assert "Every column is named above" in summary
        assert "first 40 of 151" in summary
        assert "more columns" not in summary, "the old note claimed names were missing"

    def test_the_statistics_stay_bounded(self, tmp_path):
        """Names are cheap; per-column summaries are the cost."""
        summary = self._summary(tmp_path, n_cols=151)

        stats = [l for l in summary.splitlines() if "mean=" in l]
        assert len(stats) <= 40

    def test_a_narrow_table_gains_no_note(self, tmp_path):
        summary = self._summary(tmp_path, n_cols=5)

        assert "Every column is named above" not in summary
        assert "age_20_50" in summary


class TestOutcomeValuesAreDisclosed:
    """The column whose values decide the analysis was the one never shown.

    `readmitted` is column 150 of 151, past the statistics cap, so its values
    were invisible. The model encoded it as the original UCI strings --
    `1 if x == '<30' else 0` against a column holding integers 0/1 -- every row
    became 0, its own filter emptied the frame, and train_test_split raised
    `With n_samples=0 ... the resulting train set will be empty`.
    """

    def _summary(self, tmp_path, frame, max_cols=40):
        import pandas as pd

        from kosmos.agents.research_director import ResearchDirectorAgent

        d = ResearchDirectorAgent.__new__(ResearchDirectorAgent)
        d.datasets = []
        d._CONTEXT_SAMPLE_ROWS = 200
        csv = tmp_path / "wide.csv"
        pd.DataFrame(frame).to_csv(csv, index=False)
        return d._summarise_file(str(csv), max_cols=max_cols)

    def _wide(self, n_dummies=120):
        frame = {f"onehot_{i:03d}": [0, 1] for i in range(n_dummies)}
        frame["readmitted"] = [0, 1]
        return frame

    def test_a_trailing_outcome_column_has_its_values_shown(self, tmp_path):
        summary = self._summary(tmp_path, self._wide())

        levels = [l for l in summary.splitlines() if "takes values" in l]
        assert any("readmitted" in l for l in levels), (
            "the outcome column's values were not disclosed"
        )

    def test_value_sets_are_grouped_not_listed_per_column(self, tmp_path):
        """One line per value set, not one per column -- 121 columns, few lines."""
        summary = self._summary(tmp_path, self._wide())

        levels = [l for l in summary.splitlines() if "takes values" in l]
        assert len(levels) <= 12, f"{len(levels)} lines for one value set"

    def test_a_shared_value_set_names_its_members(self, tmp_path):
        summary = self._summary(tmp_path, self._wide(n_dummies=20))

        line = next(l for l in summary.splitlines() if "takes values" in l)
        assert "onehot_000" in line
        assert "(21 columns)" in line, "a truncated list must say how many share the set"

    def test_a_high_cardinality_column_is_not_listed(self, tmp_path):
        summary = self._summary(tmp_path, {"measurement": list(range(200))})

        assert "takes values" not in summary
