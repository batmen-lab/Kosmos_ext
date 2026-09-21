"""The orchestration helpers: what counts as a table, and what gets downloaded."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "auto_task_run", ROOT / "scripts" / "auto_task_run.py"
)
auto_task_run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auto_task_run)


def test_screening_keeps_tables_and_skips_a_readme(tmp_path):
    table = tmp_path / "t.csv"
    pd.DataFrame({"age": [1, 2], "num": [0, 1]}).to_csv(table, index=False)
    readme = tmp_path / "README.md"
    readme.write_text("---\ntitle: dataset\n---\n")
    log = auto_task_run.PipelineLog(tmp_path / "log.jsonl")

    kept = auto_task_run.screen_tables([str(table), str(readme)], log)

    assert kept == [str(table)]
    written = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert any("screening" in line["message"] for line in written)


def test_screening_keeps_a_parquet_table(tmp_path):
    """HuggingFace ships parquet, and reading it as text hid the whole table."""
    pytest.importorskip("pyarrow")
    table = tmp_path / "train-00000-of-00001.parquet"
    pd.DataFrame({"age": [1, 2], "num": [0, 1]}).to_parquet(table)
    log = auto_task_run.PipelineLog(tmp_path / "log.jsonl")
    assert auto_task_run.screen_tables([str(table)], log) == [str(table)]


def test_a_converted_source_is_not_reported_as_unusable(tmp_path):
    """An `.h5ad` and the table converted from it are one dataset, not two.

    Both were candidates: the derived table was reviewed as gold and its own
    source turned up beside it as "unusable -- not a table: 0 parsed column(s)",
    because an HDF5 file is not delimited text. The source is a source.
    """
    pytest.importorskip("h5py")
    import numpy as np

    source = tmp_path / "sample.h5ad"
    with __import__("h5py").File(source, "w") as handle:
        handle.create_dataset("X", data=np.zeros((4, 3), dtype=np.float32))
        handle.create_group("obs").create_dataset("batch", data=np.array([b"b1"] * 4))
        handle.create_group("var").create_dataset("_index", data=np.array([b"G0", b"G1", b"G2"]))
    derived = tmp_path / "sample.h5ad-table.csv"
    pd.DataFrame({"G0": [0.0, 1.0], "G1": [1.0, 0.0]}).to_csv(derived, index=False)
    log = auto_task_run.PipelineLog(tmp_path / "log.jsonl")

    tabular, skipped = auto_task_run.screen_candidates([str(source), str(derived)], log)

    assert tabular == [str(derived)]
    assert len(skipped) == 1
    assert skipped[0]["converted"] is True
    assert skipped[0]["derived"] == derived.name
    assert "not a table" not in skipped[0]["reason"]
    assert "converted to sample.h5ad-table.csv" in (tmp_path / "log.jsonl").read_text()


def test_screening_says_why_each_candidate_was_skipped(tmp_path):
    """A page that says "not found" is a fact about the URL, not about the data.

    It arrived as `P_BMX.csv` from a NHANES path that no longer exists, and the
    run stopped with "none of the fetched files is a table" and no reason.
    """
    page = tmp_path / "P_BMX.csv"
    page.write_text(
        "<!DOCTYPE html>\n<html><head><title>Page Not Found | CDC</title>"
        "</head><body>404</body></html>\n"
    )
    table = tmp_path / "t.csv"
    pd.DataFrame({"age": [1, 2], "weight_kg": [70.0, 80.0]}).to_csv(table, index=False)
    log = auto_task_run.PipelineLog(tmp_path / "log.jsonl")

    tabular, skipped = auto_task_run.screen_candidates([str(page), str(table)], log)

    assert tabular == [str(table)]
    assert [entry["path"] for entry in skipped] == [str(page)]
    assert skipped[0]["reason"]
    # Printed, not only stored: the console is where a run is watched.
    assert "P_BMX.csv: " in (tmp_path / "log.jsonl").read_text()


def test_round_one_downloads_only_what_the_caller_asked_for(tmp_path, monkeypatch):
    """Eight references came back for `--fetch-limit 4`, and all eight were reviewed."""
    payload = {
        "query": "q",
        "hits": [],
        "references": ["hf://a/one", "hf://b/two", "geo://GSE1"],
        "failures": {},
    }
    calls: list[list[str]] = []

    def fake_run_command(args, check=True):
        calls.append(list(args))
        if "search-mechanical" in args:
            return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")
        body = {
            "directory": str(tmp_path / "staged"),
            "files": [
                {"path": str(tmp_path / "staged" / "t.csv"), "sha256": "deadbeef" * 8}
            ],
        }
        return subprocess.CompletedProcess(args, 0, json.dumps(body), "")

    monkeypatch.setattr(auto_task_run, "run_command", fake_run_command)
    args = argparse.Namespace(
        root=None, max_bytes=None, fetch_limit=2, max_files_per_fetch=8
    )
    log = auto_task_run.PipelineLog(tmp_path / "log.jsonl")

    auto_task_run.mechanical_round("a question", args, log)

    downloads = [call for call in calls if "fetch" in call]
    assert len(downloads) == 2


def test_a_search_call_without_a_query_term_is_reported(tmp_path):
    log = auto_task_run.PipelineLog(tmp_path / "log.jsonl")
    args = argparse.Namespace(root=None)
    text = auto_task_run.run_model_search("tu://HuggingFace_search_datasets#{}", args, log)
    assert text == ""
    written = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert "carried no query term" in written[0]["message"]


def test_the_search_terms_come_from_measurements_not_row_numbers():
    """Round two searches with the gold's own column names."""
    picked = auto_task_run._distinctive_columns(
        ["id", "__index_level_0__", "sex", "bill_length_mm", "flipper_length_mm", "year"]
    )
    assert "id" not in picked and "__index_level_0__" not in picked
    assert picked[0] == "flipper_length_mm"  # longest first
    assert len(picked) == 4


def test_listing_a_single_file_reference_says_there_is_nothing_to_list(tmp_path):
    from datafetcher.config import DataFetcherConfig
    from datafetcher.fetch import list_reference_files

    payload = list_reference_files(
        "https://example.org/data.csv", DataFetcherConfig(root=tmp_path)
    )
    assert payload["files"] == []
    assert "one file per reference" in payload["note"]


def test_an_archive_makes_its_contents_candidates():
    """UCI ships a zip holding one CSV; the CSV is the table, not the zip."""
    payload = {
        "directory": "/staged/dataset",
        "files": [
            {"path": "dataset.zip", "sha256": "a" * 64},
            {
                "path": "dataset.zip-contents/kidney_disease.csv",
                "sha256": "b" * 64,
                "derived_from": "dataset.zip",
            },
        ],
    }
    candidates = auto_task_run._candidate_records(payload)
    assert [record["path"] for record in candidates] == [
        "dataset.zip",
        "dataset.zip-contents/kidney_disease.csv",
    ]


def test_one_fetch_cannot_flood_the_review():
    """A GEO RAW.tar held 76 files; every one of them used to be reviewed."""
    payload = {
        "directory": "/staged/raw",
        "files": [
            {"path": f"RAW.tar-contents/table_{i}.csv", "sha256": f"{i:064d}"}
            for i in range(76)
        ],
    }
    assert len(auto_task_run._candidate_records(payload)) == 76
    assert len(auto_task_run._candidate_records(payload, max_files=8)) == 8


def test_single_cell_parts_are_not_capped_away():
    """Each sample in a RAW.tar is a donor: truncating the list discards donors."""
    payload = {
        "directory": "/staged/raw",
        "files": [
            {"path": f"RAW.tar-contents/GSM{i}_matrix_X.mtx.gz", "sha256": f"{i:064d}"}
            for i in range(20)
        ]
        + [{"path": f"notes_{i}.md", "sha256": f"{i + 100:064d}"} for i in range(5)],
    }
    kept = auto_task_run._candidate_records(payload, max_files=8)
    assert len(kept) == 20  # every single-cell part survives the cap
    assert len(auto_task_run._candidate_records(payload, max_files=0)) == 25


def test_a_to_csv_derivative_of_a_table_is_not_a_second_candidate():
    """Two shapes of the same rows would double count."""
    payload = {
        "directory": "/staged/geo",
        "files": [
            {"path": "GSE1_series_matrix.txt.gz", "sha256": "a" * 64},
            {
                "path": "samples_by_features.csv",
                "sha256": "b" * 64,
                "derived_from": "GSE1_series_matrix.txt.gz",
            },
        ],
    }
    candidates = auto_task_run._candidate_records(payload)
    # The source is not a table, so the conversion of it is the usable thing.
    assert [record["path"] for record in candidates] == [
        "GSE1_series_matrix.txt.gz",
        "samples_by_features.csv",
    ]
    # When the source *is* a table, the derivative is the same rows again.
    payload["files"][0]["path"] = "table.csv"
    payload["files"][1]["derived_from"] = "table.csv"
    candidates = auto_task_run._candidate_records(payload)
    assert [record["path"] for record in candidates] == ["table.csv"]


def test_a_refusal_is_rendered_as_something_to_correct_against():
    """A GEO refusal names the file to use instead; that is a next attempt."""
    failures = [
        {
            "reference": "geo://GSE194122",
            "error": (
                "GSE194122's series matrix has a header and no rows ... "
                "Supplementary files: X.h5ad.gz, Y.h5ad.gz -- fetch one with "
                "geo://GSE194122#suppl/<filename>"
            ),
        }
    ]
    report = auto_task_run.failure_report(failures)
    assert "geo://GSE194122" in report
    assert "geo://GSE194122#suppl/<filename>" in report


def test_the_retry_prompt_hands_back_the_refusal():
    from kosmos.discovery import propose_datasets

    class FakeClient:
        def __init__(self):
            self.prompts = []

        def generate_structured(self, prompt, schema, **kwargs):
            self.prompts.append(prompt)
            return {"datasets": []}

    client = FakeClient()
    propose_datasets(
        "a question",
        client=client,
        max_items=2,
        log=lambda line: None,
        retry="  geo://GSE1\n    no series matrix; use #suppl/counts.csv",
    )
    prompt = client.prompts[0]
    assert "previous identifiers were refused" in prompt
    assert "#suppl/counts.csv" in prompt
    assert "not a table" in prompt  # h5ad/mtx warning


def test_a_model_refusal_on_a_fact_the_rules_can_check_goes_to_a_person(tmp_path, monkeypatch):
    """A 202-column table was refused as "no cell-type column" while it had one.

    The rules and the model then disagree about a fact -- the label column is
    in the listing -- so the run parks it for a person instead of obeying the
    refusal, which is how a whole single-cell table disappeared. The rules are
    only allowed to overrule the model when the column they found was named by a
    person: `--hint cell_type` is a protocol, and a protocol can be checked.
    """
    path = str(tmp_path / "wide.csv")
    (tmp_path / "wide.csv").write_text("a,b,cell_type\n1,2,T\n")
    monkeypatch.setattr(
        auto_task_run,
        "mechanical_role",
        lambda *a, **k: {"role": "gold", "reason": "target present", "has_target": True},
    )
    monkeypatch.setattr(
        auto_task_run,
        "review_candidates",
        lambda *a, **k: (
            {
                path: {
                    "role": "unusable",
                    "blocker": "no_label_column",
                    "reason": "no cell-type label column",
                }
            },
            {},
        ),
    )
    args = argparse.Namespace(
        root=None,
        exclude_col=[],
        feature_prefix=(),
        adjudicate=None,
        yes=True,
        objective="predict cell type",
    )
    log = auto_task_run.PipelineLog(tmp_path / "log.jsonl")

    decisions, pending, _ = auto_task_run.dual_review(
        [path],
        "cell_type",
        args,
        client=object(),
        log=log,
        out_dir=tmp_path,
        target_decision={"source": "hint_exact"},
    )

    # Nobody answered, so the rules hold -- the table stays usable -- and the
    # contradiction is listed for a person either way.
    assert decisions[path]["role"] == "gold"
    assert decisions[path]["decided_by"] == "mechanical (pending human)"
    assert pending and pending[0]["path"] == path


def test_a_target_read_out_of_the_question_cannot_outvote_the_model(tmp_path, monkeypatch):
    """The McFarland run: the rules' "label" was the gene `WAS`.

    A gene symbol that is also an ordinary word ("the drug a cancer cell *was*
    treated with") made a gene-expression matrix look labeled, and the model —
    which read the file and said there is no drug column — was overruled. A
    column found in the question's prose is a guess, so the refusal stands; the
    model's rescue step still travels with the table.
    """
    path = str(tmp_path / "wide.csv")
    (tmp_path / "wide.csv").write_text("GENE1,GENE2,WAS\n1,2,0\n")
    monkeypatch.setattr(
        auto_task_run,
        "mechanical_role",
        lambda *a, **k: {"role": "gold", "reason": "target present", "has_target": True},
    )
    monkeypatch.setattr(
        auto_task_run,
        "review_candidates",
        lambda *a, **k: (
            {
                path: {
                    "role": "unusable",
                    "blocker": "no_label_column",
                    "reason": "a gene-expression matrix with no drug column",
                    "salvage": "join the drug labels from the original h5ad",
                }
            },
            {},
        ),
    )
    args = argparse.Namespace(
        root=None,
        exclude_col=[],
        feature_prefix=(),
        adjudicate=None,
        yes=True,
        objective="Can the drug a cancer cell was treated with be predicted?",
    )
    log = auto_task_run.PipelineLog(tmp_path / "log.jsonl")

    decisions, _pending, _ = auto_task_run.dual_review(
        [path],
        "WAS",
        args,
        client=object(),
        log=log,
        out_dir=tmp_path,
        target_decision={"source": "objective_match"},
    )

    assert decisions[path]["role"] == "unusable"
    written = (tmp_path / "log.jsonl").read_text()
    assert "the refusal stands" in written
    assert "join the drug labels from the original h5ad" in written


# --- preflight: what each proposal costs, before the bytes move -------------


def _preflight_args(**overrides):
    values = {
        "root": None,
        "max_bytes": None,
        "objective": "a question",
        "no_llm": False,
        "no_preflight": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _listing_payload(reference, files):
    return {
        "reference": reference,
        "files": files,
        "bytes": sum(entry["bytes"] for entry in files),
    }


def _proposal(reference, why="the model liked it"):
    return type("P", (), {"identifier": reference, "why": why})()


def _serving(monkeypatch, listings):
    monkeypatch.setattr(
        auto_task_run,
        "run_command",
        lambda args, check=True: subprocess.CompletedProcess(
            args, 0, json.dumps(listings[args[args.index("list-files") + 1]]), ""
        ),
    )


def test_preflight_keeps_a_candidate_whose_smallest_file_fits(tmp_path, monkeypatch):
    _serving(
        monkeypatch,
        {
            "geo://GSE1": _listing_payload(
                "geo://GSE1",
                [
                    {"path": "suppl/small.h5ad.gz", "bytes": 600 * 1024**2},
                    {"path": "suppl/big.h5ad.gz", "bytes": 2700 * 1024**2},
                ],
            )
        },
    )
    log = auto_task_run.PipelineLog(tmp_path / "log.jsonl")

    chosen, refused = auto_task_run.preflight(
        [_proposal("geo://GSE1")], _preflight_args(), None, log
    )

    assert refused == []

    assert [choice.reference for choice in chosen] == ["geo://GSE1"]
    assert "smallest 600.0 MB" in (tmp_path / "log.jsonl").read_text()


def test_preflight_drops_a_candidate_over_the_per_file_cap(tmp_path, monkeypatch):
    """A 2.9 GB file the per-file cap refuses is never transferred."""
    _serving(
        monkeypatch,
        {
            "geo://GSE1": _listing_payload(
                "geo://GSE1", [{"path": "suppl/big.h5ad.gz", "bytes": 2900 * 1024**2}]
            )
        },
    )
    log = auto_task_run.PipelineLog(tmp_path / "log.jsonl")

    chosen, _ = auto_task_run.preflight(
        [_proposal("geo://GSE1")], _preflight_args(max_bytes=2048 * 1024**2), None, log
    )

    assert chosen == []
    assert "over --max-bytes" in (tmp_path / "log.jsonl").read_text()


def test_preflight_drops_a_candidate_that_does_not_fit_the_run_budget(tmp_path, monkeypatch):
    _serving(
        monkeypatch,
        {
            "geo://GSE1": _listing_payload(
                "geo://GSE1", [{"path": "suppl/a.h5ad.gz", "bytes": 900 * 1024**2}]
            )
        },
    )
    log = auto_task_run.PipelineLog(tmp_path / "log.jsonl")
    budget = auto_task_run.DownloadBudget(500)  # 500 MB, and this one is 900 MB

    chosen, _ = auto_task_run.preflight(
        [_proposal("geo://GSE1")], _preflight_args(), None, log, budget=budget
    )

    assert chosen == []
    assert "does not fit what is left" in (tmp_path / "log.jsonl").read_text()


def test_preflight_lets_the_model_drop_an_irrelevant_candidate(tmp_path, monkeypatch):
    _serving(
        monkeypatch,
        {
            "geo://GSE1": _listing_payload(
                "geo://GSE1", [{"path": "suppl/a.csv", "bytes": 1024}]
            ),
            "hf://owner/other": _listing_payload(
                "hf://owner/other", [{"path": "train.parquet", "bytes": 2048}]
            ),
        },
    )
    log = auto_task_run.PipelineLog(tmp_path / "log.jsonl")

    class Client:
        def generate_structured(self, **_kwargs):
            return {"downloads": [{"reference": "geo://GSE1", "why": "it is the one"}]}

    chosen, _ = auto_task_run.preflight(
        [_proposal("geo://GSE1"), _proposal("hf://owner/other")],
        _preflight_args(),
        Client(),
        log,
    )

    assert [choice.reference for choice in chosen] == ["geo://GSE1"]
    assert "drop hf://owner/other" in (tmp_path / "log.jsonl").read_text()


def test_a_spent_budget_stops_further_downloads(tmp_path, monkeypatch):
    """The per-file cap bounds one artifact; this bounds the run."""
    calls: list[list[str]] = []

    def fake_run_command(args, check=True):
        calls.append(list(args))
        body = {
            "directory": str(tmp_path / "staged"),
            "files": [{"path": "t.csv", "sha256": "deadbeef" * 8, "bytes": 1024}],
        }
        return subprocess.CompletedProcess(args, 0, json.dumps(body), "")

    monkeypatch.setattr(auto_task_run, "run_command", fake_run_command)
    log = auto_task_run.PipelineLog(tmp_path / "log.jsonl")
    budget = auto_task_run.DownloadBudget(0.0005)  # smaller than one file
    budget.spend(1024)

    fetched, _ = auto_task_run.fetch_selected(
        [_proposal("hf://a/one")], None, None, "q", log, budget=budget
    )

    assert fetched == []
    assert not [call for call in calls if "fetch" in call]
    assert "download budget is spent" in (tmp_path / "log.jsonl").read_text()


def test_a_finished_download_is_charged_to_the_budget(tmp_path, monkeypatch):
    payload = {
        "directory": str(tmp_path / "staged"),
        "files": [
            {"path": "t.csv", "sha256": "deadbeef" * 8, "bytes": 5 * 1024**2},
            # A derived table was written from a file already here, not
            # downloaded: it does not cost the run's budget.
            {
                "path": "t.csv-table.csv",
                "sha256": "cafebabe" * 8,
                "bytes": 900 * 1024**2,
                "derived_from": "t.h5ad",
            },
        ],
    }
    monkeypatch.setattr(
        auto_task_run,
        "run_command",
        lambda args, check=True: subprocess.CompletedProcess(args, 0, json.dumps(payload), ""),
    )
    log = auto_task_run.PipelineLog(tmp_path / "log.jsonl")
    budget = auto_task_run.DownloadBudget(100)

    auto_task_run.fetch_selected([_proposal("hf://a/one")], None, None, "q", log, budget=budget)

    assert budget.spent == 5 * 1024**2


# --- verifying that a named file exists, before anything is transferred -----


def test_preflight_refuses_a_file_the_repository_does_not_have(tmp_path, monkeypatch):
    """The model guessed `train.csv` in a repository that holds `person.csv`.

    The listing already answers that: it says which files exist, and the run
    spends nothing. The refusal is handed back in the same shape a failed
    download produces, so the correction round can act on it.
    """
    _serving(
        monkeypatch,
        {
            "hf://KodeCharya/age_gender_height_weight_#train.csv": {
                "reference": "hf://KodeCharya/age_gender_height_weight_#train.csv",
                "selector": "train.csv",
                "files": [
                    {"path": ".gitattributes", "bytes": 2400, "size": "2.4 kB"},
                    {"path": "person.csv", "bytes": 37 * 1024**2, "size": "37.0 MB"},
                ],
            }
        },
    )
    log = auto_task_run.PipelineLog(tmp_path / "log.jsonl")

    chosen, refused = auto_task_run.preflight(
        [_proposal("hf://KodeCharya/age_gender_height_weight_#train.csv")],
        _preflight_args(),
        None,
        log,
    )

    assert chosen == []
    assert len(refused) == 1
    assert "no such file 'train.csv'" in refused[0]["error"]
    assert "person.csv (37.0 MB)" in refused[0]["error"]
    assert not [call for call in [] if "fetch" in call]


def test_preflight_refuses_a_url_that_is_a_web_page(tmp_path, monkeypatch):
    _serving(
        monkeypatch,
        {
            "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_BMX.csv": {
                "reference": "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_BMX.csv",
                "files": [{"path": "P_BMX.csv", "bytes": 20905}],
                "web_page": "Page Not Found | CDC",
            }
        },
    )
    log = auto_task_run.PipelineLog(tmp_path / "log.jsonl")

    chosen, refused = auto_task_run.preflight(
        [_proposal("https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_BMX.csv")],
        _preflight_args(),
        None,
        log,
    )

    assert chosen == []
    assert "web page (Page Not Found | CDC)" in refused[0]["error"]
    assert "nothing was downloaded" in (tmp_path / "log.jsonl").read_text()


def test_preflight_refuses_something_that_is_not_an_identifier(tmp_path):
    log = auto_task_run.PipelineLog(tmp_path / "log.jsonl")

    chosen, refused = auto_task_run.preflight(
        [_proposal("the NHANES 2017-2018 body measures file")],
        _preflight_args(),
        None,
        log,
    )

    assert chosen == []
    assert "not an identifier" in refused[0]["error"]


def test_preflight_never_takes_what_the_run_already_has(tmp_path, monkeypatch):
    """The evidence round proposed the series and re-chose the gold's own file.

    Nothing about the listing says what this run has already used, so the
    exclusion is applied both to the candidate and to the model's own choice.
    """
    _serving(
        monkeypatch,
        {
            "geo://GSE194122": {
                "reference": "geo://GSE194122",
                "selector": None,
                "files": [
                    {
                        "path": "suppl/GSE194122_cite_BMMC.h5ad.gz",
                        "bytes": 615_514_112,
                        "reference": "geo://GSE194122#suppl/GSE194122_cite_BMMC.h5ad.gz",
                    },
                    {
                        "path": "suppl/GSE194122_multiome_BMMC.h5ad.gz",
                        "bytes": 2_900_000_000,
                        "reference": "geo://GSE194122#suppl/GSE194122_multiome_BMMC.h5ad.gz",
                    },
                ],
            }
        },
    )

    class Client:
        def generate_structured(self, **_kwargs):
            # The model picks the file the run already has.
            return {
                "downloads": [
                    {
                        "reference": "geo://GSE194122#suppl/GSE194122_cite_BMMC.h5ad.gz",
                        "why": "it has the same columns",
                    }
                ]
            }

    log = auto_task_run.PipelineLog(tmp_path / "log.jsonl")
    chosen, _ = auto_task_run.preflight(
        [_proposal("geo://GSE194122")],
        _preflight_args(),
        Client(),
        log,
        exclude_names={"GSE194122_cite_BMMC.h5ad.gz-table.csv", "GSE194122_cite_BMMC.h5ad.gz"},
        exclude_refs={"geo://GSE194122#suppl/GSE194122_cite_BMMC.h5ad.gz"},
    )

    assert chosen == []
    written = (tmp_path / "log.jsonl").read_text()
    assert "was chosen but is already in this run" in written
    assert "the model kept none of the candidates" in written


def test_preflight_drops_a_candidate_that_is_the_gold_itself(tmp_path, monkeypatch):
    _serving(
        monkeypatch,
        {
            "geo://GSE194122#suppl/GSE194122_cite_BMMC.h5ad.gz": {
                "reference": "geo://GSE194122#suppl/GSE194122_cite_BMMC.h5ad.gz",
                "selector": "suppl/GSE194122_cite_BMMC.h5ad.gz",
                "files": [
                    {
                        "path": "suppl/GSE194122_cite_BMMC.h5ad.gz",
                        "bytes": 615_514_112,
                    }
                ],
            }
        },
    )
    log = auto_task_run.PipelineLog(tmp_path / "log.jsonl")

    chosen, _ = auto_task_run.preflight(
        [_proposal("geo://GSE194122#suppl/GSE194122_cite_BMMC.h5ad.gz")],
        _preflight_args(),
        None,
        log,
        exclude_names={"GSE194122_cite_BMMC.h5ad.gz"},
    )

    assert chosen == []
    assert "already in this run (same file)" in (tmp_path / "log.jsonl").read_text()


# --- an index is not the data -------------------------------------------------


def test_a_manifest_chosen_by_the_model_is_replaced_by_the_data_file():
    """The run picked `manifest/*.tsv` out of a 243 GB repository.

    They were the smallest files on offer, so "choose the smallest artifact"
    pointed straight at them; the manifest is an index of the measurements and
    contains none, and the run ended with 0 gold.
    """
    listing = {
        "reference": "hf://cfy2yue/perturbseq_normalized",
        "files": [
            {
                "path": "manifest/STUDY_SUMMARY.tsv",
                "bytes": 17_400,
                "reference": "hf://r#manifest/STUDY_SUMMARY.tsv",
            },
            {
                "path": "manifest/RELEASE_FILES.tsv",
                "bytes": 47_500,
                "reference": "hf://r#manifest/RELEASE_FILES.tsv",
            },
            {
                "path": "scPeturb/model_ready/chempert/a549__expt10__24h.h5ad",
                "bytes": 22_700_000,
                "reference": "hf://r#scPeturb/model_ready/chempert/a549__expt10__24h.h5ad",
            },
            {
                "path": "scPeturb/model_ready/chempert/bicr31__3h.h5ad",
                "bytes": 7_500_000,
                "reference": "hf://r#scPeturb/model_ready/chempert/bicr31__3h.h5ad",
            },
        ],
    }

    class Log:
        def __init__(self):
            self.messages = []

        def event(self, stage, message, **fields):
            self.messages.append(message)

    log = Log()
    kept = auto_task_run._prefer_data_over_index(
        listing, [{"reference": "hf://r#manifest/STUDY_SUMMARY.tsv"}], log
    )

    assert [entry["reference"] for entry in kept] == [
        "hf://r#scPeturb/model_ready/chempert/bicr31__3h.h5ad"
    ]
    assert "are an index of the data, not the data" in log.messages[0]


def test_a_data_file_the_model_chose_is_left_alone():
    listing = {
        "reference": "hf://r",
        "files": [
            {"path": "a.h5ad", "bytes": 1000, "reference": "hf://r#a.h5ad"},
            {"path": "manifest/x.tsv", "bytes": 10, "reference": "hf://r#manifest/x.tsv"},
        ],
    }
    log = auto_task_run.PipelineLog("/dev/null")
    kept = auto_task_run._prefer_data_over_index(
        listing, [{"reference": "hf://r#a.h5ad"}], log
    )
    assert [entry["reference"] for entry in kept] == ["hf://r#a.h5ad"]


def test_the_rescue_block_prints_what_a_person_can_do(tmp_path, capsys):
    """Exit 2 with a way back in: the manual step, and how to use it."""
    plan = {
        "gold": [],
        "supplementary": [],
        "unusable": [
            {"path": str(tmp_path / "RELEASE_FILES.tsv"), "reason": "not a table"},
            {"path": str(tmp_path / "other.csv"), "reason": "not a table"},
        ],
    }
    decisions = {
        str(tmp_path / "RELEASE_FILES.tsv"): {
            "model": {
                "role": "unusable",
                "salvage": "fetch the file named in column relative_path",
            }
        }
    }
    log = auto_task_run.PipelineLog(tmp_path / "log.jsonl")
    auto_task_run._print_rescue(plan, decisions, None, log, tmp_path / "plan.json")

    printed = capsys.readouterr().err
    assert "how to rescue this run" in printed
    assert "fetch the file named in column relative_path" in printed
    assert "--candidate" in printed
    assert "other.csv" not in printed  # no manual step was named for it
