"""The data report: the account a person reads after a run."""

from __future__ import annotations

import json

from kosmos.discovery import build_data_report, render_data_report, write_data_report
from kosmos.discovery.review import TableReview, _schema


def plan_fixture() -> dict:
    return {
        "task": {"target_column": "pce", "task_type": "regression"},
        "gold": [
            {
                "path": "data/fetched/hf/perovskite/train.parquet",
                "role": "gold",
                "reason": "target column 'pce' is present with 5 feature(s)",
                "dropped_features": ["reduced_formulas", "__index_level_0__"],
            }
        ],
        "supplementary": [
            {
                "path": "data/fetched/hf/scikit/breast_cancer.csv",
                "role": "supplementary",
                "reason": "no target column for this task",
                "dropped_features": ["Unnamed: 32"],
                "column_renames": {"concave points_mean": "concave_points_mean"},
            }
        ],
        "unusable": [
            {
                "path": "data/fetched/hf/x/README.md",
                "role": "unusable",
                "reason": "not a table: 1 parsed column(s). A table needs at least ...",
            },
            {
                "path": "data/fetched/hf/y/notes.csv",
                "role": "unusable",
                "reason": "1 column(s) are not features (free text, a row index, or empty)",
            },
            {
                "path": "data/fetched/hf/z/other.csv",
                "role": "unusable",
                "reason": "it cannot be evidence for the same model: it is missing 2 feature column(s)",
            },
        ],
    }


def decisions_fixture() -> dict:
    return {
        "data/fetched/hf/perovskite/train.parquet": {
            "role": "gold",
            "decided_by": "mechanical+model",
            "mechanical": {"role": "gold", "reason": "target column 'pce' is present"},
            "model": {
                "role": "gold",
                "reason": "the pce column is the efficiency",
                "report": (
                    "This is the raw device table of the perovskite database: one "
                    "row per measured cell, with the four J-V parameters and the "
                    "formula strings. The efficiency target is present, so it can "
                    "train and be tested on. The free-text columns are not features."
                ),
            },
        },
        "data/fetched/hf/y/notes.csv": {
            "role": "unusable",
            "decided_by": "mechanical+model",
            "mechanical": {"role": "unusable", "reason": "1 column(s) are not features"},
            "model": {
                "role": "unusable",
                "reason": "it is a single free-text column",
                "report": (
                    "The only column holds a sentence per row, which is not a "
                    "measurement. Dropping it would leave nothing to train on."
                ),
            },
        },
    }


def test_the_report_names_what_was_fetched_and_why_the_rest_went():
    report = build_data_report(
        question="How do fabrication parameters affect efficiency?",
        plan=plan_fixture(),
        decisions=decisions_fixture(),
    )
    markdown = render_data_report(report)

    assert report["summary"] == {
        "tables": 5,
        "gold": 1,
        "supplementary": 1,
        "unusable": 3,
        "awaiting_human": 0,
        # This fixture's plan names no feature columns, so there is nothing to
        # say about how wide the training table was.
        "training_columns": 0,
        "gold_columns": 0,
    }
    assert report["rejection_causes"] == {
        "not a table (fewer than two columns)": 1,
        "free text or a row index among the features": 1,
        "cannot supply the columns the gold has": 1,
    }
    assert "# Data report" in markdown
    assert "Refusals by cause" in markdown
    assert "not a table (fewer than two columns) ×1" in markdown
    # One line per dataset, with the role and the reason on it.
    assert "- `train.parquet` — **gold** —" in markdown
    assert "- `breast_cancer.csv` — **supplementary** —" in markdown
    assert "- `README.md` — **unusable** —" in markdown
    # What was left out is on the dataset's own line.
    assert "reduced_formulas" in markdown
    # The model's own paragraph is in the record, not in the one-line account.
    assert "one row per measured cell" in report["tables"][0]["model_report"]
    assert "concave points_mean" not in markdown or "→" not in markdown


def test_a_table_a_person_has_to_decide_says_so():
    pending = [
        {
            "path": "data/fetched/hf/v/ambiguous.csv",
            "mechanical_role": "unusable",
            "mechanical_reason": "1 column(s) are not features (free text ...)",
            "model_role": "gold",
            "model_reason": "the label is there under another name",
        }
    ]
    report = build_data_report(
        question="q", plan=plan_fixture(), decisions={}, pending=pending
    )
    markdown = render_data_report(report)

    assert report["summary"]["awaiting_human"] == 1
    assert "- `ambiguous.csv` — **pending** —" in markdown
    assert "**needs a person**" in markdown
    assert "the rules refused it, the model accepted it" in markdown


def test_writing_puts_both_files_beside_the_plan(tmp_path):
    markdown, record = write_data_report(
        question="q", plan=plan_fixture(), out_dir=tmp_path, decisions=decisions_fixture()
    )
    assert markdown.name == "data_report.md" and markdown.exists()
    assert record.name == "data_report.json" and record.exists()
    payload = json.loads(record.read_text())
    assert payload["summary"]["gold"] == 1
    assert payload["tables"][0]["path"].endswith("train.parquet")


def test_the_review_schema_asks_for_a_report():
    schema = _schema()
    assert "report" in schema["properties"]
    assert "blocker" in schema["properties"]
    assert TableReview(path="x", role="unusable").report == ""


def test_a_converted_table_says_what_it_was_made_from():
    """A derived table is a sample of a larger file, and the report says so.

    Reading "2,000 cells" as the whole dataset is how a person concludes the
    evidence is thin when it is a bounded sample of 90,261 cells.
    """
    plan = plan_fixture()
    plan["gold"][0] = {
        **plan["gold"][0],
        "path": "data/fetched/geo/GSE194122/multiome_BMMC.h5ad.gz-table.csv",
        "provenance": {
            "reference": "geo://GSE194122",
            "derived_from": "multiome_BMMC.h5ad.gz",
            "conversion": {
                "source": "multiome_BMMC.h5ad.gz",
                "source_cells": 90261,
                "source_genes": 14089,
                "cells_written": 2000,
                "genes_written": 14089,
                "cell_sampling": "a stratified sample of 2,000 cells (seed 42)",
                "gene_selection": "every gene",
            },
        },
    }

    report = build_data_report(question="q", plan=plan, decisions={})
    markdown = render_data_report(report)

    assert report["tables"][0]["conversion"]["source_cells"] == 90261
    assert "converted from multiome_BMMC.h5ad.gz" in markdown
    assert "2000 of 90261 cells" in markdown
    assert "14089 of 14089 genes" in markdown


def test_the_report_says_the_arms_were_trained_on_the_intersection():
    """Evidence that shares two of the gold's three columns is still evidence.

    Both arms then train on those two columns, and the report says so: a reader
    comparing "gold only" against "gold plus evidence" needs to know the
    comparison was made on the same inputs.
    """
    plan = plan_fixture()
    plan["task"]["feature_columns"] = ["pce", "bandgap"]
    plan["gold"][0]["features"] = ["pce", "bandgap", "thickness"]
    plan["supplementary"][0]["shared_features"] = ["pce", "bandgap"]

    report = build_data_report(question="q", plan=plan, decisions={})
    markdown = render_data_report(report)

    assert report["summary"]["training_columns"] == 2
    assert report["summary"]["gold_columns"] == 3
    assert "**Training columns:** 2 of the gold's 3" in markdown
    assert "intersection with the evidence" in markdown
    assert "shares 2 column(s) with the gold: pce, bandgap" in markdown


def events_fixture() -> list[dict]:
    """One gold round and one evidence round, as the log writes them."""
    return [
        {
            "stage": "retrieval",
            "message": "# retrieval: asking the model for candidate datasets",
            "proposals": [
                {"identifier": "hf://KodeCharya/age_gender_height_weight_#person.csv"}
            ],
        },
        {
            "stage": "retrieval",
            "message": "# retrieval: proposal 1/2 hf://KodeCharya/age_gender_height_weight_#person.csv (confidence 0.95)",
        },
        {
            "stage": "retrieval",
            "message": "#   why     : one row per person with gender, age, height and weight",
        },
        {
            "stage": "preflight",
            "message": "# preflight: hf://KodeCharya/age_gender_height_weight_#person.csv "
            "-> 1 file(s) on offer, smallest 37.0 MB, total 37.0 MB",
            "reference": "hf://KodeCharya/age_gender_height_weight_#person.csv",
            "files": [{"path": "person.csv", "bytes": 37 * 1024**2}],
        },
        {
            "stage": "download",
            "message": "#   ok: 1 file(s)",
            "reference": "hf://KodeCharya/age_gender_height_weight_#person.csv",
            "ok": True,
            "files": [{"path": "person.csv", "sha256": "a" * 64}],
        },
        {
            "stage": "screen",
            "message": "# screening: 1 of 1 candidate(s) are tables",
            "tabular": ["data/fetched/hf/x/person.csv"],
            "skipped": [],
        },
        {
            "stage": "supplementary",
            "message": "# supplementary: looking for other tables with these 3 column(s) of person.csv",
        },
        {
            "stage": "retrieval",
            "message": "# search 'height_cm gender age': 1 hit(s)",
            "query": "height_cm gender age",
            "hits": ["  GSE244036 -> geo://GSE244036  Whole Blood miRNA Seq of Children with Asthma"],
        },
        {
            "stage": "retrieval",
            "message": "# retrieval: proposal 1/2 hf://KodeCharya/age_gender_height_weight_#train.csv (confidence 0.60)",
        },
        {
            "stage": "preflight",
            "message": "# preflight: hf://KodeCharya/...#train.csv -> there is no 'train.csv' "
            "-- it has person.csv (37.0 MB); nothing was downloaded",
            "reference": "hf://KodeCharya/age_gender_height_weight_#train.csv",
            "skipped": "no such file",
        },
        {
            "stage": "download",
            "message": "#   FAILED: no such file 'test.csv'",
            "reference": "hf://KodeCharya/age_gender_height_weight_#test.csv",
            "ok": False,
            "error": "no such file 'test.csv' in this reference; it has person.csv",
        },
    ]


def test_each_round_says_how_it_searched_and_what_it_decided():
    """The chain, not just the verdict: 0 supplementary has many causes."""
    report = build_data_report(
        question="q", plan=plan_fixture(), decisions={}, events=events_fixture()
    )
    markdown = render_data_report(report)

    assert "## Round 1 -- the labeled table" in markdown
    assert "**proposed**" in markdown
    assert "one row per person with gender, age, height and weight" in markdown
    assert "**listing**" in markdown and "— kept" in markdown
    assert "**downloaded**" in markdown

    assert "## Round 2 -- supplementary evidence for that table" in markdown
    # The keyword and what it returned: this is the part that explains "no supp".
    assert "**keyword search** `height_cm gender age` — 1 hit(s)" in markdown
    assert "GSE244036" in markdown
    assert (
        "**refused** `hf://KodeCharya/age_gender_height_weight_#train.csv` — "
        "no such file: there is no 'train.csv'" in markdown
    )
    assert "no such file 'test.csv'" in markdown
    # The refusal also appears as a dataset line: nothing was downloaded.
    assert "**not downloaded**" in markdown

    # And a line per dataset the run touched, including the ones it never got.
    assert "- `train.parquet` — **gold** —" in markdown
    assert (
        "- `hf://KodeCharya/age_gender_height_weight_#train.csv` — "
        "**not downloaded** —" in markdown
    )


def test_the_rounds_are_separated_by_the_evidence_round_marker():
    from kosmos.discovery.data_report import rounds_from_events

    rounds = rounds_from_events(events_fixture())

    assert [p["identifier"] for p in rounds["gold"]["proposals"]] == [
        "hf://KodeCharya/age_gender_height_weight_#person.csv"
    ]
    assert [p["identifier"] for p in rounds["supplementary"]["proposals"]] == [
        "hf://KodeCharya/age_gender_height_weight_#train.csv"
    ]
    assert rounds["supplementary"]["searches"][0]["query"] == "height_cm gender age"
