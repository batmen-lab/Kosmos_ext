"""The plan file: the one contract between the fetcher and the training side."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from datafetcher import PlanTask, build_plan, load_plan, write_plan
from datafetcher.cli import main
from datafetcher.plan import DataPlan, PlanEntry, _enforce_supplementary_schema


def labeled(path, rows=60, prefix="L"):
    pd.DataFrame(
        {
            "sample_id": [f"{prefix}{i}" for i in range(rows)],
            "ENSG00000000001": [float(i) for i in range(rows)],
            "ENSG00000000002": [float(i % 5) for i in range(rows)],
            "cell_type": ["B", "T", "NK"] * (rows // 3),
        }
    ).to_csv(path, index=False)
    return path


def unlabeled(path, rows=40, prefix="U"):
    pd.DataFrame(
        {
            "sample_id": [f"{prefix}{i}" for i in range(rows)],
            "ENSG00000000001": [float(i) + 1 for i in range(rows)],
            "ENSG00000000002": [float(i % 5) for i in range(rows)],
        }
    ).to_csv(path, index=False)
    return path


def task(**overrides) -> PlanTask:
    base = {
        "objective": "predict cell type",
        "target_column": "cell_type",
        "feature_prefixes": ["ENSG"],
        "sample_id_column": "sample_id",
    }
    base.update(overrides)
    return PlanTask(**base)


def test_plan_sorts_tables_into_roles_with_reasons(tmp_path):
    gold = labeled(tmp_path / "gold.csv")
    extra = unlabeled(tmp_path / "supp.csv")
    broken = tmp_path / "broken.csv"
    broken.write_text("a,b\n1,2\n")

    plan = build_plan([gold, extra, broken], task())

    assert [e.path for e in plan.gold] == [str(gold)]
    assert [e.path for e in plan.supplementary] == [str(extra)]
    assert [e.path for e in plan.unusable] == [str(broken)]
    assert plan.gold[0].n_features == 2
    assert plan.gold[0].rows == 60
    assert "can train" in plan.gold[0].reason
    assert "correction term" in plan.supplementary[0].reason
    assert plan.version == 1
    assert "task: predict 'cell_type'" in plan.summary()


def test_without_a_target_there_are_no_roles_only_profiles(tmp_path):
    """Role assignment is 'is the label here'; without a target there is none."""
    gold = labeled(tmp_path / "gold.csv")
    plan = build_plan([gold], task(target_column=None))
    assert plan.gold == [] and plan.supplementary == [] and plan.unusable == []
    assert any("no target column decided" in note for note in plan.notes)


def test_plan_round_trips_through_a_file(tmp_path):
    gold = labeled(tmp_path / "gold.csv")
    plan = build_plan([gold], task())
    path = write_plan(plan, tmp_path / "plan.json")
    again = load_plan(path)
    assert again.model_dump(mode="json") == plan.model_dump(mode="json")


def test_plan_attaches_fetch_provenance_when_the_file_was_staged(tmp_path, web):
    """A plan should say which reference and hash a table came from."""
    from datafetcher import DataFetcherConfig, fetch

    root = tmp_path / "fetched"
    web.add("https://example.test/labeled.csv", labeled(tmp_path / "src.csv").read_text())
    fetch("https://example.test/labeled.csv", config=DataFetcherConfig(root=root))
    staged = root / "http" / "example.test__labeled.csv" / "labeled.csv"

    plan = build_plan([staged], task(), root=root)
    provenance = plan.gold[0].provenance
    assert provenance["reference"] == "https://example.test/labeled.csv"
    assert provenance["sha256"] and provenance["retrieved_at"]


def test_cli_plan_writes_a_file_and_show_plan_reads_it(tmp_path, capsys):
    gold = labeled(tmp_path / "gold.csv")
    extra = unlabeled(tmp_path / "supp.csv")
    out = tmp_path / "plan.json"
    args = [
        "--root",
        str(tmp_path),
        "plan",
        str(gold),
        str(extra),
        "--target-column",
        "cell_type",
        "--feature-prefix",
        "ENSG",
        "--sample-id-column",
        "sample_id",
        "--objective",
        "predict cell type",
        "-o",
        str(out),
    ]
    assert main(args) == 0
    printed = capsys.readouterr().out
    assert "gold: 1 table(s)" in printed and "supplementary: 1 table(s)" in printed

    payload = json.loads(out.read_text())
    assert payload["task"]["target_column"] == "cell_type"
    assert len(payload["gold"]) == 1 and len(payload["supplementary"]) == 1

    assert main(["--json", "show-plan", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["task"]["target_source"] == "cli"


def test_cli_plan_can_discover_everything_fetched(tmp_path, web, capsys):
    from datafetcher import DataFetcherConfig, fetch

    root = tmp_path / "fetched"
    web.add("https://example.test/labeled.csv", labeled(tmp_path / "src.csv").read_text())
    fetch("https://example.test/labeled.csv", config=DataFetcherConfig(root=root))

    assert (
        main(
            [
                "--root",
                str(root),
                "plan",
                "--from-root",
                "--target-column",
                "cell_type",
                "--feature-prefix",
                "ENSG",
                "--sample-id-column",
                "sample_id",
                "-o",
                str(tmp_path / "plan.json"),
            ]
        )
        == 0
    )
    payload = json.loads((tmp_path / "plan.json").read_text())
    assert len(payload["gold"]) == 1


def test_a_second_labeled_table_becomes_supplementary_evidence(tmp_path):
    """More than one labeled table: one trains, the rest are evidence."""
    big = labeled(tmp_path / "big.csv", rows=90, prefix="A")
    small = labeled(tmp_path / "small.csv", rows=30, prefix="B")
    plan = build_plan([big, small], task())

    assert [e.path for e in plan.gold] == [str(big)]  # largest wins, deterministically
    assert [e.path for e in plan.supplementary] == [str(small)]
    reason = plan.supplementary[0].reason
    assert "demoted to supplementary evidence" in reason
    assert "labels are ignored" in reason
    assert any("policy: external" in note for note in plan.notes)


def test_the_primary_labeled_table_can_be_named(tmp_path):
    big = labeled(tmp_path / "big.csv", rows=90, prefix="A")
    small = labeled(tmp_path / "small.csv", rows=30, prefix="B")
    plan = build_plan([big, small], task(), primary_labeled=str(small))
    assert [e.path for e in plan.gold] == [str(small)]
    assert [e.path for e in plan.supplementary] == [str(big)]
    assert "named as the primary" in plan.notes[0]


def test_pooling_is_still_available_and_recorded(tmp_path):
    big = labeled(tmp_path / "big.csv", rows=90, prefix="A")
    small = labeled(tmp_path / "small.csv", rows=30, prefix="B")
    plan = build_plan([big, small], task(multi_gold_policy="pool"))
    assert len(plan.gold) == 2 and plan.supplementary == []
    assert any("pooled" in note for note in plan.notes)


def test_policy_error_refuses_and_lists_the_tables(tmp_path):
    big = labeled(tmp_path / "big.csv", rows=90, prefix="A")
    small = labeled(tmp_path / "small.csv", rows=30, prefix="B")
    with pytest.raises(ValueError, match="carry the label column"):
        build_plan([big, small], task(multi_gold_policy="error"))


def test_a_demoted_table_with_too_little_in_common_is_unusable(tmp_path):
    big = labeled(tmp_path / "big.csv", rows=90, prefix="A")
    other = tmp_path / "other.csv"
    pd.DataFrame(
        {
            "sample_id": [f"X{i}" for i in range(40)],
            "ENSG00000000001": [float(i) for i in range(40)],
            "cell_type": ["B", "T"] * 20,
        }
    ).to_csv(other, index=False)
    plan = build_plan([big, other], task())
    assert [e.path for e in plan.gold] == [str(big)]
    assert plan.supplementary == []
    assert [e.path for e in plan.unusable] == [str(other)]
    assert "cannot be evidence for the same model" in plan.unusable[0].reason
    assert "shares only 1 measured column(s)" in plan.unusable[0].reason


def test_a_supplementary_table_may_carry_extra_columns(tmp_path):
    """Evidence needs the gold's features; what it carries besides them is spare.

    A mirror of the same patients often carries one more column -- its own copy
    of the label, or a `Unnamed: 32`. Refusing it for that cost a run every
    supplementary table it had.
    """
    gold = labeled(tmp_path / "gold.csv", rows=60)
    wider = tmp_path / "wider.csv"
    pd.DataFrame(
        {
            "sample_id": [f"W{i}" for i in range(30)],
            "ENSG00000000001": [float(i) for i in range(30)],
            "ENSG00000000002": [float(i % 5) for i in range(30)],
            "ENSG00000000003": [float(i % 3) for i in range(30)],
        }
    ).to_csv(wider, index=False)
    plan = build_plan([gold, wider], task())
    assert [e.path for e in plan.gold] == [str(gold)]
    assert [e.path for e in plan.supplementary] == [str(wider)]
    assert plan.unusable == []
    assert plan.supplementary[0].features == ["ENSG00000000001", "ENSG00000000002"]
    assert "ENSG00000000003" in plan.supplementary[0].dropped_features
    assert "not part of this task" in plan.supplementary[0].reason


def test_a_supplementary_table_that_shares_too_little_is_unusable(tmp_path):
    """A single overlapping column is a coincidence, not a correction."""
    gold = labeled(tmp_path / "gold.csv", rows=60)
    thin = tmp_path / "thin.csv"
    pd.DataFrame(
        {
            "sample_id": [f"T{i}" for i in range(30)],
            "ENSG00000000001": [float(i) for i in range(30)],
        }
    ).to_csv(thin, index=False)
    plan = build_plan([gold, thin], task())
    assert plan.supplementary == []
    assert [e.path for e in plan.unusable] == [str(thin)]
    assert "shares only 1 measured column(s)" in plan.unusable[0].reason


def test_a_supplementary_table_missing_one_gold_feature_is_still_evidence(tmp_path):
    """The intersection is what the correction runs at, not the gold's schema.

    A table that measured two of the gold's three columns is evidence for those
    two: refusing it threw away rows this analysis can use, which is the case
    that made "no supplementary table" the usual outcome.
    """
    gold = tmp_path / "gold.csv"
    pd.DataFrame(
        {
            "sample_id": [f"L{i}" for i in range(60)],
            "a": [float(i) for i in range(60)],
            "b": [float(i % 5) for i in range(60)],
            "c": [float(i) * 2 for i in range(60)],
            "outcome": ["B", "T", "NK"] * 20,
        }
    ).to_csv(gold, index=False)
    partial = tmp_path / "partial.csv"
    pd.DataFrame(
        {
            "sample_id": [f"U{i}" for i in range(30)],
            "a": [float(i) + 1 for i in range(30)],
            # No `c`: this table measured a and b, and the gold also measured c.
            "b": [float(i % 3) for i in range(30)],
        }
    ).to_csv(partial, index=False)
    shape = PlanTask(
        objective="predict outcome",
        target_column="outcome",
        feature_columns=["a", "b", "c"],
        sample_id_column="sample_id",
    )

    plan = build_plan([gold, partial], shape)

    assert [e.path for e in plan.supplementary] == [str(partial)]
    assert plan.supplementary[0].shared_features == ["a", "b"]
    assert "shares 2 of the gold's 3 feature(s)" in plan.supplementary[0].reason
    # Both arms train on the intersection: the comparison stays like-for-like.
    assert plan.task.feature_columns == ["a", "b"]
    assert any("shares 2 of the gold's 3 feature(s)" in note for note in plan.notes)


def test_a_column_named_differently_is_matched_not_refused(tmp_path):
    """`concave points_mean` and `concave_points_mean` are one measurement."""
    gold = labeled(tmp_path / "gold.csv", rows=60)
    mirror = tmp_path / "mirror.csv"
    pd.DataFrame(
        {
            "sample_id": [f"M{i}" for i in range(30)],
            "ENSG00000000001": [float(i) for i in range(30)],
            # Same column, spelled the way another export spells it.
            "ENSG00000000002 ": [float(i % 5) for i in range(30)],
            "Unnamed: 32": [None] * 30,
        }
    ).to_csv(mirror, index=False)
    plan = build_plan([gold, mirror], task())
    assert [e.path for e in plan.supplementary] == [str(mirror)]
    entry = plan.supplementary[0]
    assert entry.column_renames == {"ENSG00000000002 ": "ENSG00000000002"}
    assert "normalising the name" in entry.reason


def test_a_mirror_of_the_gold_is_not_evidence():
    """Same bytes, fetched from somewhere else: the rows are already labeled."""
    digest = "a" * 64
    plan = DataPlan(
        task=PlanTask(target_column="cell_type", feature_prefixes=["ENSG"]),
        gold=[
            PlanEntry(
                path="gold.csv",
                role="gold",
                reason="target present",
                features=["ENSG00000000001"],
                provenance={"sha256": digest},
            )
        ],
        supplementary=[
            PlanEntry(
                path="mirror.csv",
                role="supplementary",
                reason="no target column",
                features=["ENSG00000000001"],
                provenance={"sha256": digest},
            )
        ],
    )

    _enforce_supplementary_schema(plan)

    assert plan.supplementary == []
    assert plan.unusable[0].path == "mirror.csv"
    assert "a mirror of the primary gold" in plan.unusable[0].reason


def test_numbers_written_as_text_are_not_free_text(tmp_path):
    """The UCI cohort case: `?` in a numeric column does not make it a category.

    `chol` holds ~150 distinct numbers and two `?`; counting distinct values
    reads that as free text and refuses the whole cohort.
    """
    cohort = tmp_path / "cohort.csv"
    pd.DataFrame(
        {
            "age": [50.0, 60.0, 70.0, 80.0],
            "chol": ["200", "?", "240", "280"],
            "thal": ["fixed", "normal", "reversible", "normal"],
            "num": [0, 1, 0, 1],
        }
    ).to_csv(cohort, index=False)
    plan = build_plan(
        [cohort], task(target_column="num", feature_prefixes=[], sample_id_column=None)
    )
    assert [e.path for e in plan.gold] == [str(cohort)]
    assert plan.unusable == []
    # Only `thal` is a category; `chol` is a number with a missing measurement.
    assert "1 categorical" in plan.gold[0].reason


def test_a_review_calling_a_table_gold_cannot_widen_what_we_train_on(tmp_path):
    """Rules decide; a model proposal is recorded, never a promotion."""
    gold = labeled(tmp_path / "gold.csv", rows=60)
    other = unlabeled(tmp_path / "other.csv")
    reviews = {
        str(other): {"role": "gold", "reason": "looks like a labeled table to me"}
    }
    plan = build_plan([gold, other], task(), reviews=reviews)
    assert [e.path for e in plan.gold] == [str(gold)]
    assert [e.path for e in plan.supplementary] == [str(other)]
    assert plan.unusable == []
    # The entry says what the plan did, not what the table looked like before
    # the review: a role recorded from the pre-review decision made the file
    # explain something other than its own buckets.
    assert plan.supplementary[0].role == "supplementary"
    assert "review proposed 'gold', rules kept 'supplementary'" in plan.notes[0]


def test_the_plan_carries_the_gold_features_the_run_will_use(tmp_path):
    """One unusable column no longer costs the whole table.

    The perovskite table has `pce` plus four free-text columns; refusing it as a
    whole is what left a run with 0 gold. Now the free text is left out, and the
    plan writes the remaining columns as the task's feature list -- which is
    also what every other table is compared against.
    """
    gold = tmp_path / "cells.csv"
    pd.DataFrame(
        {
            "ff": [0.7 + i / 1000 for i in range(60)],
            "jsc": [20.0 + i / 100 for i in range(60)],
            "stack": [f"device stack variant {i}" for i in range(60)],
            "pce": [18.0 + i / 50 for i in range(60)],
        }
    ).to_csv(gold, index=False)
    plan = build_plan(
        [gold],
        task(target_column="pce", feature_prefixes=[], sample_id_column=None),
    )
    assert [e.path for e in plan.gold] == [str(gold)]
    assert plan.gold[0].features == ["ff", "jsc"]
    assert plan.gold[0].dropped_features == ["stack"]
    # The run trains on exactly these columns.
    assert plan.task.feature_columns == ["ff", "jsc"]
    assert any("training features taken from the primary gold" in note for note in plan.notes)


def _mouse_style_table(tmp_path):
    """A table written the way a mouse annotation is: `Pisd`, `Actb`, ..."""
    path = tmp_path / "mouse.csv"
    pd.DataFrame(
        {
            "sample_id": [f"M{i}" for i in range(40)],
            "Pisd": [float(i) for i in range(40)],
            "Actb": [float(i % 5) for i in range(40)],
            "Gapdh": [float(i) for i in range(40)],
            "Alb": [float(i % 3) for i in range(40)],
            "Ins1": [float(i) for i in range(40)],
            "Gcg": [float(i) for i in range(40)],
        }
    ).to_csv(path, index=False)
    return path


def test_a_mouse_table_is_not_evidence_for_a_human_gold(tmp_path, monkeypatch):
    """`Pisd` and `PISD` are orthologs: same spelling, different measurement.

    The matcher lowercases names, so without a style check a mouse table
    "shares" thousands of genes with a human one -- 12,476 of them in the Baron
    pancreas set -- and would enter the correction as if it were the same assay.
    """
    monkeypatch.delenv("KOSMOS_ALLOW_CROSS_SPECIES", raising=False)
    gold = tmp_path / "human.csv"
    pd.DataFrame(
        {
            "sample_id": [f"H{i}" for i in range(40)],
            "PISD": [float(i) for i in range(40)],
            "ACTB": [float(i % 5) for i in range(40)],
            "GAPDH": [float(i) for i in range(40)],
            "ALB": [float(i % 3) for i in range(40)],
            "INS": [float(i) for i in range(40)],
            "GCG": [float(i) for i in range(40)],
            "cell_type": ["A", "B"] * 20,
        }
    ).to_csv(gold, index=False)
    mouse = _mouse_style_table(tmp_path)
    shape = PlanTask(
        objective="predict cell type",
        target_column="cell_type",
        feature_columns=["PISD", "ACTB", "GAPDH", "ALB", "INS", "GCG"],
        sample_id_column="sample_id",
    )

    plan = build_plan([gold, mouse], shape)

    assert plan.supplementary == []
    reason = next(entry.reason for entry in plan.unusable if entry.path == str(mouse))
    assert "the organism does not" in reason
    assert "KOSMOS_ALLOW_CROSS_SPECIES" in reason


def test_cross_species_evidence_can_be_asked_for_explicitly(tmp_path, monkeypatch):
    monkeypatch.setenv("KOSMOS_ALLOW_CROSS_SPECIES", "1")
    gold = tmp_path / "human.csv"
    pd.DataFrame(
        {
            "sample_id": [f"H{i}" for i in range(40)],
            "PISD": [float(i) for i in range(40)],
            "ACTB": [float(i % 5) for i in range(40)],
            "GAPDH": [float(i) for i in range(40)],
            "ALB": [float(i % 3) for i in range(40)],
            "INS": [float(i) for i in range(40)],
            "GCG": [float(i) for i in range(40)],
            "cell_type": ["A", "B"] * 20,
        }
    ).to_csv(gold, index=False)
    mouse = _mouse_style_table(tmp_path)
    shape = PlanTask(
        objective="predict cell type",
        target_column="cell_type",
        feature_columns=["PISD", "ACTB", "GAPDH", "ALB", "INS", "GCG"],
        sample_id_column="sample_id",
    )

    plan = build_plan([gold, mouse], shape)

    assert [entry.path for entry in plan.supplementary] == [str(mouse)]
