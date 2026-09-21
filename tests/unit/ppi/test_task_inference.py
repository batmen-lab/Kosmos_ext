"""Choosing the label column: name matching first, a constrained model second."""

from __future__ import annotations

import pandas as pd

from kosmos.ppi.task_inference import (
    column_facts,
    eligible_columns,
    infer_target_column,
)


def table(path, rows=60, extra=None, label_values=("B", "T", "NK")):
    frame = {
        "sample_id": [f"s{i}" for i in range(rows)],
        "ENSG00000000001": [float(i) for i in range(rows)],
        "batch_code": [1 + (i % 2) for i in range(rows)],
        "cell_type": [label_values[i % len(label_values)] for i in range(rows)],
    }
    if extra:
        frame.update(extra)
    frame = pd.DataFrame(frame)
    frame.to_csv(path, index=False)
    return path


class FakeClient:
    """A model that returns whatever the test tells it to."""

    def __init__(self, response):
        self.response = response
        self.prompts = []

    def generate_structured(self, prompt, schema, **kwargs):
        self.prompts.append(prompt)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_a_hint_that_names_a_column_is_a_lookup_not_a_guess(tmp_path):
    path = table(tmp_path / "t.csv")
    result = infer_target_column(
        path, objective="anything", hints=["cell_type"], exclude=["sample_id"]
    )
    assert result.column == "cell_type"
    assert result.source == "hint_exact"
    assert result.confidence == "exact_match"


def test_the_question_text_alone_can_name_the_column(tmp_path):
    path = table(tmp_path / "t.csv")
    result = infer_target_column(
        path,
        objective="Predict the cell type of each sample from its expression profile",
        exclude=["sample_id"],
    )
    assert result.column == "cell_type"
    assert result.source == "objective_match"
    assert result.confidence == "inferred"


def test_a_word_in_the_question_is_not_a_gene_symbol(tmp_path):
    """`WAS` matched because "a cancer cell *was* treated" contains it.

    Every single-cell table has a column for a gene, so the rules then called an
    unlabeled perturbation table "labeled", overruled the model that had read
    the file, and trained a classifier on the table's own expression. A bare
    name this short, found only in the question's prose, is a coincidence.
    """
    path = table(
        tmp_path / "chempert.csv",
        extra={
            "WAS": [float(i % 3) for i in range(60)],
            "compound_1": ["drugA", "drugB", "vehicle"] * 20,
        },
        label_values=("drugA", "drugB", "vehicle"),
    )
    result = infer_target_column(
        path,
        objective=(
            "Can the drug a cancer cell was treated with be predicted from its "
            "transcriptome?"
        ),
        hints=["compound_1"],
        exclude=["sample_id"],
    )

    assert result.column == "compound_1"
    assert result.source == "hint_exact"


def test_a_gene_named_by_the_question_alone_is_not_a_label(tmp_path):
    """With no hint at all the prose must not promote a gene either."""
    path = table(
        tmp_path / "chempert.csv",
        extra={"WAS": [float(i % 3) for i in range(60)]},
    )
    result = infer_target_column(
        path,
        objective="Can the drug a cancer cell was treated with be predicted?",
        exclude=["sample_id"],
    )

    assert result.column != "WAS"
    # `WAS` may still be listed among the runners-up, but never at a score the
    # shortcut would act on.
    assert all(candidate.score < 0.6 for candidate in result.candidates if candidate.column == "WAS")


def test_an_identifier_is_never_accepted_as_a_label(tmp_path):
    """`sample_id` is unique in every row: a name match cannot rescue it."""
    path = table(tmp_path / "t.csv")
    facts = column_facts(path)
    eligible, rejected = eligible_columns(facts)
    assert "sample_id" not in eligible
    assert "identifier" in rejected["sample_id"]
    # Even asked for by name.
    result = infer_target_column(path, hints=["sample_id"])
    assert result.column != "sample_id"


def test_a_single_class_label_is_refused(tmp_path):
    path = table(tmp_path / "t.csv", label_values=("only",))
    result = infer_target_column(path, hints=["cell_type"], exclude=["sample_id"])
    assert result.column is None
    assert "classification label needs at least 2" in result.rejected["cell_type"]


def test_a_class_with_one_member_is_a_fact_not_a_refusal(tmp_path):
    """BMMC has cell types a 2,000-cell sample catches once.

    Refusing the label for that lost a whole single-cell table, and the split
    already degrades gracefully when a class is too small to stratify.
    """
    values = ["B"] * 58 + ["T", "NK"]
    path = table(tmp_path / "t.csv", rows=60, label_values=values)
    result = infer_target_column(path, hints=["cell_type"], exclude=["sample_id"])
    assert result.column == "cell_type"
    assert "too fine-grained" not in result.rejected.get("cell_type", "")


def test_a_continuous_column_is_refused_as_a_label_with_a_true_reason(tmp_path):
    """A measurement with many distinct values is not a label, and says so."""
    values = list(range(99)) + [0]
    path = table(tmp_path / "t.csv", rows=100, extra={"high_card": values})
    result = infer_target_column(path, hints=["cell_type"], exclude=["sample_id"])
    reason = result.rejected["high_card"]
    assert "continuous measurement, not a classification label" in reason
    assert "a class has" not in reason


def test_feature_prefixes_are_never_candidates(tmp_path):
    """A gene named in the question must not be nominated as the target."""
    path = table(tmp_path / "t.csv")
    facts = column_facts(path)
    eligible, rejected = eligible_columns(
        facts, exclude=["sample_id"], feature_prefixes=("ENSG",)
    )
    assert not [name for name in eligible if name.startswith("ENSG")]
    assert "which this task treats as features" in rejected["ENSG00000000001"]


def test_a_model_choice_must_come_from_the_eligible_columns(tmp_path):
    path = table(tmp_path / "t.csv")
    client = FakeClient({"column": "cell_type", "reason": "it is the label"})
    result = infer_target_column(
        path,
        objective="which group does this belong to?",
        exclude=["sample_id"],
        client=client,
    )
    assert result.column == "cell_type" and result.source == "llm"
    assert "cell_type" in client.prompts[0]
    assert "sample_id" not in client.prompts[0].split("- ")[-1]


def test_a_model_that_invents_a_column_is_discarded(tmp_path):
    path = table(tmp_path / "t.csv")
    client = FakeClient({"column": "patient_outcome", "reason": "plausible"})
    result = infer_target_column(
        path, objective="which group?", exclude=["sample_id"], client=client
    )
    assert result.column is None
    assert "not an eligible column" in result.rejected["__model__"]


def test_a_failed_model_call_leaves_a_usable_answer(tmp_path):
    path = table(tmp_path / "t.csv")
    client = FakeClient(RuntimeError("provider down"))
    result = infer_target_column(
        path, objective="which group?", exclude=["sample_id"], client=client
    )
    assert result.column is None
    assert "model call failed" in result.rejected["__model__"]


def test_no_hint_and_no_name_overlap_returns_nothing_with_reasons(tmp_path):
    path = table(tmp_path / "t.csv")
    result = infer_target_column(path, objective="estimate the mean of something")
    assert result.column is None
    assert result.source == "none"
    assert result.reason


def test_a_wide_table_keeps_the_json_answer_small(tmp_path):
    wide = {f"ENSG{i:011d}": [float(i + j) for j in range(60)] for i in range(60)}
    path = table(tmp_path / "t.csv", extra=wide)
    result = infer_target_column(path, hints=["cell_type"], exclude=["sample_id"])
    payload = result.to_dict()
    assert payload["target_column"] == "cell_type"
    assert len(payload["rejected"]) <= 10
    assert payload["rejected_total"] >= len(payload["rejected"])
def test_a_column_of_arrays_does_not_hide_the_rest_of_the_table(tmp_path):
    """The perovskite database carries an embedding per row; `nunique` refuses those.

    The failure read "could not read: unhashable type: 'numpy.ndarray'", so the
    table looked unreadable and the column that *could* have served as a label
    was never considered.
    """
    import numpy as np
    import pandas as pd

    from kosmos.ppi.task_inference import column_facts

    path = tmp_path / "with-embeddings.parquet"
    pd.DataFrame(
        {
            "embedding": [np.arange(4) for _ in range(3)],
            "pce": [18.0, 21.5, 19.2],
            "grade": ["low", "high", "low"],
        }
    ).to_parquet(path)

    facts = column_facts(path)

    assert set(facts) == {"embedding", "pce", "grade"}
    assert facts["grade"].n_unique == 2
    assert facts["embedding"].samples  # examples, without raising


def write_efficiency_table(path, rows=200):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    pd.DataFrame(
        {
            "spin_speed": rng.integers(1000, 5000, size=rows),
            # 15 distinct values: a coarse measurement, and few enough that a
            # classifier could also be built on it -- the genuinely ambiguous
            # case, where only the model can say which the question wants.
            "pce": 15.0 + 0.5 * rng.integers(0, 15, size=rows),
            "grade": ["A" if value > 16 else "B" for value in (15 + rng.normal(size=rows))],
        }
    ).to_csv(path, index=False)
    return path


def test_a_measured_column_is_a_regression_label(tmp_path):
    from kosmos.ppi.task_inference import infer_target_column

    path = write_efficiency_table(tmp_path / "cells.csv")
    inferred = infer_target_column(
        path,
        objective="how do fabrication parameters affect efficiency?",
        hints=["pce"],
        task_type="regression",
    )

    assert inferred.column == "pce"
    assert inferred.task_type == "regression"


def test_a_class_label_is_not_a_regression_label(tmp_path):
    from kosmos.ppi.task_inference import infer_target_column

    path = write_efficiency_table(tmp_path / "cells.csv")
    inferred = infer_target_column(
        path,
        objective="which grade is this cell?",
        hints=["grade"],
        task_type="regression",
    )
    # "grade" has two values: it cannot carry a regression task, so it is not
    # eligible and the run does not silently train a regressor on a category.
    assert inferred.column != "grade"
    assert "grade" in inferred.rejected


def test_auto_asks_the_model_which_kind_of_task_this_is(tmp_path):
    """A dozen-plus distinct values is genuinely ambiguous; the model decides."""
    from kosmos.ppi.task_inference import infer_target_column

    class FakeClient:
        def __init__(self):
            self.prompts = []

        def generate_structured(self, prompt, schema, **kwargs):
            self.prompts.append(prompt)
            return {
                "column": "pce",
                "task_type": "regression",
                "reason": "the question asks how parameters affect a measured efficiency",
            }

    path = write_efficiency_table(tmp_path / "cells.csv")
    client = FakeClient()
    inferred = infer_target_column(
        path,
        objective="how do fabrication parameters affect efficiency?",
        hints=["pce"],
        client=client,
        task_type="auto",
    )

    assert inferred.column == "pce"
    assert inferred.task_type == "regression"
    assert "task_type" in client.prompts[0]
