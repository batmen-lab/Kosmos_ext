"""The reviewer must be told which task it is reviewing for.

A perturbation question needs a per-cell *screen* -- a condition column, its
control values and the assay -- while the x->y question needs a *label* column.
Same sample, different question, so the gate has to know which one is asked;
otherwise a cell-metadata table is gated as gold and a real screen is refused.
"""

from __future__ import annotations

from kosmos.discovery.review import review_table


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.prompts: list[str] = []
        self.schemas: list[dict] = []

    def generate_structured(self, prompt, schema, **kwargs):
        self.prompts.append(prompt)
        self.schemas.append(schema)
        return self.response


PACKET = {
    "path": "/tmp/GSE90063_k562_umi_wt.txt.gz",
    "columns": ["gene", "count", "condition", "cell_type"],
    "kinds": {},
    "null_fraction": {},
    "raw_head": ["gene,count,condition,cell_type"],
}


def test_a_perturbation_review_asks_about_a_screen():
    client = FakeClient(
        {
            "header_present": True,
            "role": "gold",
            "is_screen": True,
            "has_expression": True,
            "condition_column": "condition",
            "control_labels": ["ctrl"],
            "modality": "crisprko",
            "target_column": "condition",
            "reason": "a per-cell CRISPR knockout screen with controls",
        }
    )

    review = review_table(
        "Does knocking out ETS2 change the transcriptome?",
        PACKET,
        client=client,
        task_kind="perturbation",
    )

    assert review is not None
    assert review.role == "gold"
    assert review.condition_column == "condition"
    assert review.control_labels == ["ctrl"]
    assert review.modality == "crisprko"
    # the plan's target is the condition column, so the rest of the pipeline works
    assert review.target_column == "condition"
    prompt = client.prompts[0]
    assert "perturbation-response" in prompt
    assert "condition_column" in client.schemas[0]["properties"]
    assert "is_screen" in client.schemas[0]["required"]


def test_a_gold_without_a_condition_column_is_refused():
    """The guard the mode exists for: no condition column -> not a screen."""
    client = FakeClient(
        {
            "header_present": True,
            "role": "gold",
            "is_screen": False,
            "condition_column": None,
            "reason": "cell metadata table",
        }
    )

    review = review_table("...", PACKET, client=client, task_kind="perturbation")

    assert review is not None
    assert review.role == "unusable"


def test_a_condition_column_that_is_not_visible_is_ignored():
    client = FakeClient(
        {
            "header_present": True,
            "role": "gold",
            "is_screen": True,
            "condition_column": "not_a_column",
            "reason": "screen",
        }
    )

    review = review_table("...", PACKET, client=client, task_kind="perturbation")

    assert review.condition_column is None
    assert review.role == "unusable"


def test_the_per_cell_review_is_unchanged():
    client = FakeClient(
        {"header_present": True, "role": "gold", "target_column": "cell_type", "reason": "labels"}
    )
    review = review_table("predict the cell type", PACKET, client=client, task_kind="per_cell")

    assert review.target_column == "cell_type"
    assert review.condition_column is None
    assert "is_screen" not in client.schemas[0]["properties"]


def test_a_screen_without_expression_is_refused():
    """A cell metadata table has the perturbation but not the transcriptome.

    GSE153056_ECCITE_metadata.tsv.gz was gated as `gold` -- condition column
    `gene`, controls `NT` -- and then nothing could be trained, because its
    other columns are QC metrics, not genes.
    """
    client = FakeClient(
        {
            "header_present": True,
            "role": "gold",
            "is_screen": True,
            "has_expression": False,
            "condition_column": "condition",
            "control_labels": ["NT"],
            "modality": "crisprko",
            "reason": "per-cell metadata with the targeted gene",
        }
    )

    review = review_table("...", PACKET, client=client, task_kind="perturbation")

    assert review.is_screen is True
    assert review.has_expression is False
    assert review.role == "unusable"


def test_a_screen_with_expression_is_kept():
    client = FakeClient(
        {
            "header_present": True,
            "role": "gold",
            "is_screen": True,
            "has_expression": True,
            "condition_column": "condition",
            "control_labels": ["NT"],
            "modality": "crisprko",
            "reason": "a real screen",
        }
    )

    review = review_table("...", PACKET, client=client, task_kind="perturbation")

    assert review.role == "gold"
    assert review.target_column == "condition"
