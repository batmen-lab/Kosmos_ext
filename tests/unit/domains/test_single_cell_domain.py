"""The single-cell domain: what it claims, and what it refuses.

The data pipeline here is a single-cell pipeline. A question about patients or
bulk tissue is a data question that it would answer with the wrong features and
the wrong split, so the modality is judged before any retrieval happens.
"""

from __future__ import annotations

from kosmos.domains.single_cell import (
    FILE_FORMATS,
    LABEL_COLUMNS,
    PREPROCESSING,
    is_single_cell_question,
)


class FakeClient:
    def __init__(self, response):
        self.response = response

    def generate_structured(self, prompt, schema, **kwargs):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_a_per_cell_question_is_this_domain_s_business():
    verdict = is_single_cell_question(
        "Can the cell type of a bone-marrow mononuclear cell be predicted from "
        "its gene-expression profile, and do unlabeled cells from another donor "
        "help?",
        domain="biology",
    )
    assert verdict.single_cell is True
    assert verdict.decided_by == "words"
    assert verdict.per_cell_hits > 0


def test_a_patient_question_is_not():
    """A cohort of people is not a cohort of cells."""
    verdict = is_single_cell_question(
        "Is a patient's blood pressure associated with hospital readmission, "
        "using routine clinical records?",
        domain="biology",
    )
    assert verdict.single_cell is False
    assert "not a claim" in verdict.rationale or "bulk" in verdict.rationale


def test_a_bulk_assay_is_not_per_cell_even_though_it_measures_expression():
    verdict = is_single_cell_question(
        "Does gene expression measured by bulk RNA-seq of whole tissue differ "
        "between responders and non-responders?",
        domain="biology",
    )
    assert verdict.single_cell is False
    assert verdict.bulk_hits > 0
    assert "bulk" in verdict.rationale


def test_the_domain_decides_when_the_words_do_not():
    thin = "Compare the groups and report the difference."
    assert is_single_cell_question(thin, domain="single_cell").single_cell is True
    assert is_single_cell_question(thin, domain="biology").single_cell is False


def test_the_model_settles_a_tie_and_cannot_answer_anything_else():
    tied = "The barcodes come from a microarray."
    asked = is_single_cell_question(
        tied, domain="biology", client=FakeClient({"single_cell": True})
    )
    assert asked.single_cell is True and asked.decided_by == "model"

    ignored = is_single_cell_question(
        tied, domain="biology", client=FakeClient({"single_cell": "maybe"})
    )
    assert ignored.decided_by == "domain"

    failed = is_single_cell_question(
        tied, domain="biology", client=FakeClient(RuntimeError("no model"))
    )
    assert failed.decided_by == "domain"


def test_the_conventions_the_pipeline_relies_on_are_recorded_here():
    """So the domain and the code cannot drift apart silently."""
    assert "cell_type" in LABEL_COLUMNS
    assert ".h5ad" in FILE_FORMATS
    recipe = " ".join(PREPROCESSING["recipe"]).lower()
    assert "highly variable" in recipe and "log1p" in recipe
    assert PREPROCESSING["module"] == "kosmos.ppi.singlecell"


def test_the_domain_is_registered_where_the_loop_looks():
    from kosmos.agents.skill_loader import SkillLoader

    assert "single_cell" in SkillLoader.DOMAIN_TO_BUNDLES
    assert "single_cell_analysis" in SkillLoader.DOMAIN_TO_BUNDLES["single_cell"]
