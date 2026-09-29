"""Which objective a question wants: inference (signed) or prediction (gated)."""

from __future__ import annotations

import json

from kosmos.ppi.task_ontology import classify_task, needs_external_data


class FakeClient:
    """A model that answers whatever the test told it to."""

    def __init__(self, response):
        self.response = response
        self.prompts = []

    def generate_structured(self, prompt, schema, **kwargs):
        self.prompts.append(prompt)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_a_prediction_question_gets_the_gated_objective():
    result = classify_task(
        "Can the cell type of a bone-marrow mononuclear cell be predicted from "
        "its gene-expression profile, and do unlabeled cells improve the prediction?"
    )
    assert result.kind == "prediction"
    assert result.loss_mode == "gradient_gated"
    assert result.decided_by == "words"
    assert result.rationale


def test_an_inference_question_gets_the_signed_correction():
    result = classify_task(
        "Is BCR-ABL1 expression associated with imatinib response across donors, "
        "and how large is the difference between responders and non-responders?"
    )
    assert result.kind == "inference"
    assert result.loss_mode == "signed"
    assert result.decided_by == "words"


def test_the_hypothesis_text_counts_too():
    """The plan states the task more plainly than the prompt does."""
    result = classify_task(
        "What drives resistance?",
        extra_text="Test whether the resistant clone differs from the parental line.",
    )
    assert result.kind == "inference"
    assert result.loss_mode == "signed"


def test_a_silent_question_falls_back_to_the_safe_objective():
    """No vocabulary at all: prediction, whose worst case is gold-only training."""
    result = classify_task("Tell me about these cells.")
    assert result.kind == "prediction"
    assert result.loss_mode == "gradient_gated"
    assert result.decided_by == "fallback"
    assert "safer" in result.rationale


def test_the_model_settles_a_tie_and_cannot_answer_outside_the_two_kinds():
    tie = "Predict the response and test whether it is associated with survival."
    asked = classify_task(tie, client=FakeClient({"kind": "inference", "reason": "x"}))
    assert asked.decided_by == "model"
    assert asked.kind == "inference"
    assert asked.loss_mode == "signed"

    # A model that answers with something else is not obeyed.
    ignored = classify_task(tie, client=FakeClient({"kind": "clustering"}))
    assert ignored.decided_by == "fallback"
    assert ignored.kind == "prediction"

    # A model that fails is not fatal either.
    failed = classify_task(tie, client=FakeClient(RuntimeError("no model")))
    assert failed.decided_by == "fallback"

    # A string response is accepted only when it is JSON of the right shape.
    stringly = classify_task(tie, client=FakeClient(json.dumps({"kind": "prediction"})))
    assert stringly.decided_by == "model"
    assert stringly.kind == "prediction"


def test_the_decision_is_recorded():
    payload = classify_task("Is there an association between dose and response?").to_dict()
    assert payload["kind"] == "inference"
    assert payload["decided_by"] == "words"
    assert payload["inference_hits"] >= 1
    assert payload["loss_mode"] == "signed"


def test_a_simulation_question_is_not_sent_to_the_fetcher():
    """The pre-judgment that keeps retrieval for the questions data can answer."""
    needs, why = needs_external_data(
        "Simulate the binding energy of this protein from first principles.",
        domain="biology",
    )
    assert needs is False
    assert "simulation" in why

    needs, why = needs_external_data(
        "Can the cell type of a pancreatic islet cell be predicted from its "
        "gene-expression profile across donors?",
        domain="biology",
    )
    assert needs is True
    assert "data word" in why


def test_the_domain_decides_only_when_no_word_does():
    # Nothing to go on: biology here is a data domain.
    needs, why = needs_external_data("What makes a good experiment?", domain="biology")
    assert needs is True and "domain" in why
    needs, _ = needs_external_data("What makes a good experiment?", domain="materials")
    assert needs is False


def test_the_model_settles_an_evensplit_and_cannot_overrule_a_clear_one():
    evensplit = "The proof rests on a cohort."
    asked = needs_external_data(
        evensplit, domain="biology", client=FakeClient({"needs_data": False})
    )
    assert asked == (False, "the model read the question and said so")
    # A model that fails leaves the words in charge.
    failed = needs_external_data(
        "Simulate the system.", domain="biology", client=FakeClient(RuntimeError("no"))
    )
    assert failed[0] is False


def test_an_explicit_backend_overrides_the_words():
    """`--task` is the caller's knowledge; the classifier is only a default."""
    from kosmos.ppi.task_ontology import apply_backend_override, classify_task

    # words say prediction, the caller says perturbation
    kind = classify_task("Can the cell type be predicted from expression?")
    assert kind.backend == "simple"
    forced = apply_backend_override(kind, "perturbation")
    assert forced.backend == "perturbation"
    assert forced.kind == "perturbation_response"
    assert forced.loss_mode == "graph_gated"
    assert forced.decided_by == "config"

    # words say perturbation, the caller pins the column task back
    noisy = classify_task("Does the CRISPR knockout change the transcriptome?")
    assert noisy.backend == "perturbation"
    pinned = apply_backend_override(noisy, "per_cell")
    assert pinned.backend == "simple"
    assert pinned.kind == "prediction"
    assert pinned.loss_mode == "gradient_gated"

    # auto (and unset) is a no-op
    assert apply_backend_override(noisy, None) is noisy
    assert apply_backend_override(noisy, "auto") is noisy
