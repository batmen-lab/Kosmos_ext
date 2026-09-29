"""Which kind of data task a research question is, and which objective answers it.

The discovery loop asks a question and then has to *do* something about it. Two
kinds of data task are wired here, and they want different objectives:

  **inference** -- "is this association real?", "does A affect B?", "how large is
  the difference?". The answer is an estimate with an uncertainty, so the run
  uses the **signed PPI correction**: it keeps the estimate of the labeled-only
  model unbiased while using unlabeled rows to shrink its variance, and the
  headline is the corrected metric with a bootstrap interval next to the
  gold-only baseline it corrects.

  **prediction** -- "how well can X be predicted from Y?", "can this be
  classified?", "would more data improve the prediction?". The answer is a
  model, so the run uses the **gold-anchored gradient-gated** objective: the
  synthetic rows may accelerate the labeled gradient, and are ignored wherever
  they disagree with it (`λ_t = 0` ⇒ gold-only).

Two reasons this is a module and not an `if question.startswith(...)`:

  * the classification is recorded (`decided_by`, `rationale`, the words or the
    model that settled it), because "the run used the wrong objective" has to be
    checkable after the fact;
  * the mapping from kind to objective lives in one place, so the discovery
    loop, `run.py` and the stress script cannot drift apart.

Words decide when they clearly point one way. When they are silent or evenly
split the model is asked, constrained to the two kinds; with no model, the
fallback is **prediction**, whose worst case is ordinary gold-only training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Literal

Kind = Literal["inference", "prediction", "perturbation_response"]

#: The single-cell backends a data task can be routed to. `simple` is the
#: per-cell column task (classification/regression); `perturbation` is the
#: control + perturbed gene(s) -> delta expression task.
BACKENDS = ("simple", "perturbation")


#: Language that asks for the *effect of a perturbation*, which is a different
#: object from a label: the input is a control cell plus the perturbed gene(s)
#: and the output is a vector of expression change.
PERTURBATION_WORDS = (
    "perturb",
    "perturbation",
    "knockout",
    "knock-out",
    "knock out",
    "knockdown",
    "knock-down",
    "crispr",
    "sgrna",
    "guide rna",
    "perturb-seq",
    "perturbseq",
    "overexpress",
    "activation screen",
    "gene deletion",
    "gene silencing",
    "combinatorial perturbation",
    "double knockout",
    "response to",
    "transcriptional response",
)

#: Objective each kind is trained with.
LOSS_FOR_KIND: dict[str, str] = {
    "inference": "signed",
    "prediction": "gradient_gated",
}

#: Language that asks whether something is true (an estimate), not how well
#: something can be predicted (a model).
INFERENCE_WORDS = (
    "associated",
    "association",
    "correlat",
    "effect of",
    "effect on",
    "effects of",
    "impact of",
    "influence of",
    "differ",
    "difference between",
    "enrich",
    "differential",
    "significan",
    "compare",
    "comparison",
    "is there",
    "are there",
    "does ",
    "do ",
    "how much",
    "how many",
    "relationship between",
    "driver",
    "marker of",
    "causal",
)
#: Language that asks for a model and a number on a held-out split.
PREDICTION_WORDS = (
    "predict",
    "prediction",
    "classif",
    "accuracy",
    "auroc",
    "auc",
    "f1",
    "detect",
    "diagnose",
    "screen for",
    "forecast",
    "improve the prediction",
    "generalize",
    "generalise",
    "transfer",
    "annotate",
    "cell type",
    "celltype",
)


#: Language that says the answer needs data from outside the run: samples,
#: cohorts, measurements, an assay. A question that names them is a data
#: question even before anyone looks for a dataset.
DATA_WORDS = (
    "dataset",
    "data",
    "cohort",
    "donor",
    "patient",
    "sample",
    "cell",
    "cells",
    "expression",
    "transcriptom",
    "rna-seq",
    "rna seq",
    "scrna",
    "single-cell",
    "single cell",
    "clinical",
    "biopsy",
    "record",
    "measurement",
    "assay",
    "sequencing",
    "microarray",
    "tumor",
    "tumour",
    "genome",
    "gene",
    "protein",
    "metabol",
    "ehr",
    "public data",
)
#: Language that says the answer comes from reasoning or simulation instead:
#: there is nothing to fetch, and sending the question to a repository wastes a
#: retrieval round and then fails.
NON_DATA_WORDS = (
    "simulat",
    "theoretical",
    "derivation",
    "derive ",
    "prove ",
    "proof",
    "analytically",
    "closed form",
    "literature review",
    "review the literature",
    "thought experiment",
    "toy model",
    "first principles",
)


def needs_external_data(
    question: str,
    *,
    extra_text: str = "",
    domain: str = "",
    client: Any = None,
) -> tuple[bool, str]:
    """Is this question answered with fetched data, or without it?

    The discovery loop runs questions of both kinds, and the code path is the
    right one for a simulation or a derivation: fetching would spend a retrieval
    round and an LLM call to find nothing. So the fetcher is asked only for the
    questions that name data. Words decide; the model settles a tie when there
    is one; with neither, the domain decides -- biology is a data domain here,
    and the other domains are not.
    """
    text = f"{question} {extra_text}".lower()
    data_hits = _count(text, DATA_WORDS)
    other_hits = _count(text, NON_DATA_WORDS)
    if data_hits > other_hits:
        return True, (
            f"the question names data it does not have ({data_hits} data word(s) "
            f"against {other_hits} non-data)"
        )
    if other_hits > data_hits:
        return False, (
            f"the question is answered by analysis or simulation rather than by "
            f"fetching data ({other_hits} non-data word(s) against {data_hits})"
        )
    if client is not None:
        asked = _ask_whether_data(question, extra_text, client)
        if asked is not None:
            return asked, "the model read the question and said so"
    biology = str(domain).lower() in {"biology", "bioinformatics", "medicine", "genomics"}
    return biology, (
        "no word decides it, so the domain does: "
        + ("biology is answered with data" if biology else "this domain is not")
    )


@dataclass(frozen=True)
class TaskKind:
    """What the question is, and what that means for the training run."""

    kind: Kind
    loss_mode: str
    rationale: str
    decided_by: str
    #: How many words of each kind the text carried, for the record.
    inference_hits: int = 0
    prediction_hits: int = 0
    #: Which backend answers it: the per-cell supervised trainer ("simple") or
    #: the perturbation-response backend ("perturbation").
    backend: str = "simple"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "loss_mode": self.loss_mode,
            "rationale": self.rationale,
            "decided_by": self.decided_by,
            "backend": self.backend,
            "inference_hits": self.inference_hits,
            "prediction_hits": self.prediction_hits,
        }

    @property
    def headline(self) -> str:
        """What to print as the answer's headline metric."""
        return (
            "the corrected estimate and its interval"
            if self.kind == "inference"
            else "the held-out metric of the trained model"
        )


def apply_backend_override(kind: TaskKind, backend: str | None) -> TaskKind:
    """Force the backend the caller named, keeping the objective where it fits.

    The word (and model) classifier is a *default*: it reads the question. A
    caller who names the backend knows something the sentence does not say, and
    the flag has to win -- otherwise `kosmos run --task perturbation` on a
    question whose wording is sparse would silently do the column task. The
    reverse direction is the escape hatch: a question that mentions "knockout"
    but is really about a column can be pinned back to `per_cell`.

    The objective is preserved where it still applies: forcing `simple` on a
    perturbation-looking question keeps the safe prediction objective, and
    forcing `perturbation` uses the graph-gated one.
    """
    name = str(backend or "").strip().lower().replace("-", "_")
    if name in ("", "auto", "none", "default"):
        return kind
    if name in ("per_cell", "percell", "simple", "column"):
        if kind.backend == "simple":
            return kind
        return replace(
            kind,
            kind="prediction",
            backend="simple",
            loss_mode=LOSS_FOR_KIND["prediction"],
            rationale="the caller named the per-cell (column) backend",
            decided_by="config",
        )
    if name in ("perturbation", "perturbation_response"):
        if kind.backend == "perturbation":
            return kind
        return replace(
            kind,
            kind="perturbation_response",
            backend="perturbation",
            loss_mode="graph_gated",
            rationale="the caller named the perturbation backend",
            decided_by="config",
        )
    raise ValueError(
        f"unknown task backend {backend!r}; use one of {list(BACKENDS)} or 'auto'"
    )


def _count(text: str, words) -> int:
    return sum(1 for word in words if word in text)


def classify_task(
    question: str,
    *,
    extra_text: str = "",
    client: Any = None,
    ask_model: bool = True,
) -> TaskKind:
    """Inference or prediction -- decided by words, then by the model.

    `extra_text` is where a hypothesis or a protocol description goes: the
    question alone is often one sentence, and the plan states the task more
    plainly than the prompt does.
    """
    text = f"{question} {extra_text}".lower()
    inference_hits = _count(text, INFERENCE_WORDS)
    prediction_hits = _count(text, PREDICTION_WORDS)
    perturbation_hits = _count(text, PERTURBATION_WORDS)

    if perturbation_hits and perturbation_hits >= max(inference_hits, prediction_hits):
        # A perturbation question is not a differently-worded prediction
        # question: the answer is a vector of expression change for a
        # perturbation, which the per-cell trainer cannot represent at all.
        return TaskKind(
            kind="perturbation_response",
            loss_mode="graph_gated",
            backend="perturbation",
            rationale=(
                f"the question asks what a perturbation does to expression "
                f"({perturbation_hits} perturbation word(s)); that is a response "
                f"vector, not a per-cell label"
            ),
            decided_by="words",
            inference_hits=inference_hits,
            prediction_hits=prediction_hits,
        )

    if inference_hits or prediction_hits:
        if inference_hits > prediction_hits:
            return TaskKind(
                kind="inference",
                loss_mode=LOSS_FOR_KIND["inference"],
                rationale=(
                    f"the question asks whether something is true rather than "
                    f"how well it can be predicted ({inference_hits} inferential "
                    f"word(s) against {prediction_hits} predictive)"
                ),
                decided_by="words",
                inference_hits=inference_hits,
                prediction_hits=prediction_hits,
            )
        if prediction_hits > inference_hits:
            return TaskKind(
                kind="prediction",
                loss_mode=LOSS_FOR_KIND["prediction"],
                rationale=(
                    f"the question asks for a model's held-out performance "
                    f"({prediction_hits} predictive word(s) against "
                    f"{inference_hits} inferential)"
                ),
                decided_by="words",
                inference_hits=inference_hits,
                prediction_hits=prediction_hits,
            )
        # A tie: both vocabularies appear ("predict X and test whether it is
        # associated with Y"). The model settles it when there is one.

    if client is not None and ask_model:
        asked = _ask_model(question, extra_text, client)
        if asked is not None:
            kind = asked
            return TaskKind(
                kind=kind,
                loss_mode=LOSS_FOR_KIND[kind],
                rationale=f"the model read the question and called it {kind!r}",
                decided_by="model",
                inference_hits=inference_hits,
                prediction_hits=prediction_hits,
            )

    rationale = (
        "the question carries no word of either kind, so the safer objective is "
        "used: a prediction task whose worst case is gold-only training"
        if not (inference_hits or prediction_hits)
        else "the question carries both vocabularies equally; prediction is the "
        "safer objective"
    )
    return TaskKind(
        kind="prediction",
        loss_mode=LOSS_FOR_KIND["prediction"],
        rationale=rationale,
        decided_by="fallback",
        inference_hits=inference_hits,
        prediction_hits=prediction_hits,
    )


_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["inference", "prediction"]},
        "reason": {"type": "string"},
    },
    "required": ["kind", "reason"],
}


def _ask_model(question: str, extra_text: str, client: Any) -> Kind | None:
    """One constrained call, or None. Anything outside the two kinds is dropped."""
    prompt = (
        "A researcher asks:\n\n"
        f"{question}\n"
        + (f"\nMore detail given with it:\n{extra_text}\n" if extra_text else "")
        + "\nWhich is this?\n"
        "  inference  -- the answer is whether an association/effect/difference "
        "is real, and how large it is\n"
        "  prediction -- the answer is how well something can be predicted, "
        "scored on data the model has not seen\n\n"
        "Answer with a JSON object with keys 'kind' and 'reason'."
    )
    try:
        response = client.generate_structured(
            prompt=prompt, schema=_SCHEMA, max_tokens=200, temperature=0
        )
    except Exception:  # noqa: BLE001 - a failed call is "no answer"
        return None
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            return None
    kind = str((response or {}).get("kind") or "").strip().lower()
    return kind if kind in {"inference", "prediction"} else None  # type: ignore[return-value]


_DATA_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_data": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["needs_data", "reason"],
}


def _ask_whether_data(question: str, extra_text: str, client: Any) -> bool | None:
    """One constrained call: does answering this need a dataset from outside?"""
    prompt = (
        "A researcher asks:\n\n"
        f"{question}\n"
        + (f"\nMore detail given with it:\n{extra_text}\n" if extra_text else "")
        + "\nDoes answering this need measurements from a dataset that has to be "
        "found or fetched, or is it answered by analysis, simulation or "
        "reasoning over what is already here?\n\n"
        "Answer with a JSON object with keys 'needs_data' (boolean) and 'reason'."
    )
    try:
        response = client.generate_structured(
            prompt=prompt, schema=_DATA_SCHEMA, max_tokens=200, temperature=0
        )
    except Exception:  # noqa: BLE001 - a failed call is "no answer"
        return None
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            return None
    value = (response or {}).get("needs_data")
    return value if isinstance(value, bool) else None
