"""A perturbation run must keep the annotated file of a series, not just the matrix.

The prompts prefer the smallest *data* file, so a Perturb-seq matrix comes down
and the guide/cell annotation -- where the perturbation identity actually lives --
is dropped. The run that found GSE90063 (the right screen) ended with nothing to
supervise for exactly this reason.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]


def _module():
    spec = importlib.util.spec_from_file_location(
        "auto_task_run", ROOT / "scripts" / "auto_task_run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Log:
    def __init__(self):
        self.events = []

    def event(self, stage, message, **fields):
        self.events.append((stage, message, fields))


LISTING = {
    "reference": "geo://GSE90063",
    "files": [
        {
            "reference": "geo://GSE90063#suppl/GSE90063_k562_umi_wt.txt.gz",
            "path": "GSE90063_k562_umi_wt.txt.gz",
            "bytes": 400_000_000,
        },
        {
            "reference": "geo://GSE90063#suppl/GSE90063_guide_calls.csv.gz",
            "path": "GSE90063_guide_calls.csv.gz",
            "bytes": 2_000_000,
        },
    ],
}


def _args(task_kind):
    return SimpleNamespace(task_kind=task_kind)


def test_a_perturbation_run_keeps_the_annotated_sibling():
    module = _module()
    log = Log()
    chosen = [{"reference": "geo://GSE90063#suppl/GSE90063_k562_umi_wt.txt.gz"}]

    result = module._prefer_annotated_file(LISTING, chosen, _args("perturbation"), log)

    references = [entry["reference"] for entry in result]
    # the matrix is kept (it is the measurement) ...
    assert "geo://GSE90063#suppl/GSE90063_k562_umi_wt.txt.gz" in references
    # ... and so is the file that carries the labels
    assert "geo://GSE90063#suppl/GSE90063_guide_calls.csv.gz" in references
    assert any("annotated file" in message for _, message, _ in log.events)


def test_a_per_cell_run_is_untouched():
    module = _module()
    chosen = [{"reference": "geo://GSE90063#suppl/GSE90063_k562_umi_wt.txt.gz"}]

    result = module._prefer_annotated_file(LISTING, chosen, _args("per_cell"), Log())

    assert result == chosen


def test_nothing_is_added_when_the_series_has_no_annotated_file():
    module = _module()
    listing = {"files": [LISTING["files"][0]]}
    chosen = [{"reference": "geo://GSE90063#suppl/GSE90063_k562_umi_wt.txt.gz"}]

    result = module._prefer_annotated_file(listing, chosen, _args("perturbation"), Log())

    assert result == chosen


def test_a_file_already_chosen_is_not_added_twice():
    module = _module()
    chosen = [
        {"reference": "geo://GSE90063#suppl/GSE90063_k562_umi_wt.txt.gz"},
        {"reference": "geo://GSE90063#suppl/GSE90063_guide_calls.csv.gz"},
    ]

    result = module._prefer_annotated_file(LISTING, chosen, _args("perturbation"), Log())

    assert len(result) == 2


EXPRESSION_LISTING = {
    "reference": "geo://GSE153056",
    "files": [
        {
            "reference": "geo://GSE153056#suppl/GSE153056_ECCITE_metadata.tsv.gz",
            "path": "GSE153056_ECCITE_metadata.tsv.gz",
            "bytes": 1_048_576,
        },
        {
            "reference": "geo://GSE153056#suppl/GSE153056_RAW.tar",
            "path": "GSE153056_RAW.tar",
            "bytes": 127_926_272,
        },
    ],
}


def test_a_perturbation_run_also_keeps_the_measurements_file():
    """The labels are in the metadata; the transcriptome is in the counts file."""
    module = _module()
    log = Log()
    chosen = [{"reference": "geo://GSE153056#suppl/GSE153056_ECCITE_metadata.tsv.gz"}]

    result = module._prefer_expression_file(EXPRESSION_LISTING, chosen, _args("perturbation"), log)

    references = [entry["reference"] for entry in result]
    assert "geo://GSE153056#suppl/GSE153056_RAW.tar" in references
    assert any("measurements" in message for _, message, _ in log.events)


def test_the_expression_preference_is_a_no_op_for_per_cell():
    module = _module()
    chosen = [{"reference": "geo://GSE153056#suppl/GSE153056_ECCITE_metadata.tsv.gz"}]

    result = module._prefer_expression_file(EXPRESSION_LISTING, chosen, _args("per_cell"), Log())

    assert result == chosen
