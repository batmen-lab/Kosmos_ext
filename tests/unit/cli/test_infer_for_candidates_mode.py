"""A perturbation run must reach the screen review even without a label column.

`infer_for_candidates` finds a *phenotype* column by name matching. A screen's
label is its condition column, which no phenotype vocabulary will match -- and
the function used to `raise SystemExit(2)` there, so the review (which is what
names a condition column) never ran on a real screen.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

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

    def raw(self, message):
        self.events.append(("raw", message, {}))


def _args(task_kind):
    return SimpleNamespace(
        no_llm=True,          # no model needed: this is about the abort behaviour
        scan_rows=50,
        objective="Does knocking out ETS2 change the transcriptome?",
        hint=[],
        exclude_col=[],
        feature_prefix=[],
        task_type="classification",
        task_kind=task_kind,
    )


def test_a_perturbation_run_hands_the_table_to_the_review(tmp_path):
    # A genes-by-cells matrix: columns are barcodes, so no column can be the
    # "label" and the name-matching inference finds nothing.
    path = tmp_path / "umi.txt"
    pd.DataFrame(
        [[1, 2, 3], [4, 5, 6]], columns=["AAAC-1", "AAAC-2", "AAAC-3"]
    ).to_csv(path, index=False)

    module = _module()
    target, column, decision = module.infer_for_candidates([str(path)], _args("perturbation"), Log())

    assert target == "" and column == ""
    assert decision["source"] == "none"


def test_a_per_cell_run_still_aborts_when_nothing_is_labeled(tmp_path):
    path = tmp_path / "umi.txt"
    pd.DataFrame(
        [[1, 2, 3], [4, 5, 6]], columns=["AAAC-1", "AAAC-2", "AAAC-3"]
    ).to_csv(path, index=False)

    module = _module()
    with pytest.raises(SystemExit) as exit_info:
        module.infer_for_candidates([str(path)], _args("per_cell"), Log())
    assert exit_info.value.code == 2
