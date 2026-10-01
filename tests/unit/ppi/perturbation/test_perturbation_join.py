"""A screen published as two files: labels in one, measurements in the other.

GSE153056 is a label table (barcodes plus the targeted `gene`, controls `NT`) and
an archive of counts. Neither half is a training table: the first has no
features, the second no labels. They are joined on the barcode before training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from kosmos.ppi.perturbation.assemble import (  # noqa: E402
    find_overlap,
    join_screen,
    numeric_columns,
    pick_expression_table,
)
from kosmos.ppi.perturbation.run import run_perturbation_task  # noqa: E402
from kosmos.ppi.perturbation.train import PerturbationTrainingConfig  # noqa: E402

GENES = [f"G{i:02d}" for i in range(16)]
SINGLES = [f"G{i:02d}" for i in range(6)]
CONTROL = "NT"


def _screen(cells: int = 6, seed: int = 0):
    """Labels and measurements as two files, sharing a barcode."""
    rng = np.random.default_rng(seed)
    conditions, expression = [], []
    for index, gene in enumerate(SINGLES):
        for _ in range(cells):
            value = rng.normal(0, 0.3, len(GENES))
            value[index] -= 1.5
            conditions.append(gene)
            expression.append(value)
    for _ in range(cells):
        conditions.append(CONTROL)
        expression.append(rng.normal(0, 0.3, len(GENES)))
    expression = np.stack(expression)
    barcodes = [f"BC{i:04d}-1" for i in range(len(conditions))]
    labels = pd.DataFrame({"barcode": barcodes, "condition": conditions,
                           "nCount_RNA": rng.integers(500, 2000, len(conditions))})
    measured = pd.DataFrame(expression, columns=GENES)
    measured.insert(0, "barcode", barcodes)
    return labels, measured


def test_the_two_halves_join_on_the_barcode(tmp_path):
    labels, measured = _screen()
    joined = join_screen(labels, measured, condition_column="condition")

    assert set(measured.columns[:1]) == {"barcode"}
    assert "condition" in joined.columns
    assert all(gene in joined.columns for gene in GENES)
    assert len(joined) == len(labels)
    # the label table's own columns travel with the row
    assert "nCount_RNA" in joined.columns


def test_a_barcode_suffix_does_not_break_the_join(tmp_path):
    labels, measured = _screen()
    measured["barcode"] = measured["barcode"].str.replace("-1", "", regex=False)

    key, shared, fraction = find_overlap(labels, measured)
    assert key == "barcode"
    assert fraction == 1.0


def test_a_table_without_the_labels_cannot_supervise_alone(tmp_path):
    _, measured = _screen()
    with pytest.raises(ValueError):
        join_screen(measured, measured, condition_column="condition")


def test_the_measurements_are_picked_by_column_count(tmp_path):
    labels, _ = _screen()
    # a real measurement table is wide; 80 genes stands in for the thousands
    wide = pd.DataFrame(np.random.default_rng(0).normal(size=(len(labels), 80)),
                        columns=[f"G{i:03d}" for i in range(80)])
    wide.insert(0, "barcode", labels["barcode"])
    labels_path, measured_path = tmp_path / "labels.csv", tmp_path / "counts.csv"
    labels.to_csv(labels_path, index=False)
    wide.to_csv(measured_path, index=False)

    assert numeric_columns(labels_path) <= 2
    assert pick_expression_table(labels_path, [measured_path]) == measured_path


def test_nothing_is_picked_when_the_gold_already_has_the_measurements(tmp_path):
    _, measured = _screen()
    measured_path = tmp_path / "counts.csv"
    measured.to_csv(measured_path, index=False)

    assert pick_expression_table(measured_path, []) is None


def test_the_backend_trains_on_the_joined_pair(tmp_path):
    labels, measured = _screen()
    labels_path, measured_path = tmp_path / "labels.csv", tmp_path / "counts.csv"
    labels.to_csv(labels_path, index=False)
    measured.to_csv(measured_path, index=False)

    results = run_perturbation_task(
        gold_path=labels_path,
        expression_path=measured_path,
        supplementary_paths=[measured_path],
        out_dir=tmp_path / "run",
        condition_column="condition",
        go_reference=tmp_path / "missing-go.csv",
        split_mode="mixed",
        test_fraction=0.34,
        validation_fraction=0.17,
        config=PerturbationTrainingConfig(
            epochs=1, patience=1, batch_size=8, embedding_dim=8, hidden_dim=8,
            gnn_layers=1, top_k_deg=4, seed=0,
        ),
    )

    assert set(results["arms"])  # the three GEARS arms trained
    contract = (tmp_path / "run" / "perturbation_contract.json").read_text()
    assert "condition" in contract
