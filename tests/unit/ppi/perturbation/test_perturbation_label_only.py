"""A gold with the perturbations but no transcriptome must say so.

The 2026-09-29 run gated `GSE153056_ECCITE_metadata.tsv.gz` as a screen (condition
column `gene`, controls `NT`) and then failed with "no perturbed cells with a
control baseline": its 21 numeric columns were QC metrics, so no perturbed gene
was ever in the panel. The error has to name that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kosmos.ppi.perturbation.run import run_perturbation_task


def test_a_label_only_gold_is_refused_with_a_clear_reason(tmp_path):
    frame = pd.DataFrame(
        {
            "condition": ["NT", "STAT1", "STAT1", "NFKBIA", "NT"],
            "nCount_RNA": [1000.0, 1200.0, 900.0, 1100.0, 950.0],
            "nFeature_RNA": [500.0, 540.0, 480.0, 520.0, 500.0],
            "percent_mt": [5.0, 4.0, 6.0, 4.5, 5.5],
        }
    )
    gold = tmp_path / "metadata.csv"
    frame.to_csv(gold, index=False)

    with pytest.raises(ValueError) as error:
        run_perturbation_task(
            gold_path=gold,
            out_dir=tmp_path / "run",
            condition_column="condition",
            go_reference=tmp_path / "missing-go.csv",
        )

    message = str(error.value)
    assert "has the labels and not the transcriptome" in message
    assert "STAT1" in message
