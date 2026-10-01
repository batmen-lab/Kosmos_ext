"""A screen the fetcher found: its labels, its measurements, and the panel.

GSE153056 is the case these tests are written from. The fetcher downloads the
label table (`GSE153056_ECCITE_metadata.tsv.gz`: one row per cell, the targeted
`gene`, controls `NT`) and, deep inside `GSE153056_RAW.tar`, the measurements
(`GSM4633614_ECCITE_cDNA_counts.tsv.gz`: genes in rows, cell barcodes in the
header). Neither half alone is trainable, and every failure between them looks
like "there is no data" rather than like the missing step it is:

  * the pair is found by *barcodes*, because the two files share nothing else;
  * the matrix is transposed, because that is the orientation GEO publishes;
  * a control is `NTg5` / `eGFPg1`, not only `NT`;
  * and the panel is capped, because 18,649 genes is a 348-million-edge graph.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from kosmos.domains.single_cell.tasks import profile_for  # noqa: E402
from kosmos.ppi.perturbation.assemble import (  # noqa: E402
    join_screen,
    pick_expression_partner,
)
from kosmos.ppi.perturbation.contract import is_control  # noqa: E402
from kosmos.ppi.perturbation.run import (  # noqa: E402
    guess_condition_column,
    normalise_counts,
    select_panel,
)

GENES = [f"G{i:02d}" for i in range(16)]
SINGLES = [f"G{i:02d}" for i in range(6)]
CONTROL = "NT"


def _halves(cells: int = 6, seed: int = 0):
    """The two halves of a screen: labels in one table, measurements in another."""
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
    barcodes = [f"BC{i:04d}" for i in range(len(conditions))]
    labels = pd.DataFrame(
        {"barcode": barcodes, "gene": conditions, "nCount_RNA": rng.integers(500, 2000, len(conditions))}
    )
    measured = pd.DataFrame(np.stack(expression), columns=GENES)
    measured.insert(0, "barcode", barcodes)
    return labels, measured


def test_a_genes_by_cells_matrix_joins_after_being_transposed():
    labels, measured = _halves()
    # what GEO publishes: genes in rows, barcodes in the header
    transposed = measured.set_index("barcode").T.reset_index()
    transposed = transposed.rename(columns={"index": ""})

    joined = join_screen(labels, transposed, condition_column="gene")

    assert len(joined) == len(labels)
    assert all(gene in joined.columns for gene in GENES)
    # the label table's own columns travel with the row
    assert "nCount_RNA" in joined.columns
    assert joined.attrs["expression_columns"][:2] == GENES[:2]
    assert joined.attrs["barcode_overlap"] == 1.0


def test_the_measurements_are_found_by_barcode_not_by_name(tmp_path):
    labels, measured = _halves()
    decoy = measured.copy()
    decoy["barcode"] = [f"OTHER{i}" for i in range(len(decoy))]
    labels.to_csv(tmp_path / "labels.tsv", sep="\t", index=False)
    labels_path = tmp_path / "GSE000000_metadata.tsv"
    counts_path = tmp_path / "GSM0000000_counts.tsv"    # nothing in common by name
    decoy_path = tmp_path / "wider_decoy_counts.tsv"     # wider, but other cells
    labels.to_csv(labels_path, sep="\t", index=False)
    measured.to_csv(counts_path, sep="\t", index=False)
    decoy.to_csv(decoy_path, sep="\t", index=False)

    # `min_expression_columns` is low because the fixture is 16 genes wide; the
    # rule under test is which table the *barcodes* pick, not the width.
    assert (
        pick_expression_partner(
            labels_path,
            [decoy_path, counts_path],
            min_expression_columns=4,
            min_expression_features=4,
        )
        == counts_path
    )
    # and a table that shares no barcode at all is not a partner
    assert (
        pick_expression_partner(
            labels_path, [decoy_path], min_expression_columns=4, min_expression_features=4
        )
        is None
    )


def test_a_control_may_carry_its_guide_index():
    # `NT` is the plain case; a screen writes the guide's index on the label
    assert is_control("NT")
    assert is_control("NTg5")
    assert is_control("eGFPg1")
    assert is_control("ctrl2")
    # a real gene that merely starts like a label is not a control
    assert not is_control("STAT2g1")
    assert not is_control("NTS")


def test_the_condition_hints_put_the_gene_before_the_guide():
    """A tie between hint-named columns must be broken towards the gene."""
    from kosmos.domains.single_cell.modality import CONDITION_HINTS

    assert CONDITION_HINTS.index("gene") < CONDITION_HINTS.index("guide_id")
    assert CONDITION_HINTS.index("gene") < CONDITION_HINTS.index("guide")


def test_the_condition_column_is_found_by_its_values(tmp_path):
    labels, _ = _halves()
    # ECCITE names its guide column `GO_lenti_maxID`, which no hint matches
    eccite = labels.rename(columns={"gene": "GO_lenti_maxID"})
    eccite["GO_lenti_maxID"] = eccite["GO_lenti_maxID"].replace({CONTROL: "NTg5"})
    eccite["orig.ident"] = "rep1"

    assert guess_condition_column(eccite) == "GO_lenti_maxID"
    # a hashtag/donor column with no control value is not a condition
    hashtags = pd.DataFrame({"hash.ID": [f"HTO{i}" for i in range(10)]})
    assert guess_condition_column(hashtags) is None


def test_the_panel_cap_keeps_the_perturbed_genes():
    rng = np.random.default_rng(0)
    genes = [f"G{i:03d}" for i in range(300)]
    values = rng.normal(size=(60, len(genes))).astype(np.float32)
    values[:, 5:] *= 0.01  # only the first few vary
    frame = pd.DataFrame(values, columns=genes)
    control_mask = pd.Series([True] * 20 + [False] * 40)

    selected, info = select_panel(
        frame, genes, control_mask=control_mask, always_include=["G299"], limit=50
    )

    assert len(selected) == 50
    assert "G299" in selected          # the quiet perturbed gene is kept
    assert info["measured"] == 300
    assert info["selected"] == 50
    assert info["perturbed_genes_kept"] == 1


def test_a_raw_count_matrix_is_normalised_and_a_processed_one_is_not():
    rng = np.random.default_rng(0)
    genes = [f"G{i:03d}" for i in range(600)]
    counts = rng.poisson(0.2, size=(40, len(genes))).astype(float)
    counted = pd.DataFrame(counts, columns=genes)

    normalised, verdict = normalise_counts(counted, genes)

    assert verdict["applied"] is True
    assert verdict["target_sum"] == pytest.approx(1e4)
    assert normalised[genes].to_numpy().max() <= np.log1p(1e4) + 1e-6
    # already-processed values (log-normalised, some negative) are left alone
    processed = pd.DataFrame(rng.normal(size=(40, len(genes))), columns=genes)
    same, verdict = normalise_counts(processed, genes)
    assert verdict["applied"] is False
    assert same is processed


def test_an_antibody_panel_measured_on_the_same_cells_is_not_measurements(tmp_path):
    """GSM4633615 (ADT, 5 features) shares every barcode with GSM4633614 (18,649 genes).

    Joining a label table to the antibody panel works exactly as well as joining
    it to the transcriptome -- the barcodes are the same -- and a graph built on
    `CD86` shares no gene with the gold. What tells them apart is the second
    axis: a feature panel is short in one direction.
    """
    from kosmos.ppi.perturbation.assemble import is_measurement_table

    labels, measured = _halves()
    genes_by_cells = measured.set_index("barcode").T.reset_index().rename(columns={"index": ""})
    genes_by_cells.to_csv(tmp_path / "cDNA_counts.tsv", sep="\t", index=False)
    adt = pd.DataFrame(
        {barcode: [1, 2, 3] for barcode in measured["barcode"]},
        index=["CD86", "PDL1", "CD47"],
    )
    adt.index.name = ""
    adt.to_csv(tmp_path / "ADT_counts.tsv", sep="\t")

    assert is_measurement_table(tmp_path / "cDNA_counts.tsv", min_expression_features=4)
    assert not is_measurement_table(tmp_path / "ADT_counts.tsv", min_expression_features=4)
    # the same cells, so the barcode test alone would have accepted the panel
    assert (
        pick_expression_partner(
            tmp_path / "labels.tsv",
            [tmp_path / "ADT_counts.tsv"],
            min_expression_columns=4,
            min_expression_features=4,
        )
        is None
    )


def test_a_guide_id_is_read_as_the_gene_it_targets():
    """`ATF2g1` and `ATF2g2` are two guides of one perturbation.

    A screen labels each cell with the guide it received; the panel, the
    perturbation embedding and the GO branch all speak gene symbols. Two guides
    of one gene must also land in *one* held-out group, or the split leaks.
    """
    from kosmos.ppi.perturbation.contract import build_examples, build_task, to_perturbed_gene
    from kosmos.ppi.perturbation.contract import is_control

    genes = ["G00", "G01", "G02"]
    frame = pd.DataFrame(
        {
            "guide": ["G00g1", "G00g2", "G01g3", "NTg1", "NTg2"],
            "G00": [1.0, 1.1, 0.0, 0.0, 0.0],
            "G01": [0.0, 0.0, 2.0, 0.0, 0.0],
            "G02": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    task = build_task(frame, condition_column="guide", gene_columns=genes)
    examples, report = build_examples(frame, task, min_cells_per_perturbation=1)

    assert sorted({example.label for example in examples}) == ["G00", "G01"]
    assert report["control_cells"] == 2
    # the mapping only fires when the panel measures the gene
    assert to_perturbed_gene("G00g1", {"G00"}) == "G00"
    assert to_perturbed_gene("BRD4g1", {"G00"}) == "BRD4g1"
    assert is_control("NTg1")


def test_the_gene_column_wins_over_the_guide_column():
    """ECCITE publishes `gene` and `guide_ID` side by side; the model needs `gene`."""
    from kosmos.ppi.perturbation.run import align_condition_column

    frame = pd.DataFrame(
        {
            "Unnamed: 0": [f"l1_BC{i:04d}AAAC" for i in range(20)],
            "guide_ID": [f"G{i % 3:02d}g1" for i in range(20)],
            "gene": [f"G{i % 3:02d}" for i in range(20)],
            "nCount_RNA": range(20),
        }
    )
    genes = {"G00", "G01", "G02"}

    assert align_condition_column(frame, genes, "guide_ID", ("nt",))[0] == "gene"
    assert align_condition_column(frame, genes, "gene", ("nt",))[0] == "gene"
    # and with no column named at all it still finds `gene`, not the barcodes
    assert align_condition_column(frame, genes, None, ("nt",))[0] == "gene"


def test_a_constant_prediction_has_no_perturbation_signal():
    """`top_k_overlap = 0` on a collapsed arm is a symptom, so the report names it.

    A predictor that returns the same profile for every perturbation still has a
    top-20 -- its per-gene output bias -- and that list has nothing to do with
    the perturbations. The signal statistic is what separates it from an arm
    that carries a fraction of the real between-perturbation difference.
    """
    from kosmos.ppi.perturbation.metrics import perturbation_signal

    rng = np.random.default_rng(0)
    labels = ["A", "B", "C"]
    observed = {label: rng.normal(0, 1, 50) for label in labels}
    constant = {label: np.full(50, 3.0) for label in labels}
    tracking = {label: value * 0.5 for label, value in observed.items()}

    assert perturbation_signal(constant, observed)["signal"] == 0.0
    assert perturbation_signal(tracking, observed)["signal"] == pytest.approx(0.5, abs=0.05)


def test_the_trivial_baselines_are_computed_on_the_same_metric():
    from kosmos.ppi.perturbation.metrics import baseline_metrics

    rng = np.random.default_rng(1)
    observed = {label: rng.normal(0, 0.3, 40) for label in ("A", "B", "C")}
    baselines = baseline_metrics(observed, top_k=5)

    assert set(baselines) == {"predict 0", "mean response (leave-one-out)"}
    # predicting nothing is exactly the mean squared observation
    expected = float(np.mean([np.mean(np.asarray(value) ** 2) for value in observed.values()]))
    assert baselines["predict 0"]["mse"] == pytest.approx(expected, rel=1e-6)
    assert baselines["predict 0"]["direction_accuracy"] == 0.0
    # the leave-one-out mean is not allowed to use the perturbation it predicts
    assert baselines["mean response (leave-one-out)"]["pearson"] > 0


def test_the_summary_reports_baselines_and_flags_a_constant_arm(tmp_path):
    from kosmos.ppi.perturbation.figures import write_summary

    def arm_payload(mse_deg, signal):
        return {
            "test": {
                "summary": {
                    "mse": 0.03,
                    "mse_deg": mse_deg,
                    "pearson": 0.2,
                    "spearman": 0.05,
                    "direction_accuracy": 0.8,
                    "top_k_overlap": 0.0,
                }
            },
            "epochs_run": 10,
            "signal": {
                "across_perturbation_std": 0.005,
                "observed_across_perturbation_std": 0.07,
                "signal": signal,
            },
        }

    results = {
        "task": {"n_genes": 1999, "condition_column": "gene"},
        "split_sizes": {"train": 1, "validation": 1, "test": 1},
        "arms": {
            "gears_base": arm_payload(0.35, 0.14),
            "mlp_base": arm_payload(0.34, 0.074),
        },
        "baselines": {
            "predict 0": {"mse": 0.02, "mse_deg": 0.69, "pearson": float("nan"), "direction_accuracy": 0.0, "top_k_overlap": 0.01},
            "mean response (leave-one-out)": {"mse": 0.02, "mse_deg": 0.46, "pearson": 0.36, "direction_accuracy": 0.65, "top_k_overlap": 0.34},
        },
    }

    text = write_summary(results, tmp_path).read_text()

    assert "## Reference baselines" in text
    assert "mean response (leave-one-out)" in text
    assert "No perturbation signal" in text
    assert "`mlp_base`" in text


def test_a_stopped_run_still_leaves_a_report(tmp_path):
    """An arm is an hour of work; stopping the run must not throw its numbers away."""
    from kosmos.ppi.perturbation.figures import write_summary

    results = {
        "task": {"n_genes": 1999, "condition_column": "gene"},
        "split_sizes": {"train": 13244, "validation": 1682, "test": 3417},
        "arms": {
            "gears_base": {
                "test": {
                    "summary": {
                        "mse": 0.0377,
                        "mse_deg": 0.3553,
                        "pearson": 0.173,
                        "spearman": 0.045,
                        "direction_accuracy": 0.78,
                        "top_k_overlap": 0.23,
                    }
                },
                "epochs_run": 50,
            }
        },
        "partial": True,
    }

    path = write_summary(results, tmp_path)
    text = path.read_text()

    assert "gears_base" in text
    assert "Partial run" in text
    assert "0.3553" in text


def test_a_perturbation_fetch_unpacks_the_whole_archive():
    """The counts matrix is the fifteenth member of GSE153056's RAW.tar."""
    profile = profile_for("perturbation")
    assert profile.archive_members is not None and profile.archive_members >= 22
    assert profile.files_per_fetch is not None and profile.files_per_fetch >= 22
    # the per-cell task keeps the generic caps
    assert profile_for("per_cell").archive_members is None
