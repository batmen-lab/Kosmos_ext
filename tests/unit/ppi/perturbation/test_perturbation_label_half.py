"""Perturbation needs a condition column and a matrix -- not necessarily one file."""

from __future__ import annotations

from kosmos.domains.single_cell.tasks import _perturbation_extract
from kosmos.ppi.perturbation.assemble import label_half_condition


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_a_headerless_barcode_to_guide_table_is_a_label_half(tmp_path):
    lines = []
    for i in range(60):
        barcode = f"AAACCTG{i:08d}"
        guide = ["STAT1g2", "JAK2g1", "NT", "IRF1g1"][i % 4]
        lines.append(f"{barcode}\t{guide}")
    path = _write(
        tmp_path / "GSM1_GDO_Barcodes.csv",
        "\n".join(lines) + "\n",
    )

    detected = label_half_condition(path)

    assert detected is not None
    column, controls = detected
    # headerless: pandas names the column after the first data row's value
    assert column in {"STAT1g2", "JAK2g1", "NT", "IRF1g1"}
    assert "NT" in controls


def test_the_condition_column_values_look_like_perturbations(tmp_path):
    path = _write(
        tmp_path / "labels.tsv",
        "cell\tcondition\n"
        "b1\tSTAT1g2\n"
        "b2\tSTAT1g2\n"
        "b3\tNT\n"
        "b4\tJAK2g1\n",
    )

    detected = label_half_condition(path)

    assert detected is not None
    assert detected[0] == "condition"
    assert "NT" in detected[1]


def test_a_wide_measurement_matrix_is_not_a_label_half(tmp_path):
    header = ["gene"] + [f"cell{i}" for i in range(800)]
    rows = ["\t".join(header)]
    rows += ["\t".join(["GENE1"] + ["1"] * 800), "\t".join(["GENE2"] + ["0"] * 800)]
    path = _write(tmp_path / "counts.tsv", "\n".join(rows) + "\n")

    assert label_half_condition(path) is None


def test_a_barcode_to_protein_table_is_not_a_label_half(tmp_path):
    path = _write(
        tmp_path / "ADT_Barcodes.csv",
        "AAACCTGAGCCAGAAC\tCD86\n"
        "AAACCTGAGTGGACGT\tPDL1\n"
        "AAACCTGCATGAGCGA\tCD366\n",
    )

    assert label_half_condition(path) is None


def test_a_condition_bearing_table_supervises_without_measurements():
    fields = _perturbation_extract(
        {
            "condition_column": "gene",
            "is_screen": False,
            "has_expression": False,
            "control_labels": ["NT"],
            "modality": "crisprko",
        },
        ["gene", "guide_ID"],
    )

    # It is the screen's label half, not a full screen -- still the gold's labels.
    assert fields["promote_role"] == "gold"
    assert fields["force_role"] is None


def test_a_table_with_neither_half_is_unusable():
    fields = _perturbation_extract(
        {"condition_column": None, "is_screen": False, "has_expression": False},
        ["x", "y"],
    )

    assert fields["force_role"] == "unusable"
    assert fields["promote_role"] is None


def test_a_measurement_matrix_is_kept_as_evidence():
    fields = _perturbation_extract(
        {"condition_column": None, "is_screen": False, "has_expression": True},
        [f"g{i}" for i in range(50)],
    )

    # It is not gold (no condition), but it is the evidence a perturbation
    # correction consumes -- not something to throw away.
    assert fields["promote_role"] == "supplementary"
