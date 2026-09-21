"""Single-cell files become bounded tables: chosen cells, chosen genes, chosen obs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datafetcher.singlecell import (
    SingleCellSelection,
    convert_single_cell,
    is_single_cell,
)

CELLS = 400
GENES = 30


def write_h5ad(path, *, with_label: bool = True):
    """A minimal AnnData-shaped HDF5 file: dense X, a categorical obs column."""
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(0)
    types = np.array(["T cell", "B cell", "NK cell", "rare"])[
        rng.integers(0, 4, CELLS)
    ]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=rng.random((CELLS, GENES), dtype=np.float32))
        obs = handle.create_group("obs")
        if with_label:
            group = obs.create_group("cell_type")
            categories = np.array(["B cell", "NK cell", "T cell", "rare"], dtype=object)
            group.create_dataset(
                "categories", data=np.array([c.encode() for c in categories])
            )
            index = {c: i for i, c in enumerate(categories)}
            group.create_dataset(
                "codes", data=np.array([index[t] for t in types], dtype=np.int32)
            )
        obs.create_dataset("donor_id", data=np.array([b"donor-A"] * CELLS))
        handle.create_group("var").create_dataset(
            "_index", data=np.array([f"G{i}".encode() for i in range(GENES)])
        )
    return path, types


def write_triplet(directory, *, with_metadata: bool = True):
    directory.mkdir(parents=True, exist_ok=True)
    types = np.array(["T cell", "B cell", "NK cell", "rare"])[
        np.random.default_rng(0).integers(0, 4, CELLS)
    ]
    with (directory / "matrix.mtx").open("w") as out:
        out.write("%%MatrixMarket matrix coordinate real general\n")
        out.write(f"{GENES} {CELLS} {CELLS}\n")
        for cell in range(CELLS):
            out.write(f"{(cell % GENES) + 1} {cell + 1} {float(cell % 7 + 1)}\n")
    (directory / "barcodes.tsv").write_text(
        "\n".join(f"bc{cell}" for cell in range(CELLS))
    )
    (directory / "features.tsv").write_text(
        "\n".join(f"G{i}\tGENE{i}" for i in range(GENES))
    )
    if with_metadata:
        (directory / "metadata.csv").write_text(
            "barcode,cell_type\n"
            + "\n".join(f"bc{cell},{types[cell]}" for cell in range(CELLS))
        )
    return directory / "matrix.mtx", types


def test_suffixes_are_recognised(tmp_path):
    assert is_single_cell("x/sample.h5ad")
    assert is_single_cell("x/matrix.mtx.gz")
    assert not is_single_cell("x/table.csv")


def write_modern_h5ad(path):
    """The layout a current AnnData writes: sparse X group, __categories obs.

    GSE194122's file is 90,261 x 14,087 with 141M nonzeros; this is that shape
    in miniature, and it is what the reader has to accept.
    """
    h5py = pytest.importorskip("h5py")
    import scipy.sparse as sp

    rng = np.random.default_rng(1)
    matrix = sp.random(CELLS, GENES, density=0.1, format="csr", random_state=1, dtype=np.float32)
    matrix.data = np.abs(matrix.data) + 0.1
    types = np.array(["CD14+ Mono", "CD4+ T", "NK", "rare"])[rng.integers(0, 4, CELLS)]
    levels = np.array(["CD14+ Mono", "CD4+ T", "NK", "rare"], dtype=object)
    with h5py.File(path, "w") as handle:
        x = handle.create_group("X")
        x.attrs["encoding-type"] = "csr_matrix"
        x.attrs["shape"] = np.array(matrix.shape, dtype=np.int64)
        x.create_dataset("data", data=matrix.data)
        x.create_dataset("indices", data=matrix.indices.astype(np.int32))
        x.create_dataset("indptr", data=matrix.indptr.astype(np.int32))
        obs = handle.create_group("obs")
        index = {level: i for i, level in enumerate(levels)}
        obs.create_dataset(
            "cell_type", data=np.array([index[t] for t in types], dtype=np.int8)
        )
        categories = obs.create_group("__categories")
        categories.create_dataset(
            "cell_type", data=np.array([level.encode() for level in levels])
        )
        obs.create_dataset("donor_id", data=np.array([b"donor-1"] * CELLS))
        handle.create_group("var").create_dataset(
            "_index", data=np.array([f"G{i}".encode() for i in range(GENES)])
        )
    return path, types


def test_the_layout_a_current_anndata_writes_is_read(tmp_path):
    """`X` as a CSR group with its shape in an attribute, levels in __categories."""
    path, types = write_modern_h5ad(tmp_path / "modern.h5ad")
    table, selection = convert_single_cell(
        path, selection=SingleCellSelection(max_cells=120, max_genes=6)
    )

    assert len(table) == 120
    assert {"cell_type", "donor_id"} <= set(table.columns)
    assert set(table["cell_type"]) <= set(types)
    assert len([c for c in table.columns if c.startswith("G")]) == 6
    assert any("14,087" not in note for note in selection.notes)  # small file, big file tested live


def test_an_h5ad_becomes_a_bounded_table(tmp_path):
    path, types = write_h5ad(tmp_path / "toy.h5ad")
    table, selection = convert_single_cell(
        path, selection=SingleCellSelection(max_cells=100, max_genes=5)
    )

    assert table.shape == (100, 7)  # 5 genes + cell_type + donor_id
    gene_columns = [c for c in table.columns if c.startswith("G")]
    assert len(gene_columns) == 5
    assert set(table["donor_id"]) == {"donor-A"}
    # Stratified: the rare type is small but not sampled away.
    assert "rare" in set(table["cell_type"])
    assert set(table["cell_type"]) == set(types)
    assert any("wrote 100 cells x 5 genes" in note for note in selection.notes)


def test_an_h5ad_without_the_label_column_is_still_a_table(tmp_path):
    """Unlabeled single-cell data is exactly what the evidence round wants."""
    path, _ = write_h5ad(tmp_path / "unlabeled.h5ad", with_label=False)
    table, selection = convert_single_cell(
        path, selection=SingleCellSelection(max_cells=50, max_genes=3)
    )

    assert "cell_type" not in table.columns
    assert any("has no 'cell_type' column" in note for note in selection.notes)


def test_a_perturbation_table_keeps_the_column_that_labels_it(tmp_path):
    """The drug lives in `compound_1`/`condition`, not in `cell_type`.

    Carrying only the fixed id list dropped those at conversion time, so
    `--hint compound_1` could never match: the column was gone before the
    question was asked. Numbers stay behind -- a dose is a measurement, and a
    measurement beside the gene panel is one more feature, not a label.
    """
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "chempert.h5ad"
    doses = np.array([0.1, 0.2, 0.4, 0.8] * (CELLS // 4), dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=np.random.default_rng(0).random((CELLS, GENES), dtype=np.float32))
        obs = handle.create_group("obs")
        for name, values in (
            ("compound_1", ["vehicle", "drugA", "drugB", "drugC"] * (CELLS // 4)),
            ("batch", [f"b{i % 3}" for i in range(CELLS)]),
        ):
            levels = sorted(set(values))
            group = obs.create_group(name)
            group.create_dataset("categories", data=np.array([v.encode() for v in levels]))
            index = {level: i for i, level in enumerate(levels)}
            group.create_dataset(
                "codes", data=np.array([index[v] for v in values], dtype=np.int32)
            )
        obs.create_dataset("dose", data=doses)
        obs.create_dataset("cell_id", data=np.array([f"c{i}".encode() for i in range(CELLS)]))
        handle.create_group("var").create_dataset(
            "_index", data=np.array([f"G{i}".encode() for i in range(GENES)])
        )

    table, selection = convert_single_cell(
        path, selection=SingleCellSelection(max_cells=80, max_genes=4)
    )

    assert "compound_1" in table.columns
    assert set(table["compound_1"]) <= {"vehicle", "drugA", "drugB", "drugC"}
    assert selection.facts is not None
    assert "compound_1" in selection.facts.metadata_columns
    # A measurement is not a label, and a per-cell identifier is not a label.
    assert "dose" not in table.columns
    assert "cell_id" not in table.columns
    assert any("could label a row" in note for note in selection.notes)


def test_a_triplet_with_metadata_becomes_a_labeled_table(tmp_path):
    matrix, types = write_triplet(tmp_path / "sample1")
    table, selection = convert_single_cell(
        matrix,
        selection=SingleCellSelection(max_cells=80, max_genes=4),
        metadata=matrix.parent / "metadata.csv",
    )

    assert table.shape == (80, 6)  # 4 genes + barcode + cell_type
    assert set(table["cell_type"]) <= set(types)
    assert table["barcode"].is_unique
    assert any("labels from metadata.csv" in note for note in selection.notes)


def test_a_triplet_without_metadata_is_evidence_not_gold(tmp_path):
    matrix, _ = write_triplet(tmp_path / "sample2", with_metadata=False)
    table, selection = convert_single_cell(
        matrix, selection=SingleCellSelection(max_cells=40, max_genes=3)
    )

    assert "cell_type" not in table.columns
    assert "barcode" in table.columns
    assert any("no metadata table" in note for note in selection.notes)


def test_an_mtx_without_its_siblings_is_refused(tmp_path):
    lonely = tmp_path / "matrix.mtx"
    lonely.write_text("%%MatrixMarket matrix coordinate real general\n2 2 1\n1 1 1.0\n")
    with pytest.raises(ValueError, match="barcodes/features"):
        convert_single_cell(lonely)


def test_a_fetch_registers_the_converted_table(tmp_path):
    """The fetcher converts on the way in, and records what it came from."""
    import importlib.util
    from pathlib import Path as _Path

    from datafetcher.config import DataFetcherConfig
    from datafetcher.fetch import fetch

    root = _Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "auto_task_run", root / "scripts" / "auto_task_run.py"
    )
    auto_task_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auto_task_run)

    path, _ = write_h5ad(tmp_path / "cells.h5ad")
    config = DataFetcherConfig(
        root=tmp_path / "staged", single_cell_max_cells=60, single_cell_max_genes=4
    )

    result = fetch(f"file://{path}", config=config)

    derived = [f for f in result.files if f.derived_from == "cells.h5ad"]
    assert [f.path for f in derived] == ["cells.h5ad-table.csv"]
    table = pd.read_csv(f"{result.directory}/{derived[0].path}")
    assert len(table) == 60
    assert "cell_type" in table.columns
    payload = {
        "directory": result.directory,
        "files": [
            {"path": f.path, "sha256": f.sha256, "derived_from": f.derived_from}
            for f in result.files
        ],
    }
    candidates = [record["path"] for record in auto_task_run._candidate_records(payload)]
    assert "cells.h5ad" in candidates  # the original stays visible
    assert "cells.h5ad-table.csv" in candidates  # and so does the table


def test_genes_are_not_capped_by_default(tmp_path):
    """Cells are the bounded dimension; the gene panel is the data.

    200 genes out of 14,087 is a different dataset, not a cheaper read of the
    same one: the model then learns from a panel somebody picked for it.
    """
    path, _ = write_h5ad(tmp_path / "all-genes.h5ad")
    table, selection = convert_single_cell(path)

    assert selection.max_genes == 0
    assert len([c for c in table.columns if c.startswith("G")]) == GENES
    assert selection.facts.genes_written == GENES
    assert selection.facts.genes_were_capped is False
    assert "the panel is not cut" in selection.facts.precheck_note()


def test_the_facts_say_how_much_of_the_file_the_table_is(tmp_path):
    path, _ = write_h5ad(tmp_path / "cells.h5ad")
    _, selection = convert_single_cell(path, selection=SingleCellSelection(max_cells=50))

    facts = selection.facts
    assert facts.source == "cells.h5ad"
    assert facts.source_cells == CELLS
    assert facts.cells_written == 50
    assert facts.cells_were_capped is True
    assert facts.genes_written == GENES
    assert facts.label_column == "cell_type"
    assert facts.label_present is True
    assert "50 of 400 cells" in facts.describe()
    assert "30 of 30 genes" in facts.describe()


def test_the_conversion_facts_travel_with_the_table(tmp_path):
    """The manifest records the sample, so the pre-check can say it is one."""
    from datafetcher.config import DataFetcherConfig
    from datafetcher.fetch import fetch
    from datafetcher.sample import sample_packet

    path, _ = write_h5ad(tmp_path / "cells.h5ad")
    config = DataFetcherConfig(root=tmp_path / "staged", single_cell_max_cells=40)
    result = fetch(f"file://{path}", config=config)

    derived = next(f for f in result.files if f.derived_from == "cells.h5ad")
    assert derived.conversion["cells_written"] == 40
    assert derived.conversion["source_cells"] == CELLS
    assert derived.conversion["genes_written"] == GENES

    packet = sample_packet(f"{result.directory}/{derived.path}", rows=2)
    assert packet["provenance"]["derived_from"] == "cells.h5ad"
    assert any("the panel is not cut" in note for note in packet["notes"])
    assert "40 of 400 cells" in packet["rendered"]
